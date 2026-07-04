//! DEFLATE block-scanning driver.
//!
//! Decodes each BTYPE 0/1/2 block, records every literal and back-reference
//! as an [`Event`], and keeps per-block encoder tables so writers can re-emit
//! edits without rebuilding Huffman trees.

use crate::bitstream::BitReader;

use super::constants::{CLORDER, DBASE, DEXT, LBASE, LEXT};
use super::error::DecodeError;
use super::events::{EncTable, Event, LitEvent, RefEvent};
use super::huffman::{HuffmanTable, build_tree, decode_sym, fixed_code_lengths};

#[derive(Debug)]
pub struct DecodedDeflate {
    pub output: Vec<u8>,
    pub events: Vec<Event>,
    pub lit_encs: Vec<EncTable>,
    pub dist_encs: Vec<EncTable>,
    /// `block_starts[b]` is the index into `events` of block `b`'s first
    /// event. A block owns the half-open event range
    /// `block_starts[b] .. block_starts[b + 1]` (the last block runs to
    /// `events.len()`). Block membership is a range recorded once per
    /// block, not a field on every event. Look up an event's block with
    /// [`block_of`](super::block_of).
    pub block_starts: Vec<u32>,
    /// Largest LZ77 back-reference distance found in `events`, or 1 if
    /// no back-references were emitted. Cached here so consumers (e.g.
    /// the distance-overlay colour scaler) don't need a second event scan.
    pub max_distance: u32,
}

impl DecodedDeflate {
    /// Number of DEFLATE blocks in the stream, one `block_starts` entry
    /// per block.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.block_starts.len()
    }
}

/// Receives one call per literal / back-reference the decoder emits. The
/// full [`decode_deflate`] path collects them into a `Vec<Event>`; the
/// pixels-only [`inflate`] path uses [`NoEvents`], whose no-op methods let
/// the compiler dead-code-eliminate the event construction so the shared
/// byte-producing loop carries zero recording overhead on the fast path.
trait EventSink {
    fn lit(&mut self, ev: LitEvent);
    fn reference(&mut self, ev: RefEvent);
    /// Events recorded so far: the index a new block starts at.
    fn count(&self) -> u32;
}

impl EventSink for Vec<Event> {
    #[inline]
    fn lit(&mut self, ev: LitEvent) {
        self.push(Event::Lit(ev));
    }
    #[inline]
    fn reference(&mut self, ev: RefEvent) {
        self.push(Event::Ref(ev));
    }
    #[inline]
    fn count(&self) -> u32 {
        self.len() as u32
    }
}

/// Discarding sink for [`inflate`]: every method is a no-op the optimiser
/// erases, along with the [`LitEvent`] / [`RefEvent`] built to feed it.
struct NoEvents;

impl EventSink for NoEvents {
    #[inline]
    fn lit(&mut self, _ev: LitEvent) {}
    #[inline]
    fn reference(&mut self, _ev: RefEvent) {}
    #[inline]
    fn count(&self) -> u32 {
        0
    }
}

/// Mutable state threaded through the per-block decoders: output buffer,
/// event sink, running max-distance, and the output cap.
struct DecodeState<'a, E: EventSink> {
    output: &'a mut Vec<u8>,
    events: &'a mut E,
    max_distance: &'a mut u32,
    max_output: Option<usize>,
}

impl<E: EventSink> DecodeState<'_, E> {
    /// Bail when extending `output` by `additional` bytes would exceed
    /// the cap. Inlined into the literal- and ref-emit hot paths.
    #[inline]
    fn check_room(&self, additional: usize) -> Result<(), DecodeError> {
        if let Some(max) = self.max_output
            && self.output.len() + additional > max
        {
            return Err(DecodeError::OutputTooLarge {
                decoded: self.output.len() + additional,
                max,
            });
        }
        Ok(())
    }
}

/// The non-event products of a decode: the inflated bytes plus the per-
/// block Huffman tables, block boundaries, and max back-ref distance that
/// [`decode_deflate`] surfaces (and [`inflate`] discards).
struct Inflated {
    output: Vec<u8>,
    lit_encs: Vec<EncTable>,
    dist_encs: Vec<EncTable>,
    block_starts: Vec<u32>,
    max_distance: u32,
}

/// Shared decode core, generic over the event sink so the recording and
/// pixels-only paths share one implementation of every block type and
/// every RFC 1951 validity check.
fn decode_into<E: EventSink>(
    data: &[u8],
    max_output: Option<usize>,
    events: &mut E,
) -> Result<Inflated, DecodeError> {
    let mut reader = BitReader::new(data);
    // Output grows on demand. Up-front preallocation sized from the deflate
    // length measures slower on typical photo PNGs; wasted capacity hurts
    // L2 reuse more than the doublings cost.
    let mut output: Vec<u8> = Vec::new();
    let mut lit_encs: Vec<EncTable> = Vec::new();
    let mut dist_encs: Vec<EncTable> = Vec::new();
    let mut block_starts: Vec<u32> = Vec::new();
    let mut max_distance = 1u32;
    let mut block_idx = 0u32;

    let mut state = DecodeState {
        output: &mut output,
        events,
        max_distance: &mut max_distance,
        max_output,
    };

    loop {
        // Each block owns the event range starting here.
        block_starts.push(state.events.count());
        let is_final = reader.read_bits(1);
        let block_type = reader.read_bits(2);

        match block_type {
            0b00 => {
                decode_stored_block(&mut reader, &mut state, block_idx)?;
                // A stored block carries no Huffman alphabet, but block →
                // table lookups still index by block, so push placeholder
                // empty tables sized to the standard alphabets to keep the
                // per-block indices aligned.
                lit_encs.push(EncTable::new(288));
                dist_encs.push(EncTable::new(30));
            }
            0b01 | 0b10 => {
                let (lit_lengths, dist_lengths) =
                    read_huffman_code_lengths(&mut reader, block_type)?;
                let (lit_tab, lit_enc) = build_tree(&lit_lengths)?;
                let (dist_tab, dist_enc) = build_tree(&dist_lengths)?;
                lit_encs.push(lit_enc);
                dist_encs.push(dist_enc);

                decode_huffman_block(&mut reader, &mut state, block_idx, &lit_tab, &dist_tab)?;
            }
            _ => return Err(DecodeError::ReservedBlockType { block_idx }),
        }

        block_idx += 1;
        if is_final != 0 {
            break;
        }
    }

    Ok(Inflated {
        output,
        lit_encs,
        dist_encs,
        block_starts,
        max_distance,
    })
}

/// Decode a raw deflate stream, recording every literal and back-reference
/// as an [`Event`] and keeping the per-block Huffman tables: the glass-box
/// path. `max_output` caps the inflated size in bytes (`None` for
/// unbounded); the loader passes the IHDR-derived expected output so a
/// malicious IDAT can't pump the decoder into gigabytes of allocation.
///
/// Use [`inflate`] when you only need the decompressed bytes.
pub fn decode_deflate(
    data: &[u8],
    max_output: Option<usize>,
) -> Result<DecodedDeflate, DecodeError> {
    let mut events: Vec<Event> = Vec::new();
    let inflated = decode_into(data, max_output, &mut events)?;
    Ok(DecodedDeflate {
        output: inflated.output,
        events,
        lit_encs: inflated.lit_encs,
        dist_encs: inflated.dist_encs,
        block_starts: inflated.block_starts,
        max_distance: inflated.max_distance,
    })
}

/// Decompress a raw deflate stream to its output bytes only: no event log,
/// no per-symbol allocation. The lean path behind
/// [`crate::decode`]. `max_output` behaves as in [`decode_deflate`].
pub fn inflate(data: &[u8], max_output: Option<usize>) -> Result<Vec<u8>, DecodeError> {
    let mut sink = NoEvents;
    Ok(decode_into(data, max_output, &mut sink)?.output)
}

fn decode_stored_block<E: EventSink>(
    reader: &mut BitReader,
    state: &mut DecodeState<'_, E>,
    block_idx: u32,
) -> Result<(), DecodeError> {
    reader.align_to_byte();
    let length = (reader.read_bits(8) | (reader.read_bits(8) << 8)) as u16;
    let nlen = (reader.read_bits(8) | (reader.read_bits(8) << 8)) as u16;
    // RFC 1951 §3.2.4: NLEN is the one's complement of LEN. Without this
    // check, a corrupt stored block silently emits up to 65535 garbage
    // literal events.
    if length ^ nlen != 0xFFFF {
        return Err(DecodeError::InvalidStoredBlockLength {
            length,
            nlen,
            block_idx,
        });
    }
    state.check_room(length as usize)?;
    for _ in 0..length {
        let bit_start = reader.bit_pos() as u32;
        let byte_val = reader.read_bits(8) as u8;
        let out_pos = state.output.len() as u32;
        state.events.lit(LitEvent {
            out_pos,
            symbol: byte_val,
            bit_start,
        });
        state.output.push(byte_val);
    }
    Ok(())
}

/// Parse either the fixed Huffman table (BTYPE=01) or the dynamic
/// per-block length codes + repeats (BTYPE=10). Returns the length arrays
/// ready to feed to [`build_tree`].
fn read_huffman_code_lengths(
    reader: &mut BitReader,
    block_type: u32,
) -> Result<(Vec<u32>, Vec<u32>), DecodeError> {
    if block_type == 0b01 {
        return Ok(fixed_code_lengths());
    }

    let hlit = reader.read_bits(5) + 257;
    let hdist = reader.read_bits(5) + 1;
    let hclen = reader.read_bits(4) + 4;

    let mut cl_lengths = vec![0u32; 19];
    for i in 0..hclen as usize {
        cl_lengths[CLORDER[i]] = reader.read_bits(3);
    }
    let (cl_tab, _) = build_tree(&cl_lengths)?;

    let total = (hlit + hdist) as usize;
    let mut all_lengths: Vec<u32> = Vec::with_capacity(total);
    while all_lengths.len() < total {
        let s = decode_sym(reader, &cl_tab)?;
        match s {
            0..=15 => all_lengths.push(s as u32),
            16 => {
                let last = all_lengths.last().copied().unwrap_or(0);
                let rep = (reader.read_bits(2) + 3) as usize;
                all_lengths.extend(std::iter::repeat_n(last, rep));
            }
            17 => {
                let rep = (reader.read_bits(3) + 3) as usize;
                all_lengths.extend(std::iter::repeat_n(0, rep));
            }
            _ => {
                let rep = (reader.read_bits(7) + 11) as usize;
                all_lengths.extend(std::iter::repeat_n(0, rep));
            }
        }
    }
    // First `hlit` entries are literal-alphabet code lengths, the rest
    // are distance-alphabet code lengths. `split_off` hands the tail to a
    // new Vec and shrinks the original: one allocation rather than two.
    let dl = all_lengths.split_off(hlit as usize);
    let ll = all_lengths;
    Ok((ll, dl))
}

fn decode_huffman_block<E: EventSink>(
    reader: &mut BitReader,
    state: &mut DecodeState<'_, E>,
    block_idx: u32,
    lit: &HuffmanTable,
    dist: &HuffmanTable,
) -> Result<(), DecodeError> {
    loop {
        let sym_bit_start = reader.bit_pos();
        let sym = decode_sym(reader, lit)?;

        if sym == 256 {
            return Ok(());
        }

        if sym < 256 {
            state.check_room(1)?;
            let out_pos = state.output.len() as u32;
            state.events.lit(LitEvent {
                out_pos,
                symbol: sym as u8,
                bit_start: sym_bit_start as u32,
            });
            state.output.push(sym as u8);
        } else {
            let len_idx = (sym - 257) as usize;
            if len_idx >= LBASE.len() {
                return Err(DecodeError::InvalidLengthSymbol { sym, block_idx });
            }
            let copy_len = LBASE[len_idx] + reader.read_bits(LEXT[len_idx]);

            let dist_bit_start = reader.bit_pos() as u32;
            let dist_sym_u16 = decode_sym(reader, dist)?;
            // RFC 1951: distance alphabet is 30 symbols (0..=29). Symbols
            // 30 and 31 are reserved and must not occur in valid data;
            // `DBASE`/`DEXT` only define 30 entries, so without this
            // guard a malformed stream panics with an out-of-bounds index.
            if dist_sym_u16 >= 30 {
                return Err(DecodeError::InvalidDistanceSymbol {
                    sym: dist_sym_u16,
                    block_idx,
                });
            }
            let dist_sym = dist_sym_u16 as u8;
            let distance = DBASE[dist_sym as usize] + reader.read_bits(DEXT[dist_sym as usize]);

            if distance as usize > state.output.len() {
                return Err(DecodeError::InvalidDistance {
                    distance,
                    output_len: state.output.len(),
                    block_idx,
                });
            }
            state.check_room(copy_len as usize)?;
            if distance > *state.max_distance {
                *state.max_distance = distance;
            }
            let src_start = state.output.len() - distance as usize;

            let out_pos = state.output.len() as u32;
            state.events.reference(RefEvent {
                out_pos,
                src_out_pos: src_start as u32,
                copy_len: copy_len as u16,
                dist_sym,
                dist_bit_start,
            });

            // When distance < copy_len the source range overlaps the
            // destination, and each push extends the bytes a later
            // iteration will read: the run-length-encoding pattern that
            // makes "copy 64 bytes from one byte back" expand a single
            // byte into 64. A bulk `copy_within` / `copy_from_slice`
            // would snapshot the source slice and lose that semantic.
            for i in 0..copy_len as usize {
                let byte_val = state.output[src_start + i];
                state.output.push(byte_val);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::write_bits;

    #[test]
    fn round_trips_single_stored_block() {
        // BFINAL=1, BTYPE=00, align, LEN=2 ("hi").
        let mut buf = vec![0u8; 32];
        write_bits(&mut buf, 0, 1, 1); // BFINAL
        buf[1] = 2;
        buf[2] = 0;
        buf[3] = !2;
        buf[4] = !0;
        buf[5] = b'h';
        buf[6] = b'i';

        let result = decode_deflate(&buf, None).expect("decode");
        assert_eq!(result.output, b"hi");
        assert_eq!(result.events.len(), 2);
        assert_eq!(result.num_blocks(), 1);

        // The event-free fast path must produce byte-identical output.
        let bytes = inflate(&buf, None).expect("inflate");
        assert_eq!(bytes, result.output);
    }

    #[test]
    fn rejects_reserved_block_type() {
        let mut buf = vec![0u8; 8];
        write_bits(&mut buf, 0, 1, 1);
        write_bits(&mut buf, 1, 0b11, 2);
        let err = decode_deflate(&buf, None).unwrap_err();
        assert!(matches!(
            err,
            DecodeError::ReservedBlockType { block_idx: 0 }
        ));
    }

    #[test]
    fn rejects_output_past_max_via_stored_block() {
        // BFINAL=1, BTYPE=00, LEN=2 ("hi"), but cap at 1 byte. The
        // stored-block path must bail before pushing past the cap so a
        // tiny IDAT can't inflate into gigabytes.
        let mut buf = vec![0u8; 32];
        write_bits(&mut buf, 0, 1, 1);
        buf[1] = 2;
        buf[2] = 0;
        buf[3] = !2;
        buf[4] = !0;
        buf[5] = b'h';
        buf[6] = b'i';
        let err = decode_deflate(&buf, Some(1)).unwrap_err();
        assert!(
            matches!(err, DecodeError::OutputTooLarge { decoded: 2, max: 1 }),
            "got {err:?}"
        );
    }

    #[test]
    fn rejects_stored_block_with_bad_length_complement() {
        // BFINAL=1, BTYPE=00, then byte-aligned LEN=2, NLEN=0 (not the
        // one's complement of LEN; the spec demands LEN ^ NLEN == 0xFFFF).
        let mut buf = vec![0u8; 16];
        // BFINAL=1 at bit 0, BTYPE=00 at bits 1..2.
        write_bits(&mut buf, 0, 1, 1);
        // Stored-block decoder aligns to byte boundary, then reads LEN
        // (LSB first) at byte 1.
        buf[1] = 2; // LEN low
        buf[2] = 0; // LEN high
        buf[3] = 0; // NLEN low (wrong; should be ~LEN)
        buf[4] = 0; // NLEN high
        let err = decode_deflate(&buf, None).unwrap_err();
        assert!(
            matches!(
                err,
                DecodeError::InvalidStoredBlockLength {
                    length: 2,
                    nlen: 0,
                    block_idx: 0
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn rejects_reserved_distance_symbol_30() {
        // A back-reference whose distance symbol is 30 must be rejected:
        // RFC 1951 defines only distance symbols 0..=29. Hand-encoding a
        // full dynamic block to reach that state is awkward, because a
        // single-symbol distance alphabet is degenerate under canonical
        // Huffman. Drive `decode_huffman_block` directly with synthetic
        // lit and dist tables whose dist table has sym 30 present.
        use super::super::huffman::build_tree;
        // Lit alphabet: sym 256 (EOB) and sym 257 (length-3) at clen=1,
        // two 1-bit canonical codes (code 0 and code 1).
        let mut lit_lengths = vec![0u32; 258];
        lit_lengths[256] = 1; // EOB → code 0
        lit_lengths[257] = 1; // length-3 → code 1
        let (lit_tab, _) = build_tree(&lit_lengths).expect("valid lit set");

        // Dist alphabet: sym 29 and sym 30 at clen=1: sym 29 → code 0,
        // sym 30 → code 1 (the reserved-symbol case to trigger).
        let mut dist_lengths = vec![0u32; 31];
        dist_lengths[29] = 1;
        dist_lengths[30] = 1;
        let (dist_tab, _) = build_tree(&dist_lengths).expect("valid dist set");

        // Build a buffer: bit 0 = read sym 257 (1 bit). Then no extra
        // length bits for LBASE[0]=3 (LEXT[0]=0). Then dist sym = 30
        // (1 bit, code 1 since sym 29 got code 0). Then would read
        // DEXT[30] extra bits, but decoding errors before that.
        //
        // First bit: 1 (sym 257 → length 3).
        // Next bit: 1 (dist sym 30, second 1-bit code).
        let mut buf = vec![0u8; 8];
        write_bits(&mut buf, 0, 0b11, 2);

        let mut reader = BitReader::new(&buf);
        let mut out = Vec::new();
        let mut events = Vec::new();
        let mut max_dist = 1u32;
        // We need at least 1 byte in `output` so the distance check
        // (distance > output.len()) doesn't fire first. Push a fake byte.
        out.push(0u8);
        let mut state = DecodeState {
            output: &mut out,
            events: &mut events,
            max_distance: &mut max_dist,
            max_output: None,
        };
        let err =
            decode_huffman_block(&mut reader, &mut state, 0, &lit_tab, &dist_tab).unwrap_err();
        assert!(
            matches!(
                err,
                DecodeError::InvalidDistanceSymbol {
                    sym: 30,
                    block_idx: 0
                }
            ),
            "got {err:?}"
        );
    }
}
