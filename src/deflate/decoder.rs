//! DEFLATE block-scanning driver.
//!
//! Decodes each BTYPE 0/1/2 block, records every literal and back-reference
//! as an [`Event`], and keeps per-block encoder tables so writers can re-emit
//! edits without rebuilding Huffman trees.

use crate::bitstream::BitReader;

use super::constants::{CLORDER, DBASE, DEXT, LBASE, LEXT};
use super::error::DecodeError;
use super::events::{EncTable, Event, LitEvent, RefEvent};
use super::huffman::{HuffmanTable, build_tree, decode_sym};

#[derive(Debug)]
pub struct DecodedDeflate {
    pub output: Vec<u8>,
    pub events: Vec<Event>,
    pub lit_encs: Vec<EncTable>,
    pub dist_encs: Vec<EncTable>,
    pub num_blocks: usize,
    /// Largest LZ77 back-reference distance found in `events`, or 1 if
    /// no back-references were emitted. Cached here so consumers (e.g.
    /// the distance-overlay colour scaler) don't need a second event scan.
    pub max_distance: u32,
}

pub fn decode_deflate(data: &[u8]) -> Result<DecodedDeflate, DecodeError> {
    let mut reader = BitReader::new(data);
    // Output and event vectors grow on demand. Up-front preallocation
    // sized from the deflate length measures slower on typical photo
    // PNGs — wasted capacity hurts L2 reuse more than the doublings cost.
    let mut output: Vec<u8> = Vec::new();
    let mut events: Vec<Event> = Vec::new();
    let mut lit_encs: Vec<EncTable> = Vec::new();
    let mut dist_encs: Vec<EncTable> = Vec::new();
    let mut block_idx = 0u32;
    let mut max_distance = 1u32;

    loop {
        let is_final = reader.read_bits(1);
        let block_type = reader.read_bits(2);

        match block_type {
            0b00 => {
                decode_stored_block(&mut reader, &mut output, &mut events, block_idx);
                // A stored block carries no Huffman alphabet, but every
                // event still records a `block` index that downstream code
                // uses to look up the table. Push placeholder empty tables
                // sized to the standard alphabets so indexing stays valid.
                lit_encs.push(EncTable::new(288));
                dist_encs.push(EncTable::new(30));
            }
            0b01 | 0b10 => {
                let (lit_lengths, dist_lengths) =
                    read_huffman_code_lengths(&mut reader, block_type)?;
                let (lit_tab, lit_enc) = build_tree(&lit_lengths);
                let (dist_tab, dist_enc) = build_tree(&dist_lengths);
                lit_encs.push(lit_enc);
                dist_encs.push(dist_enc);

                decode_huffman_block(
                    &mut reader,
                    &mut output,
                    &mut events,
                    block_idx,
                    &lit_tab,
                    &dist_tab,
                    &mut max_distance,
                )?;
            }
            _ => return Err(DecodeError::ReservedBlockType { block_idx }),
        }

        block_idx += 1;
        if is_final != 0 {
            break;
        }
    }

    Ok(DecodedDeflate {
        output,
        events,
        lit_encs,
        dist_encs,
        num_blocks: block_idx as usize,
        max_distance,
    })
}

fn decode_stored_block(
    reader: &mut BitReader,
    output: &mut Vec<u8>,
    events: &mut Vec<Event>,
    block_idx: u32,
) {
    reader.align_to_byte();
    let length = reader.read_bits(8) | (reader.read_bits(8) << 8);
    reader.read_bits(16); // length complement — not validated
    for _ in 0..length {
        let bit_start = reader.bit_pos() as u32;
        let byte_val = reader.read_bits(8) as u8;
        events.push(Event::Lit(LitEvent {
            out_pos: output.len() as u32,
            symbol: byte_val,
            bit_start,
            block: block_idx,
        }));
        output.push(byte_val);
    }
}

/// Parse either the fixed Huffman table (BTYPE=01) or the dynamic
/// per-block length codes + repeats (BTYPE=10). Returns the length arrays
/// ready to feed to [`build_tree`].
fn read_huffman_code_lengths(
    reader: &mut BitReader,
    block_type: u32,
) -> Result<(Vec<u32>, Vec<u32>), DecodeError> {
    if block_type == 0b01 {
        let mut ll = vec![0u32; 288];
        ll[..144].fill(8);
        ll[144..256].fill(9);
        ll[256..280].fill(7);
        ll[280..288].fill(8);
        return Ok((ll, vec![5u32; 32]));
    }

    let hlit = reader.read_bits(5) + 257;
    let hdist = reader.read_bits(5) + 1;
    let hclen = reader.read_bits(4) + 4;

    let mut cl_lengths = vec![0u32; 19];
    for i in 0..hclen as usize {
        cl_lengths[CLORDER[i]] = reader.read_bits(3);
    }
    let (cl_tab, _) = build_tree(&cl_lengths);

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
    // new Vec and shrinks the original — one allocation rather than two.
    let dl = all_lengths.split_off(hlit as usize);
    let ll = all_lengths;
    Ok((ll, dl))
}

fn decode_huffman_block(
    reader: &mut BitReader,
    output: &mut Vec<u8>,
    events: &mut Vec<Event>,
    block_idx: u32,
    lit: &HuffmanTable,
    dist: &HuffmanTable,
    max_distance: &mut u32,
) -> Result<(), DecodeError> {
    loop {
        let sym_bit_start = reader.bit_pos();
        let sym = decode_sym(reader, lit)?;

        if sym == 256 {
            return Ok(());
        }

        if sym < 256 {
            events.push(Event::Lit(LitEvent {
                out_pos: output.len() as u32,
                symbol: sym as u8,
                bit_start: sym_bit_start as u32,
                block: block_idx,
            }));
            output.push(sym as u8);
        } else {
            let len_idx = (sym - 257) as usize;
            if len_idx >= LBASE.len() {
                return Err(DecodeError::InvalidLengthSymbol { sym, block_idx });
            }
            let copy_len = LBASE[len_idx] + reader.read_bits(LEXT[len_idx]);

            let dist_bit_start = reader.bit_pos() as u32;
            let dist_sym = decode_sym(reader, dist)? as u8;
            let distance = DBASE[dist_sym as usize] + reader.read_bits(DEXT[dist_sym as usize]);

            if distance as usize > output.len() {
                return Err(DecodeError::InvalidDistance {
                    distance,
                    output_len: output.len(),
                    block_idx,
                });
            }
            if distance > *max_distance {
                *max_distance = distance;
            }
            let src_start = output.len() - distance as usize;

            events.push(Event::Ref(RefEvent {
                out_pos: output.len() as u32,
                src_out_pos: src_start as u32,
                copy_len: copy_len as u16,
                dist_sym,
                block: block_idx,
                dist_bit_start,
            }));

            // When distance < copy_len the source range overlaps the
            // destination, and each push extends the bytes a later
            // iteration will read — the run-length-encoding pattern that
            // makes "copy 64 bytes from one byte back" expand a single
            // byte into 64. A bulk `copy_within` / `copy_from_slice`
            // would snapshot the source slice and lose that semantic.
            for i in 0..copy_len as usize {
                let byte_val = output[src_start + i];
                output.push(byte_val);
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

        let result = decode_deflate(&buf).expect("decode");
        assert_eq!(result.output, b"hi");
        assert_eq!(result.events.len(), 2);
        assert_eq!(result.num_blocks, 1);
    }

    #[test]
    fn rejects_reserved_block_type() {
        let mut buf = vec![0u8; 8];
        write_bits(&mut buf, 0, 1, 1);
        write_bits(&mut buf, 1, 0b11, 2);
        let err = decode_deflate(&buf).unwrap_err();
        assert!(matches!(
            err,
            DecodeError::ReservedBlockType { block_idx: 0 }
        ));
    }
}
