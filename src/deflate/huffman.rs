//! Canonical Huffman tables for DEFLATE.
//!
//! [`build_tree`] returns a direct-indexed [`HuffmanTable`] sized to
//! `2.pow(max_code_length)` so decoding one symbol is a single peek +
//! array lookup. The encoder side ([`super::EncTable`]) is returned in
//! the same call so writers can re-emit codes at identical bit-lengths.

use crate::bitstream::BitReader;

use super::error::DecodeError;
use super::events::EncTable;

#[derive(Debug, Clone)]
pub(super) struct HuffmanTable {
    entries: Vec<LutEntry>,
    /// The width peeked on every decode.
    max_bits: u8,
}

#[derive(Debug, Clone, Copy)]
struct LutEntry {
    sym: u16,
    /// 0 for empty slots (no valid code).
    used_bits: u8,
}

/// Build a canonical Huffman table from per-symbol code lengths. Returns
/// the decoder LUT alongside the encoder side table. An all-zero / empty
/// length set yields a table with `max_bits == 0`, which [`decode_sym`]
/// treats as an empty alphabet.
///
/// Validates the Kraft-McMillan inequality before building: an
/// over-subscribed length set silently corrupts the LUT (later symbols
/// overwrite slots assigned to earlier ones), and an under-subscribed
/// set leaves valid peek patterns with no symbol — both fail loudly here
/// instead. The single-symbol/single-bit degenerate code allowed by
/// RFC 1951 §3.2.2 (used when only one distance is referenced) is the
/// one exception.
pub(super) fn build_tree(lengths: &[u32]) -> Result<(HuffmanTable, EncTable), DecodeError> {
    let max_bits = lengths
        .iter()
        .filter(|&&b| b > 0)
        .max()
        .copied()
        .unwrap_or(0) as u8;
    if max_bits == 0 {
        return Ok((
            HuffmanTable {
                entries: Vec::new(),
                max_bits: 0,
            },
            EncTable::new(lengths.len()),
        ));
    }
    if max_bits > 15 {
        return Err(DecodeError::HuffmanCodeTooLong { max_bits });
    }

    // Kraft-McMillan: sum of `2^(max_bits - l_i)` over nonzero lengths
    // is the number of LUT slots the canonical code claims. Equal to
    // `2^max_bits` is a complete code; greater is over-subscription
    // (slot collisions); less is under-subscription, which we accept
    // only for the single 1-bit code RFC 1951 §3.2.2 permits.
    let kraft_full: u32 = 1u32 << max_bits;
    let kraft_used: u32 = lengths
        .iter()
        .filter(|&&l| l > 0)
        .map(|&l| 1u32 << (max_bits as u32 - l))
        .sum();
    if kraft_used > kraft_full {
        return Err(DecodeError::OverSubscribedHuffman {
            max_bits,
            expected: kraft_full,
            actual: kraft_used,
        });
    }
    if kraft_used < kraft_full {
        let nonzero = lengths.iter().filter(|&&l| l > 0).count();
        let single_one_bit = nonzero == 1 && max_bits == 1;
        if !single_one_bit {
            return Err(DecodeError::UnderSubscribedHuffman {
                max_bits,
                actual: kraft_used,
            });
        }
    }

    // Counts per code length → first canonical code at each length.
    let mut counts = vec![0u32; max_bits as usize + 1];
    for &l in lengths {
        if l > 0 {
            counts[l as usize] += 1;
        }
    }
    let mut code = 0u16;
    let mut first_code = vec![0u16; max_bits as usize + 2];
    for bits in 1..=max_bits as usize {
        code = (code + counts[bits - 1] as u16) << 1;
        first_code[bits] = code;
    }

    let lut_size = 1usize << max_bits;
    let mut entries = vec![
        LutEntry {
            sym: 0,
            used_bits: 0
        };
        lut_size
    ];
    let mut enc = EncTable::new(lengths.len());
    let mut next_code = first_code;

    for (sym, &l) in lengths.iter().enumerate() {
        if l == 0 {
            continue;
        }
        let clen = l as u8;
        let c = next_code[l as usize];
        enc.set(sym as u16, c, clen);
        next_code[l as usize] += 1;

        // BitReader reads LSB-first; Huffman codes are MSB-first. Reverse
        // the low `clen` bits of `c` to match the peek value we'll see.
        let rev = reverse_bits(c, clen);

        // Fill every LUT slot whose low `clen` bits match `rev` — those are
        // the peek values starting with this code.
        let step = 1usize << clen;
        let mut idx = rev as usize;
        let entry = LutEntry {
            sym: sym as u16,
            used_bits: clen,
        };
        while idx < lut_size {
            entries[idx] = entry;
            idx += step;
        }
    }

    Ok((HuffmanTable { entries, max_bits }, enc))
}

#[inline(always)]
fn reverse_bits(code: u16, len: u8) -> u16 {
    if len == 0 {
        return 0;
    }
    // `u16::reverse_bits` reverses all 16 bits; shift right by `16 - len`
    // to keep only the reversed low `len` bits. Compiles to a single
    // `rbit`-style instruction on aarch64 / a `bswap+shift` on x86.
    code.reverse_bits() >> (16 - len as u32)
}

/// Decode one symbol from `reader`. Returns an error if the peeked bits
/// don't match any code in the table (gap in the canonical Huffman set or
/// truncated input decoded as zeros).
#[inline(always)]
pub(super) fn decode_sym(reader: &mut BitReader, table: &HuffmanTable) -> Result<u16, DecodeError> {
    if table.max_bits == 0 {
        return Err(DecodeError::InvalidHuffmanCode {
            max_bits: 0,
            code: 0,
        });
    }
    let peek = reader.peek_bits(table.max_bits as u32) as usize;
    let entry = table.entries[peek];
    if entry.used_bits == 0 {
        return Err(DecodeError::InvalidHuffmanCode {
            max_bits: table.max_bits,
            code: peek as u16,
        });
    }
    reader.advance(entry.used_bits as u32);
    Ok(entry.sym)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::write_bits;

    #[test]
    fn empty_or_all_zero_lengths_produce_empty_table() {
        // No nonzero lengths → no codes. Both inputs hit the same
        // `max_bits == 0` early return.
        for lengths in [&[][..], &[0, 0, 0, 0]] {
            let (tab, enc) = build_tree(lengths).expect("no codes");
            assert!(tab.max_bits == 0);
            assert!(enc.is_empty());
        }
    }

    #[test]
    fn canonical_codes_round_trip_through_decode_sym() {
        // RFC 1951 example: 8 symbols A-H, lengths 3,3,3,3,3,2,4,4
        let lengths = [3u32, 3, 3, 3, 3, 2, 4, 4];
        let (tab, enc) = build_tree(&lengths).expect("valid canonical set");
        for (sym, &len) in lengths.iter().enumerate() {
            let (code, clen) = enc.get(sym as u16).expect("sym in enc");
            assert_eq!(clen as u32, len);
            let mut buf = vec![0u8; 4];
            write_bits(&mut buf, 0, code as u32, clen);
            let mut reader = BitReader::new(&buf);
            assert_eq!(decode_sym(&mut reader, &tab).unwrap(), sym as u16);
            assert_eq!(reader.bit_pos(), clen as usize);
        }
    }

    #[test]
    fn empty_table_errors_on_decode() {
        let tab = HuffmanTable {
            entries: Vec::new(),
            max_bits: 0,
        };
        let buf = [0xFFu8; 4];
        let mut reader = BitReader::new(&buf);
        assert!(matches!(
            decode_sym(&mut reader, &tab).unwrap_err(),
            DecodeError::InvalidHuffmanCode { .. }
        ));
    }

    #[test]
    fn reverse_bits_known() {
        assert_eq!(reverse_bits(0b101, 3), 0b101);
        assert_eq!(reverse_bits(0b110, 3), 0b011);
        assert_eq!(reverse_bits(0b1000, 4), 0b0001);
        assert_eq!(reverse_bits(0b1010_1100, 8), 0b0011_0101);
    }

    #[test]
    fn lut_fills_every_short_code_slot() {
        // One symbol with a 2-bit code must occupy all 2^(max-2) = 2 slots
        // in a 4-entry LUT.
        let (tab, _) = build_tree(&[2, 2, 2, 2]).expect("valid 2-bit set");
        assert_eq!(tab.max_bits, 2);
        assert_eq!(tab.entries.len(), 4);
        for e in &tab.entries {
            assert_eq!(e.used_bits, 2);
        }
    }

    #[test]
    fn rejects_oversubscribed_lengths() {
        // Three symbols at length 1 — only two 1-bit codewords exist, so
        // the third would silently overwrite slot 0 if the build wasn't
        // guarded.
        let err = build_tree(&[1, 1, 1]).unwrap_err();
        assert!(
            matches!(
                err,
                DecodeError::OverSubscribedHuffman {
                    max_bits: 1,
                    expected: 2,
                    actual: 3,
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn rejects_undersubscribed_lengths() {
        // Two symbols at length 2 — fills two of four slots but leaves
        // two valid peek patterns with no symbol. RFC 1951's leniency
        // covers only the one-symbol/one-bit case.
        let err = build_tree(&[2, 2]).unwrap_err();
        assert!(
            matches!(
                err,
                DecodeError::UnderSubscribedHuffman {
                    max_bits: 2,
                    actual: 2,
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn accepts_single_one_bit_code() {
        // RFC 1951 §3.2.2: a distance alphabet with exactly one used
        // distance is encoded with a single 1-bit code (the other code
        // is reserved). build_tree must accept this degenerate set.
        let mut lengths = vec![0u32; 30];
        lengths[5] = 1;
        let (tab, enc) = build_tree(&lengths).expect("single 1-bit code");
        assert_eq!(tab.max_bits, 1);
        assert!(!enc.is_empty());
    }

    #[test]
    fn rejects_code_length_above_15() {
        let mut lengths = vec![0u32; 4];
        lengths[0] = 16;
        let err = build_tree(&lengths).unwrap_err();
        assert!(
            matches!(err, DecodeError::HuffmanCodeTooLong { max_bits: 16 }),
            "got {err:?}"
        );
    }
}
