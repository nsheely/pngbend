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
pub struct HuffmanTable {
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

impl HuffmanTable {
    pub fn is_empty(&self) -> bool {
        self.max_bits == 0
    }

    pub fn max_bits(&self) -> u8 {
        self.max_bits
    }
}

/// Build a canonical Huffman table from per-symbol code lengths. Returns the
/// decoder LUT alongside the encoder side table. Empty input yields an empty
/// table — check via [`HuffmanTable::is_empty`].
pub fn build_tree(lengths: &[u32]) -> (HuffmanTable, EncTable) {
    let max_bits = lengths
        .iter()
        .filter(|&&b| b > 0)
        .max()
        .copied()
        .unwrap_or(0) as u8;
    if max_bits == 0 {
        return (
            HuffmanTable {
                entries: Vec::new(),
                max_bits: 0,
            },
            EncTable::new(lengths.len()),
        );
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

    (HuffmanTable { entries, max_bits }, enc)
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
    fn empty_lengths_produce_empty_table() {
        let (tab, enc) = build_tree(&[]);
        assert!(tab.is_empty());
        assert!(enc.is_empty());
    }

    #[test]
    fn all_zero_lengths_produce_empty_table() {
        let (tab, enc) = build_tree(&[0, 0, 0, 0]);
        assert!(tab.is_empty());
        assert!(enc.is_empty());
    }

    #[test]
    fn canonical_codes_round_trip_through_decode_sym() {
        // RFC 1951 example: 8 symbols A-H, lengths 3,3,3,3,3,2,4,4
        let lengths = [3u32, 3, 3, 3, 3, 2, 4, 4];
        let (tab, enc) = build_tree(&lengths);
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
        let (tab, _) = build_tree(&[2, 2, 2, 2]);
        assert_eq!(tab.max_bits(), 2);
        assert_eq!(tab.entries.len(), 4);
        for e in &tab.entries {
            assert_eq!(e.used_bits, 2);
        }
    }
}
