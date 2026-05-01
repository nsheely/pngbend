//! Property tests for the bit-level read/write functions.
//!
//! Covers every bit width (1..=32), every byte-aligned and unaligned
//! offset, and arbitrary values — if a bit-packing change ever breaks
//! round-trip, these catch it.

use pngbend::bitstream::{BitReader, read_bits_at, write_bits};
use proptest::prelude::*;

proptest! {
    #[test]
    fn write_read_round_trip(
        value in any::<u32>(),
        n in 1u8..=32,
        offset in 0usize..1000,
    ) {
        // Mask `value` to fit in n bits so we can compare directly.
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        let v = value & mask;

        // Buffer large enough to hold (offset + n) bits plus generous padding.
        let mut buf = vec![0u8; (offset + n as usize) / 8 + 8];
        write_bits(&mut buf, offset, v, n);

        let got = read_bits_at(&buf, offset, n);
        prop_assert_eq!(got, v);
    }

    #[test]
    fn bit_reader_matches_read_bits_at(
        bytes in proptest::collection::vec(any::<u8>(), 16..64),
        sequence in proptest::collection::vec((1u32..=16, 0usize..16), 1..8),
    ) {
        // Walk the same byte buffer with BitReader.read_bits and with the
        // equivalent read_bits_at offsets; they must agree on every read.
        let mut reader = BitReader::new(&bytes);
        let mut offset = 0usize;
        for (n, _) in sequence {
            if offset + n as usize > bytes.len() * 8 {
                break;
            }
            let reader_val = reader.read_bits(n);
            // read_bits_at uses the MSB-first interpretation used by write_bits.
            // BitReader reads bits LSB-first within bytes. To cross-check, we
            // reverse the bit order of read_bits_at's result over n bits.
            let at_val = read_bits_at(&bytes, offset, n as u8);
            let at_val_lsb_first = reverse_bits(at_val, n as u8);
            prop_assert_eq!(reader_val, at_val_lsb_first);
            offset += n as usize;
        }
    }
}

fn reverse_bits(mut code: u32, len: u8) -> u32 {
    let mut rev = 0u32;
    for _ in 0..len {
        rev = (rev << 1) | (code & 1);
        code >>= 1;
    }
    rev
}
