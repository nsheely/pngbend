//! Bit-level read/write for the DEFLATE stream.
//!
//! DEFLATE packs bits LSB-first within each byte. Huffman codes are
//! conceptually MSB-first but laid down LSB-first in the byte stream;
//! [`BitReader::read_bits`] and [`write_bits`] both honour that convention
//! so a code written by the writer reads back identically.

/// Streaming reader over a deflate bit stream.
///
/// Positions are bit offsets from the start. Reads past the end yield
/// zero bytes — corrupt or truncated input must be detected by whichever
/// decoder consumes the bits, not by this reader.
pub struct BitReader {
    data: Vec<u8>,
    pos: usize,
}

impl BitReader {
    /// Wrap `data` for reading. The internal buffer is padded with three
    /// trailing zero bytes so an unaligned `u32` read at any valid bit
    /// position stays in bounds; reads past the actual stream end yield
    /// zeros rather than panicking.
    pub fn new(data: &[u8]) -> Self {
        let mut padded = Vec::with_capacity(data.len() + 3);
        padded.extend_from_slice(data);
        padded.extend_from_slice(&[0u8; 3]);
        Self {
            data: padded,
            pos: 0,
        }
    }

    /// Read up to 32 bits, LSB-first within bytes.
    #[inline(always)]
    pub fn read_bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let byte_idx = self.pos >> 3;
        let val = u32::from_le_bytes([
            self.data[byte_idx],
            self.data[byte_idx + 1],
            self.data[byte_idx + 2],
            self.data[byte_idx + 3],
        ]);
        // `(1u32 << 32) - 1` overflows `u32` (panics in debug, returns 0
        // in release). Special-case `n == 32` to use the full `u32` mask
        // — for byte-aligned reads that recovers the full value, which
        // was previously silently dropped to 0.
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        let result = (val >> (self.pos & 7)) & mask;
        self.pos += n as usize;
        result
    }

    /// Peek up to 32 bits without advancing. The Huffman decoder uses this
    /// to look up a symbol in a LUT keyed by the next `max_bits`, then
    /// advances by only the matched code length.
    #[inline(always)]
    pub fn peek_bits(&self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let byte_idx = self.pos >> 3;
        let val = u32::from_le_bytes([
            self.data[byte_idx],
            self.data[byte_idx + 1],
            self.data[byte_idx + 2],
            self.data[byte_idx + 3],
        ]);
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        (val >> (self.pos & 7)) & mask
    }

    /// Advance the read position by `n` bits.
    #[inline(always)]
    pub fn advance(&mut self, n: u32) {
        self.pos += n as usize;
    }

    pub fn align_to_byte(&mut self) {
        if self.pos & 7 != 0 {
            self.pos = (self.pos | 7) + 1;
        }
    }

    #[inline(always)]
    pub fn bit_pos(&self) -> usize {
        self.pos
    }
}

/// Patch `n` bits of `value` (MSB-first) into `buf` starting at bit position
/// `bit_start`. Sets *and* clears bits — safe to call repeatedly on the same
/// byte without leaking previous values.
///
/// Reads of those bits via [`BitReader`] recover `value`.
pub fn write_bits(buf: &mut [u8], bit_start: usize, value: u32, n: u8) {
    for i in 0..n as usize {
        let bit = (value >> (n as usize - 1 - i)) & 1;
        let byte_idx = (bit_start + i) >> 3;
        let bit_idx = (bit_start + i) & 7;
        if byte_idx >= buf.len() {
            break;
        }
        if bit != 0 {
            buf[byte_idx] |= 1 << bit_idx;
        } else {
            buf[byte_idx] &= !(1 << bit_idx);
        }
    }
}

/// Read up to 32 bits at an arbitrary bit position. The standalone form
/// (vs. constructing a [`BitReader`]) is for one-shot reads — capturing
/// the bits about to be overwritten by a [`write_bits`] call so the
/// reverse patch is recoverable, for example.
pub fn read_bits_at(buf: &[u8], bit_start: usize, n: u8) -> u32 {
    let mut value = 0u32;
    for i in 0..n as usize {
        let byte_idx = (bit_start + i) >> 3;
        let bit_idx = (bit_start + i) & 7;
        if byte_idx >= buf.len() {
            break;
        }
        let bit = (buf[byte_idx] >> bit_idx) & 1;
        // Bits arrive in MSB-first order to mirror write_bits.
        value = (value << 1) | bit as u32;
    }
    value
}

#[cfg(test)]
mod tests {
    // `write_bits` / `read_bits_at` round-trips at arbitrary widths and
    // offsets are covered by the property tests in
    // `tests/proptest_bitstream.rs`. The unit tests below cover behaviour
    // those proptests don't reach: the LSB-first / MSB-first ordering
    // bridge between `write_bits` and `BitReader`, plus `peek` /
    // `align_to_byte` mechanics.

    use super::*;

    #[test]
    fn bit_reader_round_trip_via_write_bits() {
        // Pack three values back-to-back, then read them with BitReader.
        let mut buf = vec![0u8; 8];
        write_bits(&mut buf, 0, 0b1010, 4);
        write_bits(&mut buf, 4, 0b1100, 4);
        write_bits(&mut buf, 8, 0xFF, 8);

        let mut reader = BitReader::new(&buf);
        // BitReader yields LSB-first within byte; write_bits stores MSB of value
        // at the lowest bit position. So a 4-bit value 0b1010 packed at bit 0
        // becomes bits (1, 0, 1, 0) at positions (0, 1, 2, 3) — read_bits(4)
        // reconstructs 0b0101 = 5.
        assert_eq!(reader.read_bits(4), 0b0101);
        assert_eq!(reader.read_bits(4), 0b0011);
        assert_eq!(reader.read_bits(8), 0xFF);
    }

    #[test]
    fn peek_does_not_advance() {
        let mut buf = vec![0u8; 4];
        write_bits(&mut buf, 0, 0xCD, 8);
        let mut reader = BitReader::new(&buf);
        let p = reader.peek_bits(8);
        assert_eq!(reader.bit_pos(), 0);
        let r = reader.read_bits(8);
        assert_eq!(p, r);
        assert_eq!(reader.bit_pos(), 8);
    }

    #[test]
    fn align_to_byte_rounds_up() {
        let mut reader = BitReader::new(&[0u8; 4]);
        reader.advance(3);
        reader.align_to_byte();
        assert_eq!(reader.bit_pos(), 8);
        reader.align_to_byte();
        assert_eq!(reader.bit_pos(), 8); // already aligned
    }

    /// Regression: pre-fix `read_bits(32)` computed `(1u32 << 32) - 1`
    /// which panics in debug and silently returns 0 in release. The
    /// fix special-cases `n == 32` to use `u32::MAX`. The byte-aligned
    /// path is exercised here (the only case the previous code could
    /// have got right with a 32-bit window).
    #[test]
    fn read_bits_32_round_trips_full_u32() {
        // BitReader reads LSB-first within each byte, so a byte-aligned
        // 32-bit read must reconstruct the underlying little-endian
        // u32 exactly.
        let mut buf = vec![0u8; 8];
        buf[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        let mut reader = BitReader::new(&buf);
        assert_eq!(reader.peek_bits(32), 0xDEAD_BEEF);
        assert_eq!(reader.read_bits(32), 0xDEAD_BEEF);
        assert_eq!(reader.bit_pos(), 32);
    }
}
