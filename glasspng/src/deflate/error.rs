//! DEFLATE decode errors: RFC 1951 violations (bad Huffman codes, reserved
//! symbols and block types) plus the decompression-bomb output cap.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    InvalidHuffmanCode {
        max_bits: u8,
        code: u16,
    },
    ReservedBlockType {
        block_idx: u32,
    },
    InvalidLengthSymbol {
        sym: u16,
        block_idx: u32,
    },
    InvalidDistance {
        distance: u32,
        output_len: usize,
        block_idx: u32,
    },
    /// Distance alphabet decoded a symbol >= 30. RFC 1951 reserves 30 and
    /// 31: they must not occur in a valid stream, and `DBASE`/`DEXT` only
    /// have 30 entries so indexing past would panic.
    InvalidDistanceSymbol {
        sym: u16,
        block_idx: u32,
    },
    /// Stored block (BTYPE=00) `length ^ nlen != 0xFFFF`. RFC 1951
    /// requires `nlen` to be the one's complement of `length`.
    InvalidStoredBlockLength {
        length: u16,
        nlen: u16,
        block_idx: u32,
    },
    /// Canonical Huffman code lengths sum to more than `2^max_bits` of
    /// codeword space. Kraft-McMillan inequality fails; without this
    /// check, later symbols silently overwrite LUT slots that already
    /// belonged to earlier symbols and the decoder emits the wrong byte.
    OverSubscribedHuffman {
        max_bits: u8,
        expected: u32,
        actual: u32,
    },
    /// Canonical Huffman code lengths use less than `2^max_bits` of
    /// codeword space, leaving valid bit patterns with no defined
    /// symbol. The single-symbol/single-bit case (RFC 1951 §3.2.2's
    /// degenerate distance alphabet) is the one allowed exception.
    UnderSubscribedHuffman {
        max_bits: u8,
        actual: u32,
    },
    /// A code length exceeds the 15-bit DEFLATE limit. Out-of-range
    /// lengths would size the LUT past 64 KiB entries and break the
    /// `u16` peek-width assumption in `huffman`'s bit reversal.
    HuffmanCodeTooLong {
        max_bits: u8,
    },
    /// Decoded output would grow past the caller-supplied cap. Loader
    /// passes the IHDR-derived expected size so a tiny IDAT can't
    /// expand to gigabytes (the classic decompression-bomb pattern).
    OutputTooLarge {
        decoded: usize,
        max: usize,
    },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHuffmanCode { max_bits, code } => {
                write!(f, "no Huffman match after {max_bits} bits (code={code:#b})")
            }
            Self::ReservedBlockType { block_idx } => {
                write!(f, "reserved BTYPE=11 in block {block_idx}")
            }
            Self::InvalidLengthSymbol { sym, block_idx } => {
                write!(f, "length symbol {sym} out of range in block {block_idx}")
            }
            Self::InvalidDistance {
                distance,
                output_len,
                block_idx,
            } => write!(
                f,
                "back-ref distance {distance} exceeds output length {output_len} in block {block_idx}"
            ),
            Self::InvalidDistanceSymbol { sym, block_idx } => write!(
                f,
                "reserved distance symbol {sym} (>=30) in block {block_idx}"
            ),
            Self::InvalidStoredBlockLength {
                length,
                nlen,
                block_idx,
            } => write!(
                f,
                "stored-block LEN/NLEN mismatch in block {block_idx} (LEN={length:#06x}, NLEN={nlen:#06x})"
            ),
            Self::OverSubscribedHuffman {
                max_bits,
                expected,
                actual,
            } => write!(
                f,
                "over-subscribed Huffman code: lengths consume {actual} of {expected} slots at max_bits={max_bits}"
            ),
            Self::UnderSubscribedHuffman { max_bits, actual } => write!(
                f,
                "under-subscribed Huffman code: lengths consume {actual} of {} slots at max_bits={max_bits}",
                1u32 << max_bits
            ),
            Self::HuffmanCodeTooLong { max_bits } => write!(
                f,
                "Huffman code length {max_bits} exceeds RFC 1951 limit of 15"
            ),
            Self::OutputTooLarge { decoded, max } => write!(
                f,
                "deflate output would exceed cap ({decoded} bytes, max {max})"
            ),
        }
    }
}

impl std::error::Error for DecodeError {}
