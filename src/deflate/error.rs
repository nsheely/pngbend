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
        }
    }
}

impl std::error::Error for DecodeError {}
