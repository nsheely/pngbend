//! The unified decode error, flattened across every pipeline stage.

use crate::deflate::DecodeError;
use crate::png::{ChunksError, ConvertError, FilterError, InterlaceError, Warning, ZlibError};

/// Anything that can go wrong turning PNG bytes into pixels, unified across
/// the chunk / IHDR / zlib / DEFLATE / unfilter / convert stages so callers
/// match on one type.
#[derive(Debug)]
pub enum PngError {
    /// The chunk layer rejected the bytes (bad signature, truncation, ...).
    Chunks(ChunksError),
    /// No IHDR chunk, or its fields didn't parse.
    MissingIhdr,
    /// The image has no IDAT image data.
    MissingIdat,
    /// IHDR dimensions imply an unfiltered output larger than 4 GiB, which
    /// the codec's `u32` byte positions can't address.
    OutputTooLarge { output_bytes: u64 },
    /// The zlib wrapper around the DEFLATE data was malformed.
    Zlib(ZlibError),
    /// The DEFLATE stream failed to decode.
    Deflate(DecodeError),
    /// A row-filter type was invalid or a row was malformed.
    Filter(FilterError),
    /// Pixel bytes couldn't be converted to RGBA8 (e.g. an unsupported
    /// bit depth, a missing palette, or truncated input). An out-of-range
    /// palette index is not an error; it decodes to transparent black.
    Convert(ConvertError),
    /// An interlaced (Adam7) image's data ended mid-pass. A per-pass filter
    /// or conversion failure surfaces as [`PngError::Filter`] /
    /// [`PngError::Convert`] instead, so each leaf error has a single home.
    InterlaceTruncated,
    /// A chunk CRC or the zlib Adler-32 didn't match. Only
    /// [`decode_strict`](super::decode_strict) returns this;
    /// [`decode`](super::decode) tolerates the mismatch and reports it in
    /// [`Image::warnings`](super::Image::warnings) instead.
    BadChecksum(Warning),
}

impl std::fmt::Display for PngError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chunks(e) => write!(f, "PNG chunk error: {e}"),
            Self::MissingIhdr => write!(f, "no valid IHDR chunk"),
            Self::MissingIdat => write!(f, "no IDAT image data"),
            Self::OutputTooLarge { output_bytes } => {
                write!(
                    f,
                    "unfiltered output {output_bytes} bytes exceeds 4 GiB cap"
                )
            }
            Self::Zlib(e) => write!(f, "zlib error: {e}"),
            Self::Deflate(e) => write!(f, "DEFLATE error: {e}"),
            Self::Filter(e) => write!(f, "unfilter error: {e}"),
            Self::Convert(e) => write!(f, "RGBA conversion error: {e}"),
            Self::InterlaceTruncated => write!(f, "interlaced stream ended mid-pass"),
            Self::BadChecksum(w) => write!(f, "checksum mismatch: {w}"),
        }
    }
}

impl std::error::Error for PngError {}

impl From<ChunksError> for PngError {
    fn from(e: ChunksError) -> Self {
        Self::Chunks(e)
    }
}
impl From<ZlibError> for PngError {
    fn from(e: ZlibError) -> Self {
        Self::Zlib(e)
    }
}
impl From<DecodeError> for PngError {
    fn from(e: DecodeError) -> Self {
        Self::Deflate(e)
    }
}
impl From<FilterError> for PngError {
    fn from(e: FilterError) -> Self {
        Self::Filter(e)
    }
}
impl From<ConvertError> for PngError {
    fn from(e: ConvertError) -> Self {
        Self::Convert(e)
    }
}
impl From<InterlaceError> for PngError {
    fn from(e: InterlaceError) -> Self {
        // Flatten: a per-pass filter/convert failure is the same leaf as its
        // progressive counterpart, so route it to the flat variant rather
        // than nesting it under a second Interlace layer.
        match e {
            InterlaceError::Filter(f) => Self::Filter(f),
            InterlaceError::Convert(c) => Self::Convert(c),
            InterlaceError::Truncated => Self::InterlaceTruncated,
        }
    }
}
