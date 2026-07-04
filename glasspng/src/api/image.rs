//! The public decode result types.

use crate::deflate::DecodedDeflate;
use crate::png::{PaletteEntry, PngInfo, Warning};

/// A decoded image: metadata plus tightly-packed RGBA8 pixels
/// (`info.width * info.height * 4` bytes, row-major, no padding).
#[derive(Debug, Clone)]
pub struct Image {
    pub info: PngInfo,
    /// Decoded palette for indexed PNGs; `None` for every other colour type.
    pub palette: Option<Vec<PaletteEntry>>,
    /// RGBA8, one `[r, g, b, a]` per pixel in raster order.
    pub pixels: Vec<u8>,
    /// Non-fatal issues found while decoding, e.g. a chunk whose CRC doesn't
    /// match or a stale zlib Adler-32. Advisory; the image still decoded.
    pub warnings: Vec<Warning>,
}

/// A glass-box decode: everything [`Image`] needs plus the compressed
/// representation itself. Pair the (possibly edited) `deflate.output` with
/// [`crate::png::build_zlib_stream`] and [`crate::png::write_chunks`] to
/// re-emit a valid PNG.
#[derive(Debug)]
pub struct GlassBox {
    pub info: PngInfo,
    pub palette: Option<Vec<PaletteEntry>>,
    /// The decoded DEFLATE stream: raw filtered `output` bytes, the per-
    /// literal / per-back-reference [`crate::deflate::Event`] log, the
    /// per-block Huffman encoder tables, and block boundaries.
    pub deflate: DecodedDeflate,
    /// The `output` bytes with the per-row PNG filters undone
    /// (`height * (row_stride - 1)` bytes): the input to RGBA conversion.
    pub unfiltered: Vec<u8>,
    pub warnings: Vec<Warning>,
}
