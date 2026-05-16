//! PNG format handling: chunk I/O, IHDR parsing, zlib wrapping, row-filter
//! unfiltering, and any-color-type → RGBA8 conversion.

pub mod chunks;
pub mod convert;
pub mod filter;
pub mod zlib;

pub use chunks::{
    Chunk, ChunksError, ColorType, ParsedChunks, PngInfo, concat_idat, parse_ihdr, read_chunks,
    write_chunks,
};
pub use convert::{ConvertError, PaletteEntry, decode_palette, to_rgba8, to_rgba8_rows_into};
pub use filter::{FilterError, unfilter, unfilter_rows_into};
pub use zlib::{ParsedZlib, ZlibError, adler32, build_zlib_stream, parse_zlib_stream};
