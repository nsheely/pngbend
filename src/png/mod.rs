//! PNG format handling: chunk I/O, IHDR parsing, zlib wrapping, row-filter
//! unfiltering, and any-color-type → RGBA8 conversion.

pub mod chunks;
pub mod convert;
pub mod filter;
pub mod zlib;

pub use chunks::{Chunk, ColorType, PngInfo, concat_idat, parse_ihdr, read_chunks, write_chunks};
pub use convert::{
    ConvertError, PaletteEntry, decode_palette, to_rgba8, to_rgba8_into, to_rgba8_rows_into,
};
pub use filter::{FilterError, unfilter, unfilter_into, unfilter_rows_into};
pub use zlib::{build_zlib_stream, inflate_raw};
