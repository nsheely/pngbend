//! PNG format handling: chunk I/O, IHDR parsing, zlib wrapping, row-filter
//! unfiltering, and any-color-type → RGBA8 conversion.

pub mod chunks;
pub mod convert;
pub mod filter;
pub mod interlace;
pub mod zlib;

pub use chunks::{
    Chunk, ChunkType, ChunksError, ColorType, ParsedChunks, PngInfo, Warning, concat_idat,
    parse_ihdr, read_chunks, write_chunks,
};
pub use convert::{
    ConvertError, PaletteEntry, TrnsKey, apply_color_key, decode_palette, pack, to_rgba8,
    to_rgba8_rows_into,
};
pub use filter::{FilterError, FilterStrategy, FilterType, filter, unfilter, unfilter_rows_into};
pub use interlace::{
    InterlaceError, deinterlace_to_rgba8, deinterlace_unfilter, interlaced_output_len,
};
pub use zlib::{
    ParsedZlib, ZLIB_DEFAULT_HEADER, ZlibError, adler32, build_zlib_stream, parse_zlib_stream,
};
