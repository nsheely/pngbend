//! Glass-box PNG codec.
//!
//! Decodes every colour type, bit depth, and Adam7 interlacing the format
//! defines. Encodes the byte-aligned non-indexed colour types (grey,
//! grey+alpha, RGB, RGBA at 8 or 16-bit). Additionally exposes the DEFLATE
//! *event stream* (every literal and LZ77 back-reference with its bit offset)
//! plus the per-block Huffman tables, so a consumer can edit the compressed
//! representation and re-emit a valid PNG without recompressing. The
//! [`pngbend`](https://github.com/nsheely/pngbend) editor is the reference
//! consumer of that introspection surface.
//!
//! # Quick start
//!
//! ```no_run
//! let bytes = std::fs::read("in.png").unwrap();
//!
//! // Standard: bytes -> RGBA8 pixels.
//! let img = glasspng::decode(&bytes).unwrap();
//!
//! // Encode back to PNG bytes.
//! let out = glasspng::encode(&img, &glasspng::EncodeOptions::default()).unwrap();
//!
//! // Glass-box: also get the DEFLATE event stream behind each pixel.
//! let gb = glasspng::decode_with_events(&bytes).unwrap();
//! println!("{} literals/back-refs", gb.deflate.events.len());
//! ```
//!
//! Each `png` layer runs both directions:
//! - [`bitstream`]: LSB-first bit reader/writer over a byte slice.
//! - [`deflate`]: RFC 1951 block decoder ([`deflate::decode_deflate`]) that
//!   records an [`deflate::Event`] per literal/back-reference, plus the
//!   inverse [`deflate::compress`] (greedy LZ77 + stored/fixed/dynamic-Huffman
//!   serializers, whichever is smallest).
//! - [`png`]: chunk framing, IHDR/PLTE/tRNS, row filtering both ways,
//!   RGBA conversion and its inverse [`png::pack`], Adam7, and the zlib/CRC
//!   re-emit path.
//! - [`coords`]: pixel↔byte geometry over [`png::PngInfo`].
//!
//! [`decode`] / [`decode_with_events`] / [`encode`] are the entry points;
//! the layer modules are public so callers can drive the pipeline stage by
//! stage when they need finer control. [`decode`] tolerates stale checksums
//! (so glitched files still load); [`decode_strict`] rejects them, for use
//! as a conformant decoder.

mod api;

pub use api::{
    EncodeOptions, GlassBox, Image, OutputFormat, PngError, decode, decode_strict,
    decode_with_events, encode,
};
pub use png::Warning;

pub mod bitstream;
pub mod coords;
pub mod deflate;
pub mod png;
pub mod raster;

pub use raster::Raster;
