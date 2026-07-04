//! The high-level public API, split by direction: [`decode`] /
//! [`decode_strict`] / [`decode_with_events`] live in `decode`, [`encode`] in
//! `encode`. Shared chunk-level preparation is in `prep`, the [`PngError`]
//! type in `error`, and the [`Image`] / [`GlassBox`] result types in `image`.
//!
//! - [`decode`] is the standard path: bytes in, RGBA8 pixels out.
//! - [`decode_with_events`] is the glass-box path: it additionally returns the
//!   DEFLATE event stream and per-block Huffman tables, so a consumer can edit
//!   the compressed representation and re-emit a valid PNG.

mod decode;
mod encode;
mod error;
mod image;
mod prep;

pub use decode::{decode, decode_strict, decode_with_events};
pub use encode::{EncodeOptions, encode};
pub use error::PngError;
pub use image::{GlassBox, Image};
