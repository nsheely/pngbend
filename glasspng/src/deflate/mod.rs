//! DEFLATE (RFC 1951) codec, both directions.
//!
//! Decode (`decoder`) scans blocks and records every literal and
//! back-reference as an [`Event`] so callers can map pixels back to the bits
//! that produced them. Encode is split by concern: `lz77` finds matches,
//! `huffman` builds code tables in both directions (decode LUT and
//! package-merge-optimal lengths), and `encode` serializes the three block
//! types. `constants` holds the shared length/distance tables.

pub(crate) mod constants;
mod decoder;
mod encode;
mod error;
mod events;
mod huffman;
mod lz77;

pub use constants::{DBASE, DEXT};
pub use decoder::{DecodedDeflate, decode_deflate, inflate};
pub use encode::{compress, serialize_dynamic, serialize_fixed, serialize_stored};
pub use error::DecodeError;
pub use events::{EncTable, Event, LitEvent, RefEvent, SymCode, block_of};
