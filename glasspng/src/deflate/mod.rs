//! DEFLATE (RFC 1951) codec, both directions.
//!
//! Decode (`decode`) scans blocks and records every literal and
//! back-reference as an [`Event`] so callers can map pixels back to the bits
//! that produced them. Encode (`encode`) is the mirror driver: it serializes
//! an event stream back into blocks. The two compression-only concerns it
//! leans on are separate: `lz77` finds matches, `huffman` builds code tables
//! (both directions: decode LUT and package-merge-optimal lengths).
//! `constants` holds the shared length/distance tables.

pub(crate) mod constants;
mod decode;
mod encode;
mod error;
mod events;
mod huffman;
mod lz77;

pub use constants::{DBASE, DEXT};
pub use decode::{DecodedDeflate, decode_deflate, inflate};
pub use encode::{compress, serialize_dynamic, serialize_fixed, serialize_stored};
pub use error::DecodeError;
pub use events::{EncTable, Event, LitEvent, RefEvent, SymCode, block_of};
