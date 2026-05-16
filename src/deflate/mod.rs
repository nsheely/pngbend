//! DEFLATE (RFC 1951) decoder: Huffman tables, events, error types, and
//! the block-scanning driver.
//!
//! Keeps a record of every literal and back-reference in [`Event`] form so
//! callers can map pixels back to the bits that produced them.

pub(crate) mod constants;
mod decoder;
mod error;
mod events;
mod huffman;

pub use decoder::{DecodedDeflate, decode_deflate};
pub use error::DecodeError;
pub use events::{EncTable, Event, LitEvent, RefEvent};
