//! Overlay rendering — the RGBA buffers that get alpha-composited over the
//! base image to visualise event structure.
//!
//! - [`event`] — per-event overlays (literal, distance, block). One RGBA
//!   buffer per mode, cached by [`crate::app`] and invalidated on load.
//! - [`cascade`] — the BFS-driven cascade overlay + PNG row-filter
//!   propagation, recomputed per pixel click.
//!
//! The compositor itself lives at [`crate::composite`]: it consumes one of
//! the buffers built here, but it isn't overlay generation, so it sits at
//! the crate root rather than in this module.

mod cascade;
mod event;

pub use cascade::{FilterExpansion, compute_filter_expansion, make_cascade_overlay_bytes};
pub use event::{
    make_block_overlay_bytes, make_distance_overlay_bytes, make_literal_overlay_bytes,
};
