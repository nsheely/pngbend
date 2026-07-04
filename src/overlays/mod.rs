//! Overlay rendering: RGBA buffers alpha-composited over the base image to
//! visualise event structure.
//!
//! - `event`: per-event overlays (literal, distance, block). One RGBA
//!   buffer per mode, cached by [`crate::app`] and invalidated on load.
//! - `cascade`: the BFS-driven cascade overlay + PNG row-filter
//!   propagation, recomputed per pixel click.
//!
//! The compositor at [`crate::composite`] consumes one of these buffers but
//! isn't overlay generation, so it sits at the crate root, not here.

mod cascade;
mod event;

pub use cascade::{FilterExpansion, compute_filter_expansion, make_cascade_overlay_bytes};
pub use event::{
    make_block_overlay_bytes, make_distance_overlay_bytes, make_literal_overlay_bytes,
};
