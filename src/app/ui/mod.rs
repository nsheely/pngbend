//! GUI panel rendering split by concern:
//! - [`panels`]: top menu, status bar, left pixel list, right overlay/edit.
//! - [`image`]: the central image view, texture rebuilding, and Painter-
//!   based selection markers drawn on top.

mod image;
mod panels;
