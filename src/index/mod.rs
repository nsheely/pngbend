//! Derived indices over a decoded DEFLATE stream: event lookup tables, the
//! LZ77 reverse graph, the cascade BFS scratch, and the pixel-level summary
//! displayed in the side panel.

mod cascade;
mod pixel;
mod pos_to_ev;
mod reverse_graph;

pub use cascade::{Cascade, CascadeScratch};
pub use pixel::{PixelIndex, PixelRow, build_pixel_index, valid_dist_alts};
pub use pos_to_ev::{build_pos_to_ev, event_at};
pub use reverse_graph::{ReverseGraph, build_reverse_graph};
