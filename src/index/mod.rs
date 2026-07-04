//! Derived indices over a decoded DEFLATE stream: event lookup tables, the
//! LZ77 reverse graph, the cascade BFS scratch, and the side-panel pixel
//! summary.

mod cascade_bfs;
mod pixel;
mod pos_to_ev;
mod reverse_graph;

pub use cascade_bfs::{Cascade, CascadeScratch};
pub use pixel::{DistAlt, PixelIndex, PixelRow, build_pixel_index, valid_dist_alts};
pub use pos_to_ev::{build_pos_to_ev, event_at};
pub use reverse_graph::{ReverseGraph, build_reverse_graph};
