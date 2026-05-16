//! Formats a single pixel-list row's display text on demand. Two
//! callers:
//!
//! - The sidebar's virtual-scroll callback runs this once per visible
//!   row per frame (~25 rows / frame).
//! - [`super::list_filter::FilterSpec::Generic`] runs it per candidate
//!   row when the filter text doesn't match a structured shape, sharing
//!   a reused `String` scratch so the filter rebuild stays no-alloc.
//!
//! Formatting is per-call rather than baked into [`PixelRow`] so the
//! index never holds millions of strings the user won't see.

use std::fmt::Write;

use crate::coords::PixelXY;
use crate::deflate::Event;
use crate::index::{PixelRow, event_at};

use super::io::CoreData;
use super::list_filter::FilterRef;

/// Append the sidebar display text for this row to `out`. `out` is not
/// cleared — callers that reuse a scratch buffer across rows must clear
/// it themselves.
pub(super) fn append_row_text(out: &mut String, fref: FilterRef, row: &PixelRow, c: &CoreData) {
    let idx = fref.display_index();
    let (x, y) = row.xy();
    // At sub-byte depths one byte holds several pixels — append a "×N"
    // cluster-size tag so the user sees the edit's reach. The last
    // cluster of a row may be smaller than `pixels_per_byte` when the
    // image width isn't a clean multiple. ≥ 8-bit modes get
    // `cluster_size == 1` and the tag is omitted.
    let cluster_size = (x + c.geom.pixels_per_byte()).min(c.geom.w) - x;
    let xy_text = if cluster_size > 1 {
        format!("({x:4},{y:4})×{cluster_size}")
    } else {
        format!("({x:4},{y:4})")
    };
    match fref {
        FilterRef::Lit(_) => {
            let [r, g, b] = row.rgb;
            let bpp = c.geom.bpp as usize;
            if bpp >= 3 {
                let _ = write!(out, "{idx:5}  {xy_text}  #{r:02x}{g:02x}{b:02x}");
            } else {
                let _ = write!(out, "{idx:5}  {xy_text}  {r:3}");
            }
        }
        FilterRef::Ref(_) => {
            let (dist, copy_len) = lookup_ref_metrics(c, row.xy()).unwrap_or((0, 0));
            let _ = write!(out, "{idx:5}  {xy_text}  d={dist:5} len={copy_len}");
        }
    }
}

/// Walk the pixel's channels to find the first back-reference event and
/// return its distance + copy length. The sidebar is virtual-scrolled, so
/// this runs only per rendered or filter-matched row — a few `event_at`
/// binary searches per call.
pub(super) fn lookup_ref_metrics(c: &CoreData, xy: (u32, u32)) -> Option<(usize, u16)> {
    let bpp = c.geom.bpp as usize;
    let base = c.geom.xy_to_out(PixelXY::new(xy.0, xy.1)).0 as usize;
    for ch in 0..bpp {
        let pos = base + ch;
        let Some(ev_idx) = event_at(&c.events, pos) else {
            continue;
        };
        if let Event::Ref(r) = &c.events[ev_idx as usize] {
            return Some(((r.out_pos - r.src_out_pos) as usize, r.copy_len));
        }
    }
    None
}
