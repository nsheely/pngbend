//! File I/O, threaded load, and file-dialog plumbing for the GUI app.
//!
//! - [`load`] background-loads a PNG on a worker thread and returns a
//!   fully-built [`CoreData`] + display buffer.
//! - [`dialog`] wraps `rfd` file-open/save dialogs in a channel protocol so
//!   the main thread stays responsive.

use crate::coords::ImgGeom;
use crate::deflate::{EncTable, Event};
use crate::index::{PixelIndex, ReverseGraph};
use crate::png::{PaletteEntry, PngInfo};

mod dialog;
mod load;

pub(super) use dialog::DialogResult;
pub(super) use load::{LoadError, LoadedFile};

/// All decoded data for one PNG.
pub(super) struct CoreData {
    /// Width/height/bpp/row_stride bundled so coordinate conversions
    /// take one `&ImgGeom` argument.
    pub geom: ImgGeom,
    /// Full IHDR-derived info; carries color type and bit depth so the
    /// in-app PNG filter + RGBA converter dispatch without re-parsing.
    pub info: PngInfo,
    /// Optional decoded PLTE+tRNS (present for indexed-color PNGs).
    pub palette: Option<Vec<PaletteEntry>>,
    pub output: Vec<u8>,
    pub events: Vec<Event>,
    pub lit_encs: Vec<EncTable>,
    pub dist_encs: Vec<EncTable>,
    pub num_blocks: usize,
    pub pixel_index: PixelIndex,
    pub reverse_graph: ReverseGraph,
    /// Full unfiltered PNG pixel bytes (`h * (row_stride - 1)` bytes). Held
    /// across edits so the incremental apply path can re-run only the
    /// affected rows through the row-filter inverse.
    pub unfiltered: Vec<u8>,
    /// Largest LZ77 back-reference distance in `events`, cached so the
    /// distance overlay renderer doesn't rescan the event list each time
    /// its cache entry is (re)built.
    pub max_distance: u32,
}
