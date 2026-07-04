//! File I/O, threaded load, and file-dialog plumbing for the GUI app.
//!
//! - [`load`] background-loads a PNG on a worker thread and returns a
//!   fully-built [`CoreData`] + display buffer.
//! - [`dialog`] wraps `rfd` file-open/save dialogs in a channel protocol so
//!   the main thread stays responsive.

use crate::Raster;
use crate::deflate::{EncTable, Event};
use crate::index::{PixelIndex, ReverseGraph};
use crate::png::{PaletteEntry, PngInfo};

mod dialog;
mod load;

pub(super) use dialog::DialogResult;
pub(super) use load::{LoadError, LoadedFile};

/// All decoded state for one open PNG. Three groups: the decoded DEFLATE
/// stream (flattened from [`crate::deflate::DecodedDeflate`] so hot paths
/// read `c.events` / `c.output` directly, not through a nested field), the
/// indices derived from it, and the IHDR-derived image metadata.
pub(super) struct CoreData {
    // image metadata
    /// IHDR-derived metadata + layout: colour type and bit depth for
    /// codec dispatch, geometry fields and the pixel↔byte coordinate
    /// methods for everything else.
    pub info: PngInfo,
    /// Output-byte <-> screen-pixel projection. Equivalent to `info`'s
    /// coordinate methods for progressive images, and pass-aware for Adam7
    /// interlaced ones. Everything mapping pixels to bytes goes through it.
    pub raster: Raster,
    /// Optional decoded PLTE+tRNS (present for indexed-color PNGs).
    pub palette: Option<Vec<PaletteEntry>>,

    // decoded DEFLATE stream
    pub output: Vec<u8>,
    pub events: Vec<Event>,
    pub lit_encs: Vec<EncTable>,
    pub dist_encs: Vec<EncTable>,
    /// Per-block event-start indices; see
    /// [`crate::deflate::DecodedDeflate::block_starts`]. Used with
    /// [`crate::deflate::block_of`] to resolve a clicked event's block
    /// for the per-block Huffman-table lookup.
    pub block_starts: Vec<u32>,
    /// Largest LZ77 back-reference distance in `events`, cached so the
    /// distance overlay renderer doesn't rescan the event list each time
    /// its cache entry is (re)built.
    pub max_distance: u32,

    // indices derived from the stream + metadata
    pub pixel_index: PixelIndex,
    pub reverse_graph: ReverseGraph,
    /// Full unfiltered PNG pixel bytes (`h * (row_stride - 1)` bytes). Held
    /// across edits so the incremental apply path can re-run only the
    /// affected rows through the row-filter inverse.
    pub unfiltered: Vec<u8>,
}

impl CoreData {
    /// Number of DEFLATE blocks: one `block_starts` entry per block.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.block_starts.len()
    }

    /// Whether pixel overlays can be shown. They project output positions
    /// through the progressive layout, so they're gated to non-interlaced
    /// images until the overlay renderers become pass-aware.
    #[inline]
    pub fn overlays_supported(&self) -> bool {
        !self.info.interlaced
    }
}
