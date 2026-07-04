//! The editor application: `PngBendApp`, its per-frame `eframe::App::ui` loop,
//! and the state it drives.
//!
//! State is split five ways so the frame loop can borrow the pieces
//! independently: `Document` (the file on disk + undo history), `ViewState`
//! (frame pixels, overlays, the composited texture), `ListState` (the
//! left-panel filter), `Selection` (the clicked pixel and its derived edit
//! options), and `AsyncOps` (background load + dialog channels).
//!
//! The `eframe::App::ui` impl is the readable spine: each frame it handles
//! input, polls the async channels, rebuilds the texture when dirty, and draws
//! the panels. The submodules hold the pieces: `io` (load/save), `select` +
//! `edit` (the glitch pipeline), `nav`, `list_filter`, `overlay_cache`,
//! `row_text`, `ui`, and `history`.

use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use crate::coords::PixelXY;
use crate::index::CascadeScratch;

mod edit;
mod history;
mod input;
mod io;
mod list_filter;
mod overlay_cache;
mod row_text;
mod select;
mod ui;

use history::UndoHistory;
use io::{CoreData, DialogResult, LoadedFile};
use list_filter::{FilterRef, FilterSpec, filter_all, filter_lit, filter_refs};
use overlay_cache::{OverlayCache, OverlayMode};
use select::EditOption;

#[derive(PartialEq, Eq, Clone, Copy, Default)]
pub(super) enum PixelType {
    #[default]
    Lit,
    Ref,
    All,
}

/// Persistent file-document state: the bytes on disk and how to round-trip
/// them. `core` is the materialised view of `deflate_buf`; `history` is the
/// undo stack, kept here because each entry encodes a delta on `deflate_buf`.
#[derive(Default)]
pub(in crate::app) struct Document {
    pub(in crate::app) path: Option<PathBuf>,
    pub(in crate::app) chunks: Vec<crate::png::Chunk>,
    pub(in crate::app) zlib_header: [u8; 2],
    pub(in crate::app) deflate_buf: Vec<u8>,
    pub(in crate::app) dirty: bool,
    pub(in crate::app) core: Option<CoreData>,
    pub(in crate::app) history: UndoHistory,
}

/// Frame-pixel state: the base image, the overlay buffers, the
/// composited texture handed to egui.
#[derive(Default)]
pub(in crate::app) struct ViewState {
    pub(in crate::app) base_rgba: Vec<u8>,
    /// Reused target for overlay compositing. When no overlay is shown
    /// we upload `base_rgba` directly; otherwise `composite_into` blends
    /// into this buffer and we upload it. Held across frames so the
    /// per-rebuild allocation is paid once per file load, not per click.
    pub(in crate::app) composite_scratch: Vec<u8>,
    pub(in crate::app) texture: Option<egui::TextureHandle>,
    pub(in crate::app) texture_dirty: bool,
    pub(in crate::app) overlay_mode: OverlayMode,
    pub(in crate::app) overlay_cache: OverlayCache,
    pub(in crate::app) cascade_rgba: Option<Vec<u8>>,
    pub(in crate::app) cascade_scratch: CascadeScratch,
    /// Rows that changed on the last apply / undo. When `Some`, the next
    /// texture rebuild blends only these rows in `composite_scratch`;
    /// every other row already holds the last frame's composite, still
    /// valid because nothing else moved. Consumed by `compose_into_scratch`.
    pub(in crate::app) partial_composite_rows: Option<Vec<usize>>,
}

/// The predicate a `rebuild_filter` ran, kept so the next call can detect a
/// refinement and re-test only the current view. The three parts describe one
/// rebuild, so they travel as a unit: present together or absent together.
struct FilterSnapshot {
    spec: FilterSpec,
    editable_only: bool,
    pixel_type: PixelType,
}

/// Left-panel pixel-list state: what the user is filtering, what the
/// virtual scroll is showing, and the snapshot needed to detect filter
/// refinements between keystrokes.
#[derive(Default)]
pub(in crate::app) struct ListState {
    pub(in crate::app) pixel_type: PixelType,
    pub(in crate::app) filter_text: String,
    pub(in crate::app) editable_only: bool,
    /// Indices into the source [`PixelIndex`] for every row that passes
    /// the filter. `FilterRef` is 8 bytes, far cheaper to copy than
    /// the underlying [`PixelRow`] on a multi-megapixel keystroke
    /// rebuild.
    pub(in crate::app) filtered_view: Vec<FilterRef>,
    /// Reverse map: `filtered_idx[y * w + x]` is the index in
    /// `filtered_view` of the row for pixel `(x, y)`, or `u32::MAX` if
    /// that pixel isn't currently shown. Lookup is one indexed load;
    /// resetting it between rebuilds touches only the previous view's
    /// pixels rather than the full `w * h` slot range.
    pub(in crate::app) filtered_idx: Vec<u32>,
    /// Snapshot of the previous `rebuild_filter` predicate. When the
    /// next call is a refinement (user typed another character, flipped
    /// `editable_only` on, etc.) the rebuild re-tests only the rows in
    /// `filtered_view` rather than rescanning the full `PixelIndex`:
    /// `O(previous_view.len())` instead of `O(pixel_count)`.
    pub(in crate::app) last_filter: Option<FilterSnapshot>,
    pub(in crate::app) list_scroll_to: Option<usize>,
    pub(in crate::app) list_viewport_rows: usize,
}

/// The currently selected pixel and the edit options derived from it.
#[derive(Default)]
pub(in crate::app) struct Selection {
    pub(in crate::app) sel_pixel: Option<PixelXY>,
    pub(in crate::app) backref_src: Option<PixelXY>,
    pub(in crate::app) info_text: String,
    pub(in crate::app) edit_options: Vec<EditOption>,
    pub(in crate::app) selected_edit: Option<usize>,
}

/// Background-thread channels for the file load and the file dialog.
/// A load is in flight iff `load_rx.is_some()`; the UI uses
/// [`AsyncOps::is_loading`] to drive the spinner.
#[derive(Default)]
pub(in crate::app) struct AsyncOps {
    pub(in crate::app) load_rx: Option<Receiver<Result<LoadedFile, io::LoadError>>>,
    pub(in crate::app) dialog_rx: Option<Receiver<DialogResult>>,
}

impl AsyncOps {
    pub(in crate::app) fn is_loading(&self) -> bool {
        self.load_rx.is_some()
    }
}

#[derive(Default)]
pub struct PngBendApp {
    pub(in crate::app) doc: Document,
    pub(in crate::app) view: ViewState,
    pub(in crate::app) list: ListState,
    pub(in crate::app) sel: Selection,
    pub(in crate::app) async_ops: AsyncOps,

    // Whole-app UI ephemera (shown directly in widgets, no obvious group).
    pub(in crate::app) status: String,
    pub(in crate::app) hover_info: String,
    /// Last window-title string we asked egui to set. Caching prevents
    /// rebuilding + sending a `ViewportCommand::Title` every frame when
    /// nothing about the title (path, dirty state) has changed.
    pub(in crate::app) last_title: String,
    /// Whether the Help → About modal is open.
    pub(in crate::app) show_about: bool,
}

/// Row-major index of `xy` into a `w`-wide map (`filtered_idx`).
fn pixel_lin(xy: PixelXY, w: usize) -> usize {
    xy.y as usize * w + xy.x as usize
}

impl PngBendApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, path: Option<PathBuf>) -> Self {
        // Only four fields need non-Default starting values; everything
        // else falls out of `#[derive(Default)]` on the sub-structs.
        let mut app = Self {
            doc: Document {
                zlib_header: [0x78, 0x9C], // typical zlib header, overwritten on load
                ..Default::default()
            },
            list: ListState {
                list_viewport_rows: 25, // fallback until first layout pass measures
                ..Default::default()
            },
            sel: Selection {
                info_text: "Click a pixel to inspect it.".to_string(),
                ..Default::default()
            },
            status: "Drop a PNG or use File → Open.".to_string(),
            ..Self::default()
        };
        if let Some(p) = path {
            app.open_path(p);
        }
        app
    }

    /// Clear transient per-file state: selection, edits, undo history,
    /// cached overlays, and the left-panel filtered view. Shared between
    /// `open_path` (starting a load) and `on_load_done` (finishing one) so
    /// neither site has to remember the full list.
    pub(super) fn reset_for_new_file(&mut self) {
        self.sel.sel_pixel = None;
        self.reset_edit_state();
        self.doc.history.clear();
        self.view.overlay_cache.clear();
        self.view.cascade_rgba = None;
        self.doc.dirty = false;
        // Left-panel state. Otherwise the previous file's rows keep
        // showing until the new file's rebuild_filter runs, and a stale
        // list_scroll_to from the prior selection can auto-scroll the new
        // file's list to an arbitrary row.
        self.list.filtered_view.clear();
        self.list.filtered_idx.clear();
        self.list.last_filter = None;
        self.list.list_scroll_to = None;
    }

    /// Resolve a visible-list index to the underlying `PixelRow`. `None`
    /// if the index is out of range or no file is loaded.
    pub(super) fn filtered_row(&self, i: usize) -> Option<&crate::index::PixelRow> {
        let f = self.list.filtered_view.get(i).copied()?;
        let pi = &self.doc.core.as_ref()?.pixel_index;
        Some(f.resolve(pi))
    }

    /// `(FilterRef, PixelRow)` for the visible-list index. The row is
    /// returned by value (it's `Copy`) so callers can release the borrow
    /// on `self.doc.core` before touching `&mut self`, e.g. to fire a
    /// click handler that calls `select_pixel`.
    pub(in crate::app) fn filtered_row_full(
        &self,
        i: usize,
    ) -> Option<(FilterRef, crate::index::PixelRow)> {
        let f = self.list.filtered_view.get(i).copied()?;
        let pi = &self.doc.core.as_ref()?.pixel_index;
        Some((f, *f.resolve(pi)))
    }

    /// Resolve a visible-list index to its pixel coordinate.
    pub(super) fn filtered_xy(&self, i: usize) -> Option<PixelXY> {
        Some(self.filtered_row(i)?.xy())
    }

    /// `Some(i)` if pixel `xy` is in the current filtered view at index `i`.
    #[inline]
    pub(super) fn filtered_pos(&self, xy: PixelXY) -> Option<usize> {
        let c = self.doc.core.as_ref()?;
        let w = c.info.width as usize;
        let idx = pixel_lin(xy, w);
        let slot = self.list.filtered_idx.get(idx).copied()?;
        (slot != u32::MAX).then_some(slot as usize)
    }

    #[inline]
    pub(super) fn is_in_filtered(&self, xy: PixelXY) -> bool {
        self.filtered_pos(xy).is_some()
    }

    pub(super) fn rebuild_filter(&mut self) {
        let Some(c) = self.doc.core.as_ref() else {
            self.list.filtered_view.clear();
            self.list.filtered_idx.clear();
            self.list.last_filter = None;
            return;
        };
        let pi = &c.pixel_index;
        let w = c.info.width as usize;
        let pc = w * c.info.height as usize;

        // Parse once. Structured shapes (#hex, d=N, x,y) are O(1) per row;
        // only the `Generic` fallback formats each row.
        let new_spec = FilterSpec::parse(&self.list.filter_text);
        let new_editable = self.list.editable_only;
        let new_pixel_type = self.list.pixel_type;

        // Narrowing path: can we re-test *just* the current `filtered_view`
        // rather than rescanning the full `PixelIndex`? Two conditions:
        // (1) `pixel_type` didn't change (otherwise we're filtering a
        //     different source array and the indices would be wrong);
        // (2) the predicate only got stricter: same or tighter `editable_only`
        //     AND the new spec is a refinement of the previous.
        // When either fails we fall through to the full scan.
        let can_narrow = self.list.last_filter.as_ref().is_some_and(|prev| {
            prev.pixel_type == new_pixel_type
                && (!prev.editable_only || new_editable)
                && new_spec.is_refinement_of(&prev.spec)
        });

        // Reset filtered_idx slots that were set by the previous view.
        // First call (or after an image resize) sizes the Vec to w*h;
        // subsequent calls do O(previous_view.len()) work.
        if self.list.filtered_idx.len() != pc {
            self.list.filtered_idx.clear();
            self.list.filtered_idx.resize(pc, u32::MAX);
        } else {
            for f in &self.list.filtered_view {
                let xy = f.resolve(pi).xy();
                let idx = pixel_lin(xy, w);
                if let Some(slot) = self.list.filtered_idx.get_mut(idx) {
                    *slot = u32::MAX;
                }
            }
        }

        let mut scratch = String::with_capacity(48);
        let spec = &new_spec;
        let old_editable = self
            .list
            .last_filter
            .as_ref()
            .is_some_and(|p| p.editable_only);
        if can_narrow {
            // Re-test only the rows that already passed. Rows in the old
            // view already satisfied the old editable_only (if it was set),
            // so we only have to check editable when it just flipped on.
            let old_view = std::mem::take(&mut self.list.filtered_view);
            self.list.filtered_view = old_view
                .into_iter()
                .filter(|&fref| {
                    let row = fref.resolve(pi);
                    if new_editable && !old_editable && !row.has_edit {
                        return false;
                    }
                    spec.matches(fref, row, c, &mut scratch)
                })
                .collect();
        } else {
            let editable_only = new_editable;
            let predicate = |fref: FilterRef, row: &crate::index::PixelRow| -> bool {
                if editable_only && !row.has_edit {
                    return false;
                }
                spec.matches(fref, row, c, &mut scratch)
            };
            self.list.filtered_view = match new_pixel_type {
                PixelType::Lit => filter_lit(pi, predicate),
                PixelType::Ref => filter_refs(pi, predicate),
                PixelType::All => filter_all(pi, predicate),
            };
        }

        // Write the new indices back.
        for (i, f) in self.list.filtered_view.iter().enumerate() {
            let xy = f.resolve(pi).xy();
            let idx = pixel_lin(xy, w);
            if let Some(slot) = self.list.filtered_idx.get_mut(idx) {
                *slot = i as u32;
            }
        }

        // Snapshot for the next call's refinement check.
        self.list.last_filter = Some(FilterSnapshot {
            spec: new_spec,
            editable_only: new_editable,
            pixel_type: new_pixel_type,
        });
    }
}

impl eframe::App for PngBendApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // egui's `App::ui` hands us the root `Ui` directly; panel helpers
        // mount via `Panel::show_inside`. Non-drawing helpers (keyboard
        // handling, async polling, texture rebuild) only need `ctx`.
        let ctx = ui.ctx().clone();
        self.handle_keyboard_nav(&ctx);
        self.handle_dropped_files(&ctx);
        self.try_recv_load(&ctx);
        self.poll_dialog();
        self.ensure_overlay_cached();
        self.rebuild_texture(&ctx);

        self.ui_top_menu(ui, &ctx);
        self.ui_about_window(&ctx);
        self.handle_keyboard_shortcuts(&ctx);
        self.ui_status_bar(ui);
        self.ui_left_panel(ui);

        let clicks = self.ui_right_panel(ui);
        if clicks.apply {
            self.apply_edit();
            self.view.texture_dirty = true;
        }
        if clicks.undo {
            self.undo();
            self.view.texture_dirty = true;
        }
        if clicks.save {
            self.save_png(&ctx);
        }

        self.ui_central_panel(ui, &ctx);
    }
}
