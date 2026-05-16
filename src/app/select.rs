//! Pixel selection pipeline.
//!
//! Every selection arrives at [`PngBendApp::select_pixel`] tagged with a
//! [`SelectSource`] describing how the user got here. The variants drive
//! three decisions: whether to snap to the nearest visible filtered
//! pixel, whether to scroll the list, and whether to rebuild the
//! cascade overlay.

use std::collections::HashMap;

use egui::Color32;

use crate::coords::OutPos;
use crate::deflate::{Event, RefEvent};
use crate::index::{CascadeScratch, PixelRow, event_at, valid_dist_alts};
use crate::overlays::{compute_filter_expansion, make_cascade_overlay_bytes};

use super::PngBendApp;
use super::edit::{EditAction, EditKind, Patch};
use super::io::CoreData;
use super::overlay_cache::OverlayMode;

const CH_NAMES: [&str; 4] = ["R", "G", "B", "A"];

/// What triggered a call to [`PngBendApp::select_pixel`]. Drives whether
/// the click is snapped, whether the list scrolls, and whether the cascade
/// overlay rebuilds.
#[derive(Debug, Clone, Copy)]
pub(super) enum SelectSource {
    /// User clicked the image. Snap to the nearest filtered pixel; scroll
    /// the list to follow.
    ImageClick,
    /// Keyboard navigation through the list (arrows / PgUp / PgDn / Home /
    /// End). The list owns focus; we scroll to keep the new selection in
    /// view but don't snap.
    ListNav,
    /// Selection re-derives from existing state — a list-row click or a
    /// post-redirect refocus. The originator already has focus, so neither
    /// snap nor scroll.
    Refocus,
    /// Side-panel refresh after an in-place literal swap. LZ77 topology is
    /// unchanged, so the cascade overlay painted on screen is still
    /// correct — keep it instead of rebuilding.
    AfterLiteralSwap,
}

impl SelectSource {
    fn snaps_to_filtered(self) -> bool {
        matches!(self, Self::ImageClick)
    }

    fn scrolls_list(self) -> bool {
        matches!(self, Self::ImageClick | Self::ListNav)
    }

    fn rebuilds_cascade(self) -> bool {
        !matches!(self, Self::AfterLiteralSwap)
    }
}

/// One entry in the side-panel "Available edits" list. `action` is what
/// gets applied; the rest is for the row's visual presentation.
pub(super) struct EditOption {
    pub label: String,
    pub bg_color: Color32,
    pub fg_color: Color32,
    pub action: EditAction,
}

/// One channel of the clicked pixel whose literal could be swapped.
/// Captured once at selection time so the apply path doesn't have to
/// re-resolve the event for each candidate edit.
#[derive(Debug, Clone, Copy)]
struct LitPatchSite {
    /// Channel index within the pixel (0..bpp).
    ch: usize,
    /// Bit position in the deflate stream where this literal's code lives.
    bit_start: u32,
    /// Output-byte position this literal produces.
    out_pos: u32,
}

impl PngBendApp {
    pub(super) fn select_pixel(&mut self, x: u32, y: u32, source: SelectSource) {
        let (mut sx, sy) = if source.snaps_to_filtered() {
            self.snap_to_nearest_filtered(x, y)
        } else {
            (x, y)
        };
        // Snap to the byte's cluster start so the sidebar row, info
        // text, and cascade highlight all agree on which cluster was
        // clicked. For ≥ 8-bit depths `pixels_per_byte == 1` and the
        // arithmetic is a no-op.
        if let Some(c) = self.doc.core.as_ref() {
            let ppb = c.geom.pixels_per_byte();
            sx = (sx / ppb) * ppb;
        }
        self.reset_edit_state();
        self.sel.sel_pixel = Some((sx, sy));

        let Some(c) = self.doc.core.as_ref() else {
            return;
        };
        let base_raw = c.geom.xy_to_out(crate::coords::PixelXY::new(sx, sy)).0 as usize;
        let evs = gather_pixel_events(c, base_raw);

        if evs.is_empty() {
            self.sel.info_text = format!("No event for pixel ({sx}, {sy})");
            if source.rebuilds_cascade() {
                self.view.cascade_rgba = None;
            }
            self.view.texture_dirty = true;
            return;
        }

        let SelectionBuild {
            mut info_lines,
            edit_options,
            backref_src,
            cascade_positions,
        } = build_selection_data(c, &evs, sx, sy, base_raw);

        self.sel.backref_src = backref_src;
        self.sel.edit_options = edit_options;

        if self.sel.edit_options.is_empty() {
            info_lines.push("  (no valid edits for this pixel)".to_string());
        }

        if source.rebuilds_cascade() {
            if cascade_positions.is_empty() {
                self.view.cascade_rgba = None;
            } else {
                // Split-borrow: `view.cascade_scratch` is mut-borrowed while
                // `doc.core` stays immutably borrowed — disjoint sub-struct
                // fields, so the borrow checker is happy.
                let c = self.doc.core.as_ref().expect("checked above");
                let summary = compute_cascade_for_pixel(
                    c,
                    &mut self.view.cascade_scratch,
                    &cascade_positions,
                );
                info_lines.push(summary.text);
                self.view.cascade_rgba = Some(summary.overlay);
                // Auto-show the cascade overlay only when the user hasn't
                // chosen a visualisation yet. Once they pick a mode (even
                // Cascade explicitly), subsequent clicks preserve it.
                if matches!(self.view.overlay_mode, OverlayMode::None) {
                    self.view.overlay_mode = OverlayMode::Cascade;
                }
            }
        }

        if source.scrolls_list()
            && let Some(row) = self.filtered_pos((sx, sy))
        {
            self.list.list_scroll_to = Some(row);
        }

        self.sel.info_text = info_lines.join("\n");
        self.view.texture_dirty = true;
    }

    /// Rebuild the side panel for the currently selected pixel without
    /// disturbing the cascade overlay. The literal-swap edit path uses this
    /// to refresh edit options and info text after a topology-stable edit.
    pub(super) fn refresh_selection_after_literal_swap(&mut self) {
        let Some((sx, sy)) = self.sel.sel_pixel else {
            return;
        };
        self.select_pixel(sx, sy, SelectSource::AfterLiteralSwap);
    }

    /// Find the filtered-view pixel whose `(x, y)` is closest (by squared
    /// Euclidean distance) to the click coordinates. Falls through to the
    /// raw click if the filtered view is empty or already contains it.
    fn snap_to_nearest_filtered(&self, x: u32, y: u32) -> (u32, u32) {
        if self.list.filtered_view.is_empty() || self.is_in_filtered((x, y)) {
            return (x, y);
        }
        (0..self.list.filtered_view.len())
            .filter_map(|i| self.filtered_xy(i))
            .min_by_key(|&(px, py)| {
                let dx = px as i64 - x as i64;
                let dy = py as i64 - y as i64;
                dx * dx + dy * dy
            })
            .unwrap_or((x, y))
    }

    pub(super) fn reset_edit_state(&mut self) {
        self.sel.pending_edit = None;
        self.sel.selected_edit = None;
        self.sel.edit_options.clear();
        self.sel.backref_src = None;
    }
}

// ── pure helpers ─────────────────────────────────────────────────────────────

/// Binary-search a `PixelRow` slice sorted by `(y, x)` for the row at
/// `(x, y)`. Both `pixel_index.lit` and `pixel_index.refs` are built in
/// raster order by [`crate::index::build_pixel_index`].
#[inline]
fn find_pixel_row(rows: &[PixelRow], x: u32, y: u32) -> Option<usize> {
    rows.binary_search_by_key(&(y, x), |r| (r.y(), r.x())).ok()
}

fn gather_pixel_events(c: &CoreData, base_raw: usize) -> Vec<(usize, usize)> {
    (0..c.geom.bpp as usize)
        .filter_map(|ch| {
            let pos = base_raw + ch;
            event_at(&c.events, pos).map(|idx| (ch, idx as usize))
        })
        .collect()
}

struct SelectionBuild {
    info_lines: Vec<String>,
    edit_options: Vec<EditOption>,
    backref_src: Option<(u32, u32)>,
    cascade_positions: Vec<u32>,
}

fn build_selection_data(
    c: &CoreData,
    evs: &[(usize, usize)],
    sx: u32,
    sy: u32,
    base_raw: usize,
) -> SelectionBuild {
    // `pixel_index.lit` / `.refs` are sorted by `(y, x)` at build time,
    // so resolving the selected pixel's row in either array is one
    // `O(log N)` binary search rather than a linear scan.
    let idx_str = match (
        find_pixel_row(&c.pixel_index.lit, sx, sy),
        find_pixel_row(&c.pixel_index.refs, sx, sy),
    ) {
        (Some(i), _) => format!("  lit #{}", i + 1),
        (_, Some(i)) => format!("  ref #{}", i + 1),
        _ => String::new(),
    };

    let mut info_lines = vec![format!("Pixel ({sx}, {sy}){idx_str}")];
    // Sub-byte PNGs pack several pixels into one byte; warn the user so
    // they understand a "literal swap" here recolours the whole cluster.
    // At the right edge of the image the last cluster may be smaller
    // than `pixels_per_byte` (the trailing bits are padding).
    let ppb = c.geom.pixels_per_byte();
    if ppb > 1 {
        let cluster_end = (sx + ppb).min(c.geom.w);
        let cluster_size = cluster_end - sx;
        let last_x = cluster_end - 1;
        let plural = if cluster_size > 1 { "s" } else { "" };
        info_lines.push(format!(
            "  (this byte encodes {cluster_size} pixel{plural} at y={sy}, x={sx}..{last_x} — edits affect all of them)"
        ));
    }
    let mut edit_options: Vec<EditOption> = Vec::new();
    let mut cascade_positions: Vec<u32> = Vec::new();
    // Grouped by (symbol, block) so one `EditOption` covers every channel
    // sharing those — a pixel with all three RGB bytes literal-encoded
    // in the same block becomes a single "literal X → Y" row, not three.
    let mut lit_groups: HashMap<(u8, u32), Vec<LitPatchSite>> = HashMap::new();
    let mut backref_src = None;

    for &(ch, ev_idx) in evs {
        let ch_name = CH_NAMES.get(ch).copied().unwrap_or("?");
        match &c.events[ev_idx] {
            Event::Lit(lit) => {
                cascade_positions.push(lit.out_pos);
                let le = &c.lit_encs[lit.block as usize];
                let (_, clen) = le.get(lit.symbol as u16).unwrap_or((0, 0));
                let same_count = le
                    .iter()
                    .filter(|&(s, _, cl)| cl == clen && s < 256 && s != lit.symbol as u16)
                    .count();
                info_lines.push(format!(
                    "  {ch_name}: LITERAL  val={}  block={}  {clen}-bit  {same_count} swap options",
                    lit.symbol, lit.block,
                ));
                lit_groups
                    .entry((lit.symbol, lit.block))
                    .or_default()
                    .push(LitPatchSite {
                        ch,
                        bit_start: lit.bit_start,
                        out_pos: lit.out_pos,
                    });
            }
            Event::Ref(r) => {
                cascade_positions.push(r.out_pos);
                let dist = r.out_pos - r.src_out_pos;
                let src_xy = c
                    .geom
                    .out_to_xy(OutPos(r.src_out_pos))
                    .map(|xy| (xy.x, xy.y));
                if backref_src.is_none() {
                    backref_src = src_xy;
                }
                let alts =
                    valid_dist_alts(r.block, r.dist_sym, r.out_pos, r.src_out_pos, &c.dist_encs);
                info_lines.push(format!(
                    "  {ch_name}: BACKREF  val={}  block={}  from={src_xy:?}  dist={dist}  len={}  {} redirect options",
                    c.output.get(base_raw + ch).copied().unwrap_or(0),
                    r.block,
                    r.copy_len,
                    alts.len(),
                ));
                edit_options.extend(build_ref_redirect_edits(c, r, ch_name, &alts));
            }
        }
    }

    edit_options.extend(build_lit_swap_edits(c, &lit_groups));

    SelectionBuild {
        info_lines,
        edit_options,
        backref_src,
        cascade_positions,
    }
}

fn build_ref_redirect_edits(
    c: &CoreData,
    r: &RefEvent,
    ch_name: &str,
    alts: &[(u8, u32, u32)],
) -> Vec<EditOption> {
    let de = &c.dist_encs[r.block as usize];
    let (_, old_len) = de.get(r.dist_sym as u16).unwrap_or((0, 0));
    let dist = r.out_pos - r.src_out_pos;

    alts.iter()
        .filter_map(|&(new_dsym, new_src, new_dist)| {
            let (new_code, new_len) = de.get(new_dsym as u16).unwrap_or((0, 0));
            if new_len != old_len {
                return None;
            }
            let src_xy2 = c.geom.out_to_xy(OutPos(new_src)).map(|xy| (xy.x, xy.y));
            let new_src_us = new_src as usize;
            let preview: Vec<u8> = (0..c.geom.bpp as usize)
                .filter_map(|i| c.output.get(new_src_us + i).copied())
                .collect();
            let label = format!(
                "[{ch_name}] REDIRECT  dist {dist} → {new_dist}  src {src_xy2:?}  val≈{preview:?}"
            );
            let bg = preview_color(&preview);
            Some(EditOption {
                label,
                bg_color: bg,
                fg_color: contrast_text_color(bg),
                action: EditAction {
                    patches: vec![Patch {
                        bit_start: r.dist_bit_start,
                        value: new_code as u32,
                        code_len: new_len,
                    }],
                    label: format!("redirect dist_sym → {new_dsym}"),
                    kind: EditKind::DistRedirect {
                        out_pos: r.out_pos,
                        copy_len: r.copy_len,
                        src_after: new_src,
                        dist_sym_after: new_dsym,
                    },
                },
            })
        })
        .collect()
}

/// RGB swatch for a distance redirect's target pixel — falls back to grey
/// of the channel average when fewer than 3 channels are available.
fn preview_color(preview: &[u8]) -> Color32 {
    let avg = if preview.is_empty() {
        128u8
    } else {
        (preview.iter().map(|&v| v as u32).sum::<u32>() / preview.len() as u32) as u8
    };
    Color32::from_rgb(
        preview.first().copied().unwrap_or(avg),
        preview.get(1).copied().unwrap_or(avg),
        preview.get(2).copied().unwrap_or(avg),
    )
}

fn build_lit_swap_edits(
    c: &CoreData,
    lit_groups: &HashMap<(u8, u32), Vec<LitPatchSite>>,
) -> Vec<EditOption> {
    let mut out = Vec::new();
    for ((val, blk), sites) in lit_groups {
        let le = &c.lit_encs[*blk as usize];
        let (_, clen) = le.get(*val as u16).unwrap_or((0, 0));
        let mut swappable: Vec<u16> = le
            .iter()
            .filter(|&(s, _, cl)| cl == clen && s < 256 && s != *val as u16)
            .map(|(s, _, _)| s)
            .collect();
        swappable.sort();
        let chs: String = sites
            .iter()
            .map(|s| CH_NAMES.get(s.ch).copied().unwrap_or("?"))
            .collect::<Vec<_>>()
            .join(",");
        for tgt in swappable {
            let (new_code, new_len) = le.get(tgt).unwrap_or((0, 0));
            let label = format!("[{chs}] LITERAL  {val} → {tgt}  (both {clen}-bit)");
            let bg = Color32::from_gray(tgt as u8);
            let fg = Color32::from_gray(255u8.saturating_sub(tgt as u8));
            let patches: Vec<Patch> = sites
                .iter()
                .map(|s| Patch {
                    bit_start: s.bit_start,
                    value: new_code as u32,
                    code_len: new_len,
                })
                .collect();
            let byte_updates: Vec<(u32, u8)> =
                sites.iter().map(|s| (s.out_pos, tgt as u8)).collect();
            out.push(EditOption {
                label,
                bg_color: bg,
                fg_color: fg,
                action: EditAction {
                    patches,
                    label: format!("literal {val} → {tgt}"),
                    kind: EditKind::LiteralSwap { byte_updates },
                },
            });
        }
    }
    out
}

pub(super) fn contrast_text_color(bg: Color32) -> Color32 {
    let luma = 0.299 * bg.r() as f32 + 0.587 * bg.g() as f32 + 0.114 * bg.b() as f32;
    if luma < 128.0 {
        Color32::WHITE
    } else {
        Color32::BLACK
    }
}

struct CascadeSummary {
    overlay: Vec<u8>,
    text: String,
}

fn compute_cascade_for_pixel(
    c: &CoreData,
    scratch: &mut CascadeScratch,
    positions: &[u32],
) -> CascadeSummary {
    let cascade = scratch.run(positions, &c.reverse_graph);
    let filter = compute_filter_expansion(cascade.affected, &c.output, &c.geom);
    // Skip per-row PNG filter bytes when reporting affected count.
    let row_stride = c.geom.row_stride as usize;
    let n_affected = cascade
        .affected
        .iter()
        .filter(|&&p| !(p as usize).is_multiple_of(row_stride))
        .count();
    let n_filter: u32 = filter
        .iter()
        .map(|(_, mx)| c.geom.w.saturating_sub(mx))
        .sum();
    let max_depth = cascade.max_depth;
    let overlay = make_cascade_overlay_bytes(&cascade, &filter, &c.geom);
    CascadeSummary {
        overlay,
        text: format!(
            "\n  LZ77 cascade: {n_affected} bytes  (max depth {max_depth})\n  filter halo: ~{n_filter} px (faint blue)"
        ),
    }
}
