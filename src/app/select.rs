//! Pixel selection pipeline.
//!
//! Every selection arrives at [`PngBendApp::select_pixel`] tagged with a
//! [`SelectSource`] describing how the user got here. The variants drive
//! three decisions: whether to snap to the nearest visible filtered
//! pixel, whether to scroll the list, and whether to rebuild the
//! cascade overlay.

use std::collections::HashMap;

use egui::Color32;

use crate::coords::{OutPos, PixelXY};
use crate::deflate::{EncTable, Event, RefEvent, SymCode, block_of};
use crate::index::{CascadeScratch, DistAlt, PixelRow, event_at, valid_dist_alts};
use crate::overlays::{compute_filter_expansion, make_cascade_overlay_bytes};

use super::PngBendApp;
use super::edit::{ByteWrite, EditAction, EditKind, Patch};
use super::io::CoreData;
use super::overlay_cache::OverlayMode;

/// What triggered a call to [`PngBendApp::select_pixel`]. Drives snapping,
/// list scrolling, and cascade-overlay rebuild.
#[derive(Debug, Clone, Copy)]
pub(super) enum SelectSource {
    /// User clicked the image. Snap to the nearest filtered pixel; scroll
    /// the list to follow.
    ImageClick,
    /// Keyboard navigation through the list (arrows / PgUp / PgDn / Home /
    /// End). The list owns focus; we scroll to keep the new selection in
    /// view but don't snap.
    ListNav,
    /// Selection re-derives from existing state: a list-row click or a
    /// post-redirect refocus. The originator already has focus, so neither
    /// snap nor scroll.
    Refocus,
    /// Side-panel refresh after an in-place literal swap. LZ77 topology is
    /// unchanged, so the cascade overlay painted on screen is still
    /// correct; keep it instead of rebuilding.
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
    pub(super) fn select_pixel(&mut self, xy: PixelXY, source: SelectSource) {
        let mut sel = if source.snaps_to_filtered() {
            self.snap_to_nearest_filtered(xy)
        } else {
            xy
        };
        // Snap to the byte's cluster start so the sidebar row, info
        // text, and cascade highlight all agree on which cluster was
        // clicked. For ≥ 8-bit depths `pixels_per_byte == 1` and the
        // arithmetic is a no-op.
        if let Some(c) = self.doc.core.as_ref() {
            let ppb = c.raster.pixels_per_byte();
            sel.x = (sel.x / ppb) * ppb;
        }
        self.reset_edit_state();
        self.sel.sel_pixel = Some(sel);

        let Some(c) = self.doc.core.as_ref() else {
            return;
        };
        let base_raw = c.raster.xy_to_out(sel).0 as usize;
        let evs = gather_pixel_events(c, base_raw);

        if evs.is_empty() {
            self.sel.info_text = format!("No event for pixel ({}, {})", sel.x, sel.y);
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
        } = build_selection_data(c, &evs, sel, base_raw);

        self.sel.backref_src = backref_src;
        self.sel.edit_options = edit_options;

        if self.sel.edit_options.is_empty() {
            info_lines.push("  (no valid edits for this pixel)".to_string());
        }

        if source.rebuilds_cascade() {
            // The cascade overlay projects output positions to pixels via the
            // progressive layout; skip it for interlaced images (overlays
            // aren't pass-aware yet).
            let progressive = self
                .doc
                .core
                .as_ref()
                .is_some_and(|c| c.overlays_supported());
            if cascade_positions.is_empty() || !progressive {
                self.view.cascade_rgba = None;
            } else {
                // Split-borrow: `view.cascade_scratch` mut-borrowed while
                // `doc.core` stays immutably borrowed; disjoint sub-struct
                // fields.
                let c = self.doc.core.as_ref().expect("checked above");
                let summary = compute_cascade_for_pixel(
                    c,
                    &mut self.view.cascade_scratch,
                    &cascade_positions,
                );
                info_lines.push(summary.text);
                self.view.cascade_rgba = Some(summary.overlay);
                // Auto-show the cascade overlay only until the user picks a
                // mode from the selector. `overlay_user_set` distinguishes an
                // explicit `None` from the initial default, which the bare
                // mode value can't.
                if !self.view.overlay_user_set {
                    self.view.overlay_mode = OverlayMode::Cascade;
                }
            }
        }

        if source.scrolls_list()
            && let Some(row) = self.filtered_pos(sel)
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
        let Some(sel) = self.sel.sel_pixel else {
            return;
        };
        self.select_pixel(sel, SelectSource::AfterLiteralSwap);
    }

    /// Find the filtered-view pixel whose coordinate is closest (by
    /// squared Euclidean distance) to the click. Falls through to the
    /// raw click if the filtered view is empty or already contains it.
    fn snap_to_nearest_filtered(&self, xy: PixelXY) -> PixelXY {
        if self.list.filtered_view.is_empty() || self.is_in_filtered(xy) {
            return xy;
        }
        (0..self.list.filtered_view.len())
            .filter_map(|i| self.filtered_xy(i))
            .min_by_key(|p| {
                let dx = p.x as i64 - xy.x as i64;
                let dy = p.y as i64 - xy.y as i64;
                dx * dx + dy * dy
            })
            .unwrap_or(xy)
    }

    pub(super) fn reset_edit_state(&mut self) {
        self.sel.selected_edit = None;
        self.sel.edit_options.clear();
        self.sel.backref_src = None;
    }
}

// pure helpers

/// Render an optional pixel position for the UI as `(x, y)`, or `off-image`
/// when the byte doesn't map to a pixel (a filter byte, or a source that
/// falls outside the image).
pub(in crate::app) fn fmt_pixel(p: Option<PixelXY>) -> String {
    match p {
        Some(p) => format!("({}, {})", p.x, p.y),
        None => "off-image".to_string(),
    }
}

/// Binary-search a `PixelRow` slice sorted by `(y, x)` for the row at
/// `xy`. Both `pixel_index.lit` and `pixel_index.refs` are built in
/// raster order by [`crate::index::build_pixel_index`].
#[inline]
fn find_pixel_row(rows: &[PixelRow], xy: PixelXY) -> Option<usize> {
    rows.binary_search_by_key(&(xy.y, xy.x), |r| (r.y(), r.x()))
        .ok()
}

/// A byte of the clicked pixel and the event that produced it: `ch` is
/// the channel byte offset within the pixel (`0..bpp`), `ev_idx` indexes
/// [`CoreData::events`].
#[derive(Debug, Clone, Copy)]
struct ChannelEvent {
    ch: usize,
    ev_idx: usize,
}

fn gather_pixel_events(c: &CoreData, base_raw: usize) -> Vec<ChannelEvent> {
    (0..c.info.bpp)
        .filter_map(|ch| {
            let pos = base_raw + ch;
            event_at(&c.events, pos).map(|idx| ChannelEvent {
                ch,
                ev_idx: idx as usize,
            })
        })
        .collect()
}

struct SelectionBuild {
    info_lines: Vec<String>,
    edit_options: Vec<EditOption>,
    backref_src: Option<PixelXY>,
    cascade_positions: Vec<u32>,
}

fn build_selection_data(
    c: &CoreData,
    evs: &[ChannelEvent],
    sel: PixelXY,
    base_raw: usize,
) -> SelectionBuild {
    let (sx, sy) = (sel.x, sel.y);
    // `pixel_index.lit` / `.refs` are sorted by `(y, x)` at build time,
    // so resolving the selected pixel's row in either array is one
    // `O(log N)` binary search rather than a linear scan.
    let idx_str = match (
        find_pixel_row(&c.pixel_index.lit, sel),
        find_pixel_row(&c.pixel_index.refs, sel),
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
    let ppb = c.raster.pixels_per_byte();
    if ppb > 1 {
        let cluster_end = (sx + ppb).min(c.info.width);
        let cluster_size = cluster_end - sx;
        let last_x = cluster_end - 1;
        let plural = if cluster_size > 1 { "s" } else { "" };
        info_lines.push(format!(
            "  (this byte encodes {cluster_size} pixel{plural} at y={sy}, x={sx}..{last_x}; edits affect all of them)"
        ));
    }
    let mut edit_options: Vec<EditOption> = Vec::new();
    let mut cascade_positions: Vec<u32> = Vec::new();
    // Grouped by (symbol, block) so one `EditOption` covers every channel
    // sharing those: a pixel with all three RGB bytes literal-encoded
    // in the same block becomes a single "literal X → Y" row, not three.
    let mut lit_groups: HashMap<(u8, u32), Vec<LitPatchSite>> = HashMap::new();
    let mut backref_src = None;

    for &ChannelEvent { ch, ev_idx } in evs {
        let ch_name = c.info.channel_label(ch);
        let block = block_of(&c.block_starts, ev_idx as u32);
        match &c.events[ev_idx] {
            Event::Lit(lit) => {
                cascade_positions.push(lit.out_pos);
                let le = &c.lit_encs[block as usize];
                let clen = le.get(lit.symbol as u16).map_or(0, |c| c.len);
                let same_count = same_len_lit_swaps(le, lit.symbol as u16).count();
                info_lines.push(format!(
                    "  {ch_name}: LITERAL  val={}  block={block}  {clen}-bit  {same_count} swap options",
                    lit.symbol,
                ));
                lit_groups
                    .entry((lit.symbol, block))
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
                let src_xy = c.raster.out_to_xy(OutPos(r.src_out_pos));
                if backref_src.is_none() {
                    backref_src = src_xy;
                }
                let alts =
                    valid_dist_alts(block, r.dist_sym, r.out_pos, r.src_out_pos, &c.dist_encs);
                info_lines.push(format!(
                    "  {ch_name}: BACKREF  val={}  block={block}  from={}  dist={dist}  len={}  {} redirect options",
                    c.output.get(base_raw + ch).copied().unwrap_or(0),
                    fmt_pixel(src_xy),
                    r.copy_len,
                    alts.len(),
                ));
                edit_options.extend(build_ref_redirect_edits(c, r, block, &ch_name, &alts));
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
    block: u32,
    ch_name: &str,
    alts: &[DistAlt],
) -> Vec<EditOption> {
    let de = &c.dist_encs[block as usize];
    let old_len = de.get(r.dist_sym as u16).map_or(0, |c| c.len);
    let dist = r.out_pos - r.src_out_pos;

    alts.iter()
        .filter_map(
            |&DistAlt {
                 dist_sym: new_dsym,
                 src_out_pos: new_src,
                 distance: new_dist,
             }| {
                let SymCode {
                    code: new_code,
                    len: new_len,
                } = de.get(new_dsym as u16).unwrap_or_default();
                if new_len != old_len {
                    return None;
                }
                let src_xy2 = c.raster.out_to_xy(OutPos(new_src));
                let new_src_us = new_src as usize;
                // One representative byte per channel: the high byte at
                // 16-bit (which is what `to_rgba8` displays), the whole byte
                // at 8-bit. Sampling `0..bpp` raw would splice a 16-bit R's
                // low byte in as "G".
                let bytes_per_sample = (c.info.bit_depth.max(8) / 8) as usize;
                let preview: Vec<u8> = (0..c.info.color_type.channels() as usize)
                    .filter_map(|chan| c.output.get(new_src_us + chan * bytes_per_sample).copied())
                    .collect();
                let preview_txt = preview
                    .iter()
                    .map(u8::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                let label = format!(
                    "[{ch_name}] REDIRECT  dist {dist} → {new_dist}  src {}  val≈{preview_txt}",
                    fmt_pixel(src_xy2)
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
                        label: format!("redirect distance {dist} → {new_dist}"),
                        kind: EditKind::DistRedirect {
                            out_pos: r.out_pos,
                            copy_len: r.copy_len,
                            src_after: new_src,
                            dist_sym_after: new_dsym,
                        },
                    },
                })
            },
        )
        .collect()
}

/// RGB swatch for a distance redirect's target pixel; falls back to grey
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

/// Literal symbols in `le` that share `val`'s Huffman code length (so
/// swapping one in keeps the bitstream length unchanged), excluding `val`
/// itself. The enumeration form of the same-length-swap rule; the pixel-index
/// build (`precompute_lit_swap_syms`) encodes the same rule as a bitset.
fn same_len_lit_swaps(le: &EncTable, val: u16) -> impl Iterator<Item = u16> + '_ {
    let clen = le.get(val).map_or(0, |c| c.len);
    le.iter()
        .filter(move |&(s, sc)| sc.len == clen && s < 256 && s != val)
        .map(|(s, _)| s)
}

fn build_lit_swap_edits(
    c: &CoreData,
    lit_groups: &HashMap<(u8, u32), Vec<LitPatchSite>>,
) -> Vec<EditOption> {
    let mut out = Vec::new();
    for ((val, blk), sites) in lit_groups {
        let le = &c.lit_encs[*blk as usize];
        let mut swappable: Vec<u16> = same_len_lit_swaps(le, *val as u16).collect();
        swappable.sort();
        let chs: String = sites
            .iter()
            .map(|s| c.info.channel_label(s.ch))
            .collect::<Vec<_>>()
            .join(",");
        for tgt in swappable {
            let SymCode {
                code: new_code,
                len: new_len,
            } = le.get(tgt).unwrap_or_default();
            let label = format!("[{chs}] LITERAL  {val} → {tgt}");
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
            let byte_updates: Vec<ByteWrite> = sites
                .iter()
                .map(|s| ByteWrite {
                    out_pos: s.out_pos,
                    value: tgt as u8,
                })
                .collect();
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
    let filter = compute_filter_expansion(cascade.affected, &c.output, &c.info);
    // Skip per-row PNG filter bytes when reporting affected count.
    let row_stride = c.info.row_stride;
    let n_affected = cascade
        .affected
        .iter()
        .filter(|&&p| !(p as usize).is_multiple_of(row_stride))
        .count();
    let n_filter: u32 = filter
        .iter()
        .map(|(_, mx)| c.info.width.saturating_sub(mx))
        .sum();
    let max_depth = cascade.max_depth;
    let overlay = make_cascade_overlay_bytes(&cascade, &filter, &c.info);
    CascadeSummary {
        overlay,
        text: format!(
            "\n  LZ77 cascade: {n_affected} bytes  (max depth {max_depth})\n  filter halo: ~{n_filter} px (faint blue)"
        ),
    }
}
