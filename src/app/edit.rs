//! Apply / undo / redo machinery for edits that have already been built.
//!
//! See [`super::select`] for how [`EditOption`](super::select::EditOption)s
//! and their underlying [`EditAction`]s come together from a pixel click.
//!
//! `apply_edit` dispatches on [`EditKind`]:
//!
//! - **Literal swap** — one Huffman code is replaced by another of the
//!   same length. The output byte at the patched event's position
//!   becomes the new symbol, and every LZ77 descendant inherits it via
//!   a BFS over the reverse graph. LZ77 topology is unchanged, so
//!   `events`, `reverse_graph`, and `pixel_index` all stay valid; only
//!   `output` and the unfiltered/RGBA/composite buffers for the
//!   affected rows need to refresh. Cost: `O(affected_rows)` rather
//!   than `O(image)`.
//!
//! - **Distance redirect** — one back-reference's source position moves.
//!   LZ77 topology shifts at exactly one event, but the rest of the
//!   index (every other event's bit / output positions, all literals,
//!   the per-block Huffman tables) stays valid. The redirect path
//!   patches the event in place, rebuilds the reverse graph (the only
//!   structural index that has to move), recopies the destination range
//!   from the new source, and propagates through the fresh graph. The
//!   render portion is still row-scoped.

use crate::bitstream::{read_bits_at, write_bits};
use crate::deflate::Event;
use crate::index::event_at;

use super::PngBendApp;
use super::io::CoreData;
use super::select::SelectSource;

/// The patches + kind needed to apply an edit forward and to derive its
/// inverse for undo/redo.
#[derive(Clone)]
pub(super) struct EditAction {
    /// Low-level bit writes that realise the edit in the deflate stream.
    /// Always the source of truth for what changes on disk.
    pub patches: Vec<(u32, u32, u8)>, // (bit_start, new_code, code_len)
    pub label: String,
    pub kind: EditKind,
}

/// Structural classification of an edit. Drives whether `apply_edit` can
/// take the fast in-place path or has to fall back to a full reload.
#[derive(Clone)]
pub(super) enum EditKind {
    /// Same-length Huffman-code literal swap. One entry per patched
    /// channel, naming the output byte and the new symbol value. Every
    /// patch maps to an `Event::Lit`; no LZ77 topology changes.
    LiteralSwap { byte_updates: Vec<(u32, u8)> },
    /// Distance-symbol redirect. The LZ77 source moves but the rest of
    /// the LZ77 topology (event count, every other event's `out_pos` /
    /// `copy_len` / channel role) is unchanged — same-length Huffman
    /// codes guarantee that. So we surgically update `events[i]`,
    /// recopy `output[out_pos..out_pos+copy_len]` from the new src, and
    /// propagate downstream via the existing `reverse_graph` — no
    /// `decode_deflate` needed. The graph itself does need to rebuild
    /// (one ref's outgoing edges moved from `old_src..` to
    /// `new_src..`), but that's the only structural index that does.
    DistRedirect {
        /// Output-byte offset of the redirected ref. Doubles as the
        /// "nothing before this can have changed" floor for
        /// row-scoped re-render.
        out_pos: u32,
        copy_len: u16,
        /// Target `src_out_pos` for this edit. For undo of a redirect,
        /// this is the *previous* src.
        src_after: u32,
        /// Target `dist_sym` to write into `events[i]`.
        dist_sym_after: u8,
    },
}

impl PngBendApp {
    pub(super) fn apply_edit(&mut self) {
        let Some(action) = self.sel.pending_edit.take() else {
            return;
        };
        self.status = format!("Applied: {}  at {:?}", action.label, self.sel.sel_pixel);
        let inverse = self.apply_and_capture_inverse(action);
        self.doc.history.record(inverse);

        self.doc.dirty = true;
        self.sel.pending_edit = None;
        self.sel.selected_edit = None;
    }

    pub(super) fn undo(&mut self) {
        let Some(entry) = self.doc.history.pop_undo() else {
            return;
        };
        let label = entry.label.clone();
        let inverse = self.apply_and_capture_inverse(entry);
        self.doc.history.push_redo(inverse);
        self.doc.dirty = self.doc.history.can_undo();
        self.status = format!(
            "Undone: {label}  |  undo: {}, redo: {}",
            self.doc.history.undo_len(),
            self.doc.history.redo_len()
        );
    }

    pub(super) fn redo(&mut self) {
        let Some(entry) = self.doc.history.pop_redo() else {
            return;
        };
        let label = entry.label.clone();
        let inverse = self.apply_and_capture_inverse(entry);
        self.doc.history.push_undo(inverse);
        self.doc.dirty = true;
        self.status = format!(
            "Redone: {label}  |  undo: {}, redo: {}",
            self.doc.history.undo_len(),
            self.doc.history.redo_len()
        );
    }

    /// Execute `action` and return an [`EditAction`] that inverts it,
    /// suitable for pushing onto the undo stack.
    ///
    /// Dispatches by [`EditKind`]: literal swap goes through
    /// [`Self::apply_literal_swap_incremental`], redirect through
    /// [`Self::apply_dist_redirect_incremental`].
    fn apply_and_capture_inverse(&mut self, action: EditAction) -> EditAction {
        let EditAction {
            patches,
            label,
            kind,
        } = action;
        match kind {
            EditKind::LiteralSwap { byte_updates } => {
                // Capture the pre-edit byte at each patched position *before*
                // mutating the bitstream — undo writes these back.
                let prior: Vec<(u32, u8)> = {
                    let c = self.doc.core.as_ref().expect("edit without a loaded file");
                    byte_updates
                        .iter()
                        .map(|&(out_pos, _)| (out_pos, c.output[out_pos as usize]))
                        .collect()
                };
                let inverse_patches =
                    apply_patches_capturing_prior(&mut self.doc.deflate_buf, &patches);
                self.apply_literal_swap_incremental(&byte_updates);
                EditAction {
                    patches: inverse_patches,
                    label,
                    kind: EditKind::LiteralSwap {
                        byte_updates: prior,
                    },
                }
            }
            EditKind::DistRedirect {
                out_pos,
                copy_len,
                src_after,
                dist_sym_after,
            } => {
                // Capture the redirected ref's PRE-edit (src, dist_sym) so
                // undo flips back to them. The event covering `out_pos`
                // is unchanged by a redirect — the ref still spans the
                // same output range — so an `event_at` lookup against the
                // current events list resolves it.
                let (event_idx, src_before, dist_sym_before) = {
                    let c = self.doc.core.as_ref().expect("edit without a loaded file");
                    let ev_idx = event_at(&c.events, out_pos as usize)
                        .expect("redirect targets unmapped position");
                    match &c.events[ev_idx as usize] {
                        Event::Ref(r) => (ev_idx, r.src_out_pos, r.dist_sym),
                        Event::Lit(_) => {
                            unreachable!("redirect targets a non-ref event")
                        }
                    }
                };
                let inverse_patches =
                    apply_patches_capturing_prior(&mut self.doc.deflate_buf, &patches);
                self.apply_dist_redirect_incremental(
                    event_idx,
                    out_pos,
                    copy_len,
                    src_after,
                    dist_sym_after,
                );
                EditAction {
                    patches: inverse_patches,
                    label,
                    kind: EditKind::DistRedirect {
                        out_pos,
                        copy_len,
                        src_after: src_before,
                        dist_sym_after: dist_sym_before,
                    },
                }
            }
        }
    }

    /// Apply a literal swap. For each patched channel: write the new
    /// byte at its `out_pos`, then BFS through the reverse graph and
    /// write the same byte at every descendant. Re-runs the inverse
    /// PNG filter, RGBA conversion, and composite over only the rows
    /// the propagation touched.
    ///
    /// The propagation needs no seen-set: LZ77 is a tree rooted at each
    /// literal event, so the fan-outs of distinct `byte_updates` cannot
    /// overlap. Each visited byte is written exactly once regardless of
    /// traversal order.
    fn apply_literal_swap_incremental(&mut self, byte_updates: &[(u32, u8)]) {
        let Some(c) = self.doc.core.as_mut() else {
            return;
        };
        let mut rows = RowTracker::new(&c.info);
        // Stack, not queue. Traversal order doesn't matter (every visited
        // byte takes the same `new_byte` write), and in debug a `Vec`
        // push/pop costs about a quarter of `VecDeque`'s ring-buffer
        // bookkeeping.
        let mut stack: Vec<u32> = Vec::new();

        for &(out_pos, new_byte) in byte_updates {
            let pos_us = out_pos as usize;
            c.output[pos_us] = new_byte;
            rows.mark(pos_us);

            // Keep the originating Lit event's `symbol` field in sync with
            // the patched bitstream so anything else reading `events`
            // (overlays, select_pixel) sees the new value.
            if let Some(ev_idx) = event_at(&c.events, pos_us)
                && let Event::Lit(lit) = &mut c.events[ev_idx as usize]
            {
                lit.symbol = new_byte;
            }

            propagate_lz77(
                &c.reverse_graph,
                pos_us,
                new_byte,
                &mut c.output,
                &mut stack,
                &mut rows,
            );
        }

        if rows.first_affected == usize::MAX {
            // No-op edit (no byte_updates / empty patch list).
            self.view.texture_dirty = true;
            return;
        }

        match render_affected_rows(
            c,
            &mut self.view.base_rgba,
            rows.first_affected,
            &rows.touched,
        ) {
            Ok(rebuilt) => {
                // Row-scoped composite next frame — overlays stay valid
                // because LZ77 topology didn't move.
                self.view.partial_composite_rows = Some(rebuilt);
                self.view.texture_dirty = true;
                self.refresh_selection_after_literal_swap();
            }
            Err(e) => {
                self.status = format!("render after edit: {e}");
            }
        }
    }

    /// Apply a distance redirect.
    ///
    /// The LZ77 topology change is narrow: exactly one event's
    /// `(src_out_pos, dist_sym)` moves. Every other event's bit
    /// position, output range, and block index stays valid; so do
    /// `pixel_index`'s xy and editability flags and the per-block
    /// Huffman tables. That makes a `decode_deflate` + full re-index
    /// unnecessary — we update the affected pieces in place.
    ///
    /// Steps:
    /// 1. Patch `events[event_idx]` with the new `src_out_pos` and
    ///    `dist_sym`.
    /// 2. Rebuild `reverse_graph` against the patched events. This must
    ///    happen **before** the BFS in step 4 — for refs whose source
    ///    range overlaps their destination (`src + copy_len > out_pos`),
    ///    edges land within the destination range, and a stale graph
    ///    would let the BFS overwrite correctly recopied bytes through
    ///    wrong edges.
    /// 3. Recopy `output[out_pos..out_pos+copy_len]` from the new src.
    /// 4. Propagate the new bytes through the fresh `reverse_graph` to
    ///    every downstream ref that copies from this destination range.
    /// 5. Refresh the cached `max_distance` — one ref's distance changed.
    /// 6. Row-scoped unfilter + RGBA + composite on the touched rows;
    ///    the rest of `unfiltered` and `base_rgba` stays byte-identical.
    ///    `pixel_index.rgb` for the touched rows goes intentionally
    ///    stale — see the comment at step 6.
    fn apply_dist_redirect_incremental(
        &mut self,
        event_idx: u32,
        out_pos: u32,
        copy_len: u16,
        src_after: u32,
        dist_sym_after: u8,
    ) {
        let Some(c) = self.doc.core.as_mut() else {
            return;
        };

        // 1. Patch the event in place.
        if let Event::Ref(r) = &mut c.events[event_idx as usize] {
            r.src_out_pos = src_after;
            r.dist_sym = dist_sym_after;
        } else {
            // EditAction should only ever target a Ref event. Bail
            // rather than risk corrupting state.
            return;
        }

        // 2. Rebuild reverse_graph against the patched events. CSR
        //    can't be cheaply mutated in place; rebuilding is correct
        //    and cheap enough until a sparse delta-graph lands.
        c.reverse_graph = crate::index::build_reverse_graph(&c.events, c.output.len());

        // 3. Recopy output bytes for this ref's destination span. The
        //    new src is upstream of `out_pos` (LZ77 never copies from
        //    later bytes), so positions in `[src_after..out_pos)` are
        //    stable while we read them. When the ref overlaps its
        //    destination (`src_after + copy_len > out_pos`), forward
        //    iteration is required: a write at `dst_base + off` can be
        //    read by a later iteration as `src_base + off'` when
        //    `src_base + off' >= dst_base`, mirroring the run-length
        //    expansion the LZ77 decoder uses to populate these bytes.
        let copy_len = copy_len as usize;
        let dst_base = out_pos as usize;
        let src_base = src_after as usize;
        for off in 0..copy_len {
            c.output[dst_base + off] = c.output[src_base + off];
        }

        // 4. Propagate through the fresh reverse_graph from each newly-
        //    written output byte. `propagate_lz77` walks descendants
        //    and writes them with the byte's value. The bytes in the
        //    destination range can have different values per offset, so
        //    each is its own BFS rather than a single multi-seed run.
        let mut rows = RowTracker::new(&c.info);
        let mut stack: Vec<u32> = Vec::new();
        for off in 0..copy_len {
            let pos = dst_base + off;
            rows.mark(pos);
            propagate_lz77(
                &c.reverse_graph,
                pos,
                c.output[pos],
                &mut c.output,
                &mut stack,
                &mut rows,
            );
        }

        // 5. Refresh `max_distance`. The redirected ref's distance
        //    moved, which might raise (or lower, in undo) the cached
        //    maximum. `O(events)` scan, runs once per redirect.
        c.max_distance = c
            .events
            .iter()
            .filter_map(|e| match e {
                Event::Ref(r) => Some(r.out_pos - r.src_out_pos),
                _ => None,
            })
            .max()
            .unwrap_or(1);

        if rows.first_affected == usize::MAX {
            self.view.texture_dirty = true;
            return;
        }

        // 6. Row-scoped render. Refreshes `unfiltered` and `base_rgba`
        //    for every row the propagation touched. `pixel_index.rgb`
        //    for those rows goes stale, but the sidebar only reads
        //    `rgb` when formatting a visible row's display text and
        //    that path can format directly from `c.output` instead, so
        //    no follow-up sync is needed.
        match render_affected_rows(
            c,
            &mut self.view.base_rgba,
            rows.first_affected,
            &rows.touched,
        ) {
            Ok(rebuilt) => {
                self.view.partial_composite_rows = Some(rebuilt);
                self.view.texture_dirty = true;
            }
            Err(e) => {
                self.status = format!("render after redirect: {e}");
            }
        }

        // The distance overlay's colour ramp depends on each ref's
        // distance; the redirected ref now has a different one, so the
        // cached distance overlay is stale. The cascade overlay was
        // computed against the previous reverse_graph topology, so
        // that's stale too. Literal and block overlays remain valid:
        // every literal kept its symbol and every event kept its block.
        self.view.overlay_cache.invalidate_distance();
        self.view.cascade_rgba = None;

        if let Some((x, y)) = self.sel.sel_pixel {
            self.select_pixel(x, y, SelectSource::Refocus);
        }
    }

    pub(super) fn assemble_png_bytes(&self, zlib: &[u8]) -> Vec<u8> {
        // Re-emit exactly one IDAT with the (possibly edited) deflate stream;
        // copy every other chunk through unchanged.
        let out_chunks: Vec<crate::png::Chunk> = self
            .doc
            .chunks
            .iter()
            .scan(false, |idat_seen, c| {
                if &c.typ == b"IDAT" {
                    if *idat_seen {
                        return Some(None); // skip trailing IDATs
                    }
                    *idat_seen = true;
                    Some(Some(crate::png::Chunk {
                        typ: *b"IDAT",
                        data: zlib.to_vec(),
                    }))
                } else {
                    Some(Some(crate::png::Chunk {
                        typ: c.typ,
                        data: c.data.clone(),
                    }))
                }
            })
            .flatten()
            .collect();
        crate::png::write_chunks(&out_chunks)
    }
}

// ── free helpers ─────────────────────────────────────────────────────────

/// Tracks which rows had any byte change during one edit, for the
/// row-scoped re-render afterwards. Bundles `touched` (per-row flag),
/// `first_affected` (lowest touched row, drives the row-scoped unfilter
/// loop's start position), and the `(row_stride, h)` geometry needed to
/// classify a byte position.
struct RowTracker {
    touched: Vec<bool>,
    first_affected: usize,
    row_stride: usize,
    h: usize,
}

impl RowTracker {
    fn new(info: &crate::png::PngInfo) -> Self {
        Self {
            touched: vec![false; info.height as usize],
            first_affected: usize::MAX,
            row_stride: info.row_stride,
            h: info.height as usize,
        }
    }

    /// Mark the row containing byte offset `pos` as touched and update
    /// `first_affected` so [`render_affected_rows`] knows where to start.
    #[inline]
    fn mark(&mut self, pos: usize) {
        let row = pos / self.row_stride;
        if row < self.h && !self.touched[row] {
            self.touched[row] = true;
            if row < self.first_affected {
                self.first_affected = row;
            }
        }
    }
}

/// BFS through the LZ77 reverse graph from `out_pos`, writing
/// `new_byte` into every descendant byte. The caller owns `stack` so it
/// can be reused across several seed positions in the same edit (each
/// invocation clears it first), and `rows` accumulates the set of rows
/// touched across the entire edit.
#[inline]
fn propagate_lz77(
    reverse_graph: &crate::index::ReverseGraph,
    out_pos: usize,
    new_byte: u8,
    output: &mut [u8],
    stack: &mut Vec<u32>,
    rows: &mut RowTracker,
) {
    stack.clear();
    stack.push(out_pos as u32);
    while let Some(pos) = stack.pop() {
        for &dst in reverse_graph.neighbors(pos) {
            let di = dst as usize;
            // SAFETY: `build_reverse_graph` only emits edges to byte
            // positions inside the output buffer, so `di < output.len()`.
            unsafe {
                *output.get_unchecked_mut(di) = new_byte;
            }
            rows.mark(di);
            stack.push(dst);
        }
    }
}

/// Row-scoped render pipeline shared by the literal-swap and redirect
/// paths. Inverse-filters every row flagged in `row_touched` (plus any
/// downstream rows that chain through Up/Avg/Paeth filter types), then
/// converts those rows in `base_rgba`. Returns the rebuilt set so the
/// caller can hand it to `partial_composite_rows` and keep the texture
/// rebuild row-scoped as well.
fn render_affected_rows(
    core: &mut CoreData,
    base_rgba: &mut [u8],
    first_affected: usize,
    row_touched: &[bool],
) -> Result<Vec<usize>, String> {
    let mut rebuilt = Vec::with_capacity(row_touched.iter().filter(|b| **b).count() + 4);
    crate::png::unfilter_rows_into(
        &core.output,
        &core.info,
        &mut core.unfiltered,
        first_affected,
        |y| row_touched.get(y).copied().unwrap_or(false),
        |y| rebuilt.push(y),
    )
    .map_err(|e| format!("unfilter: {e}"))?;
    crate::png::to_rgba8_rows_into(
        &core.unfiltered,
        &core.info,
        core.palette.as_deref(),
        base_rgba,
        rebuilt.iter().copied(),
    )
    .map_err(|e| format!("rgba: {e}"))?;
    Ok(rebuilt)
}

/// Write each `(bit_start, value, code_len)` patch into `buf`, capturing
/// the bits that were overwritten so the caller can stash the inverse
/// patch list onto the undo stack.
fn apply_patches_capturing_prior(
    buf: &mut [u8],
    patches: &[(u32, u32, u8)],
) -> Vec<(u32, u32, u8)> {
    let mut inverse = Vec::with_capacity(patches.len());
    for &(bit_start, value, code_len) in patches {
        let bs = bit_start as usize;
        let prev = read_bits_at(buf, bs, code_len);
        inverse.push((bit_start, prev, code_len));
        write_bits(buf, bs, value, code_len);
    }
    inverse
}

#[cfg(test)]
mod tests {
    use super::apply_patches_capturing_prior;
    use proptest::prelude::*;

    proptest! {
        /// The undo invariant: applying an edit's patches forward, then
        /// applying the captured inverse, restores `deflate_buf`
        /// byte-for-byte. Real `EditAction.patches` are non-overlapping
        /// (one patch per channel for literal swaps, one per redirect),
        /// so the generator lays patches out sequentially with a 1-bit
        /// gap to mirror that.
        #[test]
        fn forward_then_inverse_restores_buffer(
            buf in proptest::collection::vec(any::<u8>(), 4..64usize),
            specs in proptest::collection::vec((1u8..=16, any::<u32>()), 1..7),
        ) {
            let max_bit = buf.len() * 8;
            let mut bit_cursor: u32 = 0;
            let mut patches: Vec<(u32, u32, u8)> = Vec::new();
            for (cl, v) in specs {
                let cl_u32 = cl as u32;
                if (bit_cursor + cl_u32) as usize > max_bit {
                    break;
                }
                let mask = if cl == 32 { u32::MAX } else { (1u32 << cl) - 1 };
                patches.push((bit_cursor, v & mask, cl));
                bit_cursor += cl_u32 + 1;
            }
            prop_assume!(!patches.is_empty());

            let original = buf.clone();
            let mut work = buf;
            let inverse = apply_patches_capturing_prior(&mut work, &patches);
            let _ = apply_patches_capturing_prior(&mut work, &inverse);
            prop_assert_eq!(work, original);
        }
    }
}
