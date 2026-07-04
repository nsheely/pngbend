//! Pixel-level summaries for the side panel.
//!
//! Walks the image in raster order, consults [`super::build_pos_to_ev`]
//! for the event that produced each pixel's first byte, and splits pixels
//! into:
//! - **literal rows**: any channel is a literal whose Huffman code has a
//!   same-length swap alternative within this block.
//! - **back-reference rows**: no channel is a literal, at least one is
//!   produced by a redirectable back-ref.
//!
//! Rows store only structural data (`xy`, `rgb`, `editable`); the
//! sidebar's virtual-scroll callback formats each visible row's display
//! text on demand, so the index never holds per-row `String`s for
//! millions of unseen pixels. The filter UI matches against a reused
//! scratch `String` for the same reason.
//!
//! Per-block editability is summarised up front: a `[bool; 256]` per lit
//! alphabet (one slot per literal symbol) and a `u32` bitmask per dist
//! alphabet (30 symbols → 30 bits). Each entry is one load / bit-test in
//! the per-pixel inner loop. Built in `O(alphabet)` per block by counting
//! Huffman-length buckets, not `O(alphabet²)`.

use rayon::prelude::*;

use crate::coords::PixelXY;
use crate::deflate::{DBASE, DEXT};
use crate::deflate::{EncTable, Event, SymCode};

/// One row in the pixel-list side panel: image-space coordinates, the
/// pixel's RGB swatch colour, and whether it has at least one valid edit.
/// `Copy` is cheap (8 bytes) so call sites pass by value.
///
/// Coordinates are packed as `u16`: image dimensions are capped at
/// `u16::MAX` at load, saving 4 bytes per row vs. two `u32`s on a
/// multi-megapixel index. The [`PixelRow::xy`] accessor returns a
/// [`PixelXY`] for the coordinate-safe app surface.
#[derive(Debug, Clone, Copy)]
pub struct PixelRow {
    xy: (u16, u16),
    pub rgb: [u8; 3],
    /// `true` when this pixel has at least one applicable edit: a
    /// same-Huffman-width literal swap, or a redirectable back-ref
    /// alternative. The sidebar greys non-`has_edit` rows out; the
    /// "Editable only" filter hides them.
    pub has_edit: bool,
}

impl PixelRow {
    pub fn new(xy: PixelXY, rgb: [u8; 3], has_edit: bool) -> Self {
        debug_assert!(
            xy.x <= u16::MAX as u32 && xy.y <= u16::MAX as u32,
            "PixelRow coordinates must fit in u16",
        );
        Self {
            xy: (xy.x as u16, xy.y as u16),
            rgb,
            has_edit,
        }
    }

    #[inline]
    pub fn xy(self) -> PixelXY {
        PixelXY::new(self.xy.0 as u32, self.xy.1 as u32)
    }

    #[inline]
    pub fn x(self) -> u32 {
        self.xy.0 as u32
    }

    #[inline]
    pub fn y(self) -> u32 {
        self.xy.1 as u32
    }
}

/// Per-pixel summaries for the "Literals" and "Backrefs" radio buttons.
/// `n_lit_with_edit` is counted during build so the sidebar's pixel-count
/// label needs no second pass over `lit`. Every entry in `refs` is already
/// filtered to redirectable refs, so each is editable by construction.
pub struct PixelIndex {
    pub lit: Vec<PixelRow>,
    pub refs: Vec<PixelRow>,
    pub n_lit_with_edit: usize,
}

/// One literal symbol (0..=255) per slot; `true` if this block's alphabet
/// holds at least one other literal with the same Huffman length (the swap
/// edit has a target). 256 bytes per block, contiguous.
type LitSwapSet = [bool; 256];

/// Bitmask over distance symbols (0..=29): bit `i` is set if dist-symbol
/// `i` has at least one compatible redirect target in this block (same
/// Huffman length and same `DEXT`). One `u32` per block.
type DistRedirMask = u32;

pub fn build_pixel_index(
    events: &[Event],
    output: &[u8],
    pos_to_ev: &[u32],
    lit_encs: &[EncTable],
    dist_encs: &[EncTable],
    block_starts: &[u32],
    raster: &crate::Raster,
) -> PixelIndex {
    let info = raster.info();
    let w = info.width as usize;
    let h = info.height as usize;
    let bpp = info.bpp;
    let pixels_per_byte = raster.pixels_per_byte() as usize;

    let lit_swap_syms = precompute_lit_swap_syms(lit_encs);
    let blk_redir_syms = precompute_redirectable_dist_syms(dist_encs);

    // Per-event class byte. The hot per-pixel loop resolves events by
    // output position (`pos_to_ev`), so every per-channel read adds memory
    // traffic. Folding each event's block-derived editability into one byte
    // lets `classify_pixel` do a single `u8` load per channel instead of
    // chasing `events`, a block table, and the per-block Huffman sets, and
    // runs `is_ref_redirectable` once per event, not once per copied byte.
    // Blocks own disjoint, contiguous event ranges, so classification fans
    // out across them with the block index as a free per-range constant, no
    // per-event block lookup. Dropped when the build returns; nothing
    // long-lived carries a per-event block.
    let event_class: Vec<EventClass> = (0..block_starts.len())
        .into_par_iter()
        .flat_map_iter(|block| {
            let start = block_starts[block] as usize;
            let end = block_starts
                .get(block + 1)
                .map_or(events.len(), |&e| e as usize);
            let lit_set = lit_swap_syms.get(block);
            let blk_redir = &blk_redir_syms;
            events[start..end].iter().map(move |e| match e {
                Event::Lit(lit) => {
                    if lit_set.is_some_and(|s| s[lit.symbol as usize]) {
                        EventClass::LitSwap
                    } else {
                        EventClass::LitNoSwap
                    }
                }
                Event::Ref(r) => {
                    if is_ref_redirectable(r, block, blk_redir, dist_encs) {
                        EventClass::RefRedir
                    } else {
                        EventClass::Skip
                    }
                }
            })
        })
        .collect();

    // Rows are independent after the two precomputes: split across rayon
    // workers and stitch the per-row collections back in `y` order so
    // `lit` / `refs` stay sorted by `(y, x)` as downstream code (binary
    // search in `select.rs`, merge-sort in `filter_all`) expects.
    //
    // At sub-byte depths multiple x-values share one byte, hence one event.
    // Step `x` by `pixels_per_byte` so the list emits one entry per byte,
    // named by the cluster's first pixel. For ≥ 8-bit depths
    // `pixels_per_byte == 1` and behaviour is unchanged.
    let per_row: Vec<RowBuckets> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut buckets = RowBuckets::new(w);
            for x in (0..w).step_by(pixels_per_byte) {
                let base = raster.xy_to_out(PixelXY::new(x as u32, y as u32)).0 as usize;
                if base >= output.len() {
                    continue;
                }
                let Some(kind) = classify_pixel(base, bpp, pos_to_ev, &event_class) else {
                    continue;
                };

                let r = output.get(base).copied().unwrap_or(0);
                let g = output.get(base + 1).copied().unwrap_or(r);
                let b = output.get(base + 2).copied().unwrap_or(r);
                let row = PixelRow::new(
                    PixelXY::new(x as u32, y as u32),
                    [r, g, b],
                    matches!(kind, PixelKind::Lit { has_swap: true } | PixelKind::Ref),
                );

                match kind {
                    PixelKind::Lit { has_swap } => {
                        if has_swap {
                            buckets.n_lit_with_edit += 1;
                        }
                        buckets.lit.push(row);
                    }
                    PixelKind::Ref => buckets.refs.push(row),
                }
            }
            buckets
        })
        .collect();

    let pixel_count = w * h;
    let mut lit = Vec::with_capacity(pixel_count / 4);
    let mut refs = Vec::with_capacity(pixel_count / 4);
    let mut n_lit_with_edit = 0usize;
    for b in per_row {
        lit.extend_from_slice(&b.lit);
        refs.extend_from_slice(&b.refs);
        n_lit_with_edit += b.n_lit_with_edit;
    }

    PixelIndex {
        lit,
        refs,
        n_lit_with_edit,
    }
}

/// Per-row scratch the parallel build_pixel_index loop produces. Stitching
/// these back in `y` order preserves the raster sort order downstream.
struct RowBuckets {
    lit: Vec<PixelRow>,
    refs: Vec<PixelRow>,
    n_lit_with_edit: usize,
}

impl RowBuckets {
    fn new(w: usize) -> Self {
        // A few hundred entries per row on a typical photo; quarter-row
        // capacity avoids the first three doublings.
        let cap = (w / 4).max(8);
        Self {
            lit: Vec::with_capacity(cap),
            refs: Vec::with_capacity(cap),
            n_lit_with_edit: 0,
        }
    }
}

enum PixelKind {
    /// A literal pixel. `has_swap` is true when the block's alphabet holds
    /// at least one other literal with the same Huffman code length, so the
    /// byte can be rewritten without disturbing bit-alignment.
    Lit { has_swap: bool },
    /// A back-reference pixel. Only emitted for refs whose distance has at
    /// least one same-width alternative, so every `Ref` row is editable.
    Ref,
}

/// Per-event class for the pixel index's hot loop, one byte each. Encodes only
/// what the per-pixel classifier needs (literal vs redirectable ref, plus
/// whether a literal has a same-length swap), so the loop reads one byte per
/// channel and never touches `events` or the per-block Huffman tables. A
/// non-redirectable ref is `Skip`, like an unowned channel.
#[derive(Clone, Copy)]
#[repr(u8)]
enum EventClass {
    Skip,
    LitNoSwap,
    LitSwap,
    RefRedir,
}

/// Classify a pixel from its channel-owning events' class bytes.
/// - Any channel owned by a literal event → `Lit`, `has_swap` if that
///   literal has a same-length swap alternative.
/// - Else any channel owned by a redirectable back-ref → `Ref`.
/// - Else the pixel is invisible to the side panel.
#[inline]
fn classify_pixel(
    base: usize,
    bpp: usize,
    pos_to_ev: &[u32],
    event_class: &[EventClass],
) -> Option<PixelKind> {
    let mut lit_seen = false;
    let mut lit_has_swap = false;
    let mut ref_seen = false;

    for ch in 0..bpp {
        let pos = base + ch;
        if pos >= pos_to_ev.len() {
            break;
        }
        let ev_idx = pos_to_ev[pos];
        if ev_idx == u32::MAX {
            continue;
        }
        match event_class
            .get(ev_idx as usize)
            .copied()
            .unwrap_or(EventClass::Skip)
        {
            EventClass::LitNoSwap => lit_seen = true,
            EventClass::LitSwap => {
                lit_seen = true;
                lit_has_swap = true;
            }
            EventClass::RefRedir => ref_seen = true,
            EventClass::Skip => {}
        }
    }

    if lit_seen {
        Some(PixelKind::Lit {
            has_swap: lit_has_swap,
        })
    } else if ref_seen {
        Some(PixelKind::Ref)
    } else {
        None
    }
}

/// For each block, which literal symbols (0..=255) have at least one
/// same-length alternative in its Huffman alphabet. `counts[c]` is the
/// number of literal symbols with Huffman length `c`; a symbol is swappable
/// iff its bucket has more than one entry. Two `O(alphabet)` passes per
/// block (count, then mark).
fn precompute_lit_swap_syms(lit_encs: &[EncTable]) -> Vec<LitSwapSet> {
    lit_encs
        .iter()
        .map(|le| {
            let raw = le.raw();
            let lit_slice = &raw[..raw.len().min(256)];
            let mut counts = [0u16; 16];
            for &SymCode { len: clen, .. } in lit_slice {
                if clen != 0 && (clen as usize) < counts.len() {
                    counts[clen as usize] = counts[clen as usize].saturating_add(1);
                }
            }
            let mut valid = [false; 256];
            for (sym, &SymCode { len: clen, .. }) in lit_slice.iter().enumerate() {
                if clen != 0 && (clen as usize) < counts.len() && counts[clen as usize] > 1 {
                    valid[sym] = true;
                }
            }
            valid
        })
        .collect()
}

/// For each block, which dist-symbols have a compatible redirect target
/// (same Huffman length AND same extra-bits count). Bucketed by
/// `(clen, dext)`; symbols in a bucket of size > 1 are redirectable.
/// Returns a `u32` mask per block; bit `i` = dist-symbol `i` is editable.
fn precompute_redirectable_dist_syms(dist_encs: &[EncTable]) -> Vec<DistRedirMask> {
    dist_encs
        .iter()
        .map(|de| {
            let raw = de.raw();
            // DEFLATE: dist alphabet is 0..30, dist clen ≤ 15, DEXT ≤ 13.
            // Symbols 30/31 are reserved (RFC 1951); ignore them even if a
            // fixed-Huffman block's `EncTable` allocates 32 slots, since
            // `DEXT` is only defined for 0..30.
            let limit = raw.len().min(30);
            let raw = &raw[..limit];
            let mut counts = [[0u8; 16]; 16];
            for (sym, &SymCode { len: clen, .. }) in raw.iter().enumerate() {
                if clen == 0 {
                    continue;
                }
                let dext = DEXT[sym] as usize;
                if (clen as usize) < 16 && dext < 16 {
                    counts[clen as usize][dext] = counts[clen as usize][dext].saturating_add(1);
                }
            }
            let mut mask: u32 = 0;
            for (sym, &SymCode { len: clen, .. }) in raw.iter().enumerate() {
                if clen == 0 {
                    continue;
                }
                let dext = DEXT[sym] as usize;
                if (clen as usize) < 16 && dext < 16 && counts[clen as usize][dext] > 1 {
                    mask |= 1u32 << sym;
                }
            }
            mask
        })
        .collect()
}

/// A redirect target for a back-reference: point it at `dist_sym` and its
/// source moves to `src_out_pos`, `distance` bytes back. `src_out_pos` and
/// `distance` are both `u32`, so the named fields keep them from being
/// transposed at a call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DistAlt {
    pub dist_sym: u8,
    pub src_out_pos: u32,
    pub distance: u32,
}

/// Iterate dist-symbols in `de` that are valid redirect targets for the
/// back-reference at `(out_pos, src_out_pos)` with current symbol `cur_sym`.
/// Valid: same Huffman length, same `DEXT` extra-bits class, and a
/// non-negative new source position.
///
/// `None` only when `cur_sym` isn't present in `de`: nothing to redirect
/// from.
fn compatible_dist_alts<'a>(
    de: &'a EncTable,
    cur_sym: u8,
    out_pos: u32,
    src_out_pos: u32,
) -> Option<impl Iterator<Item = DistAlt> + 'a> {
    // Symbols 30/31 are reserved (RFC 1951) and have no `DEXT`/`DBASE`
    // entries; refuse them up front rather than panic at the lookup.
    if (cur_sym as usize) >= DBASE.len() {
        return None;
    }
    let cur_clen = de.get(cur_sym as u16)?.len;
    let cur_dext = DEXT[cur_sym as usize];
    let distance = out_pos - src_out_pos;
    let extra_val = distance.saturating_sub(DBASE[cur_sym as usize]);
    // `de.raw()` may run to 32 entries for a fixed-Huffman block, but
    // `DBASE`/`DEXT` are only defined for 0..30, so clip the iteration.
    let limit = de.raw().len().min(DBASE.len());
    Some(de.raw()[..limit].iter().enumerate().filter_map(
        move |(sym, &SymCode { len: clen, .. })| {
            if clen == 0 || (sym as u8) == cur_sym || clen != cur_clen || DEXT[sym] != cur_dext {
                return None;
            }
            let new_dist = DBASE[sym] + extra_val;
            let new_src_signed = out_pos as i64 - new_dist as i64;
            (new_src_signed >= 0).then_some(DistAlt {
                dist_sym: sym as u8,
                src_out_pos: new_src_signed as u32,
                distance: new_dist,
            })
        },
    ))
}

fn is_ref_redirectable(
    r: &crate::deflate::RefEvent,
    block: usize,
    blk_redir_syms: &[DistRedirMask],
    dist_encs: &[EncTable],
) -> bool {
    // Per-block bitset rules out symbols that have no alternatives anywhere
    // in the table, before we touch the per-event distance math.
    let Some(&mask) = blk_redir_syms.get(block) else {
        return false;
    };
    if (r.dist_sym as u32) >= 32 || mask & (1u32 << r.dist_sym) == 0 {
        return false;
    }
    compatible_dist_alts(&dist_encs[block], r.dist_sym, r.out_pos, r.src_out_pos)
        .is_some_and(|mut it| it.next().is_some())
}

/// Redirect alternatives for a single back-reference.
pub fn valid_dist_alts(
    block: u32,
    dist_sym: u8,
    out_pos: u32,
    src_out_pos: u32,
    dist_encs: &[EncTable],
) -> Vec<DistAlt> {
    dist_encs
        .get(block as usize)
        .and_then(|de| compatible_dist_alts(de, dist_sym, out_pos, src_out_pos))
        .map(|it| it.collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist_table_of(entries: &[(u16, u16, u8)]) -> EncTable {
        let mut de = EncTable::new(30);
        for &(sym, code, clen) in entries {
            de.set(sym, code, clen);
        }
        de
    }

    #[test]
    fn valid_dist_alts_filters_by_hlen_and_dext() {
        let de = dist_table_of(&[
            (1, 0b00, 3),  // DBASE=2,  DEXT=0
            (2, 0b01, 3),  // DBASE=3,  DEXT=0 → valid alt
            (4, 0b10, 3),  // DBASE=5,  DEXT=1 → wrong DEXT
            (5, 0b110, 4), // wrong clen
        ]);
        let alts = valid_dist_alts(0, 1, 100, 98, &[de]);
        assert_eq!(alts.len(), 1);
        assert_eq!(alts[0].dist_sym, 2);
        assert_eq!(alts[0].distance, 3);
    }

    #[test]
    fn valid_dist_alts_skips_negative_src() {
        let de = dist_table_of(&[
            (0, 0b0, 1),  // DBASE=1
            (15, 0b1, 1), // DBASE=193, too large for out_pos=10
        ]);
        assert!(valid_dist_alts(0, 0, 10, 9, &[de]).is_empty());
    }

    #[test]
    fn precompute_lit_swap_matches_naive() {
        // Sanity-check the O(n) bucket reshape against the obvious O(n²) form.
        let mut le = EncTable::new(288);
        le.set(5, 0b00, 3);
        le.set(6, 0b01, 3); // same clen=3 → both swappable
        le.set(7, 0b10, 4); // lonely clen=4
        le.set(9, 0b110, 5);
        le.set(10, 0b111, 5); // same clen=5 → both swappable
        let got = precompute_lit_swap_syms(std::slice::from_ref(&le));
        let mut expected = [false; 256];
        for sym in [5u8, 6, 9, 10] {
            expected[sym as usize] = true;
        }
        assert_eq!(got[0], expected);
    }

    #[test]
    fn precompute_dist_redir_ignores_reserved_symbols_30_31() {
        // Fixed-Huffman blocks build a 32-slot dist EncTable. Symbols 30
        // and 31 are reserved (RFC 1951) and have no `DEXT`/`DBASE`
        // entries; the precompute must skip them rather than panic.
        let mut de = EncTable::new(32);
        for sym in 0u16..32 {
            de.set(sym, sym, 5);
        }
        let masks = precompute_redirectable_dist_syms(&[de]);
        // No bits set above sym=29.
        assert_eq!(masks[0] & !((1u32 << 30) - 1), 0);
    }

    #[test]
    fn precompute_dist_redir_sets_expected_bits() {
        let de = dist_table_of(&[
            (1, 0b00, 3),  // DBASE=2,  DEXT=0
            (2, 0b01, 3),  // DBASE=3,  DEXT=0, same (clen,dext) as sym 1
            (4, 0b10, 3),  // DBASE=5,  DEXT=1, alone in (3, 1) bucket
            (5, 0b110, 4), // alone in (4, 1) bucket
        ]);
        let masks = precompute_redirectable_dist_syms(&[de]);
        assert_eq!(masks[0] & (1u32 << 1), 1u32 << 1);
        assert_eq!(masks[0] & (1u32 << 2), 1u32 << 2);
        assert_eq!(masks[0] & (1u32 << 4), 0);
        assert_eq!(masks[0] & (1u32 << 5), 0);
    }
}
