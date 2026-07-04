//! Event-driven overlay renderers.
//!
//! Each renderer produces a full `w * h * 4` RGBA buffer the GUI alpha-
//! composites over the base image. Buffers live in
//! [`crate::app::overlay_cache::OverlayCache`] across frames, invalidated
//! on file load and on any edit that changes the event list.
//!
//! For multi-byte back-references (`copy_len` up to 258), the renderer
//! walks one pixel at a time, not one byte: the per-pixel `div+mod` by
//! `row_stride` runs once per pixel, then `+= bpp` skips to the next until
//! the row boundary.

use crate::coords::OutPos;
use crate::deflate::Event;
use crate::png::PngInfo;

pub fn make_literal_overlay_bytes(events: &[Event], info: &PngInfo) -> Vec<u8> {
    make_overlay(events, info, |_, e| match e {
        Event::Lit(_) => Some([80, 255, 80, 200]),
        Event::Ref(_) => None,
    })
}

/// `max_distance` is the longest back-reference distance in `events`, the
/// upper end of the colour ramp. Pass the value cached on
/// [`crate::deflate::DecodedDeflate`] rather than rescanning `events`.
pub fn make_distance_overlay_bytes(events: &[Event], info: &PngInfo, max_distance: u32) -> Vec<u8> {
    let max_d = max_distance.max(1) as f32;
    make_overlay(events, info, |_, e| match e {
        Event::Lit(_) => Some([255, 255, 255, 180]),
        Event::Ref(r) => Some(distance_colour((r.out_pos - r.src_out_pos) as usize, max_d)),
    })
}

pub fn make_block_overlay_bytes(
    events: &[Event],
    info: &PngInfo,
    block_starts: &[u32],
    num_blocks: usize,
) -> Vec<u8> {
    let nb = num_blocks.max(1);
    let palette: Vec<[u8; 4]> = (0..nb).map(|i| block_color(i, nb)).collect();
    // Events are in stream order, so the block index only advances. A
    // running cursor over `block_starts` beats a binary search per event.
    let mut block = 0usize;
    make_overlay(events, info, |i, _e| {
        while block + 1 < block_starts.len() && i as u32 >= block_starts[block + 1] {
            block += 1;
        }
        Some(palette[block % palette.len()])
    })
}

// helpers

/// Walk events once, painting each with the colour the caller picks.
/// `color_for(index, event)` returns `Some(rgba)` to paint or `None` to
/// skip: the literal overlay ignores refs, the ref overlays tint each kind
/// differently, and the block overlay uses the index to advance its block
/// cursor. Literals paint a single pixel; back-refs paint every pixel their
/// copy span touches.
fn make_overlay<F>(events: &[Event], info: &PngInfo, mut color_for: F) -> Vec<u8>
where
    F: FnMut(usize, &Event) -> Option<[u8; 4]>,
{
    let mut rgba = vec![0u8; info.width as usize * info.height as usize * 4];
    for (i, e) in events.iter().enumerate() {
        let Some(color) = color_for(i, e) else {
            continue;
        };
        match e {
            Event::Lit(lit) => paint_at(&mut rgba, OutPos(lit.out_pos), info, color),
            Event::Ref(r) => paint_ref_pixels(
                &mut rgba,
                r.out_pos as usize,
                r.copy_len as usize,
                info,
                color,
            ),
        }
    }
    rgba
}

/// Paint `color` into the RGBA slot for byte offset `pos`. No-op for filter
/// bytes or out-of-range positions.
#[inline]
fn paint_at(rgba: &mut [u8], pos: OutPos, info: &PngInfo, color: [u8; 4]) {
    if let Some(base) = info.rgba_index(pos)
        && base + 4 <= rgba.len()
    {
        rgba[base..base + 4].copy_from_slice(&color);
    }
}

/// Paint every pixel touched by the back-reference whose output span is
/// `[out_pos, out_pos + copy_len)`. Walks at pixel granularity, so each
/// pixel is painted exactly once even when the ref covers all `bpp`
/// channels.
fn paint_ref_pixels(
    rgba: &mut [u8],
    out_pos: usize,
    copy_len: usize,
    info: &PngInfo,
    color: [u8; 4],
) {
    let stride = info.row_stride;
    let bpp = info.bpp;
    let w = info.width as usize;
    let h = info.height as usize;
    let end = out_pos + copy_len;
    let mut pos = out_pos;
    while pos < end {
        let row = pos / stride;
        if row >= h {
            break;
        }
        let col = pos % stride;
        if col == 0 {
            // Per-row PNG filter byte, not a pixel channel.
            pos += 1;
            continue;
        }
        let x = (col - 1) / bpp;
        // `(col - 1) / bpp < w` is guaranteed for `row_stride == 1 + w*bpp`,
        // so no `x < w` guard needed.
        let base = (row * w + x) * 4;
        if base + 4 <= rgba.len() {
            rgba[base..base + 4].copy_from_slice(&color);
        }
        // Skip this pixel's remaining channel bytes to avoid repainting it.
        // Filter bytes and row boundaries fall to the outer `col == 0`
        // branch next iteration.
        let channel = (col - 1) % bpp;
        pos += bpp - channel;
    }
}

/// Two-segment cool→warm ramp: short distances cyan-ish, long ones red.
/// `t = dist / max_d` clamped to `[0, 1]` is split at 0.5: the lower
/// half ramps blue → cyan, the upper half cyan → red.
fn distance_colour(dist: usize, max_d: f32) -> [u8; 4] {
    let t = (dist as f32 / max_d).min(1.0);
    let (rv, gv, bv) = if t < 0.5 {
        let s = t / 0.5;
        (0u8, (255.0 * s) as u8, 255u8)
    } else {
        let s = (t - 0.5) / 0.5;
        (
            (255.0 * s) as u8,
            (255.0 * (1.0 - 200.0 / 255.0 * s)) as u8,
            (255.0 * (1.0 - s)) as u8,
        )
    };
    [rv, gv, bv, 200]
}

/// HSV-to-RGBA for the block overlay palette.
fn block_color(i: usize, nb: usize) -> [u8; 4] {
    let hue = (i as f32 / nb as f32) * 360.0;
    let h60 = hue / 60.0;
    let hi = h60 as usize % 6;
    let f = h60 - h60.floor();
    let p = (255.0 * (1.0 - f)) as u8;
    let q = (255.0 * f) as u8;
    match hi {
        0 => [255, q, 0, 160],
        1 => [255 - p, 255, 0, 160],
        2 => [0, 255, q, 160],
        3 => [0, 255 - p, 255, 160],
        4 => [q, 0, 255, 160],
        _ => [255, 0, 255 - q, 160],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate::{Event, LitEvent, RefEvent};

    fn lit(out_pos: u32) -> Event {
        Event::Lit(LitEvent {
            out_pos,
            symbol: 0,
            bit_start: 0,
        })
    }

    fn refe(out_pos: u32, src_out_pos: u32, copy_len: u16) -> Event {
        Event::Ref(RefEvent {
            out_pos,
            src_out_pos,
            copy_len,
            dist_sym: 0,
            dist_bit_start: 0,
        })
    }

    #[test]
    fn paint_ref_pixels_covers_full_row_no_redundant_writes() {
        // 4×1 RGB image: row_stride = 1 + 4*3 = 13. Filter byte at col 0.
        // A ref covering the whole row (bytes 1..13) should paint all 4
        // pixels exactly once each.
        let info = PngInfo::new(4, 1, 8, crate::png::ColorType::Rgb);
        let mut rgba = vec![0u8; 4 * 4];
        paint_ref_pixels(&mut rgba, 1, 12, &info, [1, 2, 3, 255]);
        for px in 0..4 {
            let base = px * 4;
            assert_eq!(&rgba[base..base + 4], &[1, 2, 3, 255], "px={px}");
        }
    }

    #[test]
    fn paint_ref_pixels_crosses_row_boundary() {
        // 2×2 RGB. row_stride = 7. Ref spans bytes 4..12: last pixel of
        // row 0 + filter + both pixels of row 1.
        let info = PngInfo::new(2, 2, 8, crate::png::ColorType::Rgb);
        let mut rgba = vec![0u8; 2 * 2 * 4];
        paint_ref_pixels(&mut rgba, 4, 8, &info, [9, 9, 9, 9]);
        // Row 0 pixel 1 should be painted (not pixel 0).
        assert_eq!(&rgba[0..4], &[0, 0, 0, 0]);
        assert_eq!(&rgba[4..8], &[9, 9, 9, 9]);
        // Row 1 both pixels painted.
        assert_eq!(&rgba[8..12], &[9, 9, 9, 9]);
        assert_eq!(&rgba[12..16], &[9, 9, 9, 9]);
    }

    #[test]
    fn distance_overlay_marks_lit_pixels_white() {
        let info = PngInfo::new(2, 1, 8, crate::png::ColorType::Rgb);
        let rgba = make_distance_overlay_bytes(&[lit(1)], &info, 1);
        assert_eq!(&rgba[0..4], &[255, 255, 255, 180]);
    }

    #[test]
    fn block_overlay_colours_by_block() {
        let info = PngInfo::new(1, 2, 8, crate::png::ColorType::Rgb);
        // Two events; block 0 owns event 0, block 1 owns event 1.
        let rgba = make_block_overlay_bytes(
            &[
                lit(1),        // row 0 pixel 0, block 0
                refe(5, 1, 3), // row 1 pixel 0, block 1
            ],
            &info,
            &[0, 1],
            2,
        );
        assert_ne!(rgba[3], 0, "row 0 has some alpha");
        assert_ne!(rgba[7], 0, "row 1 has some alpha");
    }

    #[test]
    fn literal_overlay_skips_refs() {
        // A ref at pixel 0 must not be painted green by the literal overlay.
        let info = PngInfo::new(2, 1, 8, crate::png::ColorType::Rgb);
        let rgba = make_literal_overlay_bytes(&[refe(1, 0, 3)], &info);
        assert_eq!(&rgba[0..4], &[0, 0, 0, 0]);
    }
}
