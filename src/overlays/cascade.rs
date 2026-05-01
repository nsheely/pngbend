//! Cascade overlay: yellow/orange/red for LZ77 fan-out from a clicked pixel,
//! plus a faint blue halo for PNG row-filter propagation.

use crate::coords::{ImgGeom, OutPos};
use crate::index::Cascade;

/// Per-row min pixel-x reached by PNG row-filter propagation from
/// LZ77-affected bytes. `min_x[row] == None` means that row is unaffected;
/// otherwise pixels `[min_x[row], w-1]` are filter-affected beyond what
/// LZ77 already covers directly.
pub struct FilterExpansion {
    pub min_x: Vec<Option<u32>>,
}

impl FilterExpansion {
    pub fn iter(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.min_x
            .iter()
            .enumerate()
            .filter_map(|(row, opt)| opt.map(|x| (row as u32, x)))
    }

    pub fn is_empty(&self) -> bool {
        self.min_x.iter().all(Option::is_none)
    }
}

/// Propagate LZ77-affected byte positions through PNG row filters.
pub fn compute_filter_expansion(
    affected: &[u32],
    output: &[u8],
    geom: &ImgGeom,
) -> FilterExpansion {
    let h = geom.h as usize;
    let row_stride = geom.row_stride as usize;

    // Row filter type for each row.
    let row_ft: Vec<u8> = (0..h)
        .map(|r| output.get(r * row_stride).copied().unwrap_or(0))
        .collect();

    // Min pixel-x hit by LZ77 in each row (sentinel `u32::MAX` = unset).
    let mut lz77_min_x: Vec<u32> = vec![u32::MAX; h];
    for &pos in affected {
        if let Some(xy) = geom.out_to_xy(OutPos(pos)) {
            let row = xy.y as usize;
            if row < h && xy.x < lz77_min_x[row] {
                lz77_min_x[row] = xy.x;
            }
        }
    }

    if lz77_min_x.iter().all(|&x| x == u32::MAX) {
        return FilterExpansion {
            min_x: vec![None; h],
        };
    }

    let mut min_x: Vec<Option<u32>> = vec![None; h];
    let mut carry: Option<u32> = None;

    for row in 0..h {
        let ft = row_ft[row];
        let lz77 = (lz77_min_x[row] != u32::MAX).then_some(lz77_min_x[row]);

        // Filters 2 (Up), 3 (Average), 4 (Paeth) read the prior row, so any
        // affected pixel in the prior row carries forward.
        let effective = match (carry, lz77) {
            (Some(c), Some(m)) if matches!(ft, 2..=4) => Some(c.min(m)),
            (Some(c), None) if matches!(ft, 2..=4) => Some(c),
            (Some(_), _) => {
                carry = None;
                lz77
            }
            (None, l) => l,
        };

        if let Some(mx) = effective {
            min_x[row] = Some(mx);
            if row + 1 < h && matches!(row_ft[row + 1], 2..=4) {
                carry = Some(mx);
            }
        }
    }

    FilterExpansion { min_x }
}

/// RGBA overlay showing cascade footprint.
///
/// Depth 0 = bright yellow. Shallow = orange. Deep = red. Filter-halo
/// pixels painted faint blue.
pub fn make_cascade_overlay_bytes(
    cascade: &Cascade,
    filter: &FilterExpansion,
    geom: &ImgGeom,
) -> Vec<u8> {
    let w = geom.w as usize;
    let h = geom.h as usize;
    let mut rgba = vec![0u8; w * h * 4];
    let max_d = cascade.max_depth.max(1);

    // Blue filter-halo first, so cascade yellows/reds paint over it.
    for (row, min_x) in filter.iter() {
        if (row as usize) >= h {
            continue;
        }
        let row = row as usize;
        for x in (min_x as usize)..w {
            let base = (row * w + x) * 4;
            rgba[base] = 80;
            rgba[base + 1] = 120;
            rgba[base + 2] = 255;
            rgba[base + 3] = 40;
        }
    }

    for &pos in cascade.affected {
        let Some(base) = geom.rgba_index(OutPos(pos)) else {
            continue;
        };
        let d = cascade.depth(pos).unwrap_or(0);
        let colour = cascade_colour(d, max_d);
        rgba[base..base + 4].copy_from_slice(&colour);
    }

    rgba
}

#[inline]
fn cascade_colour(depth: u32, max_d: u32) -> [u8; 4] {
    if depth == 0 {
        return [255, 255, 80, 220]; // bright yellow — the seed
    }
    let t = (depth as f32 / max_d as f32).min(1.0);
    let g = (200.0 * (1.0 - t)) as u8;
    [255, g, 0, 180]
}
