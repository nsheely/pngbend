//! Pass-aware output-byte <-> screen-pixel projection.
//!
//! For a progressive image the mapping is [`PngInfo`]'s own coordinate
//! methods. For an Adam7-interlaced image the decoded `output` is seven
//! concatenated sub-images, so a screen pixel's byte lives in a pass-
//! dependent location. [`Raster`] unifies both: build it once from a
//! [`PngInfo`] and every consumer projects through it instead of assuming
//! one contiguous raster.

use crate::coords::{OutPos, PixelXY};
use crate::png::PngInfo;
use crate::png::interlace::{PASSES, pass_dims};

/// Adam7 pass (0-indexed) owning each pixel of the repeating 8x8 tile,
/// indexed `[y % 8][x % 8]` (PNG spec Figure 9).
const PASS_OF: [[u8; 8]; 8] = [
    [0, 5, 3, 5, 1, 5, 3, 5],
    [6, 6, 6, 6, 6, 6, 6, 6],
    [4, 5, 4, 5, 4, 5, 4, 5],
    [6, 6, 6, 6, 6, 6, 6, 6],
    [2, 5, 3, 5, 2, 5, 3, 5],
    [6, 6, 6, 6, 6, 6, 6, 6],
    [4, 5, 4, 5, 4, 5, 4, 5],
    [6, 6, 6, 6, 6, 6, 6, 6],
];

#[derive(Debug, Clone, Copy, Default)]
struct PassGeom {
    x0: u32,
    y0: u32,
    dx: u32,
    dy: u32,
    width: u32,
    height: u32,
    row_stride: usize,
    byte_offset: usize,
}

/// Projects between output-byte positions and screen pixels for one image,
/// whether progressive or Adam7-interlaced.
#[derive(Debug, Clone)]
pub struct Raster {
    info: PngInfo,
    /// Per-pass geometry when interlaced; empty passes have `width`/`height`
    /// 0. Unused (and left default) for progressive images.
    passes: [PassGeom; 7],
}

impl Raster {
    pub fn new(info: PngInfo) -> Self {
        let mut passes = [PassGeom::default(); 7];
        if info.interlaced {
            let mut byte_offset = 0usize;
            for (p, geom) in passes.iter_mut().enumerate() {
                let (w, h) = pass_dims(info.width, info.height, p);
                let (x0, y0, dx, dy) = PASSES[p];
                let row_stride = if w > 0 && h > 0 {
                    PngInfo::new(w, h, info.bit_depth, info.color_type).row_stride
                } else {
                    0
                };
                *geom = PassGeom {
                    x0,
                    y0,
                    dx,
                    dy,
                    width: w,
                    height: h,
                    row_stride,
                    byte_offset,
                };
                byte_offset += h as usize * row_stride;
            }
        }
        Self { info, passes }
    }

    pub fn info(&self) -> &PngInfo {
        &self.info
    }

    #[inline]
    pub fn pixels_per_byte(&self) -> u32 {
        self.info.pixels_per_byte()
    }

    /// Output byte holding the first channel of screen pixel `xy`.
    #[inline]
    pub fn xy_to_out(&self, xy: PixelXY) -> OutPos {
        if !self.info.interlaced {
            return self.info.xy_to_out(xy);
        }
        let p = PASS_OF[(xy.y % 8) as usize][(xy.x % 8) as usize] as usize;
        let g = &self.passes[p];
        let sub_x = (xy.x - g.x0) / g.dx;
        let sub_y = (xy.y - g.y0) / g.dy;
        let byte_in_row = (sub_x * self.info.bits_per_pixel) / 8;
        OutPos((g.byte_offset + sub_y as usize * g.row_stride + 1 + byte_in_row as usize) as u32)
    }

    /// Screen pixel a byte belongs to, or `None` for per-row filter bytes
    /// and out-of-range positions.
    #[inline]
    pub fn out_to_xy(&self, pos: OutPos) -> Option<PixelXY> {
        if !self.info.interlaced {
            return self.info.out_to_xy(pos);
        }
        let p = pos.0 as usize;
        for g in &self.passes {
            if g.width == 0 || g.height == 0 {
                continue;
            }
            let size = g.height as usize * g.row_stride;
            if p < g.byte_offset || p >= g.byte_offset + size {
                continue;
            }
            let local = p - g.byte_offset;
            let col = local % g.row_stride;
            if col == 0 {
                return None; // per-row filter byte
            }
            let sub_x = ((col - 1) as u32 * 8) / self.info.bits_per_pixel;
            if sub_x >= g.width {
                return None;
            }
            let row = (local / g.row_stride) as u32;
            return Some(PixelXY::new(g.x0 + sub_x * g.dx, g.y0 + row * g.dy));
        }
        None
    }

    /// Byte offset of the pixel at `pos` in a packed `w*h*4` RGBA buffer.
    #[inline]
    pub fn rgba_index(&self, pos: OutPos) -> Option<usize> {
        let xy = self.out_to_xy(pos)?;
        Some((xy.y as usize * self.info.width as usize + xy.x as usize) * 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::png::ColorType;

    fn round_trips(info: PngInfo) {
        let r = Raster::new(info);
        // Every screen pixel maps to a byte that maps back to it.
        for y in 0..info.height {
            for x in 0..info.width {
                let xy = PixelXY::new(x, y);
                let pos = r.xy_to_out(xy);
                assert_eq!(r.out_to_xy(pos), Some(xy), "xy {x},{y}");
            }
        }
    }

    #[test]
    fn progressive_matches_pnginfo() {
        let info = PngInfo::new(7, 5, 8, ColorType::Rgb);
        let r = Raster::new(info);
        for pos in (0..(info.height as usize * info.row_stride) as u32).map(OutPos) {
            assert_eq!(r.out_to_xy(pos), info.out_to_xy(pos));
        }
        round_trips(info);
    }

    #[test]
    fn interlaced_round_trips_pixels() {
        for (w, h, ct, bd) in [
            (2, 2, ColorType::Greyscale, 8),
            (8, 8, ColorType::Rgb, 8),
            (13, 9, ColorType::Rgba, 8),
            (5, 7, ColorType::GreyAlpha, 16),
        ] {
            let mut info = PngInfo::new(w, h, bd, ct);
            info.interlaced = true;
            round_trips(info);
        }
    }
}
