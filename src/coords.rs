//! Coordinate newtypes for the two integer kinds that flow through the
//! pipeline:
//!
//! - [`OutPos`] — byte offset within the unfiltered PNG output stream
//!   (length = `h * row_stride`; includes the per-row filter byte at
//!   column 0).
//! - [`PixelXY`] — image-space pixel coordinate (filter bytes excluded).
//!
//! Conversions go through [`ImgGeom`] which bundles `row_stride`,
//! `bpp`, `bits_per_pixel`, `w`, and `h`. At sub-byte depths (1/2/4-bit
//! greyscale or indexed) one byte holds multiple pixels — [`out_to_xy`]
//! returns the *first* pixel of that byte's cluster, and [`xy_to_out`]
//! collapses every pixel in a cluster to the same byte position.

/// Byte offset in the unfiltered PNG output (the deflate decoder's `output`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct OutPos(pub u32);

/// Image-space pixel coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PixelXY {
    pub x: u32,
    pub y: u32,
}

impl PixelXY {
    pub const fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }
}

/// Image geometry needed to convert between coordinate spaces.
#[derive(Debug, Clone, Copy)]
pub struct ImgGeom {
    pub w: u32,
    pub h: u32,
    /// Bytes per pixel, rounded up — used as the filter-byte offset for
    /// Sub/Average/Paeth predictors. For sub-byte depths this is 1
    /// (PNG spec §9.4: filter-byte offset rounds up to whole bytes).
    pub bpp: u32,
    /// True bits per pixel: `bit_depth * channels`. 1/2/4 for sub-byte
    /// greyscale and indexed; 8/16/24/32/48/64 for the byte-aligned
    /// types. Drives the pixel↔byte arithmetic in [`out_to_xy`] /
    /// [`xy_to_out`].
    pub bits_per_pixel: u32,
    /// Bytes per row including the leading filter byte.
    pub row_stride: u32,
}

impl ImgGeom {
    pub fn new(w: u32, h: u32, bits_per_pixel: u32) -> Self {
        // Bytes per pixel for filter offset: round up to whole bytes, min 1.
        let bpp = bits_per_pixel.div_ceil(8).max(1);
        // Row data bytes: round up so 9 pixels at 1 bpp give 2 bytes, etc.
        let row_data_bytes = (w * bits_per_pixel).div_ceil(8);
        Self {
            w,
            h,
            bpp,
            bits_per_pixel,
            row_stride: 1 + row_data_bytes,
        }
    }

    /// Convert a byte offset in the output stream to pixel coordinates.
    /// Returns `None` for column 0 of any row (the per-row PNG filter
    /// byte). For sub-byte depths the returned `x` is the first pixel
    /// of the byte's packed cluster.
    #[inline]
    pub fn out_to_xy(&self, pos: OutPos) -> Option<PixelXY> {
        let p = pos.0;
        let row = p / self.row_stride;
        let col = p % self.row_stride;
        if col == 0 || row >= self.h {
            return None;
        }
        // pixel index = byte_in_row * 8 / bits_per_pixel.
        // 8-bit RGB (bits_per_pixel=24): byte 4 → (3*8)/24 = 1, second
        // pixel ✓. 1-bit greyscale (bits_per_pixel=1): byte 1 → 0,
        // cluster start ✓.
        let x = ((col - 1) * 8) / self.bits_per_pixel;
        if x < self.w {
            Some(PixelXY::new(x, row))
        } else {
            None
        }
    }

    /// Byte offset of the first channel of pixel `(x, y)` in the output
    /// stream. For sub-byte depths multiple x-values collapse to the
    /// same byte (the cluster they share).
    #[inline]
    pub fn xy_to_out(&self, xy: PixelXY) -> OutPos {
        let byte_in_row = (xy.x * self.bits_per_pixel) / 8;
        OutPos(xy.y * self.row_stride + 1 + byte_in_row)
    }

    /// Number of pixels packed into one byte at this depth. `1` for
    /// every byte-aligned colour mode; `2`, `4`, or `8` for 4/2/1-bit
    /// greyscale and indexed.
    #[inline]
    pub fn pixels_per_byte(&self) -> u32 {
        if self.bits_per_pixel >= 8 {
            1
        } else {
            8 / self.bits_per_pixel
        }
    }

    /// Byte offset into a packed `w * h * 4` RGBA buffer for the pixel at
    /// `pos`, or `None` if `pos` is a filter byte or out of range.
    #[inline]
    pub fn rgba_index(&self, pos: OutPos) -> Option<usize> {
        let xy = self.out_to_xy(pos)?;
        Some((xy.y as usize * self.w as usize + xy.x as usize) * 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_to_xy_filter_byte_returns_none() {
        // 4×3 8-bit RGB: bits_per_pixel = 24.
        let g = ImgGeom::new(4, 3, 24);
        assert_eq!(g.out_to_xy(OutPos(0)), None);
        assert_eq!(g.out_to_xy(OutPos(g.row_stride)), None);
    }

    #[test]
    fn out_to_xy_round_trips_via_xy_to_out() {
        // 7×5 8-bit RGBA: bits_per_pixel = 32.
        let g = ImgGeom::new(7, 5, 32);
        for y in 0..g.h {
            for x in 0..g.w {
                let xy = PixelXY::new(x, y);
                let pos = g.xy_to_out(xy);
                assert_eq!(g.out_to_xy(pos), Some(xy));
            }
        }
    }

    #[test]
    fn out_to_xy_past_image_returns_none() {
        let g = ImgGeom::new(4, 3, 24);
        // pos at end of last row + 1 → row >= h
        let past = OutPos(g.h * g.row_stride);
        assert!(g.out_to_xy(past).is_none());
    }

    #[test]
    fn one_bit_greyscale_packs_eight_pixels_per_byte() {
        // 1-bit greyscale, width 16, height 1. Row is 2 data bytes plus filter.
        let g = ImgGeom::new(16, 1, 1);
        assert_eq!(g.bpp, 1);
        assert_eq!(g.row_stride, 3);
        assert_eq!(g.pixels_per_byte(), 8);
        // Every pixel in 0..8 collapses to byte 1 (first data byte).
        for x in 0..8 {
            assert_eq!(g.xy_to_out(PixelXY::new(x, 0)), OutPos(1));
        }
        // Pixels 8..16 land in byte 2.
        for x in 8..16 {
            assert_eq!(g.xy_to_out(PixelXY::new(x, 0)), OutPos(2));
        }
        // out_to_xy returns the first pixel of each byte's cluster.
        assert_eq!(g.out_to_xy(OutPos(1)), Some(PixelXY::new(0, 0)));
        assert_eq!(g.out_to_xy(OutPos(2)), Some(PixelXY::new(8, 0)));
    }

    #[test]
    fn four_bit_indexed_packs_two_pixels_per_byte() {
        // 4-bit indexed, width 5, height 1. 5 pixels = 3 bytes (ceil).
        let g = ImgGeom::new(5, 1, 4);
        assert_eq!(g.bpp, 1);
        assert_eq!(g.row_stride, 4);
        assert_eq!(g.pixels_per_byte(), 2);
        assert_eq!(g.xy_to_out(PixelXY::new(0, 0)), OutPos(1));
        assert_eq!(g.xy_to_out(PixelXY::new(1, 0)), OutPos(1));
        assert_eq!(g.xy_to_out(PixelXY::new(2, 0)), OutPos(2));
        assert_eq!(g.xy_to_out(PixelXY::new(4, 0)), OutPos(3));
    }
}
