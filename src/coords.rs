//! Coordinate newtypes for the two integer kinds that flow through the
//! pipeline:
//!
//! - [`OutPos`] — byte offset within the unfiltered PNG output stream
//!   (length = `h * (1 + w * bpp)`; includes the per-row filter byte at
//!   column 0).
//! - [`PixelXY`] — image-space pixel coordinate (filter bytes excluded).
//!
//! Conversions go through [`ImgGeom`] which bundles `row_stride`,
//! `bpp`, `w`, and `h`.

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
    pub bpp: u32,
    pub row_stride: u32,
}

impl ImgGeom {
    pub fn new(w: u32, h: u32, bpp: u32) -> Self {
        Self {
            w,
            h,
            bpp,
            row_stride: 1 + w * bpp,
        }
    }

    /// Convert a byte offset in the output stream to pixel coordinates.
    /// Returns `None` for column 0 of any row (the per-row PNG filter byte).
    #[inline]
    pub fn out_to_xy(&self, pos: OutPos) -> Option<PixelXY> {
        let p = pos.0;
        let row = p / self.row_stride;
        let col = p % self.row_stride;
        if col == 0 || row >= self.h {
            return None;
        }
        let x = (col - 1) / self.bpp;
        if x < self.w {
            Some(PixelXY::new(x, row))
        } else {
            None
        }
    }

    /// Byte offset of the first channel of pixel `(x, y)` in the output stream.
    #[inline]
    pub fn xy_to_out(&self, xy: PixelXY) -> OutPos {
        OutPos(xy.y * self.row_stride + 1 + xy.x * self.bpp)
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
        let g = ImgGeom::new(4, 3, 3);
        assert_eq!(g.out_to_xy(OutPos(0)), None);
        assert_eq!(g.out_to_xy(OutPos(g.row_stride)), None);
    }

    #[test]
    fn out_to_xy_round_trips_via_xy_to_out() {
        let g = ImgGeom::new(7, 5, 4);
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
        let g = ImgGeom::new(4, 3, 3);
        // pos at end of last row + 1 → row >= h
        let past = OutPos(g.h * g.row_stride);
        assert!(g.out_to_xy(past).is_none());
    }
}
