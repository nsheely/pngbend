//! Adam7 interlacing.
//!
//! An interlaced PNG's `output` is seven concatenated sub-images (passes),
//! each an ordinary filtered image with its own dimensions and stride.
//! Decoding reuses the per-image [`unfilter`] and [`to_rgba8`] on each pass,
//! then scatters the pass's pixels to their positions in the full raster.

use super::chunks::PngInfo;
use super::convert::{ConvertError, PaletteEntry, to_rgba8};
use super::filter::{FilterError, unfilter};

/// The seven Adam7 passes as `(x0, y0, dx, dy)`: pass `p` owns the pixels
/// `(x0 + i*dx, y0 + j*dy)` (PNG spec Figure 9).
pub(crate) const PASSES: [(u32, u32, u32, u32); 7] = [
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
];

/// Failure while reassembling an interlaced image: one of the per-pass
/// [`unfilter`] / [`to_rgba8`] steps failed, or the stream was too short.
#[derive(Debug)]
pub enum InterlaceError {
    Filter(FilterError),
    Convert(ConvertError),
    Truncated,
}

impl std::fmt::Display for InterlaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Filter(e) => write!(f, "interlace unfilter: {e}"),
            Self::Convert(e) => write!(f, "interlace convert: {e}"),
            Self::Truncated => write!(f, "interlaced stream ended mid-pass"),
        }
    }
}

impl std::error::Error for InterlaceError {}

impl From<FilterError> for InterlaceError {
    fn from(e: FilterError) -> Self {
        Self::Filter(e)
    }
}
impl From<ConvertError> for InterlaceError {
    fn from(e: ConvertError) -> Self {
        Self::Convert(e)
    }
}

/// Pixel dimensions of pass `p` for a `width x height` image; either may be
/// 0 (the pass is empty and contributes no bytes).
pub(crate) fn pass_dims(width: u32, height: u32, p: usize) -> (u32, u32) {
    let (x0, y0, dx, dy) = PASSES[p];
    let w = if width > x0 {
        (width - x0).div_ceil(dx)
    } else {
        0
    };
    let h = if height > y0 {
        (height - y0).div_ceil(dy)
    } else {
        0
    };
    (w, h)
}

/// A sub-image [`PngInfo`] for pass `p`, or `None` if the pass is empty.
fn pass_info(info: &PngInfo, p: usize) -> Option<PngInfo> {
    let (w, h) = pass_dims(info.width, info.height, p);
    (w > 0 && h > 0).then(|| PngInfo::new(w, h, info.bit_depth, info.color_type))
}

/// Byte length an Adam7 `output` decodes to: the sum of each non-empty
/// pass's `height * row_stride`. The decompression-bomb cap uses this in
/// place of the progressive `height * row_stride`.
pub fn interlaced_output_len(info: &PngInfo) -> usize {
    (0..7)
        .filter_map(|p| pass_info(info, p))
        .map(|pi| pi.height as usize * pi.row_stride)
        .sum()
}

/// The byte slice pass `p` occupies in `output`, advancing `offset`.
fn take_pass<'a>(
    output: &'a [u8],
    offset: &mut usize,
    pi: &PngInfo,
) -> Result<&'a [u8], InterlaceError> {
    let bytes = pi.height as usize * pi.row_stride;
    let end = offset
        .checked_add(bytes)
        .filter(|&e| e <= output.len())
        .ok_or(InterlaceError::Truncated)?;
    let slice = &output[*offset..end];
    *offset = end;
    Ok(slice)
}

/// Decode an Adam7 `output` to `width * height * 4` RGBA8.
pub fn deinterlace_to_rgba8(
    output: &[u8],
    info: &PngInfo,
    palette: Option<&[PaletteEntry]>,
) -> Result<Vec<u8>, InterlaceError> {
    let w = info.width as usize;
    let mut rgba = vec![0u8; w * info.height as usize * 4];
    let mut offset = 0usize;
    for (p, &(x0, y0, dx, dy)) in PASSES.iter().enumerate() {
        let Some(pi) = pass_info(info, p) else {
            continue;
        };
        let stream = take_pass(output, &mut offset, &pi)?;
        let pass_rgba = to_rgba8(&unfilter(stream, &pi)?, &pi, palette)?;
        let (pw, ph) = (pi.width as usize, pi.height as usize);
        for py in 0..ph {
            let full_y = y0 as usize + py * dy as usize;
            for px in 0..pw {
                let full_x = x0 as usize + px * dx as usize;
                let src = (py * pw + px) * 4;
                let dst = (full_y * w + full_x) * 4;
                rgba[dst..dst + 4].copy_from_slice(&pass_rgba[src..src + 4]);
            }
        }
    }
    Ok(rgba)
}

/// The per-pass unfiltered bytes, concatenated in pass order, for the
/// glass-box path. Each pass contributes `pass_height * (row_stride - 1)`
/// bytes.
pub fn deinterlace_unfilter(output: &[u8], info: &PngInfo) -> Result<Vec<u8>, InterlaceError> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    for p in 0..7 {
        let Some(pi) = pass_info(info, p) else {
            continue;
        };
        let stream = take_pass(output, &mut offset, &pi)?;
        out.extend_from_slice(&unfilter(stream, &pi)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::png::ColorType;

    #[test]
    fn pass_dims_for_a_2x2_image() {
        // Only passes 1, 6, 7 are non-empty for 2x2.
        let dims: Vec<(u32, u32)> = (0..7).map(|p| pass_dims(2, 2, p)).collect();
        assert_eq!(dims[0], (1, 1)); // pass 1: (0,0)
        assert_eq!(dims[1], (0, 1)); // pass 2: x0=4 > width
        assert_eq!(dims[5], (1, 1)); // pass 6: (1,0)
        assert_eq!(dims[6], (2, 1)); // pass 7: (0,1),(1,1)
    }

    #[test]
    fn interlaced_len_sums_nonempty_passes() {
        let info = PngInfo::new(2, 2, 8, ColorType::Greyscale);
        // pass1 grey8 1x1: stride 2, 1 row = 2; pass6 1x1: 2; pass7 2x1:
        // stride 3, 1 row = 3. Total 7.
        assert_eq!(interlaced_output_len(&info), 7);
    }
}
