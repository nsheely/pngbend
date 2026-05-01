//! Convert unfiltered PNG bytes to `w*h*4` RGBA8 for display.
//!
//! Supports the four non-palette color types at 8-bit and 16-bit depth.
//! Indexed (palette) PNGs are supported at 8-bit with an optional PLTE+tRNS
//! decoded into a 256-entry palette. Sub-byte (1/2/4 bit) depths return an
//! error; callers should fall back to `image::load_from_memory` for those.

use super::chunks::{ColorType, PngInfo};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvertError {
    UnsupportedDepth {
        color_type: ColorType,
        bit_depth: u8,
    },
    MissingPalette,
    TruncatedInput {
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedDepth {
                color_type,
                bit_depth,
            } => write!(
                f,
                "unsupported depth {bit_depth} for color type {color_type:?}"
            ),
            Self::MissingPalette => write!(f, "indexed PNG but no palette decoded"),
            Self::TruncatedInput { expected, actual } => {
                write!(
                    f,
                    "unfiltered input too short: expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for ConvertError {}

/// Palette entry — RGBA, with alpha defaulting to 255 when no tRNS is present.
pub type PaletteEntry = [u8; 4];

/// Convert unfiltered bytes to `w*h*4` RGBA8.
pub fn to_rgba8(
    unfiltered: &[u8],
    info: &PngInfo,
    palette: Option<&[PaletteEntry]>,
) -> Result<Vec<u8>, ConvertError> {
    let n_pixels = info.width as usize * info.height as usize;
    let mut rgba = vec![0u8; n_pixels * 4];
    to_rgba8_into(unfiltered, info, palette, &mut rgba)?;
    Ok(rgba)
}

/// Like [`to_rgba8`] but writes into a pre-allocated `rgba` buffer so
/// the caller can amortise the `w * h * 4` allocation across reloads.
pub fn to_rgba8_into(
    unfiltered: &[u8],
    info: &PngInfo,
    palette: Option<&[PaletteEntry]>,
    rgba: &mut [u8],
) -> Result<(), ConvertError> {
    let n_pixels = info.width as usize * info.height as usize;
    let expected = n_pixels * info.bpp;
    if unfiltered.len() < expected {
        return Err(ConvertError::TruncatedInput {
            expected,
            actual: unfiltered.len(),
        });
    }
    let w = info.width as usize;
    let h = info.height as usize;
    for y in 0..h {
        to_rgba8_row_unchecked(unfiltered, info, palette, rgba, y, w)?;
    }
    Ok(())
}

/// Convert `rows` (in whatever order — typically sorted) from the unfiltered
/// buffer into their RGBA slots in `rgba`. Used by the incremental edit
/// path: a literal swap touches a handful of rows, not the whole image.
pub fn to_rgba8_rows_into(
    unfiltered: &[u8],
    info: &PngInfo,
    palette: Option<&[PaletteEntry]>,
    rgba: &mut [u8],
    rows: impl IntoIterator<Item = usize>,
) -> Result<(), ConvertError> {
    let n_pixels = info.width as usize * info.height as usize;
    let expected = n_pixels * info.bpp;
    if unfiltered.len() < expected {
        return Err(ConvertError::TruncatedInput {
            expected,
            actual: unfiltered.len(),
        });
    }
    let w = info.width as usize;
    for y in rows {
        to_rgba8_row_unchecked(unfiltered, info, palette, rgba, y, w)?;
    }
    Ok(())
}

/// Convert one row worth of unfiltered pixels into four-byte RGBA. Bounds
/// are assumed valid — callers use the full-image wrappers above which
/// check them. `w` is the image width in pixels (passed in so the caller
/// saves a field load per row on hot paths).
fn to_rgba8_row_unchecked(
    unfiltered: &[u8],
    info: &PngInfo,
    palette: Option<&[PaletteEntry]>,
    rgba: &mut [u8],
    y: usize,
    w: usize,
) -> Result<(), ConvertError> {
    let rgba_base = y * w * 4;
    match (info.color_type, info.bit_depth) {
        (ColorType::Rgba, 8) => {
            let src_base = y * w * 4;
            rgba[rgba_base..rgba_base + w * 4]
                .copy_from_slice(&unfiltered[src_base..src_base + w * 4]);
        }
        (ColorType::Rgb, 8) => {
            let src_base = y * w * 3;
            for i in 0..w {
                let s = src_base + i * 3;
                let d = rgba_base + i * 4;
                write_rgba(
                    rgba,
                    d,
                    unfiltered[s],
                    unfiltered[s + 1],
                    unfiltered[s + 2],
                    255,
                );
            }
        }
        (ColorType::Greyscale, 8) => {
            let src_base = y * w;
            for i in 0..w {
                write_grey(rgba, rgba_base + i * 4, unfiltered[src_base + i], 255);
            }
        }
        (ColorType::GreyAlpha, 8) => {
            let src_base = y * w * 2;
            for i in 0..w {
                let s = src_base + i * 2;
                write_grey(rgba, rgba_base + i * 4, unfiltered[s], unfiltered[s + 1]);
            }
        }
        (ColorType::Indexed, 8) => {
            let pal = palette.ok_or(ConvertError::MissingPalette)?;
            let src_base = y * w;
            for i in 0..w {
                let idx = unfiltered[src_base + i] as usize;
                let c = pal.get(idx).copied().unwrap_or([0, 0, 0, 0]);
                let d = rgba_base + i * 4;
                rgba[d..d + 4].copy_from_slice(&c);
            }
        }
        // 16-bit color types: take the high byte of each big-endian sample.
        (ColorType::Rgba, 16) => {
            let src_base = y * w * 8;
            for i in 0..w {
                let s = src_base + i * 8;
                let d = rgba_base + i * 4;
                write_rgba(
                    rgba,
                    d,
                    unfiltered[s],
                    unfiltered[s + 2],
                    unfiltered[s + 4],
                    unfiltered[s + 6],
                );
            }
        }
        (ColorType::Rgb, 16) => {
            let src_base = y * w * 6;
            for i in 0..w {
                let s = src_base + i * 6;
                let d = rgba_base + i * 4;
                write_rgba(
                    rgba,
                    d,
                    unfiltered[s],
                    unfiltered[s + 2],
                    unfiltered[s + 4],
                    255,
                );
            }
        }
        (ColorType::Greyscale, 16) => {
            let src_base = y * w * 2;
            for i in 0..w {
                write_grey(rgba, rgba_base + i * 4, unfiltered[src_base + i * 2], 255);
            }
        }
        (ColorType::GreyAlpha, 16) => {
            let src_base = y * w * 4;
            for i in 0..w {
                let s = src_base + i * 4;
                write_grey(rgba, rgba_base + i * 4, unfiltered[s], unfiltered[s + 2]);
            }
        }
        (ct, bd) => {
            return Err(ConvertError::UnsupportedDepth {
                color_type: ct,
                bit_depth: bd,
            });
        }
    }
    Ok(())
}

/// Write one RGBA pixel `(r, g, b, a)` at `dst[d..d+4]`. A
/// `copy_from_slice` so LLVM emits a single 4-byte store.
#[inline(always)]
fn write_rgba(dst: &mut [u8], d: usize, r: u8, g: u8, b: u8, a: u8) {
    dst[d..d + 4].copy_from_slice(&[r, g, b, a]);
}

/// Write a greyscale pixel — replicate the luma to R/G/B and store
/// alpha — as one 4-byte slot.
#[inline(always)]
fn write_grey(dst: &mut [u8], d: usize, luma: u8, alpha: u8) {
    write_rgba(dst, d, luma, luma, luma, alpha);
}

/// Decode a PLTE chunk into a 256-entry palette. `plte` must be a multiple of
/// 3 bytes (RGB triples); `trns` if present supplies per-index alphas.
pub fn decode_palette(plte: &[u8], trns: Option<&[u8]>) -> Vec<PaletteEntry> {
    let mut pal = Vec::with_capacity(256);
    let mut i = 0;
    while i + 2 < plte.len() {
        pal.push([plte[i], plte[i + 1], plte[i + 2], 255]);
        i += 3;
    }
    if let Some(t) = trns {
        for (idx, &a) in t.iter().enumerate() {
            if idx < pal.len() {
                pal[idx][3] = a;
            }
        }
    }
    pal
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(w: u32, h: u32, color: ColorType, bit_depth: u8) -> PngInfo {
        let channels = color.channels() as usize;
        let bpp = ((channels * bit_depth as usize) / 8).max(1);
        PngInfo {
            width: w,
            height: h,
            bit_depth,
            color_type: color,
            bpp,
            row_stride: 1 + w as usize * bpp,
        }
    }

    #[test]
    fn rgba8_passthrough() {
        let unf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let out = to_rgba8(&unf, &info(2, 1, ColorType::Rgba, 8), None).unwrap();
        assert_eq!(out, unf);
    }

    #[test]
    fn rgb8_adds_opaque_alpha() {
        let unf = vec![10, 20, 30, 40, 50, 60];
        let out = to_rgba8(&unf, &info(2, 1, ColorType::Rgb, 8), None).unwrap();
        assert_eq!(out, vec![10, 20, 30, 255, 40, 50, 60, 255]);
    }

    #[test]
    fn grey8_replicates_to_rgb_with_opaque_alpha() {
        let unf = vec![128];
        let out = to_rgba8(&unf, &info(1, 1, ColorType::Greyscale, 8), None).unwrap();
        assert_eq!(out, vec![128, 128, 128, 255]);
    }

    #[test]
    fn grey_alpha_8() {
        let unf = vec![64, 200];
        let out = to_rgba8(&unf, &info(1, 1, ColorType::GreyAlpha, 8), None).unwrap();
        assert_eq!(out, vec![64, 64, 64, 200]);
    }

    #[test]
    fn indexed_8_uses_palette() {
        let unf = vec![0, 1, 2];
        let pal = vec![[10, 20, 30, 255], [40, 50, 60, 200], [70, 80, 90, 0]];
        let out = to_rgba8(&unf, &info(3, 1, ColorType::Indexed, 8), Some(&pal)).unwrap();
        assert_eq!(out, vec![10, 20, 30, 255, 40, 50, 60, 200, 70, 80, 90, 0]);
    }

    #[test]
    fn indexed_without_palette_errors() {
        let unf = vec![0];
        let err = to_rgba8(&unf, &info(1, 1, ColorType::Indexed, 8), None).unwrap_err();
        assert!(matches!(err, ConvertError::MissingPalette));
    }

    #[test]
    fn sub_byte_depth_unsupported() {
        // Provide enough bytes so we hit the match arm instead of tripping
        // the TruncatedInput guard first.
        let unf = vec![0xFF; 8];
        let bad = PngInfo {
            width: 8,
            height: 1,
            bit_depth: 1,
            color_type: ColorType::Greyscale,
            bpp: 1,
            row_stride: 2,
        };
        let err = to_rgba8(&unf, &bad, None).unwrap_err();
        assert!(matches!(err, ConvertError::UnsupportedDepth { .. }));
    }

    #[test]
    fn rgb16_takes_high_byte() {
        // 1 RGB 16-bit pixel: R=0x1234, G=0x5678, B=0x9ABC. Stored big-endian.
        let unf = vec![0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC];
        let out = to_rgba8(&unf, &info(1, 1, ColorType::Rgb, 16), None).unwrap();
        assert_eq!(out, vec![0x12, 0x56, 0x9A, 255]);
    }

    #[test]
    fn truncated_input_errors() {
        let unf = vec![1, 2];
        let err = to_rgba8(&unf, &info(2, 1, ColorType::Rgba, 8), None).unwrap_err();
        assert!(matches!(err, ConvertError::TruncatedInput { .. }));
    }

    #[test]
    fn decode_palette_without_trns_has_opaque_alpha() {
        let plte = vec![1, 2, 3, 4, 5, 6];
        let pal = decode_palette(&plte, None);
        assert_eq!(pal, vec![[1, 2, 3, 255], [4, 5, 6, 255]]);
    }

    #[test]
    fn decode_palette_with_trns_sets_alpha() {
        let plte = vec![10, 20, 30, 40, 50, 60];
        let trns = vec![100, 200];
        let pal = decode_palette(&plte, Some(&trns));
        assert_eq!(pal, vec![[10, 20, 30, 100], [40, 50, 60, 200]]);
    }
}
