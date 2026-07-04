//! Convert unfiltered PNG bytes to `w*h*4` RGBA8 for display.
//!
//! Covers every PNG colour mode the format defines:
//! - RGB / RGBA / Greyscale / GreyAlpha at 8-bit and 16-bit depth
//! - Indexed (palette) at 1, 2, 4, or 8-bit depth
//! - Greyscale at 1, 2, or 4-bit depth (scaled to 8-bit luma per PNG §13.10)
//!
//! Sub-byte depths pack pixels MSB-first within each byte and pad rows
//! to a whole-byte boundary; the unfilter step runs on bytes regardless
//! of pixel depth, so this layer just unpacks.

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
    /// Allocating the RGBA buffer would overflow `usize` or exceed
    /// `u32::MAX` bytes. Returned when callers pass IHDR dimensions
    /// without the loader's prior dimension check.
    OutputTooLarge {
        width: u32,
        height: u32,
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
            Self::OutputTooLarge { width, height } => write!(
                f,
                "RGBA output would exceed limit for {width}×{height} image"
            ),
        }
    }
}

impl std::error::Error for ConvertError {}

/// Palette entry: RGBA, with alpha defaulting to 255 when no tRNS is present.
pub type PaletteEntry = [u8; 4];

/// Verify `unfiltered` has the full `h * (row_stride - 1)` bytes the
/// row-walking code will read. Shared between [`to_rgba8`] and
/// [`to_rgba8_rows_into`] so the contract is one definition.
fn require_full_unfiltered(unfiltered: &[u8], info: &PngInfo) -> Result<(), ConvertError> {
    let expected = info.height as usize * (info.row_stride - 1);
    if unfiltered.len() < expected {
        Err(ConvertError::TruncatedInput {
            expected,
            actual: unfiltered.len(),
        })
    } else {
        Ok(())
    }
}

/// Convert unfiltered bytes to `w*h*4` RGBA8. Rejects pathological
/// dimensions before allocating (caps at the loader's `u32::MAX`
/// limit) and pre-checks the unfiltered input length against the
/// expected `h * (row_stride - 1)` row-data bytes; for sub-byte depths
/// that's smaller than `n_pixels * bpp` because multiple pixels pack
/// into a byte.
pub fn to_rgba8(
    unfiltered: &[u8],
    info: &PngInfo,
    palette: Option<&[PaletteEntry]>,
) -> Result<Vec<u8>, ConvertError> {
    let total = (info.width as usize)
        .checked_mul(info.height as usize)
        .and_then(|p| p.checked_mul(4))
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or(ConvertError::OutputTooLarge {
            width: info.width,
            height: info.height,
        })?;
    require_full_unfiltered(unfiltered, info)?;
    let mut rgba = vec![0u8; total];
    let w = info.width as usize;
    for y in 0..info.height as usize {
        to_rgba8_row_unchecked(unfiltered, info, palette, &mut rgba, y, w)?;
    }
    Ok(rgba)
}

/// Convert `rows` (in whatever order, typically sorted) from the unfiltered
/// buffer into their RGBA slots in `rgba`. Used by the incremental edit
/// path: a literal swap touches a handful of rows, not the whole image.
pub fn to_rgba8_rows_into(
    unfiltered: &[u8],
    info: &PngInfo,
    palette: Option<&[PaletteEntry]>,
    rgba: &mut [u8],
    rows: impl IntoIterator<Item = usize>,
) -> Result<(), ConvertError> {
    require_full_unfiltered(unfiltered, info)?;
    let w = info.width as usize;
    for y in rows {
        to_rgba8_row_unchecked(unfiltered, info, palette, rgba, y, w)?;
    }
    Ok(())
}

/// Convert one row worth of unfiltered pixels into four-byte RGBA. Bounds
/// are assumed valid; callers use the full-image wrappers above which
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
        // Sub-byte depths: PNG packs MSB-first within each byte, rows
        // padded to a whole byte. Greyscale values scale to 8-bit luma
        // via the spec's "value * (255 / (2^bd - 1))" rule (1-bit:
        // *255, 2-bit: *85, 4-bit: *17). Indexed values index PLTE
        // directly without scaling.
        (ColorType::Greyscale, 1) => {
            let src_base = y * w.div_ceil(8);
            for i in 0..w {
                let byte = unfiltered[src_base + i / 8];
                let bit = (byte >> (7 - (i % 8))) & 1;
                write_grey(rgba, rgba_base + i * 4, bit * 255, 255);
            }
        }
        (ColorType::Greyscale, 2) => {
            let src_base = y * w.div_ceil(4);
            for i in 0..w {
                let byte = unfiltered[src_base + i / 4];
                let sample = (byte >> ((3 - (i % 4)) * 2)) & 0x3;
                write_grey(rgba, rgba_base + i * 4, sample * 85, 255);
            }
        }
        (ColorType::Greyscale, 4) => {
            let src_base = y * w.div_ceil(2);
            for i in 0..w {
                let byte = unfiltered[src_base + i / 2];
                let sample = (byte >> ((1 - (i % 2)) * 4)) & 0xF;
                write_grey(rgba, rgba_base + i * 4, sample * 17, 255);
            }
        }
        (ColorType::Indexed, 1) => {
            let pal = palette.ok_or(ConvertError::MissingPalette)?;
            let src_base = y * w.div_ceil(8);
            for i in 0..w {
                let byte = unfiltered[src_base + i / 8];
                let idx = ((byte >> (7 - (i % 8))) & 1) as usize;
                let c = pal.get(idx).copied().unwrap_or([0, 0, 0, 0]);
                let d = rgba_base + i * 4;
                rgba[d..d + 4].copy_from_slice(&c);
            }
        }
        (ColorType::Indexed, 2) => {
            let pal = palette.ok_or(ConvertError::MissingPalette)?;
            let src_base = y * w.div_ceil(4);
            for i in 0..w {
                let byte = unfiltered[src_base + i / 4];
                let idx = ((byte >> ((3 - (i % 4)) * 2)) & 0x3) as usize;
                let c = pal.get(idx).copied().unwrap_or([0, 0, 0, 0]);
                let d = rgba_base + i * 4;
                rgba[d..d + 4].copy_from_slice(&c);
            }
        }
        (ColorType::Indexed, 4) => {
            let pal = palette.ok_or(ConvertError::MissingPalette)?;
            let src_base = y * w.div_ceil(2);
            for i in 0..w {
                let byte = unfiltered[src_base + i / 2];
                let idx = ((byte >> ((1 - (i % 2)) * 4)) & 0xF) as usize;
                let c = pal.get(idx).copied().unwrap_or([0, 0, 0, 0]);
                let d = rgba_base + i * 4;
                rgba[d..d + 4].copy_from_slice(&c);
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

/// Write a greyscale pixel (replicate the luma to R/G/B and store
/// alpha) as one 4-byte slot.
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

/// A tRNS colour key: pixels whose samples equal this value decode to
/// transparent (PNG spec §11.3.2, colour types Greyscale and RGB only).
/// Indexed transparency is folded into the palette by [`decode_palette`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrnsKey {
    Grey(u16),
    Rgb(u16, u16, u16),
}

impl TrnsKey {
    /// The key as an 8-bit RGB triple in the space [`to_rgba8`] produces,
    /// so a matching pixel is found by RGB comparison. Sub-byte greyscale
    /// scales by the spec's `255 / (2^bd - 1)` rule; 16-bit takes the high
    /// byte, matching the decoder's downsample.
    fn to_rgb8(self, bit_depth: u8) -> [u8; 3] {
        let scale = |s: u16| -> u8 {
            match bit_depth {
                16 => (s >> 8) as u8,
                8 => s as u8,
                4 => (s * 17) as u8,
                2 => (s * 85) as u8,
                _ => (s * 255) as u8,
            }
        };
        match self {
            Self::Grey(g) => {
                let v = scale(g);
                [v, v, v]
            }
            Self::Rgb(r, g, b) => [scale(r), scale(g), scale(b)],
        }
    }
}

/// Set alpha to 0 on every pixel whose RGB equals the tRNS colour `key`.
/// Call after [`to_rgba8`] for Greyscale / RGB images that carry a tRNS
/// chunk. Exact for 1/2/4/8-bit; approximate for 16-bit (matches the
/// high-byte decode).
pub fn apply_color_key(rgba: &mut [u8], info: &PngInfo, key: TrnsKey) {
    let target = key.to_rgb8(info.bit_depth);
    for px in rgba.chunks_exact_mut(4) {
        if px[..3] == target {
            px[3] = 0;
        }
    }
}

/// Pack RGBA8 pixels back into the raw (unfiltered) byte layout for
/// `info`'s colour type, the inverse of [`to_rgba8`]. Supports the byte-
/// aligned non-indexed types (Grey/RGB/GreyAlpha/RGBA at 8 and 16 bit);
/// RGBA8 is a straight copy. Indexed and sub-byte depths return
/// [`ConvertError::UnsupportedDepth`] (they need a palette or quantisation
/// the caller must supply). 16-bit packing writes each 8-bit sample as
/// `[v, v]`, so it round-trips the high-byte decode losslessly.
///
/// Grey targets take the R channel and RGB targets drop alpha, so the
/// input must be consistent with the target (`R == G == B` for grey,
/// `A == 255` for the alpha-less types) to be lossless.
pub fn pack(rgba: &[u8], info: &PngInfo) -> Result<Vec<u8>, ConvertError> {
    let w = info.width as usize;
    let h = info.height as usize;
    let expected = w * h * 4;
    if rgba.len() < expected {
        return Err(ConvertError::TruncatedInput {
            expected,
            actual: rgba.len(),
        });
    }
    let row_bytes = info.row_stride - 1;
    let mut out = vec![0u8; h * row_bytes];
    for y in 0..h {
        let px = &rgba[y * w * 4..y * w * 4 + w * 4];
        let row = &mut out[y * row_bytes..y * row_bytes + row_bytes];
        pack_row(px, info, row, w)?;
    }
    Ok(out)
}

fn pack_row(px: &[u8], info: &PngInfo, row: &mut [u8], w: usize) -> Result<(), ConvertError> {
    match (info.color_type, info.bit_depth) {
        (ColorType::Rgba, 8) => row.copy_from_slice(px),
        (ColorType::Rgb, 8) => {
            for i in 0..w {
                row[i * 3..i * 3 + 3].copy_from_slice(&px[i * 4..i * 4 + 3]);
            }
        }
        (ColorType::Greyscale, 8) => {
            for i in 0..w {
                row[i] = px[i * 4];
            }
        }
        (ColorType::GreyAlpha, 8) => {
            for i in 0..w {
                row[i * 2] = px[i * 4];
                row[i * 2 + 1] = px[i * 4 + 3];
            }
        }
        (ColorType::Rgba, 16) => {
            for i in 0..w {
                for c in 0..4 {
                    let v = px[i * 4 + c];
                    row[i * 8 + c * 2] = v;
                    row[i * 8 + c * 2 + 1] = v;
                }
            }
        }
        (ColorType::Rgb, 16) => {
            for i in 0..w {
                for c in 0..3 {
                    let v = px[i * 4 + c];
                    row[i * 6 + c * 2] = v;
                    row[i * 6 + c * 2 + 1] = v;
                }
            }
        }
        (ColorType::Greyscale, 16) => {
            for i in 0..w {
                let v = px[i * 4];
                row[i * 2] = v;
                row[i * 2 + 1] = v;
            }
        }
        (ColorType::GreyAlpha, 16) => {
            for i in 0..w {
                let (v, a) = (px[i * 4], px[i * 4 + 3]);
                row[i * 4] = v;
                row[i * 4 + 1] = v;
                row[i * 4 + 2] = a;
                row[i * 4 + 3] = a;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn info(w: u32, h: u32, color: ColorType, bit_depth: u8) -> PngInfo {
        PngInfo::new(w, h, bit_depth, color)
    }

    /// `to_rgba8(pack(rgba)) == rgba` for every byte-aligned non-indexed
    /// type, given an input consistent with the target (grey has R==G==B,
    /// alpha-less types have A==255).
    #[test]
    fn pack_then_decode_round_trips() {
        for (color, depth) in [
            (ColorType::Rgba, 8),
            (ColorType::Rgb, 8),
            (ColorType::Greyscale, 8),
            (ColorType::GreyAlpha, 8),
            (ColorType::Rgba, 16),
            (ColorType::Rgb, 16),
            (ColorType::Greyscale, 16),
            (ColorType::GreyAlpha, 16),
        ] {
            let (w, h) = (4usize, 3usize);
            let g = info(w as u32, h as u32, color, depth);
            let has_alpha = matches!(color, ColorType::Rgba | ColorType::GreyAlpha);
            let is_grey = matches!(color, ColorType::Greyscale | ColorType::GreyAlpha);
            let rgba: Vec<u8> = (0..w * h)
                .flat_map(|i| {
                    let v = (i * 9 + 1) as u8;
                    let (r, gc, b) = if is_grey {
                        (v, v, v)
                    } else {
                        (v, v ^ 0x5A, v ^ 0x33)
                    };
                    let a = if has_alpha { (i * 4 + 2) as u8 } else { 255 };
                    [r, gc, b, a]
                })
                .collect();
            let raw = pack(&rgba, &g).unwrap();
            assert_eq!(raw.len(), h * (g.row_stride - 1));
            let back = to_rgba8(&raw, &g, None).unwrap();
            assert_eq!(back, rgba, "{color:?} {depth}-bit");
        }
    }

    #[test]
    fn pack_rejects_indexed_and_subbyte() {
        let rgba = vec![0u8; 4 * 4];
        assert!(matches!(
            pack(&rgba, &info(2, 2, ColorType::Indexed, 8)),
            Err(ConvertError::UnsupportedDepth { .. })
        ));
        assert!(matches!(
            pack(&rgba, &info(2, 2, ColorType::Greyscale, 4)),
            Err(ConvertError::UnsupportedDepth { .. })
        ));
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
    fn one_bit_greyscale_unpacks_msb_first() {
        // One byte 0b10110001 = pixels 1, 0, 1, 1, 0, 0, 0, 1.
        let unf = vec![0b1011_0001];
        let out = to_rgba8(&unf, &info(8, 1, ColorType::Greyscale, 1), None).unwrap();
        let lumas: Vec<u8> = out.chunks_exact(4).map(|p| p[0]).collect();
        assert_eq!(lumas, vec![255, 0, 255, 255, 0, 0, 0, 255]);
    }

    #[test]
    fn two_bit_greyscale_scales_to_eight_bit() {
        // One byte 0b00_01_10_11 = samples 0, 1, 2, 3 → lumas 0, 85, 170, 255.
        let unf = vec![0b00_01_10_11];
        let out = to_rgba8(&unf, &info(4, 1, ColorType::Greyscale, 2), None).unwrap();
        let lumas: Vec<u8> = out.chunks_exact(4).map(|p| p[0]).collect();
        assert_eq!(lumas, vec![0, 85, 170, 255]);
    }

    #[test]
    fn four_bit_greyscale_scales_to_eight_bit() {
        // One byte 0xA5 → high nibble 0xA → luma 0xAA, low nibble 0x5 → 0x55.
        let unf = vec![0xA5];
        let out = to_rgba8(&unf, &info(2, 1, ColorType::Greyscale, 4), None).unwrap();
        let lumas: Vec<u8> = out.chunks_exact(4).map(|p| p[0]).collect();
        assert_eq!(lumas, vec![0xAA, 0x55]);
    }

    #[test]
    fn one_bit_indexed_uses_palette() {
        // Byte 0b1010_0000, pal[0]=red, pal[1]=blue → R B R B R R R R.
        let unf = vec![0b1010_0000];
        let pal = vec![[255, 0, 0, 255], [0, 0, 255, 255]];
        let out = to_rgba8(&unf, &info(8, 1, ColorType::Indexed, 1), Some(&pal)).unwrap();
        // First pixel comes from bit 7 (MSB) = 1 → blue.
        assert_eq!(&out[0..4], &[0, 0, 255, 255]);
        assert_eq!(&out[4..8], &[255, 0, 0, 255]);
        assert_eq!(&out[8..12], &[0, 0, 255, 255]);
    }

    #[test]
    fn four_bit_indexed_uses_palette() {
        let unf = vec![0x12];
        let mut pal = vec![[0, 0, 0, 255]; 16];
        pal[1] = [10, 20, 30, 255];
        pal[2] = [40, 50, 60, 255];
        let out = to_rgba8(&unf, &info(2, 1, ColorType::Indexed, 4), Some(&pal)).unwrap();
        assert_eq!(&out[0..4], &[10, 20, 30, 255]); // high nibble = 1
        assert_eq!(&out[4..8], &[40, 50, 60, 255]); // low nibble = 2
    }

    #[test]
    fn sub_byte_row_padding_handled() {
        // 9-pixel 1-bit row → 2 bytes (last byte has 7 padding bits).
        // Pixels: 1 0 1 0 1 0 1 0 | 1 ? ? ? ? ? ? ?
        let unf = vec![0b1010_1010, 0b1000_0000];
        let out = to_rgba8(&unf, &info(9, 1, ColorType::Greyscale, 1), None).unwrap();
        let lumas: Vec<u8> = out.chunks_exact(4).map(|p| p[0]).collect();
        assert_eq!(lumas, vec![255, 0, 255, 0, 255, 0, 255, 0, 255]);
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
