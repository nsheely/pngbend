//! Chunk-level decode preparation shared by both decode entry points: parse
//! the container, read IHDR + palette + tRNS, gather IDAT, and compute the
//! decompression-bomb cap. The zlib/DEFLATE decode (where the two paths
//! diverge) is left to the caller.

use crate::png::{
    Chunk, ChunkType, ColorType, PaletteEntry, PngInfo, TrnsKey, Warning, concat_idat,
    decode_palette, interlaced_output_len, parse_ihdr, read_chunks,
};

use super::PngError;

/// The result of [`prepare`]. Holds `idat` owned so the caller can borrow a
/// zlib view without a copy.
pub(super) struct Prep {
    pub(super) info: PngInfo,
    pub(super) palette: Option<Vec<PaletteEntry>>,
    pub(super) trns_key: Option<TrnsKey>,
    pub(super) idat: Vec<u8>,
    pub(super) warnings: Vec<Warning>,
    /// IHDR-implied inflated size, the decompression-bomb cap.
    pub(super) cap: usize,
}

pub(super) fn prepare(bytes: &[u8]) -> Result<Prep, PngError> {
    let parsed = read_chunks(bytes)?;
    let info = parse_ihdr(&parsed.chunks).ok_or(PngError::MissingIhdr)?;
    // Cap inflation at the IHDR-implied size (RFC 2083: IDAT decodes to
    // exactly `height * (1 + width * bpp)` bytes, or the sum of the Adam7
    // pass sizes when interlaced) so an adversarial IDAT can't pump the
    // decoder into gigabytes.
    let cap = if info.interlaced {
        let n = interlaced_output_len(&info);
        if n > u32::MAX as usize {
            return Err(PngError::OutputTooLarge {
                output_bytes: n as u64,
            });
        }
        n
    } else {
        expected_output_len(&info)?
    };
    let palette = read_palette(&parsed.chunks);
    let trns_key = parse_trns_key(&parsed.chunks, &info);
    let idat = concat_idat(&parsed.chunks);
    if idat.is_empty() {
        return Err(PngError::MissingIdat);
    }
    Ok(Prep {
        info,
        palette,
        trns_key,
        idat,
        warnings: parsed.warnings,
        cap,
    })
}

/// The tRNS colour key for a Greyscale or RGB image, or `None` (indexed
/// transparency is folded into the palette by `read_palette`).
fn parse_trns_key(chunks: &[Chunk], info: &PngInfo) -> Option<TrnsKey> {
    let d = &chunks.iter().find(|c| c.typ == ChunkType::TRNS)?.data;
    let be = |i: usize| u16::from_be_bytes([d[i], d[i + 1]]);
    match info.color_type {
        ColorType::Greyscale if d.len() >= 2 => Some(TrnsKey::Grey(be(0))),
        ColorType::Rgb if d.len() >= 6 => Some(TrnsKey::Rgb(be(0), be(2), be(4))),
        _ => None,
    }
}

/// `height * (1 + width * bpp)`, guarded to fit the codec's `u32` byte
/// positions.
fn expected_output_len(info: &PngInfo) -> Result<usize, PngError> {
    let output_bytes = u64::from(info.height) * (1 + u64::from(info.width) * info.bpp as u64);
    if output_bytes > u32::MAX as u64 {
        return Err(PngError::OutputTooLarge { output_bytes });
    }
    Ok(output_bytes as usize)
}

/// Decode PLTE (+ optional tRNS) into a palette, or `None` when the image
/// carries no palette chunk.
fn read_palette(chunks: &[Chunk]) -> Option<Vec<PaletteEntry>> {
    let plte = chunks.iter().find(|c| c.typ == ChunkType::PLTE)?;
    let trns = chunks
        .iter()
        .find(|c| c.typ == ChunkType::TRNS)
        .map(|c| c.data.as_slice());
    Some(decode_palette(&plte.data, trns))
}
