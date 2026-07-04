//! The decode entry points: [`decode`] (bytes -> RGBA8), [`decode_strict`]
//! (the same but rejecting bad checksums, for use as a conformant decoder),
//! and [`decode_with_events`] (the glass-box path that also returns the
//! DEFLATE event stream and per-block Huffman tables).
//!
//! Safe by default: inflation is capped at the exact output the IHDR
//! dimensions call for, so a decompression-bomb PNG (a tiny IDAT that would
//! expand to gigabytes) is rejected rather than allocated.

use crate::deflate::{decode_deflate, inflate};
use crate::png::{
    Warning, adler32, apply_color_key, deinterlace_to_rgba8, deinterlace_unfilter,
    parse_zlib_stream, to_rgba8, unfilter,
};

use super::prep::prepare;
use super::{GlassBox, Image, PngError};

/// Decode a PNG to RGBA8 pixels.
///
/// The common path when you just want an image. For bit-level editing of
/// the compressed stream, use [`decode_with_events`] instead.
///
/// ```no_run
/// let bytes = std::fs::read("in.png").unwrap();
/// let img = glasspng::decode(&bytes).unwrap();
/// assert_eq!(img.pixels.len(), img.info.width as usize * img.info.height as usize * 4);
/// ```
pub fn decode(bytes: &[u8]) -> Result<Image, PngError> {
    let mut prep = prepare(bytes)?;
    let zlib = parse_zlib_stream(&prep.idat)?;
    prep.warnings.extend(zlib.warnings);

    // The lean path: `inflate` produces only the output bytes, skipping the
    // per-symbol event log the glass-box path builds.
    let output = inflate(zlib.deflate_buf, Some(prep.cap))?;
    if zlib.stored_adler != adler32(&output) {
        prep.warnings.push(Warning::StaleImageAdler);
    }

    let mut pixels = if prep.info.interlaced {
        deinterlace_to_rgba8(&output, &prep.info, prep.palette.as_deref())?
    } else {
        let unfiltered = unfilter(&output, &prep.info)?;
        to_rgba8(&unfiltered, &prep.info, prep.palette.as_deref())?
    };
    if let Some(key) = prep.trns_key {
        apply_color_key(&mut pixels, &prep.info, key);
    }
    Ok(Image {
        info: prep.info,
        palette: prep.palette,
        pixels,
        warnings: prep.warnings,
    })
}

/// Decode like [`decode`], but reject any image whose chunk CRC or zlib
/// Adler-32 fails instead of tolerating it, so glasspng acts as a
/// conformant decoder. [`decode`] (and the glass-box path) tolerate stale
/// checksums so glitched files still load.
pub fn decode_strict(bytes: &[u8]) -> Result<Image, PngError> {
    let img = decode(bytes)?;
    // Every warning glasspng emits is an integrity mismatch (chunk CRC,
    // zlib FCHECK, or the image Adler-32), so a non-empty list is a bad
    // checksum.
    match img.warnings.first() {
        Some(w) => Err(PngError::BadChecksum(w.clone())),
        None => Ok(img),
    }
}

/// Decode a PNG and keep the DEFLATE event stream and per-block Huffman
/// tables: the glass-box path behind [`crate::deflate::Event`]-level
/// editing.
pub fn decode_with_events(bytes: &[u8]) -> Result<GlassBox, PngError> {
    let mut prep = prepare(bytes)?;
    let zlib = parse_zlib_stream(&prep.idat)?;
    prep.warnings.extend(zlib.warnings);

    let deflate = decode_deflate(zlib.deflate_buf, Some(prep.cap))?;
    if zlib.stored_adler != adler32(&deflate.output) {
        prep.warnings.push(Warning::StaleImageAdler);
    }

    // For interlaced images `unfiltered` is the per-pass raw bytes
    // concatenated (there is no single progressive raster to undo).
    let unfiltered = if prep.info.interlaced {
        deinterlace_unfilter(&deflate.output, &prep.info)?
    } else {
        unfilter(&deflate.output, &prep.info)?
    };
    Ok(GlassBox {
        info: prep.info,
        palette: prep.palette,
        deflate,
        unfiltered,
        warnings: prep.warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate::{Event, serialize_stored};
    use crate::png::{Chunk, ChunkType, build_zlib_stream, write_chunks};

    /// Hand-assemble a 2×2 8-bit RGB PNG whose IDAT is a single stored
    /// (uncompressed) DEFLATE block, so the fixture needs no compressor.
    /// Filter byte 0 (None) on each row means the raw output *is* the
    /// filtered stream, so the pixel values round-trip verbatim.
    fn tiny_rgb_png() -> (Vec<u8>, [[u8; 3]; 4]) {
        let px = [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]];
        // row_stride = 1 filter byte + 2 px * 3 bytes = 7; two rows.
        let mut output = Vec::new();
        for row in 0..2 {
            output.push(0); // filter: None
            for col in 0..2 {
                output.extend_from_slice(&px[row * 2 + col]);
            }
        }
        // Stored DEFLATE block: BFINAL=1, BTYPE=00 -> 0x01; then LEN / NLEN.
        let len = output.len() as u16;
        let mut deflate = vec![0x01];
        deflate.extend_from_slice(&len.to_le_bytes());
        deflate.extend_from_slice(&(!len).to_le_bytes());
        deflate.extend_from_slice(&output);
        let zlib = build_zlib_stream(&deflate, [0x78, 0x01], &output);

        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&2u32.to_be_bytes()); // width
        ihdr.extend_from_slice(&2u32.to_be_bytes()); // height
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // depth, RGB, comp, filter, interlace
        let png = write_chunks(&[
            Chunk {
                typ: ChunkType::IHDR,
                data: ihdr,
            },
            Chunk {
                typ: ChunkType::IDAT,
                data: zlib,
            },
            Chunk {
                typ: ChunkType::IEND,
                data: Vec::new(),
            },
        ]);
        (png, px)
    }

    #[test]
    fn decode_yields_rgba_pixels() {
        let (png, px) = tiny_rgb_png();
        let img = decode(&png).expect("decode");
        assert_eq!(img.info.width, 2);
        assert_eq!(img.info.height, 2);
        assert_eq!(img.pixels.len(), 2 * 2 * 4);
        for (i, expect) in px.iter().enumerate() {
            let got = &img.pixels[i * 4..i * 4 + 4];
            assert_eq!(got, &[expect[0], expect[1], expect[2], 255], "pixel {i}");
        }
        assert!(img.warnings.is_empty(), "clean fixture: {:?}", img.warnings);
    }

    #[test]
    fn decode_with_events_exposes_the_stream() {
        let (png, _) = tiny_rgb_png();
        let gb = decode_with_events(&png).expect("decode_with_events");
        // Stored block emits one literal event per output byte (14 bytes).
        assert_eq!(gb.deflate.output.len(), 14);
        assert_eq!(gb.deflate.events.len(), 14);
        assert!(gb.deflate.events.iter().all(|e| matches!(e, Event::Lit(_))));
        assert_eq!(gb.deflate.num_blocks(), 1);
        assert_eq!(gb.unfiltered.len(), 2 * 2 * 3);
    }

    #[test]
    fn truncated_input_is_an_error_not_a_panic() {
        assert!(matches!(decode(&[]), Err(PngError::Chunks(_))));
        assert!(matches!(decode(b"not a png"), Err(PngError::Chunks(_))));
    }

    #[test]
    fn decode_applies_trns_colour_key() {
        // 3×1 greyscale [10, 20, 30] with a tRNS key of 20: the middle
        // pixel decodes transparent, the others opaque.
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&3u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 0, 0, 0, 0]); // depth 8, greyscale
        let filtered = vec![0u8, 10, 20, 30]; // filter None + 3 samples
        let idat = build_zlib_stream(&serialize_stored(&filtered), [0x78, 0x9c], &filtered);
        let png = write_chunks(&[
            Chunk {
                typ: ChunkType::IHDR,
                data: ihdr,
            },
            Chunk {
                typ: ChunkType::TRNS,
                data: vec![0, 20], // 16-bit BE key; low byte 20
            },
            Chunk {
                typ: ChunkType::IDAT,
                data: idat,
            },
            Chunk {
                typ: ChunkType::IEND,
                data: Vec::new(),
            },
        ]);
        let img = decode(&png).expect("decode");
        assert_eq!(&img.pixels[0..4], &[10, 10, 10, 255]);
        assert_eq!(&img.pixels[4..8], &[20, 20, 20, 0]);
        assert_eq!(&img.pixels[8..12], &[30, 30, 30, 255]);
    }

    #[test]
    fn decode_reassembles_adam7_interlaced() {
        // 2×2 greyscale, interlace=1. Non-empty passes are 1 (0,0), 6
        // (1,0), 7 (0,1)+(1,1); each row is filter byte 0 + samples.
        let (a, b, c, d) = (10u8, 20u8, 30u8, 40u8);
        let output = vec![0, a, 0, b, 0, c, d]; // pass1, pass6, pass7
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&2u32.to_be_bytes());
        ihdr.extend_from_slice(&2u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 0, 0, 0, 1]); // depth 8, grey, interlace 1
        let idat = build_zlib_stream(&serialize_stored(&output), [0x78, 0x9c], &output);
        let png = write_chunks(&[
            Chunk {
                typ: ChunkType::IHDR,
                data: ihdr,
            },
            Chunk {
                typ: ChunkType::IDAT,
                data: idat,
            },
            Chunk {
                typ: ChunkType::IEND,
                data: Vec::new(),
            },
        ]);
        let img = decode(&png).expect("decode interlaced");
        assert!(img.info.interlaced);
        assert_eq!(&img.pixels[0..4], &[a, a, a, 255]); // (0,0)
        assert_eq!(&img.pixels[4..8], &[b, b, b, 255]); // (1,0)
        assert_eq!(&img.pixels[8..12], &[c, c, c, 255]); // (0,1)
        assert_eq!(&img.pixels[12..16], &[d, d, d, 255]); // (1,1)
    }
}
