//! High-level decode entry points: the two-tier public surface.
//!
//! - [`decode`] is the standard path: bytes in, RGBA8 pixels out.
//! - [`decode_with_events`] is the glass-box path: it additionally returns
//!   the DEFLATE event stream and per-block Huffman tables, so a consumer
//!   can edit the compressed representation and re-emit a valid PNG.
//!
//! Both are safe by default: the inflated size is capped at the exact
//! output the IHDR dimensions call for, so a decompression-bomb PNG (a
//! tiny IDAT that would expand to gigabytes) is rejected rather than
//! allocated.

use crate::deflate::{DecodeError, DecodedDeflate, compress, decode_deflate, inflate};
use crate::png::{
    Chunk, ChunkType, ChunksError, ColorType, ConvertError, FilterError, FilterStrategy,
    InterlaceError, PaletteEntry, PngInfo, TrnsKey, Warning, ZLIB_DEFAULT_HEADER, ZlibError,
    adler32, apply_color_key, build_zlib_stream, concat_idat, decode_palette, deinterlace_to_rgba8,
    deinterlace_unfilter, filter, interlaced_output_len, pack, parse_ihdr, parse_zlib_stream,
    read_chunks, to_rgba8, unfilter, write_chunks,
};

/// A decoded image: metadata plus tightly-packed RGBA8 pixels
/// (`info.width * info.height * 4` bytes, row-major, no padding).
#[derive(Debug, Clone)]
pub struct Image {
    pub info: PngInfo,
    /// Decoded palette for indexed PNGs; `None` for every other colour type.
    pub palette: Option<Vec<PaletteEntry>>,
    /// RGBA8, one `[r, g, b, a]` per pixel in raster order.
    pub pixels: Vec<u8>,
    /// Non-fatal issues found while decoding, e.g. a chunk whose CRC doesn't
    /// match or a stale zlib Adler-32. Advisory; the image still decoded.
    pub warnings: Vec<Warning>,
}

/// A glass-box decode: everything [`Image`] needs plus the compressed
/// representation itself. Pair the (possibly edited) `deflate.output` with
/// [`crate::png::build_zlib_stream`] and [`crate::png::write_chunks`] to
/// re-emit a valid PNG.
#[derive(Debug)]
pub struct GlassBox {
    pub info: PngInfo,
    pub palette: Option<Vec<PaletteEntry>>,
    /// The decoded DEFLATE stream: raw filtered `output` bytes, the per-
    /// literal / per-back-reference [`crate::deflate::Event`] log, the
    /// per-block Huffman encoder tables, and block boundaries.
    pub deflate: DecodedDeflate,
    /// The `output` bytes with the per-row PNG filters undone
    /// (`height * (row_stride - 1)` bytes): the input to RGBA conversion.
    pub unfiltered: Vec<u8>,
    pub warnings: Vec<Warning>,
}

/// Anything that can go wrong turning PNG bytes into pixels, unified across
/// the chunk / IHDR / zlib / DEFLATE / unfilter / convert stages so callers
/// match on one type.
#[derive(Debug)]
pub enum PngError {
    /// The chunk layer rejected the bytes (bad signature, truncation, ...).
    Chunks(ChunksError),
    /// No IHDR chunk, or its fields didn't parse.
    MissingIhdr,
    /// The image has no IDAT image data.
    MissingIdat,
    /// IHDR dimensions imply an unfiltered output larger than 4 GiB, which
    /// the codec's `u32` byte positions can't address.
    OutputTooLarge { output_bytes: u64 },
    /// The zlib wrapper around the DEFLATE data was malformed.
    Zlib(ZlibError),
    /// The DEFLATE stream failed to decode.
    Deflate(DecodeError),
    /// A row-filter type was invalid or a row was malformed.
    Filter(FilterError),
    /// Pixel bytes couldn't be converted to RGBA8 (e.g. an unsupported
    /// bit depth, a missing palette, or truncated input). An out-of-range
    /// palette index is not an error; it decodes to transparent black.
    Convert(ConvertError),
    /// An interlaced (Adam7) image's data ended mid-pass. A per-pass filter
    /// or conversion failure surfaces as [`PngError::Filter`] /
    /// [`PngError::Convert`] instead, so each leaf error has a single home.
    InterlaceTruncated,
    /// A chunk CRC or the zlib Adler-32 didn't match. Only [`decode_strict`]
    /// returns this; [`decode`] tolerates the mismatch and reports it in
    /// [`Image::warnings`] instead.
    BadChecksum(Warning),
}

impl std::fmt::Display for PngError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chunks(e) => write!(f, "PNG chunk error: {e}"),
            Self::MissingIhdr => write!(f, "no valid IHDR chunk"),
            Self::MissingIdat => write!(f, "no IDAT image data"),
            Self::OutputTooLarge { output_bytes } => {
                write!(
                    f,
                    "unfiltered output {output_bytes} bytes exceeds 4 GiB cap"
                )
            }
            Self::Zlib(e) => write!(f, "zlib error: {e}"),
            Self::Deflate(e) => write!(f, "DEFLATE error: {e}"),
            Self::Filter(e) => write!(f, "unfilter error: {e}"),
            Self::Convert(e) => write!(f, "RGBA conversion error: {e}"),
            Self::InterlaceTruncated => write!(f, "interlaced stream ended mid-pass"),
            Self::BadChecksum(w) => write!(f, "checksum mismatch: {w}"),
        }
    }
}

impl std::error::Error for PngError {}

impl From<ChunksError> for PngError {
    fn from(e: ChunksError) -> Self {
        Self::Chunks(e)
    }
}
impl From<ZlibError> for PngError {
    fn from(e: ZlibError) -> Self {
        Self::Zlib(e)
    }
}
impl From<DecodeError> for PngError {
    fn from(e: DecodeError) -> Self {
        Self::Deflate(e)
    }
}
impl From<FilterError> for PngError {
    fn from(e: FilterError) -> Self {
        Self::Filter(e)
    }
}
impl From<ConvertError> for PngError {
    fn from(e: ConvertError) -> Self {
        Self::Convert(e)
    }
}
impl From<InterlaceError> for PngError {
    fn from(e: InterlaceError) -> Self {
        // Flatten: a per-pass filter/convert failure is the same leaf as its
        // progressive counterpart, so route it to the flat variant rather
        // than nesting it under a second Interlace layer.
        match e {
            InterlaceError::Filter(f) => Self::Filter(f),
            InterlaceError::Convert(c) => Self::Convert(c),
            InterlaceError::Truncated => Self::InterlaceTruncated,
        }
    }
}

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

/// The pixel formats [`encode`] can produce from RGBA8 input: the byte-aligned
/// non-indexed colour types. Indexed and sub-byte outputs, which `pack` can't
/// build, are simply not representable here rather than rejected at runtime.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OutputFormat {
    #[default]
    Rgba8,
    Rgb8,
    Grey8,
    GreyAlpha8,
    Rgba16,
    Rgb16,
    Grey16,
    GreyAlpha16,
}

impl OutputFormat {
    /// The `(colour type, bit depth)` this format writes to IHDR and packs to.
    fn dims(self) -> (ColorType, u8) {
        match self {
            Self::Rgba8 => (ColorType::Rgba, 8),
            Self::Rgb8 => (ColorType::Rgb, 8),
            Self::Grey8 => (ColorType::Greyscale, 8),
            Self::GreyAlpha8 => (ColorType::GreyAlpha, 8),
            Self::Rgba16 => (ColorType::Rgba, 16),
            Self::Rgb16 => (ColorType::Rgb, 16),
            Self::Grey16 => (ColorType::Greyscale, 16),
            Self::GreyAlpha16 => (ColorType::GreyAlpha, 16),
        }
    }
}

/// Output format and filtering for [`encode`].
#[derive(Debug, Clone, Copy, Default)]
pub struct EncodeOptions {
    pub format: OutputFormat,
    pub filter: FilterStrategy,
}

/// Encode an [`Image`]'s RGBA8 pixels to PNG bytes: `pack` -> `filter` ->
/// DEFLATE -> zlib -> chunks, the mirror of [`decode`]. The output colour
/// type / depth come from `options` (default RGBA8). Targets [`pack`] can't
/// produce from RGBA8 (indexed, sub-byte) return [`PngError::Convert`].
///
/// The DEFLATE stream is compressed with greedy LZ77, emitted as whichever
/// of stored / fixed-Huffman / dynamic-Huffman is smallest (see [`compress`]).
/// `decode(encode(img)).pixels == img.pixels`.
pub fn encode(image: &Image, options: &EncodeOptions) -> Result<Vec<u8>, PngError> {
    let (color_type, bit_depth) = options.format.dims();
    let info = PngInfo::new(image.info.width, image.info.height, bit_depth, color_type);
    let raw = pack(&image.pixels, &info)?;
    let filtered = filter(&raw, &info, options.filter);
    let deflate = compress(&filtered);
    let idat = build_zlib_stream(&deflate, ZLIB_DEFAULT_HEADER, &filtered);
    let chunks = [
        Chunk {
            typ: ChunkType::IHDR,
            data: ihdr_bytes(&info),
        },
        Chunk {
            typ: ChunkType::IDAT,
            data: idat,
        },
        Chunk {
            typ: ChunkType::IEND,
            data: Vec::new(),
        },
    ];
    Ok(write_chunks(&chunks))
}

/// The 13-byte IHDR payload for `info` (compression / filter / interlace
/// all 0).
fn ihdr_bytes(info: &PngInfo) -> Vec<u8> {
    let mut d = Vec::with_capacity(13);
    d.extend_from_slice(&info.width.to_be_bytes());
    d.extend_from_slice(&info.height.to_be_bytes());
    d.push(info.bit_depth);
    d.push(info.color_type.to_byte());
    d.extend_from_slice(&[0, 0, 0]);
    d
}

/// The chunk-level work shared by both entry points: parse the container,
/// read IHDR + palette, and gather the IDAT bytes, leaving the zlib/DEFLATE
/// decode (which is where the two paths diverge) to the caller. Returns the
/// `idat` owned so the caller can borrow a zlib view without a copy.
struct Prep {
    info: PngInfo,
    palette: Option<Vec<PaletteEntry>>,
    trns_key: Option<TrnsKey>,
    idat: Vec<u8>,
    warnings: Vec<Warning>,
    /// IHDR-implied inflated size, the decompression-bomb cap.
    cap: usize,
}

fn prepare(bytes: &[u8]) -> Result<Prep, PngError> {
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
fn read_palette(chunks: &[crate::png::Chunk]) -> Option<Vec<PaletteEntry>> {
    let plte = chunks.iter().find(|c| c.typ == ChunkType::PLTE)?;
    let trns = chunks
        .iter()
        .find(|c| c.typ == ChunkType::TRNS)
        .map(|c| c.data.as_slice());
    Some(decode_palette(&plte.data, trns))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate::Event;
    use crate::png::{Chunk, build_zlib_stream, write_chunks};

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
    fn encode_then_decode_round_trips_pixels() {
        // Arbitrary 3×2 RGBA8 image; encode with each filter strategy and
        // confirm the decoded pixels come back identical.
        let (w, h) = (3u32, 2u32);
        let pixels: Vec<u8> = (0..w * h * 4).map(|i| (i * 13 + 7) as u8).collect();
        let img = Image {
            info: PngInfo::new(w, h, 8, ColorType::Rgba),
            palette: None,
            pixels: pixels.clone(),
            warnings: Vec::new(),
        };
        for filter in [
            FilterStrategy::MinSad,
            FilterStrategy::Fixed(crate::png::FilterType::Paeth),
            FilterStrategy::Fixed(crate::png::FilterType::None),
        ] {
            let opts = EncodeOptions {
                filter,
                ..Default::default()
            };
            let bytes = encode(&img, &opts).expect("encode");
            let back = decode(&bytes).expect("decode");
            assert_eq!(back.info.width, w);
            assert_eq!(back.info.height, h);
            assert_eq!(back.pixels, pixels, "filter {filter:?}");
        }
    }

    #[test]
    fn encode_as_rgb_drops_alpha_and_round_trips() {
        // RGB output target: input alpha 255, pixels survive the trip.
        let (w, h) = (2u32, 2u32);
        let pixels: Vec<u8> = (0..w * h)
            .flat_map(|i| [(i * 20) as u8, (i * 20 + 5) as u8, (i * 20 + 9) as u8, 255])
            .collect();
        let img = Image {
            info: PngInfo::new(w, h, 8, ColorType::Rgb),
            palette: None,
            pixels: pixels.clone(),
            warnings: Vec::new(),
        };
        let opts = EncodeOptions {
            format: OutputFormat::Rgb8,
            ..Default::default()
        };
        let back = decode(&encode(&img, &opts).unwrap()).unwrap();
        assert_eq!(back.pixels, pixels);
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
        let idat = build_zlib_stream(
            &crate::deflate::serialize_stored(&filtered),
            [0x78, 0x9c],
            &filtered,
        );
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
        let idat = build_zlib_stream(
            &crate::deflate::serialize_stored(&output),
            [0x78, 0x9c],
            &output,
        );
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
