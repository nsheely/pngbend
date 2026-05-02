//! PNG chunk parsing and IHDR interpretation.

#[derive(Debug)]
pub struct Chunk {
    pub typ: [u8; 4],
    pub data: Vec<u8>,
}

/// PNG color type per the IHDR spec. Carried on [`PngInfo`] so the
/// row-filter inverse and the colour-type → RGBA converter can dispatch
/// off it without re-parsing IHDR each call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorType {
    Greyscale,
    Rgb,
    Indexed,
    GreyAlpha,
    Rgba,
}

impl ColorType {
    fn from_byte(b: u8) -> Option<Self> {
        Some(match b {
            0 => Self::Greyscale,
            2 => Self::Rgb,
            3 => Self::Indexed,
            4 => Self::GreyAlpha,
            6 => Self::Rgba,
            _ => return None,
        })
    }

    pub fn channels(self) -> u32 {
        match self {
            Self::Greyscale | Self::Indexed => 1,
            Self::GreyAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

#[derive(Clone, Copy)]
pub struct PngInfo {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub color_type: ColorType,
    /// Bytes per pixel in the unfiltered stream.
    pub bpp: usize,
    /// 1 + width * bpp (the extra byte is the per-row filter type).
    pub row_stride: usize,
}

const PNG_SIG: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];

/// Fatal parse errors from [`read_chunks`]. Bad CRCs aren't here — they
/// surface as [`BadCrc`] warnings inside [`ParsedChunks`] so a glitched
/// file with stale checksums still loads.
#[derive(Debug)]
pub enum ChunksError {
    MissingSignature,
    Truncated,
}

impl std::fmt::Display for ChunksError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingSignature => write!(f, "PNG signature missing"),
            Self::Truncated => write!(f, "PNG truncated mid-chunk"),
        }
    }
}

impl std::error::Error for ChunksError {}

/// Output of [`read_chunks`]: the chunk list plus any per-chunk CRC
/// warnings. Derefs to `[Chunk]` so callers that don't care about
/// warnings can pass `&parsed` straight through to [`parse_ihdr`].
#[derive(Debug, Default)]
pub struct ParsedChunks {
    pub chunks: Vec<Chunk>,
    pub warnings: Vec<String>,
}

impl std::ops::Deref for ParsedChunks {
    type Target = [Chunk];
    fn deref(&self) -> &[Chunk] {
        &self.chunks
    }
}

/// Returns `true` if `data` begins with the 8-byte PNG signature.
pub fn verify_png_signature(data: &[u8]) -> bool {
    data.starts_with(&PNG_SIG)
}

pub fn read_chunks(data: &[u8]) -> Result<ParsedChunks, ChunksError> {
    if !verify_png_signature(data) {
        return Err(ChunksError::MissingSignature);
    }
    let mut out = ParsedChunks::default();
    let mut pos = 8; // skip signature
    while pos + 12 <= data.len() {
        let length =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let typ: [u8; 4] = [data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]];
        let end = pos + 8 + length;
        if end + 4 > data.len() {
            return Err(ChunksError::Truncated);
        }
        let chunk_data = &data[pos + 8..end];
        let stored = u32::from_be_bytes([data[end], data[end + 1], data[end + 2], data[end + 3]]);
        let computed = crc32(&typ, chunk_data);
        if stored != computed {
            let typ_str = std::str::from_utf8(&typ).unwrap_or("????");
            out.warnings
                .push(format!("stale checksum on {typ_str} chunk"));
        }
        out.chunks.push(Chunk {
            typ,
            data: chunk_data.to_vec(),
        });
        pos = end + 4; // skip CRC
    }
    Ok(out)
}

/// Concatenate every `IDAT` chunk's payload into a single buffer. The
/// PNG spec allows splitting the compressed stream across multiple IDATs;
/// the whole deflate stream is the concatenation in chunk order.
///
/// Two passes — sum lengths, then `extend_from_slice` into a `Vec` of
/// the exact capacity. One allocation total.
pub fn concat_idat(chunks: &[Chunk]) -> Vec<u8> {
    let total: usize = chunks
        .iter()
        .filter(|c| &c.typ == b"IDAT")
        .map(|c| c.data.len())
        .sum();
    let mut out = Vec::with_capacity(total);
    for c in chunks.iter().filter(|c| &c.typ == b"IDAT") {
        out.extend_from_slice(&c.data);
    }
    out
}

pub fn write_chunks(chunks: &[Chunk]) -> Vec<u8> {
    let mut out = PNG_SIG.to_vec();
    for chunk in chunks {
        let len = chunk.data.len() as u32;
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(&chunk.typ);
        out.extend_from_slice(&chunk.data);
        let crc = crc32(&chunk.typ, &chunk.data);
        out.extend_from_slice(&crc.to_be_bytes());
    }
    out
}

pub fn parse_ihdr(chunks: &[Chunk]) -> Option<PngInfo> {
    let ihdr = chunks.iter().find(|c| &c.typ == b"IHDR")?;
    // IHDR is exactly 13 bytes per the PNG spec: 4 width, 4 height,
    // bit-depth, color-type, compression-method, filter-method,
    // interlace-method. Anything shorter is a malformed chunk.
    if ihdr.data.len() < 13 {
        return None;
    }
    let width = u32::from_be_bytes([ihdr.data[0], ihdr.data[1], ihdr.data[2], ihdr.data[3]]);
    let height = u32::from_be_bytes([ihdr.data[4], ihdr.data[5], ihdr.data[6], ihdr.data[7]]);
    let bit_depth = ihdr.data[8];
    let color_type = ColorType::from_byte(ihdr.data[9])?;
    // PNG spec: compression-method and filter-method must each be 0
    // (the only values defined). Interlace-method is 0 (none) or 1
    // (Adam7); this editor doesn't support interlaced PNGs, so 1 is a
    // clean rejection rather than a silent miscompose.
    let compression_method = ihdr.data[10];
    let filter_method = ihdr.data[11];
    let interlace_method = ihdr.data[12];
    if compression_method != 0 || filter_method != 0 || interlace_method != 0 {
        return None;
    }
    let channels = color_type.channels() as usize;
    let bpp = ((channels * bit_depth as usize) / 8).max(1);
    Some(PngInfo {
        width,
        height,
        bit_depth,
        color_type,
        bpp,
        row_stride: 1 + width as usize * bpp,
    })
}

// ── CRC-32 (ISO 3309 / PNG) ──────────────────────────────────────────────────

pub(super) fn crc32(typ: &[u8; 4], data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for n in 0u32..256 {
            let mut c = n;
            for _ in 0..8 {
                if c & 1 != 0 {
                    c = 0xEDB88320 ^ (c >> 1);
                } else {
                    c >>= 1;
                }
            }
            t[n as usize] = c;
        }
        t
    });
    let mut crc: u32 = 0xFFFFFFFF;
    for &b in typ.iter().chain(data.iter()) {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFFFFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_vector_for_iend() {
        // PNG IEND chunk has 0-byte data; CRC of the type tag 'IEND' alone
        // equals 0xAE426082.
        assert_eq!(crc32(b"IEND", &[]), 0xAE426082);
    }

    #[test]
    fn read_write_chunks_round_trip() {
        let chunks = vec![
            Chunk {
                typ: *b"IHDR",
                data: vec![1, 2, 3, 4, 5, 6, 7, 8, 8, 6, 0, 0, 0],
            },
            Chunk {
                typ: *b"IDAT",
                data: vec![0xAA; 100],
            },
            Chunk {
                typ: *b"IEND",
                data: vec![],
            },
        ];
        let bytes = write_chunks(&chunks);
        let parsed = read_chunks(&bytes).expect("parse round-tripped chunks");
        assert_eq!(parsed.len(), chunks.len());
        for (orig, got) in chunks.iter().zip(parsed.iter()) {
            assert_eq!(orig.typ, got.typ);
            assert_eq!(orig.data, got.data);
        }
    }

    #[test]
    fn parse_ihdr_decodes_rgba8() {
        // width=2, height=3, bit_depth=8, color=6 (RGBA)
        let mut data = vec![0u8; 13];
        data[0..4].copy_from_slice(&2u32.to_be_bytes());
        data[4..8].copy_from_slice(&3u32.to_be_bytes());
        data[8] = 8;
        data[9] = 6;
        let chunks = vec![Chunk {
            typ: *b"IHDR",
            data,
        }];
        let info = parse_ihdr(&chunks).expect("parse");
        assert_eq!(info.width, 2);
        assert_eq!(info.height, 3);
        assert_eq!(info.bpp, 4);
        assert_eq!(info.row_stride, 1 + 2 * 4);
        assert_eq!(info.color_type, ColorType::Rgba);
        assert_eq!(info.bit_depth, 8);
    }

    #[test]
    fn parse_ihdr_rejects_unknown_color_type() {
        let mut data = vec![0u8; 13];
        data[0..4].copy_from_slice(&1u32.to_be_bytes());
        data[4..8].copy_from_slice(&1u32.to_be_bytes());
        data[8] = 8;
        data[9] = 9; // unknown
        let chunks = vec![Chunk {
            typ: *b"IHDR",
            data,
        }];
        assert!(parse_ihdr(&chunks).is_none());
    }

    /// Regression: pre-fix the length check was `< 10`, leaving
    /// 11/12-byte IHDRs to read past their data when validating the
    /// trailing compression / filter / interlace bytes.
    #[test]
    fn parse_ihdr_rejects_short_ihdr_data() {
        for short_len in [0usize, 9, 10, 11, 12] {
            let chunks = vec![Chunk {
                typ: *b"IHDR",
                data: vec![0u8; short_len],
            }];
            assert!(
                parse_ihdr(&chunks).is_none(),
                "IHDR with {short_len} bytes must be rejected"
            );
        }
    }

    #[test]
    fn parse_ihdr_rejects_invalid_compression_method() {
        let mut data = vec![0u8; 13];
        data[0..4].copy_from_slice(&1u32.to_be_bytes());
        data[4..8].copy_from_slice(&1u32.to_be_bytes());
        data[8] = 8;
        data[9] = 6;
        data[10] = 1; // PNG spec only defines compression-method 0
        let chunks = vec![Chunk {
            typ: *b"IHDR",
            data,
        }];
        assert!(parse_ihdr(&chunks).is_none());
    }

    #[test]
    fn parse_ihdr_rejects_interlaced_png() {
        let mut data = vec![0u8; 13];
        data[0..4].copy_from_slice(&1u32.to_be_bytes());
        data[4..8].copy_from_slice(&1u32.to_be_bytes());
        data[8] = 8;
        data[9] = 6;
        data[12] = 1; // Adam7 interlace — not supported by this editor
        let chunks = vec![Chunk {
            typ: *b"IHDR",
            data,
        }];
        assert!(parse_ihdr(&chunks).is_none());
    }

    #[test]
    fn read_chunks_rejects_missing_signature() {
        let buf = b"NOT_A_PNG_FILE_AT_ALL_NOPE_NOPE";
        let err = read_chunks(buf).unwrap_err();
        assert!(matches!(err, ChunksError::MissingSignature), "got {err:?}");
        assert!(!verify_png_signature(buf));
        assert!(verify_png_signature(&PNG_SIG));
    }

    #[test]
    fn read_chunks_surfaces_bad_chunk_crc_as_warning() {
        // Build a one-chunk PNG, then flip a bit in the IEND CRC.
        // pngbend is an editor, not a viewer: a stale CRC must surface
        // as a warning so the user can still load the file.
        let chunks = vec![Chunk {
            typ: *b"IEND",
            data: vec![],
        }];
        let mut bytes = write_chunks(&chunks);
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        let parsed = read_chunks(&bytes).expect("CRC mismatch must not fail the parse");
        assert_eq!(parsed.chunks.len(), 1);
        assert_eq!(parsed.warnings.len(), 1);
        assert!(parsed.warnings[0].contains("IEND"));
    }
}
