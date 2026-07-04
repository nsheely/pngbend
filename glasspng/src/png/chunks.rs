//! PNG chunk parsing and IHDR interpretation.

/// A four-byte PNG chunk type tag (`IHDR`, `IDAT`, ...). A newtype so
/// comparisons read `c.typ == ChunkType::IDAT` not `&c.typ == b"IDAT"`, the
/// named tags live in one place, and `Display` prints the tag or `????`
/// when it isn't UTF-8.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkType(pub [u8; 4]);

impl ChunkType {
    pub const IHDR: Self = Self(*b"IHDR");
    pub const IDAT: Self = Self(*b"IDAT");
    pub const IEND: Self = Self(*b"IEND");
    pub const PLTE: Self = Self(*b"PLTE");
    pub const TRNS: Self = Self(*b"tRNS");

    #[inline]
    pub fn as_bytes(&self) -> &[u8; 4] {
        &self.0
    }
}

impl std::fmt::Display for ChunkType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(std::str::from_utf8(&self.0).unwrap_or("????"))
    }
}

#[derive(Debug)]
pub struct Chunk {
    pub typ: ChunkType,
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

    /// IHDR colour-type byte (PNG spec Table 11.1).
    pub fn to_byte(self) -> u8 {
        match self {
            Self::Greyscale => 0,
            Self::Rgb => 2,
            Self::Indexed => 3,
            Self::GreyAlpha => 4,
            Self::Rgba => 6,
        }
    }

    pub fn channels(self) -> u32 {
        match self {
            Self::Greyscale | Self::Indexed => 1,
            Self::GreyAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    /// Whether `bit_depth` is legal for this colour type (PNG spec Table
    /// 11.1). Greyscale allows 1/2/4/8/16, indexed 1/2/4/8, and the
    /// truecolour and alpha types only 8/16.
    fn allows_bit_depth(self, bit_depth: u8) -> bool {
        match self {
            Self::Greyscale => matches!(bit_depth, 1 | 2 | 4 | 8 | 16),
            Self::Indexed => matches!(bit_depth, 1 | 2 | 4 | 8),
            Self::Rgb | Self::GreyAlpha | Self::Rgba => matches!(bit_depth, 8 | 16),
        }
    }

    /// Per-channel display names, one per sample in output order.
    /// Greyscale luma is `Y`; indexed samples are palette indices.
    fn channel_names(self) -> &'static [&'static str] {
        match self {
            Self::Greyscale => &["Y"],
            Self::GreyAlpha => &["Y", "A"],
            Self::Rgb => &["R", "G", "B"],
            Self::Rgba => &["R", "G", "B", "A"],
            Self::Indexed => &["idx"],
        }
    }
}

/// Parsed IHDR plus every derived layout quantity the rest of the app
/// needs. This is the single geometry type: codec dispatch reads
/// `color_type` / `bit_depth`, buffer indexing reads `bpp` /
/// `row_stride`, and the pixel↔byte coordinate methods (defined in
/// [`crate::coords`]) read `bits_per_pixel`.
///
/// The derived fields are computed in exactly one place
/// ([`PngInfo::new`]) so the row-layout formula cannot drift between
/// call sites. Construct via `new`; the fields are `pub` for read
/// access.
#[derive(Debug, Clone, Copy)]
pub struct PngInfo {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub color_type: ColorType,
    /// Bytes per pixel rounded up to whole bytes (PNG spec §9.4: the
    /// filter-offset bytes-per-pixel, `1` for sub-byte depths).
    pub bpp: usize,
    /// True bits per pixel: `bit_depth * channels`. 1/2/4 for sub-byte
    /// greyscale and indexed; 8/16/24/32/48/64 for the byte-aligned
    /// types. Drives the pixel↔byte coordinate arithmetic.
    pub bits_per_pixel: u32,
    /// `1 + ceil(width * bits_per_pixel / 8)`: leading filter byte
    /// plus the row's packed data bytes. For interlaced images this is the
    /// stride of the *full* image; each Adam7 pass has its own smaller
    /// stride (see [`crate::png::interlace`]).
    pub row_stride: usize,
    /// Adam7 interlacing (IHDR interlace-method 1). When set, the decoded
    /// `output` is seven concatenated sub-images rather than one raster,
    /// and `row_stride` describes only the reassembled full image.
    pub interlaced: bool,
}

impl PngInfo {
    /// Human label for the `byte_offset`-th byte within a pixel, for the
    /// selection info panel. `byte_offset` is a byte index in `0..bpp`
    /// (the unit the event loop walks), *not* a channel index; the two
    /// coincide only at 8-bit depth.
    ///
    /// - 8-bit: one byte per channel → the channel name (`R`, `G`, ...).
    /// - 16-bit: two big-endian bytes per channel → the channel name
    ///   plus ` hi` / ` lo`, so byte 1 of 16-bit RGB reads `R lo`, not
    ///   the wrong `G`.
    /// - Sub-byte: the whole pixel cluster lives in one byte; there is
    ///   no per-channel offset, so it's named by colour type.
    pub fn channel_label(&self, byte_offset: usize) -> String {
        let names = self.color_type.channel_names();
        if self.bit_depth < 8 {
            return names[0].to_string();
        }
        let bytes_per_sample = (self.bit_depth / 8) as usize; // 1 or 2
        let sample = byte_offset / bytes_per_sample;
        let name = names.get(sample).copied().unwrap_or("?");
        if bytes_per_sample == 2 {
            let half = if byte_offset.is_multiple_of(2) {
                " hi"
            } else {
                " lo"
            };
            format!("{name}{half}")
        } else {
            name.to_string()
        }
    }

    /// Compute every derived layout field from the four IHDR quantities.
    /// The only place the row-layout formulas live.
    pub fn new(width: u32, height: u32, bit_depth: u8, color_type: ColorType) -> Self {
        let bits_per_pixel = bit_depth as u32 * color_type.channels();
        // PNG spec §9.4: filter-byte offset uses bytes-per-pixel rounded
        // up to whole bytes (so 1 for sub-byte depths).
        let bpp = (bits_per_pixel as usize).div_ceil(8).max(1);
        // Row data bytes round up too: 9 pixels at 1 bit/pixel is 2 bytes.
        let row_data_bytes = (width as usize * bits_per_pixel as usize).div_ceil(8);
        Self {
            width,
            height,
            bit_depth,
            color_type,
            bpp,
            bits_per_pixel,
            row_stride: 1 + row_data_bytes,
            interlaced: false,
        }
    }
}

const PNG_SIG: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];

/// Fatal parse errors from [`read_chunks`]. Bad CRCs aren't here; they
/// surface as a [`Warning`] inside [`ParsedChunks`] so a glitched file with
/// stale checksums still loads.
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
/// warnings.
#[derive(Debug, Default)]
pub struct ParsedChunks {
    pub chunks: Vec<Chunk>,
    pub warnings: Vec<Warning>,
}

/// A non-fatal integrity issue found while decoding: the image still decoded,
/// but a checksum didn't match. [`crate::decode`] collects these; every
/// variant is a checksum/CRC mismatch, which is what lets
/// [`crate::decode_strict`] reject a non-empty warning list as a bad checksum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Warning {
    /// A chunk's CRC-32 didn't match its data.
    ChunkCrc { typ: ChunkType },
    /// The zlib header's FCHECK bits don't validate.
    ZlibHeaderChecksum,
    /// The zlib Adler-32 trailer didn't match the decoded image data.
    StaleImageAdler,
}

impl std::fmt::Display for Warning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChunkCrc { typ } => write!(f, "stale checksum on {typ} chunk"),
            Self::ZlibHeaderChecksum => write!(f, "stale checksum on PNG image-data header"),
            Self::StaleImageAdler => write!(f, "stale zlib checksum on PNG image data"),
        }
    }
}

/// Returns `true` if `data` begins with the 8-byte PNG signature.
fn verify_png_signature(data: &[u8]) -> bool {
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
        let typ = ChunkType([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
        let end = pos + 8 + length;
        if end + 4 > data.len() {
            return Err(ChunksError::Truncated);
        }
        let chunk_data = &data[pos + 8..end];
        let stored = u32::from_be_bytes([data[end], data[end + 1], data[end + 2], data[end + 3]]);
        let computed = crc32(&typ, chunk_data);
        if stored != computed {
            out.warnings.push(Warning::ChunkCrc { typ });
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
/// Two passes: sum lengths, then `extend_from_slice` into a `Vec` of
/// the exact capacity. One allocation total.
pub fn concat_idat(chunks: &[Chunk]) -> Vec<u8> {
    let total: usize = chunks
        .iter()
        .filter(|c| c.typ == ChunkType::IDAT)
        .map(|c| c.data.len())
        .sum();
    let mut out = Vec::with_capacity(total);
    for c in chunks.iter().filter(|c| c.typ == ChunkType::IDAT) {
        out.extend_from_slice(&c.data);
    }
    out
}

pub fn write_chunks(chunks: &[Chunk]) -> Vec<u8> {
    let mut out = PNG_SIG.to_vec();
    for chunk in chunks {
        let len = chunk.data.len() as u32;
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(chunk.typ.as_bytes());
        out.extend_from_slice(&chunk.data);
        let crc = crc32(&chunk.typ, &chunk.data);
        out.extend_from_slice(&crc.to_be_bytes());
    }
    out
}

pub fn parse_ihdr(chunks: &[Chunk]) -> Option<PngInfo> {
    let ihdr = chunks.iter().find(|c| c.typ == ChunkType::IHDR)?;
    // IHDR is exactly 13 bytes per the PNG spec: 4 width, 4 height,
    // bit-depth, color-type, compression-method, filter-method,
    // interlace-method. Anything shorter is a malformed chunk.
    if ihdr.data.len() < 13 {
        return None;
    }
    let width = u32::from_be_bytes([ihdr.data[0], ihdr.data[1], ihdr.data[2], ihdr.data[3]]);
    let height = u32::from_be_bytes([ihdr.data[4], ihdr.data[5], ihdr.data[6], ihdr.data[7]]);
    // PNG spec §11.2.2: "Zero is an invalid value." Reject before any
    // downstream arithmetic that would divide or multiply by them.
    if width == 0 || height == 0 {
        return None;
    }
    let bit_depth = ihdr.data[8];
    let color_type = ColorType::from_byte(ihdr.data[9])?;
    // Reject bit-depth / colour-type pairings the spec forbids (PNG spec
    // Table 11.1) at the header, so no illegal `PngInfo` reaches the
    // converter.
    if !color_type.allows_bit_depth(bit_depth) {
        return None;
    }
    // PNG spec: compression-method and filter-method must each be 0 (the
    // only values defined). Interlace-method is 0 (none) or 1 (Adam7).
    let compression_method = ihdr.data[10];
    let filter_method = ihdr.data[11];
    let interlace_method = ihdr.data[12];
    if compression_method != 0 || filter_method != 0 || interlace_method > 1 {
        return None;
    }
    let mut info = PngInfo::new(width, height, bit_depth, color_type);
    info.interlaced = interlace_method == 1;
    Some(info)
}

// CRC-32 (ISO 3309 / PNG)

pub(super) fn crc32(typ: &ChunkType, data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for n in 0u32..256 {
            let mut c = n;
            for _ in 0..8 {
                if c & 1 != 0 {
                    // 0xEDB88320: the reversed CRC-32 polynomial (PNG spec 15.3).
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
    for &b in typ.as_bytes().iter().chain(data.iter()) {
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
        assert_eq!(crc32(&ChunkType::IEND, &[]), 0xAE426082);
    }

    #[test]
    fn channel_label_names_bytes_by_colour_mode() {
        // 8-bit RGB: one byte per channel.
        let rgb8 = PngInfo::new(1, 1, 8, ColorType::Rgb);
        assert_eq!(rgb8.channel_label(0), "R");
        assert_eq!(rgb8.channel_label(1), "G");
        assert_eq!(rgb8.channel_label(2), "B");

        // 16-bit RGB: two big-endian bytes per channel. Byte 1 must read
        // "R lo", not "G".
        let rgb16 = PngInfo::new(1, 1, 16, ColorType::Rgb);
        assert_eq!(rgb16.channel_label(0), "R hi");
        assert_eq!(rgb16.channel_label(1), "R lo");
        assert_eq!(rgb16.channel_label(2), "G hi");
        assert_eq!(rgb16.channel_label(5), "B lo");

        // Greyscale luma is Y, not R.
        assert_eq!(
            PngInfo::new(1, 1, 8, ColorType::Greyscale).channel_label(0),
            "Y"
        );
        assert_eq!(
            PngInfo::new(1, 1, 16, ColorType::GreyAlpha).channel_label(3),
            "A lo"
        );

        // Sub-byte: one packed byte, named by colour type.
        assert_eq!(
            PngInfo::new(8, 1, 1, ColorType::Greyscale).channel_label(0),
            "Y"
        );
        assert_eq!(
            PngInfo::new(8, 1, 4, ColorType::Indexed).channel_label(0),
            "idx"
        );
    }

    #[test]
    fn read_write_chunks_round_trip() {
        let chunks = vec![
            Chunk {
                typ: ChunkType::IHDR,
                data: vec![1, 2, 3, 4, 5, 6, 7, 8, 8, 6, 0, 0, 0],
            },
            Chunk {
                typ: ChunkType::IDAT,
                data: vec![0xAA; 100],
            },
            Chunk {
                typ: ChunkType::IEND,
                data: vec![],
            },
        ];
        let bytes = write_chunks(&chunks);
        let parsed = read_chunks(&bytes).expect("parse round-tripped chunks");
        assert_eq!(parsed.chunks.len(), chunks.len());
        for (orig, got) in chunks.iter().zip(parsed.chunks.iter()) {
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
            typ: ChunkType::IHDR,
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
            typ: ChunkType::IHDR,
            data,
        }];
        assert!(parse_ihdr(&chunks).is_none());
    }

    /// IHDR shorter than the full 13 bytes must be rejected: an 11- or
    /// 12-byte IHDR would otherwise read past its data when validating
    /// the trailing compression / filter / interlace bytes.
    #[test]
    fn parse_ihdr_rejects_short_ihdr_data() {
        for short_len in [0usize, 9, 10, 11, 12] {
            let chunks = vec![Chunk {
                typ: ChunkType::IHDR,
                data: vec![0u8; short_len],
            }];
            assert!(
                parse_ihdr(&chunks).is_none(),
                "IHDR with {short_len} bytes must be rejected"
            );
        }
    }

    #[test]
    fn parse_ihdr_rejects_zero_width_or_height() {
        // PNG spec §11.2.2 requires non-zero width and height.
        for (w, h) in [(0u32, 1u32), (1, 0), (0, 0)] {
            let mut data = vec![0u8; 13];
            data[0..4].copy_from_slice(&w.to_be_bytes());
            data[4..8].copy_from_slice(&h.to_be_bytes());
            data[8] = 8;
            data[9] = 6;
            let chunks = vec![Chunk {
                typ: ChunkType::IHDR,
                data,
            }];
            assert!(
                parse_ihdr(&chunks).is_none(),
                "{w}×{h} IHDR must be rejected"
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
            typ: ChunkType::IHDR,
            data,
        }];
        assert!(parse_ihdr(&chunks).is_none());
    }

    #[test]
    fn parse_ihdr_accepts_adam7_and_rejects_reserved_interlace() {
        let ihdr = |interlace: u8| {
            let mut data = vec![0u8; 13];
            data[0..4].copy_from_slice(&1u32.to_be_bytes());
            data[4..8].copy_from_slice(&1u32.to_be_bytes());
            data[8] = 8;
            data[9] = 6;
            data[12] = interlace;
            vec![Chunk {
                typ: ChunkType::IHDR,
                data,
            }]
        };
        assert!(!parse_ihdr(&ihdr(0)).unwrap().interlaced);
        assert!(parse_ihdr(&ihdr(1)).unwrap().interlaced);
        assert!(parse_ihdr(&ihdr(2)).is_none()); // reserved
    }

    #[test]
    fn parse_ihdr_validates_bit_depth_against_color_type() {
        let ihdr = |depth: u8, color: u8| {
            let mut data = vec![0u8; 13];
            data[0..4].copy_from_slice(&1u32.to_be_bytes());
            data[4..8].copy_from_slice(&1u32.to_be_bytes());
            data[8] = depth;
            data[9] = color;
            vec![Chunk {
                typ: ChunkType::IHDR,
                data,
            }]
        };
        // Illegal pairings (PNG spec Table 11.1) and undefined depths.
        for (depth, color) in [
            (1, 2),  // RGB @ 1-bit
            (4, 2),  // RGB @ 4-bit
            (16, 3), // Indexed @ 16-bit
            (2, 6),  // RGBA @ 2-bit
            (4, 4),  // GreyAlpha @ 4-bit
            (3, 0),  // undefined bit depth
            (0, 0),  // undefined bit depth
        ] {
            assert!(
                parse_ihdr(&ihdr(depth, color)).is_none(),
                "depth {depth} colour {color} must be rejected"
            );
        }
        // Legal pairings still parse.
        for (depth, color) in [(1, 0), (16, 0), (8, 3), (16, 2), (8, 6)] {
            assert!(
                parse_ihdr(&ihdr(depth, color)).is_some(),
                "depth {depth} colour {color} must be accepted"
            );
        }
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
            typ: ChunkType::IEND,
            data: vec![],
        }];
        let mut bytes = write_chunks(&chunks);
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        let parsed = read_chunks(&bytes).expect("CRC mismatch must not fail the parse");
        assert_eq!(parsed.chunks.len(), 1);
        assert_eq!(parsed.warnings.len(), 1);
        assert_eq!(
            parsed.warnings[0],
            Warning::ChunkCrc {
                typ: ChunkType::IEND
            }
        );
    }
}
