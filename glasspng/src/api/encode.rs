//! Encode an [`Image`] back to PNG bytes: the mirror of the decode pipeline.

use crate::deflate::compress;
use crate::png::{
    Chunk, ChunkType, FilterStrategy, OutputFormat, PngInfo, ZLIB_DEFAULT_HEADER,
    build_zlib_stream, filter, pack, write_chunks,
};

use super::{Image, PngError};

/// Output format and filtering for [`encode`].
#[derive(Debug, Clone, Copy, Default)]
pub struct EncodeOptions {
    pub format: OutputFormat,
    pub filter: FilterStrategy,
}

/// Encode an [`Image`]'s RGBA8 pixels to PNG bytes: `pack` -> `filter` ->
/// DEFLATE -> zlib -> chunks, the mirror of [`decode`](super::decode). The
/// output format comes from `options` (default RGBA8); [`OutputFormat`] names
/// only the formats the encoder can produce (the byte-aligned non-indexed
/// colour types), so an unsupported target isn't representable.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::png::{ColorType, FilterType};
    use crate::{Image, decode};

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
            FilterStrategy::Fixed(FilterType::Paeth),
            FilterStrategy::Fixed(FilterType::None),
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
}
