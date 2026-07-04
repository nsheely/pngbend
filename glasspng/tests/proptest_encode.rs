//! Property-based round-trip tests for the encoder.
//!
//! `decode(encode(img)) == img` is the encoder's core contract. The unit
//! tests cover fixed examples and the decoder is fuzzed on its own; these
//! properties fuzz the *encode* path — pack, filter selection, LZ77,
//! Huffman-mode choice, zlib framing, chunk writing — over random
//! dimensions, pixels, output formats, and filter strategies.
//!
//! Every output format is lossless once the RGBA8 input is made consistent
//! with it (grey formats need R==G==B; alpha-less formats need A==255), so
//! the assertion is exact equality, not tolerance.

use glasspng::png::{ColorType, FilterStrategy, FilterType, PngInfo};
use glasspng::{EncodeOptions, Image, OutputFormat, decode, encode};
use proptest::prelude::*;

const FORMATS: &[OutputFormat] = &[
    OutputFormat::Rgba8,
    OutputFormat::Rgb8,
    OutputFormat::Grey8,
    OutputFormat::GreyAlpha8,
    OutputFormat::Rgba16,
    OutputFormat::Rgb16,
    OutputFormat::Grey16,
    OutputFormat::GreyAlpha16,
];

const FILTERS: &[FilterStrategy] = &[
    FilterStrategy::MinSad,
    FilterStrategy::Fixed(FilterType::None),
    FilterStrategy::Fixed(FilterType::Sub),
    FilterStrategy::Fixed(FilterType::Up),
    FilterStrategy::Fixed(FilterType::Average),
    FilterStrategy::Fixed(FilterType::Paeth),
];

/// Random image dimensions plus a matching run of RGBA8 pixel bytes.
fn image() -> impl Strategy<Value = (u32, u32, Vec<u8>)> {
    (1u32..=40, 1u32..=40).prop_flat_map(|(w, h)| {
        let n = (w * h * 4) as usize;
        (Just(w), Just(h), prop::collection::vec(any::<u8>(), n..=n))
    })
}

/// Force RGBA8 pixels to be lossless under `fmt`: grey formats need the three
/// colour channels equal, alpha-less formats need opaque alpha.
fn normalize(pixels: &mut [u8], fmt: OutputFormat) {
    use OutputFormat::*;
    let grey = matches!(fmt, Grey8 | Grey16 | GreyAlpha8 | GreyAlpha16);
    let opaque = matches!(fmt, Rgb8 | Rgb16 | Grey8 | Grey16);
    for px in pixels.chunks_exact_mut(4) {
        if grey {
            px[1] = px[0];
            px[2] = px[0];
        }
        if opaque {
            px[3] = 255;
        }
    }
}

fn round_trip(w: u32, h: u32, pixels: Vec<u8>, format: OutputFormat, filter: FilterStrategy) {
    let img = Image {
        info: PngInfo::new(w, h, 8, ColorType::Rgba),
        palette: None,
        pixels: pixels.clone(),
        warnings: Vec::new(),
    };
    let options = EncodeOptions { format, filter };
    let bytes = encode(&img, &options).expect("encode");
    let back = decode(&bytes).expect("decode round-trip");
    assert_eq!(back.info.width, w);
    assert_eq!(back.info.height, h);
    assert_eq!(back.pixels, pixels, "{format:?} filter {filter:?}");
}

proptest! {
    /// Every supported output format round-trips exactly, using the default
    /// MinSad filter selection.
    #[test]
    fn encode_round_trips_every_format(
        (w, h, mut pixels) in image(),
        format in prop::sample::select(FORMATS),
    ) {
        normalize(&mut pixels, format);
        round_trip(w, h, pixels, format, FilterStrategy::MinSad);
    }

    /// Every filter strategy round-trips exactly on RGBA8 (the lossless-for-
    /// any-input format), exercising all five predictors and the heuristic.
    #[test]
    fn encode_round_trips_every_filter(
        (w, h, pixels) in image(),
        filter in prop::sample::select(FILTERS),
    ) {
        round_trip(w, h, pixels, OutputFormat::Rgba8, filter);
    }
}
