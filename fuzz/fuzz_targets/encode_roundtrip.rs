#![no_main]

//! Oracle fuzz of the encode path: if arbitrary bytes decode to an image,
//! re-encode it and require an exact round-trip. Drives pack, filter
//! selection, LZ77, Huffman-mode choice, zlib framing, and chunk writing over
//! *real decoded images*, complementing the randomized-pixel encoder proptests.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if glasspng_fuzz::oversized(data) {
        return;
    }
    let Ok(img) = glasspng::decode(data) else {
        return;
    };
    let bytes = glasspng::encode(&img, &glasspng::EncodeOptions::default())
        .expect("encoding a freshly decoded image must succeed");
    let back = glasspng::decode(&bytes).expect("our own encode must decode");
    assert_eq!(back.pixels, img.pixels, "encode round-trip changed pixels");
    assert_eq!(
        (back.info.width, back.info.height),
        (img.info.width, img.info.height),
    );
});
