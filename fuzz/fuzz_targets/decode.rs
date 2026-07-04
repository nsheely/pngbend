#![no_main]

//! Fuzz glasspng's public decode surface. Arbitrary bytes must return `Ok` or
//! `Err`, never panic. Exercises chunk framing, IHDR, zlib/DEFLATE,
//! unfiltering, RGBA conversion, Adam7 interlacing, and tRNS colour-keying.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if glasspng_fuzz::oversized(data) {
        return;
    }
    // Lean path: bytes -> RGBA8.
    let _ = glasspng::decode(data);
    // Glass-box path: additionally records the DEFLATE event stream and
    // per-block Huffman tables.
    let _ = glasspng::decode_with_events(data);
});
