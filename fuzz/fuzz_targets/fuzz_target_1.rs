//! Walks arbitrary bytes through the full load pipeline — chunks,
//! IHDR, zlib wrapper, DEFLATE, unfilter, and RGBA conversion — so a
//! panic anywhere in the chain is found. Mirrors what
//! `app::io::load::load_file` does, minus the GUI bits.

#![no_main]

use libfuzzer_sys::fuzz_target;
use pngbend::deflate::decode_deflate;
use pngbend::png::{
    concat_idat, decode_palette, parse_ihdr, parse_zlib_stream, read_chunks, to_rgba8, unfilter,
};

fuzz_target!(|data: &[u8]| {
    let Ok(parsed) = read_chunks(data) else {
        return;
    };
    let Some(info) = parse_ihdr(&parsed) else {
        return;
    };
    // Mirror the loader's MAX_DIMENSION + u32::MAX output check so the
    // fuzzer doesn't burn time on legitimate-but-huge allocations.
    if info.width > u16::MAX as u32 || info.height > u16::MAX as u32 {
        return;
    }
    let output_bytes =
        u64::from(info.height) * (1 + u64::from(info.width) * info.bpp as u64);
    if output_bytes > 16 * 1024 * 1024 {
        return;
    }
    let idat = concat_idat(&parsed);
    if idat.is_empty() {
        return;
    }
    let Ok(zlib) = parse_zlib_stream(&idat) else {
        return;
    };
    // Cap deflate output at 4 MiB — adversarial input shouldn't pump the
    // decoder into runaway allocations.
    let Ok(decoded) = decode_deflate(zlib.deflate_buf, Some(4 * 1024 * 1024)) else {
        return;
    };
    let Ok(unfiltered) = unfilter(&decoded.output, &info) else {
        return;
    };
    let palette = parsed
        .iter()
        .find(|c| c.typ == *b"PLTE")
        .map(|p| decode_palette(&p.data, None));
    let _ = to_rgba8(&unfiltered, &info, palette.as_deref());
});
