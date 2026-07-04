//! Walks arbitrary bytes through the full load pipeline (chunks, IHDR,
//! zlib wrapper, DEFLATE, unfilter, RGBA conversion) to catch a panic
//! anywhere in the chain. Mirrors `app::io::load::load_file` minus the
//! GUI.

#![no_main]

use libfuzzer_sys::fuzz_target;
use pngbend::deflate::decode_deflate;
use pngbend::png::{
    ChunkType, concat_idat, decode_palette, parse_ihdr, parse_zlib_stream, read_chunks, to_rgba8,
    unfilter,
};

fuzz_target!(|data: &[u8]| {
    let Ok(parsed) = read_chunks(data) else {
        return;
    };
    let Some(info) = parse_ihdr(&parsed.chunks) else {
        return;
    };
    // Mirror the loader's MAX_DIMENSION + u32::MAX output check so the
    // fuzzer doesn't burn time on huge allocations.
    if info.width > u16::MAX as u32 || info.height > u16::MAX as u32 {
        return;
    }
    let output_bytes = u64::from(info.height) * (1 + u64::from(info.width) * info.bpp as u64);
    if output_bytes > 16 * 1024 * 1024 {
        return;
    }
    let idat = concat_idat(&parsed.chunks);
    if idat.is_empty() {
        return;
    }
    let Ok(zlib) = parse_zlib_stream(&idat) else {
        return;
    };
    // Cap deflate output at 4 MiB so adversarial input can't pump the
    // decoder into runaway allocations.
    let Ok(decoded) = decode_deflate(zlib.deflate_buf, Some(4 * 1024 * 1024)) else {
        return;
    };
    let Ok(unfiltered) = unfilter(&decoded.output, &info) else {
        return;
    };
    let palette = parsed
        .chunks
        .iter()
        .find(|c| c.typ == ChunkType::PLTE)
        .map(|p| decode_palette(&p.data, None));
    let _ = to_rgba8(&unfiltered, &info, palette.as_deref());
});
