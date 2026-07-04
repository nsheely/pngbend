//! Shared harness helpers for the glasspng fuzz targets.

/// Whether `data`'s IHDR declares more than 16 MiB of unfiltered output.
///
/// `glasspng::decode` already caps inflation at the IHDR-implied size (so a
/// decompression bomb is rejected, not allocated), but that cap can still be
/// up to 4 GiB. Skipping the largest declared images keeps the fuzzer spending
/// its time on structure rather than big-but-valid allocations, and avoids
/// libFuzzer's RSS limit reporting them as false OOM "crashes".
pub fn oversized(data: &[u8]) -> bool {
    let Ok(parsed) = glasspng::png::read_chunks(data) else {
        return false;
    };
    let Some(info) = glasspng::png::parse_ihdr(&parsed.chunks) else {
        return false;
    };
    let out = u64::from(info.height) * (1 + u64::from(info.width) * info.bpp as u64);
    info.width > u16::MAX as u32 || info.height > u16::MAX as u32 || out > 16 * 1024 * 1024
}
