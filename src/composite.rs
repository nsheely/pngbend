//! Alpha compositor: the final step of the texture-rebuild pipeline,
//! consuming a base RGBA buffer and an overlay RGBA buffer.
//!
//! Integer math end-to-end. The hot loop blends four RGBA pixels per
//! iteration via `wide::u32x4` (16-byte SIMD on every target). Wider lanes
//! (`u32x8`) only help on AVX2 builds and regress the portable build by
//! ~10 %; `u32x4` keeps the binary portable and lets LLVM widen on AVX2
//! hardware automatically.
//!
//! All blending assumes the *base* is fully opaque (`ba == 255`). That holds
//! for every output of [`crate::png::to_rgba8`] and its interlaced counterpart
//! [`crate::png::deinterlace_to_rgba8`] (RGB, greyscale, and indexed without
//! tRNS), the buffers the GUI passes here. Under it the
//! source-over formula collapses to
//! `out_rgb = (fo*oa + fb*(255 - oa)) / 255`, with no per-pixel divide.
//! RGBA / greyscale-alpha / palette+tRNS sources don't satisfy it; output
//! is still well-defined (`out_a = 255`) and fine for overlay
//! visualisation.

use wide::u32x4;

/// Alpha-composite `overlay` over `base` (both RGBA, `w * h * 4` bytes)
/// and write the result into `out`. `out` is resized to match `base.len()`,
/// so the GUI passes the same `Vec` every frame to avoid the per-rebuild
/// allocation.
///
/// See the module docs for the `ba == 255` invariant and the alpha
/// trade-off on RGBA / greyscale-alpha sources.
pub fn composite_into(base: &[u8], overlay: &[u8], out: &mut Vec<u8>) {
    assert_eq!(base.len(), overlay.len());
    debug_assert_eq!(base.len() % 4, 0);
    out.clear();
    out.resize(base.len(), 0);
    composite_ba255_simd(base, overlay, out);
}

/// Allocating wrapper around [`composite_into`], for tests and benches
/// without the reusable scratch the GUI's frame loop uses.
pub fn composite_rgba(base: &[u8], overlay: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(base.len());
    composite_into(base, overlay, &mut out);
    out
}

/// Row-scoped composite. Re-blends only the listed `rows` in `out` and
/// assumes every other row already holds a valid composite from a prior
/// full [`composite_into`] call. The incremental edit path uses this to
/// skip blending the rows it didn't touch (typically >95 % of them).
pub fn composite_rows_into(
    base: &[u8],
    overlay: &[u8],
    out: &mut [u8],
    w: u32,
    rows: impl IntoIterator<Item = usize>,
) {
    assert_eq!(out.len(), base.len());
    assert_eq!(overlay.len(), base.len());
    let row_bytes = w as usize * 4;
    for row in rows {
        let start = row * row_bytes;
        let end = start + row_bytes;
        if end > out.len() {
            break;
        }
        composite_ba255_simd(
            &base[start..end],
            &overlay[start..end],
            &mut out[start..end],
        );
    }
}

/// Vectorised composite assuming `base` alpha is 255 everywhere.
///
/// 4 RGBA pixels (16 bytes) per `u32x4` iteration; scalar tail handles
/// the 0..3 pixel remainder. Two muls + one add + one shift per lane
/// per channel; no `u64` divide.
fn composite_ba255_simd(base: &[u8], overlay: &[u8], out: &mut [u8]) {
    let n_pixels = base.len() / 4;
    let n_chunks = n_pixels / 4;

    let mask_ff = u32x4::splat(0xFF);
    let c_255 = u32x4::splat(255);
    let c_1 = u32x4::splat(1);
    // Pre-shifted fully-opaque alpha for the packed store.
    let alpha_fill = u32x4::splat(255u32 << 24);

    for chunk in 0..n_chunks {
        let start = chunk * 16;
        let b = load_u32x4_le(&base[start..start + 16]);
        let o = load_u32x4_le(&overlay[start..start + 16]);

        // Extract channels. Each `u32x4` lane now holds one channel of
        // one pixel in its low 8 bits.
        let o_r = o & mask_ff;
        let o_g = (o >> 8) & mask_ff;
        let o_b = (o >> 16) & mask_ff;
        let o_a = (o >> 24) & mask_ff;
        let b_r = b & mask_ff;
        let b_g = (b >> 8) & mask_ff;
        let b_b = (b >> 16) & mask_ff;
        let inv_a = c_255 - o_a;

        // Source-over assuming ba=255:
        //   out_rgb = (fo*oa + fb*(255 - oa)) / 255
        // Exact integer form of the /255 divide:
        //   (x + 1 + (x >> 8)) >> 8
        // which is bit-exact for x ∈ [0, 65025]. Max sum is 65025
        // (since oa and inv_oa sum to 255). Three u32 ops per channel;
        // no divide.
        let sr = o_r * o_a + b_r * inv_a;
        let sg = o_g * o_a + b_g * inv_a;
        let sb = o_b * o_a + b_b * inv_a;
        let r = (sr + c_1 + (sr >> 8)) >> 8;
        let g = (sg + c_1 + (sg >> 8)) >> 8;
        let bl = (sb + c_1 + (sb >> 8)) >> 8;

        // Pack back: lane = (A=255)<<24 | B<<16 | G<<8 | R.
        let packed = r | (g << 8) | (bl << 16) | alpha_fill;
        store_u32x4_le(&mut out[start..start + 16], packed);
    }

    // Scalar tail for pixels not covered by the 4-at-a-time SIMD loop.
    let tail_start = n_chunks * 16;
    composite_ba255_scalar(
        &base[tail_start..],
        &overlay[tail_start..],
        &mut out[tail_start..],
    );
}

/// Scalar fallback. Shares the ba=255 fast-path math so tail pixels
/// match the SIMD body bit-exactly.
fn composite_ba255_scalar(base: &[u8], overlay: &[u8], out: &mut [u8]) {
    for oi in (0..base.len()).step_by(4) {
        let oa = overlay[oi + 3] as u32;
        let inv = 255 - oa;
        for c in 0..3 {
            let fo = overlay[oi + c] as u32;
            let fb = base[oi + c] as u32;
            let sum = fo * oa + fb * inv;
            // Exact integer /255 for sum ∈ [0, 65025].
            out[oi + c] = ((sum + 1 + (sum >> 8)) >> 8) as u8;
        }
        out[oi + 3] = 255;
    }
}

#[inline(always)]
fn load_u32x4_le(bytes: &[u8]) -> u32x4 {
    // `bytes.len() >= 16` is a precondition. Reads are explicit
    // little-endian so the source RGBA byte order matches the lane
    // layout the math expects regardless of host endianness.
    u32x4::new([
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
    ])
}

#[inline(always)]
fn store_u32x4_le(out: &mut [u8], v: u32x4) {
    let a = v.to_array();
    out[0..4].copy_from_slice(&a[0].to_le_bytes());
    out[4..8].copy_from_slice(&a[1].to_le_bytes());
    out[8..12].copy_from_slice(&a[2].to_le_bytes());
    out[12..16].copy_from_slice(&a[3].to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba(r: u8, g: u8, b: u8, a: u8) -> [u8; 4] {
        [r, g, b, a]
    }

    /// Scalar reference implementation. The SIMD body must agree with
    /// this byte-for-byte under the `ba == 255` invariant; that's what
    /// the round-trip tests below check.
    fn reference_ba255(base: &[u8], overlay: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; base.len()];
        composite_ba255_scalar(base, overlay, &mut out);
        out
    }

    #[test]
    fn simd_matches_scalar_four_pixels() {
        let base = vec![
            100, 150, 200, 255, // pixel 0
            50, 75, 125, 255, // pixel 1
            10, 20, 30, 255, // pixel 2
            200, 220, 240, 255, // pixel 3
        ];
        let overlay = vec![
            255, 0, 0, 128, // pixel 0: red at 50% alpha
            0, 255, 0, 64, // pixel 1: green at ~25%
            0, 0, 255, 200, // pixel 2: blue at ~78%
            128, 128, 128, 0, // pixel 3: transparent overlay
        ];
        let mut simd_out = Vec::new();
        composite_into(&base, &overlay, &mut simd_out);
        assert_eq!(simd_out, reference_ba255(&base, &overlay));
    }

    #[test]
    fn simd_handles_oa_zero_and_255() {
        // Ensure edge lanes (fully opaque, fully transparent) produce the
        // right extremes: opaque → overlay, transparent → base.
        let base = [10, 20, 30, 255, 40, 50, 60, 255];
        let overlay = [200, 210, 220, 255, 100, 110, 120, 0];
        // Scalar oracle, processed end-to-end via the SIMD path's tail
        // scalar (since n_chunks = 0 for 2 pixels).
        let mut out = vec![0u8; 8];
        composite_ba255_simd(&base, &overlay, &mut out);
        // Pixel 0: overlay opaque → overlay RGB, alpha=255.
        assert_eq!(&out[0..4], &rgba(200, 210, 220, 255));
        // Pixel 1: overlay transparent → base RGB, alpha=255.
        assert_eq!(&out[4..8], &rgba(40, 50, 60, 255));
    }

    #[test]
    fn simd_respects_tail_not_multiple_of_four() {
        // 5 pixels: first 4 go through SIMD body, last one through scalar tail.
        let base = vec![1u8; 5 * 4];
        let overlay = vec![2u8; 5 * 4];
        // Force overlay alpha to 128 on every pixel.
        let mut o = overlay.clone();
        for i in 0..5 {
            o[i * 4 + 3] = 128;
        }
        let mut out = Vec::new();
        composite_into(&base, &o, &mut out);
        let ref_out = reference_ba255(&base, &o);
        assert_eq!(out, ref_out);
    }

    #[test]
    fn rows_composite_only_touches_listed_rows() {
        // 2 rows × 4 pixels, composite only row 1.
        let w = 4u32;
        let base = vec![10u8; 8 * 4];
        let overlay = vec![200u8; 8 * 4];
        let mut out = base.clone(); // pretend prior-frame composite
        composite_rows_into(&base, &overlay, &mut out, w, std::iter::once(1));
        // Row 0 unchanged.
        assert_eq!(&out[..16], &base[..16]);
        // Row 1 composited.
        let mut ref_row = vec![0u8; 16];
        composite_ba255_scalar(&base[16..32], &overlay[16..32], &mut ref_row);
        assert_eq!(&out[16..32], &ref_row[..]);
    }
}
