//! Leaf helpers for the apply paths: the row-scoped re-render pipeline and
//! the bit-level patch/inverse-capture used for undo.

use crate::bitstream::{read_bits_at, write_bits};

use super::super::io::CoreData;
use super::Patch;

/// Row-scoped render pipeline shared by the literal-swap and redirect
/// paths. Inverse-filters every row flagged in `row_touched` (plus any
/// downstream rows that chain through Up/Avg/Paeth filter types), then
/// converts those rows in `base_rgba`. Returns the rebuilt set so the
/// caller can hand it to `partial_composite_rows` and keep the texture
/// rebuild row-scoped as well.
pub(super) fn render_affected_rows(
    core: &mut CoreData,
    base_rgba: &mut [u8],
    first_affected: usize,
    row_touched: &[bool],
) -> Result<Vec<usize>, String> {
    // Interlaced output has no single progressive raster to patch row by
    // row, so reassemble the whole image from the (edited) passes into
    // `base_rgba`. We return no rebuilt rows: overlays are disabled for
    // interlaced images, so the texture re-uploads `base_rgba` wholesale
    // instead of compositing the returned row list.
    if core.info.interlaced {
        let full =
            crate::png::deinterlace_to_rgba8(&core.output, &core.info, core.palette.as_deref())
                .map_err(|e| format!("deinterlace: {e}"))?;
        base_rgba.copy_from_slice(&full);
        return Ok(Vec::new());
    }
    let mut rebuilt = Vec::with_capacity(row_touched.iter().filter(|b| **b).count() + 4);
    crate::png::unfilter_rows_into(
        &core.output,
        &core.info,
        &mut core.unfiltered,
        first_affected,
        |y| row_touched.get(y).copied().unwrap_or(false),
        |y| rebuilt.push(y),
    )
    .map_err(|e| format!("unfilter: {e}"))?;
    crate::png::to_rgba8_rows_into(
        &core.unfiltered,
        &core.info,
        core.palette.as_deref(),
        base_rgba,
        rebuilt.iter().copied(),
    )
    .map_err(|e| format!("rgba: {e}"))?;
    Ok(rebuilt)
}

/// Write each patch into `buf`, capturing the bits that were overwritten
/// so the caller can stash the inverse patch list onto the undo stack.
pub(super) fn apply_patches_capturing_prior(buf: &mut [u8], patches: &[Patch]) -> Vec<Patch> {
    let mut inverse = Vec::with_capacity(patches.len());
    for &Patch {
        bit_start,
        value,
        code_len,
    } in patches
    {
        let bs = bit_start as usize;
        let prev = read_bits_at(buf, bs, code_len);
        inverse.push(Patch {
            bit_start,
            value: prev,
            code_len,
        });
        write_bits(buf, bs, value, code_len);
    }
    inverse
}

#[cfg(test)]
mod tests {
    use super::super::Patch;
    use super::apply_patches_capturing_prior;
    use proptest::prelude::*;

    proptest! {
        /// The undo invariant: applying an edit's patches forward, then
        /// applying the captured inverse, restores `deflate_buf`
        /// byte-for-byte. Real `EditAction.patches` are non-overlapping
        /// (one patch per channel for literal swaps, one per redirect),
        /// so the generator lays patches out sequentially with a 1-bit
        /// gap to mirror that.
        #[test]
        fn forward_then_inverse_restores_buffer(
            buf in proptest::collection::vec(any::<u8>(), 4..64usize),
            specs in proptest::collection::vec((1u8..=16, any::<u32>()), 1..7),
        ) {
            let max_bit = buf.len() * 8;
            let mut bit_cursor: u32 = 0;
            let mut patches: Vec<Patch> = Vec::new();
            for (cl, v) in specs {
                let cl_u32 = cl as u32;
                if (bit_cursor + cl_u32) as usize > max_bit {
                    break;
                }
                let mask = if cl == 32 { u32::MAX } else { (1u32 << cl) - 1 };
                patches.push(Patch { bit_start: bit_cursor, value: v & mask, code_len: cl });
                bit_cursor += cl_u32 + 1;
            }
            prop_assume!(!patches.is_empty());

            let original = buf.clone();
            let mut work = buf;
            let inverse = apply_patches_capturing_prior(&mut work, &patches);
            let _ = apply_patches_capturing_prior(&mut work, &inverse);
            prop_assert_eq!(work, original);
        }
    }
}
