//! PNG row-filter unfiltering.
//!
//! A decoded PNG deflate stream is `h` rows of `1 + w * bpp` bytes. Each
//! row begins with a filter type (0..=4); the remaining `w * bpp` bytes
//! are the filtered pixel data. [`unfilter`] applies the inverse filter
//! per row to produce `h * w * bpp` contiguous raw pixel bytes (no
//! filter prefix).
//!
//! [`unfilter_rows_into`] is the row-scoped variant the incremental
//! editor uses: re-run only the rows whose filtered bytes changed, plus
//! any downstream rows that read from them via filter types 2/3/4.

use super::chunks::PngInfo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterError {
    TruncatedRow {
        row: usize,
    },
    UnknownFilter {
        row: usize,
        filter: u8,
    },
    /// Allocating the unfiltered buffer would overflow `usize` or
    /// exceed `u32::MAX` bytes. Returned (rather than aborting on OOM)
    /// when callers pass IHDR dimensions without the loader's prior
    /// dimension check.
    OutputTooLarge {
        rows: usize,
        row_bytes: usize,
    },
}

impl std::fmt::Display for FilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TruncatedRow { row } => write!(f, "row {row} truncated in PNG stream"),
            Self::UnknownFilter { row, filter } => {
                write!(f, "unknown filter type {filter} at row {row}")
            }
            Self::OutputTooLarge { rows, row_bytes } => write!(
                f,
                "unfilter output would exceed limit ({rows} rows × {row_bytes} bytes)"
            ),
        }
    }
}

impl std::error::Error for FilterError {}

/// Apply the inverse of PNG row filters to the decoded PNG stream. Returns a
/// new `h * row_bytes` buffer with the per-row filter bytes stripped.
///
/// `info.row_stride` includes the filter byte; `row_bytes = row_stride - 1`
/// is the width of the raw pixel row in bytes.
pub fn unfilter(stream: &[u8], info: &PngInfo) -> Result<Vec<u8>, FilterError> {
    let h = info.height as usize;
    let row_bytes = info.row_stride - 1;
    // Reject pathological dimensions before allocating. Cap at u32::MAX
    // matches the loader's `output_bytes > u32::MAX` check (event
    // positions are u32). Library callers without that prior check
    // get a structured error instead of an OOM abort.
    let total = h
        .checked_mul(row_bytes)
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or(FilterError::OutputTooLarge { rows: h, row_bytes })?;
    let mut out = vec![0u8; total];
    unfilter_rows_into(stream, info, &mut out, 0, |_| true, |_| ())?;
    Ok(out)
}

/// Re-run the inverse filter for just the rows whose raw bytes changed,
/// propagating down through any filter-2/3/4 chain that reads from the
/// updated `prev` row. `unfiltered` must be a full `h * row_bytes` buffer
/// in a state consistent with the *un*modified portion of `stream`.
///
/// `changed_row` returns `true` when row `y`'s filtered bytes differ from
/// the last time `unfiltered` was computed. The caller tracks this; the
/// incremental edit path knows because it records every `output`
/// position it wrote to.
///
/// Returns the number of rows re-unfiltered. Used for telemetry and as
/// input to `to_rgba8_rows_into`.
pub fn unfilter_rows_into(
    stream: &[u8],
    info: &PngInfo,
    unfiltered: &mut [u8],
    first_affected: usize,
    mut changed_row: impl FnMut(usize) -> bool,
    mut on_rebuilt: impl FnMut(usize),
) -> Result<usize, FilterError> {
    let h = info.height as usize;
    let row_stride = info.row_stride;
    let row_bytes = row_stride - 1;
    let bpp = info.bpp;
    assert_eq!(unfiltered.len(), h * row_bytes, "unfiltered buffer misized");
    if first_affected >= h {
        return Ok(0);
    }

    let mut rebuilt = 0usize;
    let mut prev_was_updated = false;
    let mut zero_prev = vec![0u8; row_bytes];

    for y in first_affected..h {
        let stream_start = y * row_stride;
        let stream_end = stream_start + row_stride;
        if stream_end > stream.len() {
            return Err(FilterError::TruncatedRow { row: y });
        }
        let filter = stream[stream_start];
        let reads_prev = matches!(filter, 2..=4);
        let chain = reads_prev && prev_was_updated;
        let need = changed_row(y) || chain;

        if !need {
            prev_was_updated = false;
            continue;
        }

        let filt = &stream[stream_start + 1..stream_end];
        let row_off = y * row_bytes;
        let (prev, curr): (&[u8], &mut [u8]) = if y == 0 {
            (&zero_prev, &mut unfiltered[row_off..row_off + row_bytes])
        } else {
            let (head, tail) = unfiltered.split_at_mut(row_off);
            let prev = &head[row_off - row_bytes..];
            (prev, &mut tail[..row_bytes])
        };

        apply_unfilter(filter, filt, prev, curr, bpp)
            .map_err(|_| FilterError::UnknownFilter { row: y, filter })?;
        on_rebuilt(y);
        rebuilt += 1;
        prev_was_updated = true;
        // zero_prev only used for y == 0; no-op hint for the borrow checker.
        let _ = &mut zero_prev;
    }
    Ok(rebuilt)
}

#[inline]
fn apply_unfilter(
    filter: u8,
    filt: &[u8],
    prev: &[u8],
    curr: &mut [u8],
    bpp: usize,
) -> Result<(), ()> {
    match filter {
        0 => curr.copy_from_slice(filt),
        1 => sub_unfilter(filt, curr, bpp),
        2 => up_unfilter(filt, prev, curr),
        3 => avg_unfilter(filt, prev, curr, bpp),
        4 => paeth_unfilter(filt, prev, curr, bpp),
        _ => return Err(()),
    }
    Ok(())
}

#[inline]
fn sub_unfilter(filt: &[u8], curr: &mut [u8], bpp: usize) {
    for i in 0..filt.len() {
        let left = if i >= bpp { curr[i - bpp] } else { 0 };
        curr[i] = filt[i].wrapping_add(left);
    }
}

#[inline]
fn up_unfilter(filt: &[u8], prev: &[u8], curr: &mut [u8]) {
    // Per-byte wrapping add. LLVM auto-vectorises this at `-O3` into a
    // 128-bit SIMD loop equivalent to a hand-rolled `wide::u8x16`, so
    // there's no benefit to writing the SIMD by hand here.
    for i in 0..filt.len() {
        curr[i] = filt[i].wrapping_add(prev[i]);
    }
}

#[inline]
fn avg_unfilter(filt: &[u8], prev: &[u8], curr: &mut [u8], bpp: usize) {
    for i in 0..filt.len() {
        let left = if i >= bpp { curr[i - bpp] } else { 0 };
        let up = prev[i];
        // PNG spec: floor((left + up) / 2), computed without overflow.
        let avg = ((left as u16 + up as u16) / 2) as u8;
        curr[i] = filt[i].wrapping_add(avg);
    }
}

#[inline]
fn paeth_unfilter(filt: &[u8], prev: &[u8], curr: &mut [u8], bpp: usize) {
    for i in 0..filt.len() {
        let left = if i >= bpp { curr[i - bpp] } else { 0 };
        let up = prev[i];
        let upleft = if i >= bpp { prev[i - bpp] } else { 0 };
        curr[i] = filt[i].wrapping_add(paeth_predictor(left, up, upleft));
    }
}

#[inline]
fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    // Reference implementation from PNG spec.
    let a = a as i32;
    let b = b as i32;
    let c = c as i32;
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc {
        a as u8
    } else if pb <= pc {
        b as u8
    } else {
        c as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::png::ColorType;

    fn info(w: u32, h: u32, color: ColorType, bit_depth: u8) -> PngInfo {
        let channels = color.channels() as usize;
        let bpp = ((channels * bit_depth as usize) / 8).max(1);
        PngInfo {
            width: w,
            height: h,
            bit_depth,
            color_type: color,
            bpp,
            row_stride: 1 + w as usize * bpp,
        }
    }

    #[test]
    fn filter_none_passes_through() {
        // 2×1 RGB image, filter=0, raw bytes unchanged.
        let stream = vec![0, 10, 20, 30, 40, 50, 60];
        let info = info(2, 1, ColorType::Rgb, 8);
        let out = unfilter(&stream, &info).unwrap();
        assert_eq!(out, vec![10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn filter_sub_recovers_left_deltas() {
        // Original row: [10, 20, 30, 40] bpp=2. Sub stores differences from
        // bpp-byte-left: [10, 20, 30-10, 40-20] = [10, 20, 20, 20].
        let stream = vec![1, 10, 20, 20, 20];
        let info = info(2, 1, ColorType::GreyAlpha, 8);
        let out = unfilter(&stream, &info).unwrap();
        assert_eq!(out, vec![10, 20, 30, 40]);
    }

    #[test]
    fn filter_up_recovers_from_prev_row() {
        // Row 0 None: [5, 10]. Row 1 Up (filter=2) stored as [1, 3] →
        // recovered raw[i] = stored[i] + prev[i] = [1+5, 3+10] = [6, 13].
        let stream = vec![0, 5, 10, 2, 1, 3];
        let info = info(2, 2, ColorType::Greyscale, 8);
        let out = unfilter(&stream, &info).unwrap();
        assert_eq!(out, vec![5, 10, 6, 13]);
    }

    #[test]
    fn filter_average_matches_spec() {
        // 1-pixel-wide grey image: row 0 = [10], row 1 Average stored as x
        // where x = raw - floor((left + up)/2) = raw - floor((0+10)/2) = raw - 5.
        // If raw=20, stored=15.
        let stream = vec![0, 10, 3, 15];
        let info = info(1, 2, ColorType::Greyscale, 8);
        let out = unfilter(&stream, &info).unwrap();
        assert_eq!(out, vec![10, 20]);
    }

    #[test]
    fn filter_paeth_matches_spec() {
        // Paeth on a monotone increasing pattern: predictor picks best of
        // (left, up, upleft). For a flat image, raw=left=up=upleft, so
        // predictor=any and filtered byte = 0.
        let stream = vec![0, 42, 42, 4, 0, 0];
        let info = info(2, 2, ColorType::Greyscale, 8);
        let out = unfilter(&stream, &info).unwrap();
        assert_eq!(out, vec![42, 42, 42, 42]);
    }

    #[test]
    fn filter_rejects_unknown_type() {
        let stream = vec![99, 0];
        let info = info(1, 1, ColorType::Greyscale, 8);
        let err = unfilter(&stream, &info).unwrap_err();
        assert!(matches!(err, FilterError::UnknownFilter { filter: 99, .. }));
    }

    #[test]
    fn filter_rejects_truncated_row() {
        let stream = vec![0, 1]; // claims a 1×2 image (2 rows of 2 bytes each)
        let info = info(1, 2, ColorType::Greyscale, 8);
        let err = unfilter(&stream, &info).unwrap_err();
        assert!(matches!(err, FilterError::TruncatedRow { .. }));
    }

    #[test]
    fn paeth_predictor_picks_closest() {
        // p = a + b - c, then pick whichever of (a,b,c) is closest to p with
        // ties broken a, b, c.
        assert_eq!(paeth_predictor(1, 2, 3), 1); // p=0, pa=1, pb=2, pc=3 → a
        assert_eq!(paeth_predictor(5, 10, 7), 7); // p=8, pa=3, pb=2, pc=1 → c
        assert_eq!(paeth_predictor(10, 10, 10), 10); // all equal → a
        assert_eq!(paeth_predictor(0, 255, 0), 255); // b wins cleanly
    }
}
