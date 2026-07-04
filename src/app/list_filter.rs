//! Filter state for the left-panel pixel list.
//!
//! The list stores *indices* into the source [`PixelIndex`] arrays rather
//! than cloning full [`PixelRow`]s. On a multi-megapixel photo this turns
//! every filter keystroke from "clone N strings" into "copy N u32s".
//!
//! Predicates take `FnMut(FilterRef, &PixelRow) -> bool` so callers can
//! amortise a scratch buffer across rows: the text-matching path formats
//! each row on demand and reuses the same `String` scratch across the
//! entire rebuild.

use crate::index::{PixelIndex, PixelRow};

use super::io::CoreData;
use super::row_text::{append_row_text, lookup_ref_metrics};

/// Reference to a row in either the `lit` or `refs` array of the parent
/// [`PixelIndex`]. 8 bytes (one `u32` index plus the enum discriminant),
/// so the `Vec<FilterRef>` of a multi-megapixel filtered view stays
/// cheap to copy and merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FilterRef {
    Lit(u32),
    Ref(u32),
}

impl FilterRef {
    /// Resolve to the underlying row. The returned borrow has the same
    /// lifetime as `pi`.
    #[inline]
    pub fn resolve(self, pi: &PixelIndex) -> &PixelRow {
        match self {
            Self::Lit(i) => &pi.lit[i as usize],
            Self::Ref(i) => &pi.refs[i as usize],
        }
    }

    /// 1-based display index for the sidebar row: `i + 1` for either arm.
    #[inline]
    pub fn display_index(self) -> u32 {
        match self {
            Self::Lit(i) | Self::Ref(i) => i + 1,
        }
    }
}

/// Build a filtered view for the "Literals" radio.
pub(super) fn filter_lit(
    pi: &PixelIndex,
    mut pred: impl FnMut(FilterRef, &PixelRow) -> bool,
) -> Vec<FilterRef> {
    let mut out = Vec::new();
    for (i, row) in pi.lit.iter().enumerate() {
        let f = FilterRef::Lit(i as u32);
        if pred(f, row) {
            out.push(f);
        }
    }
    out
}

/// Build a filtered view for the "Backrefs" radio.
pub(super) fn filter_refs(
    pi: &PixelIndex,
    mut pred: impl FnMut(FilterRef, &PixelRow) -> bool,
) -> Vec<FilterRef> {
    let mut out = Vec::new();
    for (i, row) in pi.refs.iter().enumerate() {
        let f = FilterRef::Ref(i as u32);
        if pred(f, row) {
            out.push(f);
        }
    }
    out
}

/// Build a filtered view for the "All" radio. Both source arrays are
/// individually sorted by `(y, x)`, so a merge step produces a globally
/// sorted output without ever materialising a combined `Vec<PixelRow>`.
pub(super) fn filter_all(
    pi: &PixelIndex,
    mut pred: impl FnMut(FilterRef, &PixelRow) -> bool,
) -> Vec<FilterRef> {
    let mut out = Vec::with_capacity(pi.lit.len() + pi.refs.len());
    let (mut li, mut ri) = (0usize, 0usize);
    while li < pi.lit.len() || ri < pi.refs.len() {
        let take_lit = match (pi.lit.get(li), pi.refs.get(ri)) {
            (Some(l), Some(r)) => sort_key(l) <= sort_key(r),
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        if take_lit {
            let row = &pi.lit[li];
            let f = FilterRef::Lit(li as u32);
            li += 1;
            if pred(f, row) {
                out.push(f);
            }
        } else {
            let row = &pi.refs[ri];
            let f = FilterRef::Ref(ri as u32);
            ri += 1;
            if pred(f, row) {
                out.push(f);
            }
        }
    }
    out
}

#[inline]
fn sort_key(r: &PixelRow) -> (u32, u32) {
    (r.y(), r.x())
}

// structured filter predicates
//
// The filter box accepts a few structured shapes (`#rrggbb`, `d=N`,
// `len=N`, `x,y`, ...). Parsing them once per keystroke makes the
// per-row check `O(1)` arithmetic on fields the row already carries:
// no per-row formatting, lower-casing, or substring scan. Anything the
// parser doesn't recognise falls into the [`FilterSpec::Generic`] arm,
// which is the only one that touches the formatter at all.

#[derive(Debug, Clone)]
pub(super) enum FilterSpec {
    /// Empty filter: every row passes.
    All,
    /// `#rrggbb` (or shorter, byte-aligned). `n_bytes` of `bytes` must
    /// match the start of `PixelRow.rgb`.
    HexPrefix { n_bytes: u8, bytes: [u8; 3] },
    /// `d=N` / `dist=N`: a ref row whose LZ77 distance equals N.
    Dist(usize),
    /// `len=N`: a ref row whose copy length equals N.
    Len(u16),
    /// `X,Y`: exact pixel coordinates.
    Coord(u32, u32),
    /// `,Y`: any pixel with y == Y.
    CoordRow(u32),
    /// `X,`: any pixel with x == X.
    CoordCol(u32),
    /// Fallback: format each row's display text, lowercase, and substring
    /// match. `String` is pre-lowercased at parse time.
    Generic(String),
}

impl FilterSpec {
    /// Parse the filter box text. Whitespace is trimmed; patterns that
    /// don't match any structured form fall back to `Generic`.
    pub(super) fn parse(text: &str) -> Self {
        let t = text.trim();
        if t.is_empty() {
            return FilterSpec::All;
        }

        if let Some(hex) = t.strip_prefix('#')
            && let Some(spec) = parse_hex_prefix(hex)
        {
            return spec;
        }
        if let Some(num) = t.strip_prefix("d=").or_else(|| t.strip_prefix("dist="))
            && let Ok(n) = num.parse::<usize>()
        {
            return FilterSpec::Dist(n);
        }
        if let Some(num) = t.strip_prefix("len=")
            && let Ok(n) = num.parse::<u16>()
        {
            return FilterSpec::Len(n);
        }
        if let Some((xpart, ypart)) = t.split_once(',') {
            let xparse = xpart.trim().parse::<u32>();
            let yparse = ypart.trim().parse::<u32>();
            match (
                xparse,
                yparse,
                xpart.trim().is_empty(),
                ypart.trim().is_empty(),
            ) {
                (Ok(x), Ok(y), _, _) => return FilterSpec::Coord(x, y),
                (Err(_), Ok(y), true, _) => return FilterSpec::CoordRow(y),
                (Ok(x), Err(_), _, true) => return FilterSpec::CoordCol(x),
                _ => {}
            }
        }

        FilterSpec::Generic(t.to_ascii_lowercase())
    }

    /// Evaluate against one row. `scratch` is reused across rows; only
    /// the `Generic` arm actually writes into it, so warm `HexPrefix` /
    /// `Dist` / `Coord` matches don't touch the allocator.
    pub(super) fn matches(
        &self,
        fref: FilterRef,
        row: &PixelRow,
        c: &CoreData,
        scratch: &mut String,
    ) -> bool {
        match self {
            FilterSpec::All => true,
            FilterSpec::HexPrefix { n_bytes, bytes } => {
                let n = *n_bytes as usize;
                row.rgb[..n] == bytes[..n]
            }
            FilterSpec::Dist(d) => {
                matches!(fref, FilterRef::Ref(_))
                    && lookup_ref_metrics(c, row.xy()).is_some_and(|(dist, _)| dist == *d)
            }
            FilterSpec::Len(n) => {
                matches!(fref, FilterRef::Ref(_))
                    && lookup_ref_metrics(c, row.xy()).is_some_and(|(_, len)| len == *n)
            }
            FilterSpec::Coord(x, y) => row.x() == *x && row.y() == *y,
            FilterSpec::CoordRow(y) => row.y() == *y,
            FilterSpec::CoordCol(x) => row.x() == *x,
            FilterSpec::Generic(needle) => {
                scratch.clear();
                append_row_text(scratch, fref, row, c);
                scratch.make_ascii_lowercase();
                scratch.contains(needle.as_str())
            }
        }
    }

    /// True if every row matching `self` also matches `other`: its match
    /// set is a subset of `other`'s. When `rebuild_filter` detects this on
    /// a keystroke, it only re-tests rows currently in `filtered_view`
    /// instead of rescanning the whole `PixelIndex`.
    ///
    /// Conservative: returns `false` for any cross-type transition (e.g.
    /// `Coord(x,y)` → `CoordRow(y)`) where correctness would need a more
    /// involved check, so we fall back to a safe full rebuild on those.
    pub(super) fn is_refinement_of(&self, other: &FilterSpec) -> bool {
        use FilterSpec::*;
        match (other, self) {
            // `All` matches everything, so anything is a refinement.
            (All, _) => true,
            // Refining a specific filter to `All` would widen the match set.
            (_, All) => false,
            (Generic(a), Generic(b)) => b.contains(a.as_str()),
            (
                HexPrefix {
                    n_bytes: na,
                    bytes: ba,
                },
                HexPrefix {
                    n_bytes: nb,
                    bytes: bb,
                },
            ) => {
                let na = *na as usize;
                let nb = *nb as usize;
                nb >= na && bb[..na] == ba[..na]
            }
            (Dist(a), Dist(b)) => a == b,
            (Len(a), Len(b)) => a == b,
            (Coord(x1, y1), Coord(x2, y2)) => x1 == x2 && y1 == y2,
            (CoordRow(a), CoordRow(b)) => a == b,
            (CoordCol(a), CoordCol(b)) => a == b,
            _ => false,
        }
    }
}

/// Parse `rrggbb` (or a byte-aligned prefix: `rr` / `rrgg`) into a byte
/// prefix. Non-hex or odd length returns `None` so the caller can fall
/// through to the generic substring match.
fn parse_hex_prefix(s: &str) -> Option<FilterSpec> {
    if s.is_empty() || s.len() > 6 || !s.len().is_multiple_of(2) {
        return None;
    }
    let mut bytes = [0u8; 3];
    for (i, byte) in bytes.iter_mut().take(s.len() / 2).enumerate() {
        let hi = hex_digit(s.as_bytes()[i * 2])?;
        let lo = hex_digit(s.as_bytes()[i * 2 + 1])?;
        *byte = (hi << 4) | lo;
    }
    Some(FilterSpec::HexPrefix {
        n_bytes: (s.len() / 2) as u8,
        bytes,
    })
}

#[inline]
fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(x: u32, y: u32, has_edit: bool) -> PixelRow {
        PixelRow::new(crate::coords::PixelXY::new(x, y), [0, 0, 0], has_edit)
    }

    fn pi() -> PixelIndex {
        PixelIndex {
            lit: vec![row(0, 0, true), row(2, 1, false), row(5, 3, true)],
            refs: vec![row(1, 0, true), row(3, 2, true)],
            n_lit_with_edit: 2,
        }
    }

    #[test]
    fn filter_lit_picks_passing_rows() {
        let v = filter_lit(&pi(), |_f, r| r.has_edit);
        assert_eq!(v, vec![FilterRef::Lit(0), FilterRef::Lit(2)]);
    }

    #[test]
    fn filter_refs_picks_passing_rows() {
        // pick just the second ref row
        let v = filter_refs(&pi(), |f, _| matches!(f, FilterRef::Ref(1)));
        assert_eq!(v, vec![FilterRef::Ref(1)]);
    }

    #[test]
    fn filter_all_is_sorted_and_respects_predicate() {
        let v = filter_all(&pi(), |_, _| true);
        // All rows in y,x order: (0,0)Lit0 → (1,0)Ref0 → (2,1)Lit1 →
        // (3,2)Ref1 → (5,3)Lit2
        assert_eq!(
            v,
            vec![
                FilterRef::Lit(0),
                FilterRef::Ref(0),
                FilterRef::Lit(1),
                FilterRef::Ref(1),
                FilterRef::Lit(2),
            ]
        );
    }

    #[test]
    fn filter_all_filters() {
        let v = filter_all(&pi(), |_, r| r.has_edit);
        // Exclude Lit(1) (has_edit=false).
        assert_eq!(
            v,
            vec![
                FilterRef::Lit(0),
                FilterRef::Ref(0),
                FilterRef::Ref(1),
                FilterRef::Lit(2),
            ]
        );
    }

    #[test]
    fn display_index_is_one_based() {
        assert_eq!(FilterRef::Lit(0).display_index(), 1);
        assert_eq!(FilterRef::Ref(7).display_index(), 8);
    }

    // FilterSpec parsing

    #[test]
    fn parse_empty_and_whitespace_is_all() {
        assert!(matches!(FilterSpec::parse(""), FilterSpec::All));
        assert!(matches!(FilterSpec::parse("   "), FilterSpec::All));
    }

    #[test]
    fn parse_hex_prefix_variants() {
        match FilterSpec::parse("#ff") {
            FilterSpec::HexPrefix { n_bytes: 1, bytes } => assert_eq!(bytes, [0xff, 0, 0]),
            other => panic!("expected HexPrefix(1), got {other:?}"),
        }
        match FilterSpec::parse("#aabb") {
            FilterSpec::HexPrefix { n_bytes: 2, bytes } => assert_eq!(bytes, [0xaa, 0xbb, 0]),
            other => panic!("expected HexPrefix(2), got {other:?}"),
        }
        match FilterSpec::parse("#112233") {
            FilterSpec::HexPrefix { n_bytes: 3, bytes } => {
                assert_eq!(bytes, [0x11, 0x22, 0x33]);
            }
            other => panic!("expected HexPrefix(3), got {other:?}"),
        }
    }

    #[test]
    fn parse_malformed_hex_falls_back_to_generic() {
        // Odd length, not byte-aligned.
        assert!(matches!(FilterSpec::parse("#fff"), FilterSpec::Generic(_)));
        // Non-hex character.
        assert!(matches!(FilterSpec::parse("#zz"), FilterSpec::Generic(_)));
    }

    #[test]
    fn parse_dist_and_len() {
        assert!(matches!(FilterSpec::parse("d=42"), FilterSpec::Dist(42)));
        assert!(matches!(
            FilterSpec::parse("dist=100"),
            FilterSpec::Dist(100)
        ));
        assert!(matches!(FilterSpec::parse("len=5"), FilterSpec::Len(5)));
        // Non-numeric payload falls back.
        assert!(matches!(FilterSpec::parse("d=abc"), FilterSpec::Generic(_)));
    }

    #[test]
    fn parse_coord_variants() {
        assert!(matches!(
            FilterSpec::parse("100,200"),
            FilterSpec::Coord(100, 200)
        ));
        assert!(matches!(
            FilterSpec::parse(" 100 , 200 "),
            FilterSpec::Coord(100, 200)
        ));
        assert!(matches!(FilterSpec::parse(",50"), FilterSpec::CoordRow(50)));
        assert!(matches!(FilterSpec::parse("7,"), FilterSpec::CoordCol(7)));
    }

    #[test]
    fn parse_bare_integer_is_generic() {
        // Single integer is ambiguous (could be x, y, or part of #hex),
        // so fall back to the substring path rather than guess.
        assert!(matches!(FilterSpec::parse("42"), FilterSpec::Generic(_)));
    }

    #[test]
    fn parse_generic_lowercases_text() {
        match FilterSpec::parse("D=ABC") {
            FilterSpec::Generic(s) => assert_eq!(s, "d=abc"),
            other => panic!("expected Generic, got {other:?}"),
        }
    }

    // is_refinement_of

    #[test]
    fn refinement_generic_prefix_growth() {
        let old = FilterSpec::parse("ab");
        let new = FilterSpec::parse("ab1");
        assert!(new.is_refinement_of(&old));
        // Back-stepping is not a refinement.
        assert!(!old.is_refinement_of(&new));
    }

    #[test]
    fn refinement_generic_same_is_refinement() {
        let a = FilterSpec::parse("abc");
        assert!(a.is_refinement_of(&a));
    }

    #[test]
    fn refinement_generic_disjoint_not_refinement() {
        let a = FilterSpec::parse("abc");
        let b = FilterSpec::parse("xyz");
        assert!(!b.is_refinement_of(&a));
    }

    #[test]
    fn refinement_hexprefix_extension() {
        let old = FilterSpec::parse("#aa");
        let new = FilterSpec::parse("#aabb");
        assert!(new.is_refinement_of(&old));
        assert!(!old.is_refinement_of(&new));
    }

    #[test]
    fn refinement_hexprefix_divergent() {
        // Different first byte, not a refinement.
        let a = FilterSpec::parse("#aa");
        let b = FilterSpec::parse("#bb");
        assert!(!b.is_refinement_of(&a));
    }

    #[test]
    fn refinement_from_all() {
        let old = FilterSpec::parse("");
        let new = FilterSpec::parse("ab");
        // Anything refines All.
        assert!(new.is_refinement_of(&old));
        // All does NOT refine a specific filter (it would widen).
        assert!(!old.is_refinement_of(&new));
    }

    #[test]
    fn refinement_cross_type_conservative_false() {
        let coord = FilterSpec::parse("10,20");
        let row = FilterSpec::parse(",20");
        // Coord IS a refinement of CoordRow semantically, but we return
        // false for cross-type to keep the check simple and correct.
        assert!(!coord.is_refinement_of(&row));
        assert!(!row.is_refinement_of(&coord));
    }

    #[test]
    fn refinement_structured_same() {
        assert!(FilterSpec::Dist(42).is_refinement_of(&FilterSpec::Dist(42)));
        assert!(FilterSpec::Len(5).is_refinement_of(&FilterSpec::Len(5)));
        assert!(FilterSpec::Coord(10, 20).is_refinement_of(&FilterSpec::Coord(10, 20)));
    }

    #[test]
    fn refinement_structured_different_values() {
        assert!(!FilterSpec::Dist(42).is_refinement_of(&FilterSpec::Dist(43)));
    }
}
