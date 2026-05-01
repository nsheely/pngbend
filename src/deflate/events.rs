//! One record per literal or back-reference emitted by the decoder.
//!
//! Also houses [`EncTable`], the encoder-side companion to
//! [`super::HuffmanTable`]. Writers use it to re-emit codes at the same
//! bit-lengths as the original; downstream indices use it to find swap /
//! redirect alternatives for each symbol.

/// Dense encoder-side table: symbol → `(canonical code, code length in
/// bits)`, indexed directly by symbol. Lit blocks have alphabet 0..=287,
/// dist blocks 0..=29.
///
/// `clen == 0` is the sentinel for "symbol not present in this block";
/// unambiguous because DEFLATE never assigns a zero-length code to a
/// present symbol (1 bit is the minimum).
///
/// One contiguous `Box<[(u16, u8)]>` — 864 bytes per lit alphabet, 90
/// bytes per dist alphabet. Membership-check and "iterate every present
/// symbol" become a linear scan with one comparison per slot, which is
/// what the redirect-alternative search in [`crate::index::pixel`]
/// hammers per click.
#[derive(Debug, Clone, Default)]
pub struct EncTable {
    entries: Box<[(u16, u8)]>,
}

impl EncTable {
    /// A table sized to hold a `n_symbols`-sized alphabet. All symbols
    /// start absent (`clen == 0`).
    pub fn new(n_symbols: usize) -> Self {
        Self {
            entries: vec![(0u16, 0u8); n_symbols].into_boxed_slice(),
        }
    }

    /// Insert or overwrite `sym`'s entry. No-op if `sym` is outside the
    /// table's alphabet.
    #[inline]
    pub fn set(&mut self, sym: u16, code: u16, clen: u8) {
        if let Some(slot) = self.entries.get_mut(sym as usize) {
            *slot = (code, clen);
        }
    }

    /// `Some((code, clen))` if `sym` is present in this block's alphabet,
    /// else `None`.
    #[inline]
    pub fn get(&self, sym: u16) -> Option<(u16, u8)> {
        let &(code, clen) = self.entries.get(sym as usize)?;
        (clen != 0).then_some((code, clen))
    }

    /// True when no symbol has been assigned a code (either a freshly
    /// constructed empty table or a stored block's placeholder).
    pub fn is_empty(&self) -> bool {
        self.entries.iter().all(|&(_, clen)| clen == 0)
    }

    /// Iterate `(symbol, code, clen)` for every present symbol.
    pub fn iter(&self) -> impl Iterator<Item = (u16, u16, u8)> + '_ {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(sym, &(code, clen))| (clen != 0).then_some((sym as u16, code, clen)))
    }

    /// Raw `(code, clen)` slice indexed by symbol. Lets tight loops scan
    /// `clen` directly; absent slots have `clen == 0` so the present-check
    /// is a single comparison rather than a branch through `Option`.
    pub fn raw(&self) -> &[(u16, u8)] {
        &self.entries
    }
}

/// One literal symbol emitted by the decoder.
///
/// Position fields are `u32`: the loader rejects images whose unfiltered
/// output would exceed 4 GiB, so byte and bit offsets always fit in 32
/// bits. Keeps the `Event` enum below at 24 bytes per element —
/// significant on multi-million-event indices.
#[derive(Debug, Clone)]
pub struct LitEvent {
    pub out_pos: u32,
    pub bit_start: u32,
    pub block: u32,
    pub symbol: u8,
}

#[derive(Debug, Clone)]
pub struct RefEvent {
    pub out_pos: u32,
    pub src_out_pos: u32,
    pub dist_bit_start: u32,
    pub block: u32,
    pub copy_len: u16,
    pub dist_sym: u8,
}

#[derive(Debug, Clone)]
pub enum Event {
    Lit(LitEvent),
    Ref(RefEvent),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_table_misses_everything() {
        let t = EncTable::default();
        assert!(t.is_empty());
        assert_eq!(t.get(0), None);
        assert_eq!(t.iter().count(), 0);
    }

    #[test]
    fn set_then_get_round_trip() {
        let mut t = EncTable::new(288);
        t.set(42, 0b1011, 4);
        t.set(100, 0b111, 3);
        assert_eq!(t.get(42), Some((0b1011, 4)));
        assert_eq!(t.get(100), Some((0b111, 3)));
        assert_eq!(t.get(7), None); // unset
        assert_eq!(t.iter().count(), 2);
        assert!(!t.is_empty());
    }

    #[test]
    fn set_outside_alphabet_is_noop() {
        let mut t = EncTable::new(30);
        t.set(50, 0xABC, 7);
        assert_eq!(t.get(50), None);
    }

    #[test]
    fn raw_exposes_dense_layout() {
        let mut t = EncTable::new(10);
        t.set(3, 5, 2);
        assert_eq!(t.raw().len(), 10);
        assert_eq!(t.raw()[3], (5, 2));
        assert_eq!(t.raw()[0], (0, 0));
    }
}
