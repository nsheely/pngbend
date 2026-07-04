//! One record per literal or back-reference emitted by the decoder.
//!
//! Also houses [`EncTable`], the encoder-side companion to
//! [`super::HuffmanTable`]. Writers use it to re-emit codes at the same
//! bit-lengths as the original; downstream indices use it to find swap /
//! redirect alternatives for each symbol.

/// A symbol's canonical Huffman encoding: the `code` bits and their bit
/// `len`. Returned by [`EncTable::get`]. A present symbol always has
/// `len >= 1` (DEFLATE never assigns a zero-length code).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SymCode {
    pub code: u16,
    pub len: u8,
}

/// Dense encoder-side table: symbol → `(canonical code, code length in
/// bits)`, indexed directly by symbol. Lit blocks have alphabet 0..=287,
/// dist blocks 0..=29.
///
/// `len == 0` is the sentinel for "symbol not present in this block";
/// unambiguous because DEFLATE never assigns a zero-length code to a
/// present symbol (1 bit is the minimum).
///
/// One contiguous `Box<[SymCode]>`: 864 bytes per lit alphabet, 90 bytes per
/// dist alphabet (each [`SymCode`] is 4 bytes). Membership-check and "iterate
/// every present symbol" become a linear scan with one comparison per slot,
/// which is what a consumer's redirect-alternative search hammers per click.
#[derive(Debug, Clone, Default)]
pub struct EncTable {
    entries: Box<[SymCode]>,
}

impl EncTable {
    /// Table for an `n_symbols` alphabet; all symbols start absent
    /// (`len == 0`).
    pub fn new(n_symbols: usize) -> Self {
        Self {
            entries: vec![SymCode::default(); n_symbols].into_boxed_slice(),
        }
    }

    /// Insert or overwrite `sym`'s entry. No-op if `sym` is outside the
    /// table's alphabet.
    #[inline]
    pub fn set(&mut self, sym: u16, code: u16, len: u8) {
        if let Some(slot) = self.entries.get_mut(sym as usize) {
            *slot = SymCode { code, len };
        }
    }

    /// The [`SymCode`] for `sym`, or `None` if the symbol is absent from
    /// this block's alphabet.
    #[inline]
    pub fn get(&self, sym: u16) -> Option<SymCode> {
        let &sc = self.entries.get(sym as usize)?;
        (sc.len != 0).then_some(sc)
    }

    /// True when no symbol has been assigned a code (either a freshly
    /// constructed empty table or a stored block's placeholder).
    pub fn is_empty(&self) -> bool {
        self.entries.iter().all(|sc| sc.len == 0)
    }

    /// Iterate `(symbol, SymCode)` for every present symbol.
    pub fn iter(&self) -> impl Iterator<Item = (u16, SymCode)> + '_ {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(sym, &sc)| (sc.len != 0).then_some((sym as u16, sc)))
    }

    /// Raw [`SymCode`] slice indexed by symbol. Lets tight loops scan `len`
    /// directly; absent slots have `len == 0` so the present-check is a single
    /// comparison rather than a branch through `Option`.
    pub fn raw(&self) -> &[SymCode] {
        &self.entries
    }
}

/// One literal symbol emitted by the decoder.
///
/// Position fields are `u32`: the loader rejects images whose unfiltered
/// output would exceed 4 GiB, so byte and bit offsets always fit in 32
/// bits. There is no `block` field: a block is a contiguous *range* of
/// events, recorded once as [`super::DecodedDeflate::block_starts`] rather than
/// stamped onto every element. That keeps the `Event` enum at 20 bytes, so
/// per-event scans over a multi-million-event stream move less memory.
#[derive(Debug, Clone)]
pub struct LitEvent {
    pub out_pos: u32,
    pub bit_start: u32,
    pub symbol: u8,
}

#[derive(Debug, Clone)]
pub struct RefEvent {
    pub out_pos: u32,
    pub src_out_pos: u32,
    pub dist_bit_start: u32,
    pub copy_len: u16,
    pub dist_sym: u8,
}

#[derive(Debug, Clone)]
pub enum Event {
    Lit(LitEvent),
    Ref(RefEvent),
}

/// Given the per-block event-start indices (`block_starts[b]` is the
/// index of block `b`'s first event) and an event index, return the
/// block that event belongs to. `O(log blocks)`; blocks number in the
/// dozens-to-hundreds, so this is cheap even at click frequency.
#[inline]
pub fn block_of(block_starts: &[u32], ev_idx: u32) -> u32 {
    // Largest b with block_starts[b] <= ev_idx.
    (block_starts.partition_point(|&s| s <= ev_idx) as u32).saturating_sub(1)
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
        assert_eq!(
            t.get(42),
            Some(SymCode {
                code: 0b1011,
                len: 4
            })
        );
        assert_eq!(
            t.get(100),
            Some(SymCode {
                code: 0b111,
                len: 3
            })
        );
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
        assert_eq!(t.raw()[3], SymCode { code: 5, len: 2 });
        assert_eq!(t.raw()[0], SymCode { code: 0, len: 0 });
    }
}
