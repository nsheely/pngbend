//! Reverse LZ77 dependency graph in compressed-sparse-row form.
//!
//! For every source byte position, lists every output position copied
//! from it. The cascade BFS in [`super::CascadeScratch`] walks this graph
//! to find every byte downstream of a given output position: the fan-out
//! of an edit there.
//!
//! Two contiguous `Vec<u32>`s: `offsets` sized `output_len + 1` and `edges`
//! sized to the total ref-byte count. Drop is `O(1)` (two frees), no
//! per-source allocation.
//!
//! Build is two passes:
//!
//! - **Pass 1** counts each source position's out-degree with a
//!   range-update and prefix sum: a ref of length `L` writes two cells
//!   (at `src` and `src + L`) instead of `L` increments. For typical
//!   `copy_len ≈ 5` that's ~2.5× fewer scattered writes to cold cache
//!   lines; the closing prefix sum is a dense sequential scan.
//! - **Pass 2** fills `edges`. A `cursor` Vec (cloned from `offsets`)
//!   holds the next write index per source, so each ref byte touches one
//!   cold cache line instead of three.

use crate::deflate::Event;

#[derive(Debug, Default)]
pub struct ReverseGraph {
    /// `offsets[src..=src+1]` bounds the slice of `edges` belonging to `src`.
    /// `offsets.len() == output_len + 1`.
    offsets: Vec<u32>,
    edges: Vec<u32>,
}

impl ReverseGraph {
    #[inline]
    pub fn neighbors(&self, src: u32) -> &[u32] {
        let si = src as usize;
        if si + 1 >= self.offsets.len() {
            return &[];
        }
        let start = self.offsets[si] as usize;
        let end = self.offsets[si + 1] as usize;
        &self.edges[start..end]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub fn build_reverse_graph(events: &[Event], output_len: usize) -> ReverseGraph {
    // Pass 1: count out-degree of every source position via range update.
    //
    // For a ref with `src_out_pos = s, copy_len = L`, every cell in
    // `degree[s..s+L]` should gain 1. Recording the delta at the two
    // endpoints (`+1` at `s`, `−1` at `s+L`) and prefix-summing yields the
    // same degrees with `O(events)` scattered writes instead of
    // `O(Σ copy_len)`.
    //
    // The decoder guarantees `s + L ≤ output_len` for every ref (the source
    // is already in `output` when the ref is emitted), so both writes are in
    // bounds against `delta` of length `output_len + 1`. The unchecked
    // writes skip `Vec::index_mut`'s precondition chain, a noticeable
    // fraction of build time on multi-million-event inputs per `perf`.
    let mut delta: Vec<i32> = vec![0; output_len + 1];
    for e in events {
        if let Event::Ref(r) = e {
            let s = r.src_out_pos as usize;
            let end = s + r.copy_len as usize;
            debug_assert!(end <= output_len, "ref source extends past output");
            unsafe {
                *delta.get_unchecked_mut(s) += 1;
                *delta.get_unchecked_mut(end) -= 1;
            }
        }
    }

    // Prefix sum delta → degree, then shift into `offsets` in the same
    // pass. `offsets[0] = 0` (exclusive prefix);
    // `offsets[i+1] = offsets[i] + degree[i]`. Both `delta[i]` and
    // `offsets[i+1]` are in bounds for `i ∈ 0..output_len`.
    let mut offsets: Vec<u32> = vec![0; output_len + 1];
    let mut running: i32 = 0;
    let mut acc: u32 = 0;
    for i in 0..output_len {
        debug_assert!(i + 1 < offsets.len());
        unsafe {
            running += *delta.get_unchecked(i);
            acc = acc.wrapping_add(running as u32);
            *offsets.get_unchecked_mut(i + 1) = acc;
        }
    }
    let total_edges = acc as usize;

    drop(delta);

    // Pass 2: fill edges. `cursor` is a writable copy of `offsets` that
    // tracks the next free edge slot per source. Each ref byte does one
    // cursor read/increment + one edges write, leaving the canonical
    // `offsets` untouched for [`ReverseGraph::neighbors`] queries.
    //
    // Invariants: `slot_idx = s + offset < s + L ≤ output_len <
    // cursor.len()`, so the cursor lookup is in bounds. `cursor[slot_idx]`
    // starts at `offsets[slot_idx]` and increments at most
    // `degree[slot_idx]` times, never past `offsets[slot_idx + 1] ≤
    // total_edges`.
    let mut cursor: Vec<u32> = offsets.clone();
    let mut edges: Vec<u32> = vec![0u32; total_edges];
    for e in events {
        if let Event::Ref(r) = e {
            let s = r.src_out_pos as usize;
            let dst_base = r.out_pos as usize;
            for offset in 0..r.copy_len as usize {
                let slot_idx = s + offset;
                debug_assert!(slot_idx < cursor.len());
                unsafe {
                    let slot = *cursor.get_unchecked(slot_idx) as usize;
                    debug_assert!(slot < edges.len());
                    *edges.get_unchecked_mut(slot) = (dst_base + offset) as u32;
                    *cursor.get_unchecked_mut(slot_idx) = (slot as u32).wrapping_add(1);
                }
            }
        }
    }

    ReverseGraph { offsets, edges }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate::{Event, LitEvent, RefEvent};

    fn refe(out_pos: u32, src_out_pos: u32, copy_len: u16) -> Event {
        Event::Ref(RefEvent {
            out_pos,
            src_out_pos,
            copy_len,
            dist_sym: 0,
            dist_bit_start: 0,
        })
    }

    fn lit(out_pos: u32) -> Event {
        Event::Lit(LitEvent {
            out_pos,
            symbol: 0,
            bit_start: 0,
        })
    }

    #[test]
    fn empty_events_empty_graph() {
        let g = build_reverse_graph(&[], 0);
        assert!(g.is_empty());
    }

    #[test]
    fn single_ref_single_byte() {
        // src=0, copy_len=1, dst=3. One edge 0 → 3.
        let events = vec![lit(0), lit(1), lit(2), refe(3, 0, 1)];
        let g = build_reverse_graph(&events, 4);
        assert_eq!(g.neighbors(0), &[3]);
        assert_eq!(g.neighbors(1), &[]);
    }

    #[test]
    fn overlapping_refs_produce_sorted_edges() {
        // Two refs both reading from src=0 with differing lengths.
        // Produces edges 0→2, 0→3, 1→4, 1→5.
        let events = vec![
            lit(0),
            lit(1),
            refe(2, 0, 2), // 0→2, 1→3
            refe(4, 0, 2), // 0→4, 1→5
        ];
        let g = build_reverse_graph(&events, 6);
        let mut n0 = g.neighbors(0).to_vec();
        let mut n1 = g.neighbors(1).to_vec();
        n0.sort();
        n1.sort();
        assert_eq!(n0, vec![2, 4]);
        assert_eq!(n1, vec![3, 5]);
    }

    #[test]
    fn range_update_handles_ref_at_output_end() {
        // Ref writes positions [2, 3] copied from src [0, 1]. delta[0]+=1,
        // delta[2]-=1, and output_len=4 means delta[..5] is sized correctly.
        let events = vec![lit(0), lit(1), refe(2, 0, 2)];
        let g = build_reverse_graph(&events, 4);
        assert_eq!(g.neighbors(0), &[2]);
        assert_eq!(g.neighbors(1), &[3]);
        // Nothing was written at output_len itself.
        assert_eq!(g.neighbors(4), &[]);
    }
}
