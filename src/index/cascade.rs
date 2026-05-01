//! Epoch-versioned BFS scratch for the "what bytes downstream depend on
//! this edit" query.
//!
//! `CascadeScratch` is held on the app across clicks. Its `depth_map`
//! is a dense per-output-byte table allocated once per file; bumping
//! the `epoch` byte at the start of each run invalidates every stale
//! entry in O(1) without re-zeroing.
//!
//! Each slot is a packed `u16`: high byte = run epoch, low byte = BFS
//! depth (saturating at 255). 2 bytes per output byte → 24 MB for a
//! 12 MB output buffer, half what `(epoch: u32, depth: u8)` would take,
//! so the BFS reads from a table that fits in L3.
//!
//! The epoch is 8 bits and wraps every 256 runs. On wrap the map is
//! zeroed once and the epoch restarts at 1: ~0.8 ms per 256-run cycle
//! (≈ 3 µs/run amortised).

use super::reverse_graph::ReverseGraph;

/// Packed `(epoch << 8) | depth`. The epoch lives in the high byte; the
/// comparison the hot BFS loop does is `slot & EPOCH_MASK == epoch_shifted`.
type Slot = u16;

const EPOCH_SHIFT: u32 = 8;
const EPOCH_MASK: Slot = 0xFF00;
const DEPTH_MASK: Slot = 0x00FF;

pub struct CascadeScratch {
    depth_map: Vec<Slot>,
    /// Doubles as BFS queue and final affected-set output. Nodes are
    /// pushed FIFO; `run` walks them via a local `head` cursor. One Vec
    /// instead of a `VecDeque` frontier + a separate output array.
    affected: Vec<u32>,
    /// Stored already shifted into the high byte so the hot per-node
    /// comparison is a single `==` against `slot & EPOCH_MASK`.
    epoch_shifted: Slot,
}

#[inline(always)]
fn pack(epoch_shifted: Slot, depth: u32) -> Slot {
    epoch_shifted | (depth.min(DEPTH_MASK as u32) as Slot)
}

#[inline(always)]
fn unpack_depth(slot: Slot) -> u32 {
    (slot & DEPTH_MASK) as u32
}

impl Default for CascadeScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl CascadeScratch {
    pub fn new() -> Self {
        Self {
            depth_map: Vec::new(),
            affected: Vec::new(),
            epoch_shifted: 0,
        }
    }

    /// Run BFS from `start` over `rev`. The returned view borrows this
    /// scratch; reusing the scratch invalidates it.
    pub fn run<'a>(&'a mut self, start: &[u32], rev: &ReverseGraph) -> Cascade<'a> {
        let n = rev.len();
        if self.depth_map.len() != n {
            self.depth_map.clear();
            self.depth_map.resize(n, 0);
            self.epoch_shifted = 0;
        }
        // Advance the epoch. With only 8 bits of epoch we wrap every 256
        // runs; on wrap we zero the entire table once and restart at
        // epoch=1 so the new run's slots are unambiguously fresh.
        self.epoch_shifted = self.epoch_shifted.wrapping_add(1 << EPOCH_SHIFT);
        if self.epoch_shifted & EPOCH_MASK == 0 {
            self.depth_map.fill(0);
            self.epoch_shifted = 1 << EPOCH_SHIFT;
        }
        let ep = self.epoch_shifted;
        self.affected.clear();
        let mut max_depth = 0u32;

        for &p in start {
            let pi = p as usize;
            if pi >= n {
                continue;
            }
            // SAFETY: `pi < n == depth_map.len()`.
            let slot = unsafe { self.depth_map.get_unchecked_mut(pi) };
            if *slot & EPOCH_MASK != ep {
                *slot = pack(ep, 0);
                self.affected.push(p);
            }
        }

        // BFS using `affected` itself as a Vec-queue: push at the end,
        // walk a local `head` index forward. Each node is enqueued once
        // (the visited stamp guards), so when `head` catches up to `len`
        // we're done.
        let mut head = 0usize;
        while head < self.affected.len() {
            let pos = self.affected[head];
            head += 1;
            // `pos` was pushed after its slot was stamped with `ep`, so the
            // unchecked read is safe.
            let d = unpack_depth(unsafe { *self.depth_map.get_unchecked(pos as usize) });
            for &dst in rev.neighbors(pos) {
                let di = dst as usize;
                // SAFETY: reverse-graph invariants give `di < n`.
                let slot = unsafe { self.depth_map.get_unchecked_mut(di) };
                if *slot & EPOCH_MASK != ep {
                    let nd = d + 1;
                    *slot = pack(ep, nd);
                    if nd > max_depth {
                        max_depth = nd;
                    }
                    self.affected.push(dst);
                }
            }
        }

        Cascade {
            affected: &self.affected,
            depth_map: &self.depth_map,
            epoch_shifted: ep,
            max_depth,
        }
    }
}

/// Borrowed view of a completed cascade BFS.
pub struct Cascade<'a> {
    pub affected: &'a [u32],
    depth_map: &'a [Slot],
    epoch_shifted: Slot,
    pub max_depth: u32,
}

impl Cascade<'_> {
    /// BFS depth for `pos`, or `None` if unvisited this run.
    #[inline]
    pub fn depth(&self, pos: u32) -> Option<u32> {
        let slot = *self.depth_map.get(pos as usize)?;
        (slot & EPOCH_MASK == self.epoch_shifted).then(|| unpack_depth(slot))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate::{Event, RefEvent};
    use crate::index::build_reverse_graph;

    /// Build a CSR reverse graph from adjacency lists, for testing.
    fn rev_from_adj(adj: &[Vec<u32>]) -> ReverseGraph {
        let events: Vec<Event> = adj
            .iter()
            .enumerate()
            .flat_map(|(src, dsts)| {
                dsts.iter().map(move |&dst| {
                    Event::Ref(RefEvent {
                        out_pos: dst,
                        src_out_pos: src as u32,
                        copy_len: 1,
                        dist_sym: 0,
                        block: 0,
                        dist_bit_start: 0,
                    })
                })
            })
            .collect();
        build_reverse_graph(&events, adj.len())
    }

    #[test]
    fn single_seed_no_edges() {
        let rev = rev_from_adj(&vec![Vec::new(); 10]);
        let mut scratch = CascadeScratch::new();
        let cascade = scratch.run(&[5], &rev);
        assert_eq!(cascade.affected, &[5]);
        assert_eq!(cascade.max_depth, 0);
        assert_eq!(cascade.depth(5), Some(0));
        assert_eq!(cascade.depth(4), None);
    }

    #[test]
    fn chain() {
        let rev = rev_from_adj(&[vec![1], vec![2], vec![3], vec![]]);
        let mut scratch = CascadeScratch::new();
        let cascade = scratch.run(&[0], &rev);
        assert_eq!(cascade.depth(0), Some(0));
        assert_eq!(cascade.depth(3), Some(3));
        assert_eq!(cascade.max_depth, 3);
        assert_eq!(cascade.affected.len(), 4);
    }

    #[test]
    fn diamond_takes_shortest_path() {
        let rev = rev_from_adj(&[vec![1, 2], vec![3], vec![3], vec![]]);
        let mut scratch = CascadeScratch::new();
        let cascade = scratch.run(&[0], &rev);
        assert_eq!(cascade.depth(3), Some(2));
    }

    #[test]
    fn dedupes_seeds() {
        let rev = rev_from_adj(&vec![Vec::new(); 10]);
        let mut scratch = CascadeScratch::new();
        let cascade = scratch.run(&[3, 3, 3], &rev);
        assert_eq!(cascade.affected, &[3]);
    }

    #[test]
    fn scratch_reuses_across_runs() {
        // Two unrelated runs on the same scratch must give independent results
        // — no leak from the first run's depths.
        let rev_a = rev_from_adj(&[vec![1], vec![2], vec![]]);
        let rev_b = rev_from_adj(&[vec![], vec![], vec![0]]);
        let mut scratch = CascadeScratch::new();

        let a_depths: Vec<u32> = {
            let cascade = scratch.run(&[0], &rev_a);
            (0..3)
                .map(|p| cascade.depth(p).unwrap_or(u32::MAX))
                .collect()
        };
        assert_eq!(a_depths, vec![0, 1, 2]);

        let b_depths: Vec<u32> = {
            let cascade = scratch.run(&[2], &rev_b);
            (0..3)
                .map(|p| cascade.depth(p).unwrap_or(u32::MAX))
                .collect()
        };
        assert_eq!(b_depths, vec![1, u32::MAX, 0]);
    }

    #[test]
    fn depth_saturates_at_255() {
        // Build a 300-node chain; deepest depth must not wrap.
        let mut adj = vec![Vec::new(); 300];
        for (i, dsts) in adj.iter_mut().enumerate().take(299) {
            dsts.push(i as u32 + 1);
        }
        let rev = rev_from_adj(&adj);
        let mut scratch = CascadeScratch::new();
        let cascade = scratch.run(&[0], &rev);
        // The true depth is 299 but we clamp to 255 in storage.
        assert!(cascade.max_depth >= 255);
        assert_eq!(cascade.depth(299), Some(255));
    }

    #[test]
    fn epoch_wraparound_is_clean() {
        // Run 300+ cascades on the same scratch to trigger the 8-bit epoch
        // wrap; a stale slot from the pre-wrap era must not masquerade as
        // visited in the post-wrap run.
        let rev = rev_from_adj(&[vec![1], vec![], vec![3], vec![]]);
        let mut scratch = CascadeScratch::new();
        // Alternate seeds so different slots are left dirty across runs.
        for i in 0..400 {
            let seed = if i & 1 == 0 { 0 } else { 2 };
            let _ = scratch.run(&[seed], &rev);
        }
        // Final sanity: a seed at the "other" position still visits cleanly.
        let c = scratch.run(&[2], &rev);
        assert_eq!(c.depth(2), Some(0));
        assert_eq!(c.depth(3), Some(1));
        assert_eq!(c.depth(0), None);
        assert_eq!(c.depth(1), None);
    }
}
