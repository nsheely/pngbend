//! Greedy LZ77 match-finder for the encoder.
//!
//! Hash chains over 3-byte prefixes locate back-references within the 32 KiB
//! window; each position emits a back-reference (match length >= 3) or a
//! literal, producing an [`Event`] stream for the serializers.

use super::constants::{DBASE, symbol_index};
use super::events::{Event, LitEvent, RefEvent};

const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const WINDOW: usize = 32768;
const HASH_BITS: usize = 15;
const HASH_SIZE: usize = 1 << HASH_BITS;
/// Cap on hash-chain walk length per position: bounds worst-case time on
/// pathological inputs at a small ratio cost.
const MAX_CHAIN: usize = 128;

#[inline]
fn hash3(d: &[u8], i: usize) -> usize {
    ((d[i] as usize) << 10 ^ (d[i + 1] as usize) << 5 ^ (d[i + 2] as usize)) & (HASH_SIZE - 1)
}

/// Distance symbol (0..=29) for a back-reference distance (1..=32768).
fn distance_to_sym(dist: u32) -> u8 {
    symbol_index(&DBASE, dist) as u8
}

/// Greedy LZ77 over `data`: emit a back-reference at each position with a
/// match of length >= 3 in the window, else a literal.
pub(super) fn lz77(data: &[u8]) -> Vec<Event> {
    let mut events = Vec::new();
    if data.len() < MIN_MATCH {
        for (i, &b) in data.iter().enumerate() {
            events.push(Event::Lit(LitEvent {
                out_pos: i as u32,
                bit_start: 0,
                symbol: b,
            }));
        }
        return events;
    }
    let mut head = vec![u32::MAX; HASH_SIZE];
    let mut prev = vec![u32::MAX; data.len()];

    let insert = |i: usize, head: &mut [u32], prev: &mut [u32]| {
        if i + MIN_MATCH <= data.len() {
            let h = hash3(data, i);
            prev[i] = head[h];
            head[h] = i as u32;
        }
    };

    let mut i = 0;
    while i < data.len() {
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        if i + MIN_MATCH <= data.len() {
            let max_len = (data.len() - i).min(MAX_MATCH);
            let mut cand = head[hash3(data, i)];
            let mut chain = 0;
            while cand != u32::MAX && chain < MAX_CHAIN {
                let c = cand as usize;
                if i - c > WINDOW {
                    break;
                }
                let mut l = 0;
                while l < max_len && data[c + l] == data[i + l] {
                    l += 1;
                }
                if l > best_len {
                    best_len = l;
                    best_dist = i - c;
                    if l >= max_len {
                        break;
                    }
                }
                cand = prev[c];
                chain += 1;
            }
        }
        if best_len >= MIN_MATCH {
            events.push(Event::Ref(RefEvent {
                out_pos: i as u32,
                src_out_pos: (i - best_dist) as u32,
                dist_bit_start: 0,
                copy_len: best_len as u16,
                dist_sym: distance_to_sym(best_dist as u32),
            }));
            for j in i..i + best_len {
                insert(j, &mut head, &mut prev);
            }
            i += best_len;
        } else {
            events.push(Event::Lit(LitEvent {
                out_pos: i as u32,
                bit_start: 0,
                symbol: data[i],
            }));
            insert(i, &mut head, &mut prev);
            i += 1;
        }
    }
    events
}
