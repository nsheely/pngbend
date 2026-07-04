//! Map an output-byte offset to the event that produced it.
//!
//! Two flavours:
//! - [`event_at`]: `O(log N)` binary search over the events list, for
//!   sparse runtime queries (a few per click, a handful per visible
//!   sidebar row).
//! - [`build_pos_to_ev`]: dense `Vec<u32>` indexed by byte. Builds in
//!   `O(events)`, answers in `O(1)`, for when the same lookup runs for
//!   every byte, as `build_pixel_index` does during load.
//!
//! The decoder emits events in strictly increasing `out_pos` order and
//! each output range (`out_pos..out_pos + len`) abuts the next event's
//! start, so a binary search by `out_pos` plus a range check resolves any
//! byte to its single owning event.

use crate::deflate::Event;

/// Sentinel in the dense [`build_pos_to_ev`] map: "no event wrote this
/// output position". A well-formed stream claims every byte with exactly
/// one event, so the sentinel only surfaces on truncated or malformed
/// input.
const POS_UNSET: u32 = u32::MAX;

/// Index of the event that produced output byte `pos`, or `None` if no
/// event covers it (e.g. truncated input). `O(log events.len())`.
///
/// `events` must be sorted by `out_pos`; the decoder emits them that way.
pub fn event_at(events: &[Event], pos: usize) -> Option<u32> {
    if events.is_empty() {
        return None;
    }
    let pos_u32 = u32::try_from(pos).ok()?;
    // partition_point gives the smallest index whose `out_pos > pos`. The
    // event we want is the one immediately before that, if any.
    let next = events.partition_point(|e| event_out_pos(e) <= pos_u32);
    let i = next.checked_sub(1)?;
    // SAFETY/RANGE: i < events.len() because next > 0 here.
    let e = &events[i];
    let len = match e {
        Event::Lit(_) => 1,
        Event::Ref(r) => r.copy_len as u32,
    };
    let end = event_out_pos(e) + len;
    (pos_u32 < end).then_some(i as u32)
}

#[inline]
fn event_out_pos(e: &Event) -> u32 {
    match e {
        Event::Lit(l) => l.out_pos,
        Event::Ref(r) => r.out_pos,
    }
}

/// Dense byte → event-index map. `out[i]` is the event that wrote byte
/// `i`, or `POS_UNSET` for unwritten positions. Built once during load
/// and consumed by [`super::build_pixel_index`]; not kept at runtime.
pub fn build_pos_to_ev(events: &[Event], output_len: usize) -> Vec<u32> {
    let mut pos_to_ev = vec![POS_UNSET; output_len];
    // Soundness gate for the unchecked writes below (same rationale as
    // [`super::build_reverse_graph`]). Every write must land within
    // `output_len`; the decoder guarantees it, but `Event`'s fields are
    // public, so a caller could pass an out-of-range event from safe code.
    // On violation return the unwritten map rather than risk an out-of-bounds
    // write. One sequential pass, cheap next to the fill below.
    let in_bounds = events.iter().all(|e| match e {
        Event::Lit(l) => (l.out_pos as usize) < output_len,
        Event::Ref(r) => (r.out_pos as usize).saturating_add(r.copy_len as usize) <= output_len,
    });
    if !in_bounds {
        return pos_to_ev;
    }
    for (i, e) in events.iter().enumerate() {
        match e {
            Event::Lit(lit) => {
                let out_pos = lit.out_pos as usize;
                debug_assert_eq!(pos_to_ev[out_pos], POS_UNSET, "events must not overlap");
                // SAFETY: the gate above proved `out_pos < output_len`.
                unsafe {
                    *pos_to_ev.get_unchecked_mut(out_pos) = i as u32;
                }
            }
            Event::Ref(r) => {
                let start = r.out_pos as usize;
                let end = start + r.copy_len as usize;
                // SAFETY: the gate above proved `end ≤ output_len = len`.
                let span = unsafe { pos_to_ev.get_unchecked_mut(start..end) };
                span.fill(i as u32);
            }
        }
    }
    pos_to_ev
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate::{LitEvent, RefEvent};

    fn lit(out_pos: u32) -> Event {
        Event::Lit(LitEvent {
            out_pos,
            symbol: 0,
            bit_start: 0,
        })
    }

    fn refe(out_pos: u32, copy_len: u16) -> Event {
        Event::Ref(RefEvent {
            out_pos,
            src_out_pos: 0,
            copy_len,
            dist_sym: 0,
            dist_bit_start: 0,
        })
    }

    #[test]
    fn event_at_matches_dense_map() {
        // Mixed stream: literals then a ref then a literal. event_at and
        // build_pos_to_ev must agree at every covered byte.
        let events = vec![lit(0), lit(1), refe(2, 4), lit(6)];
        let dense = build_pos_to_ev(&events, 7);
        for (pos, &expected) in dense.iter().enumerate() {
            assert_eq!(event_at(&events, pos), Some(expected));
        }
    }

    #[test]
    fn out_of_range_event_yields_unwritten_map_without_ub() {
        // Event positions are public, so safe code can build an event that
        // writes past output_len. The gate must return the all-unset map
        // rather than index out of bounds (UB in release with the unchecked
        // writes).
        assert_eq!(build_pos_to_ev(&[lit(10_000)], 4), vec![POS_UNSET; 4]);
        assert_eq!(build_pos_to_ev(&[refe(2, 100)], 4), vec![POS_UNSET; 4]);
    }

    #[test]
    fn event_at_returns_none_past_end() {
        let events = vec![lit(0), refe(1, 3)];
        assert_eq!(event_at(&events, 4), None);
        assert_eq!(event_at(&events, 100), None);
    }

    #[test]
    fn event_at_handles_empty_events() {
        assert_eq!(event_at(&[], 0), None);
    }

    #[test]
    fn event_at_returns_index_for_first_byte_of_ref() {
        // The ref at out_pos=10, len=5 must answer for every byte in 10..15.
        let events = vec![refe(10, 5)];
        for pos in 10..15 {
            assert_eq!(event_at(&events, pos), Some(0), "pos={pos}");
        }
        // Just past the ref → None.
        assert_eq!(event_at(&events, 15), None);
        // Before the ref → None.
        assert_eq!(event_at(&events, 9), None);
    }
}
