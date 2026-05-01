//! End-to-end test: load a real PNG, decode its IDAT stream, and verify the
//! library's public surface produces sensible output.

use std::collections::VecDeque;
use std::path::PathBuf;

use pngbend::bitstream::{read_bits_at, write_bits};
use pngbend::deflate::{Event, decode_deflate};
use pngbend::index::{build_pos_to_ev, build_reverse_graph, valid_dist_alts};
use pngbend::png::{
    concat_idat, inflate_raw, parse_ihdr, read_chunks, unfilter, unfilter_rows_into,
};

fn sample_path() -> PathBuf {
    let manifest =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set under cargo test");
    PathBuf::from(manifest)
        .join("..")
        .join("samples")
        .join("checksum_sunset.png")
}

/// Read the integration-test sample, or skip the test cleanly if it isn't
/// on disk. Lets `cargo test` pass on a fresh clone without samples — the
/// per-test logic still runs whenever a sample is present.
fn read_sample_or_skip(test_name: &str) -> Option<Vec<u8>> {
    let path = sample_path();
    match std::fs::read(&path) {
        Ok(raw) => Some(raw),
        Err(e) => {
            eprintln!(
                "skipping {test_name}: cannot read {} ({e}). Drop a PNG at that path to run the full integration suite.",
                path.display()
            );
            None
        }
    }
}

#[test]
fn loads_sample_png_end_to_end() {
    let Some(raw) = read_sample_or_skip("loads_sample_png_end_to_end") else {
        return;
    };

    let chunks = read_chunks(&raw);
    assert!(!chunks.is_empty(), "no PNG chunks parsed");
    assert!(
        chunks.iter().any(|c| &c.typ == b"IHDR"),
        "no IHDR chunk found"
    );
    assert!(
        chunks.iter().any(|c| &c.typ == b"IDAT"),
        "no IDAT chunk found"
    );

    let info = parse_ihdr(&chunks).expect("parse IHDR");
    assert!(info.width > 0 && info.height > 0);
    assert!(info.bpp >= 1 && info.bpp <= 4);

    // Concatenate IDAT, strip 2-byte zlib header + 4-byte Adler trailer.
    let idat = concat_idat(&chunks);
    assert!(idat.len() > 6, "IDAT too short");
    let deflate = &idat[2..idat.len() - 4];

    let decoded = decode_deflate(deflate).expect("decode deflate");
    let expected_min = info.height as usize * info.row_stride;
    assert_eq!(
        decoded.output.len(),
        expected_min,
        "output length should equal h * (1 + w*bpp)"
    );
    assert!(decoded.num_blocks >= 1);
    assert_eq!(decoded.lit_encs.len(), decoded.num_blocks);
    assert_eq!(decoded.dist_encs.len(), decoded.num_blocks);

    let lit_count = decoded
        .events
        .iter()
        .filter(|e| matches!(e, Event::Lit(_)))
        .count();
    let ref_count = decoded
        .events
        .iter()
        .filter(|e| matches!(e, Event::Ref(_)))
        .count();
    assert!(lit_count > 0, "expected at least one literal event");
    // A typical photo PNG is highly back-referenced; at least one Ref is
    // a sane lower bound for a real-world sample.
    assert!(ref_count > 0, "expected at least one back-ref event");

    // pos_to_ev and reverse_graph should also build cleanly.
    let pos_to_ev = build_pos_to_ev(&decoded.events, decoded.output.len());
    assert_eq!(pos_to_ev.len(), decoded.output.len());

    let rev = build_reverse_graph(&decoded.events, decoded.output.len());
    assert_eq!(rev.len(), decoded.output.len());

    // inflate_raw returns the same bytes as decoded.output.
    let inflated = inflate_raw(deflate).expect("inflate_raw");
    assert_eq!(inflated, decoded.output);
}

#[test]
fn malformed_deflate_returns_error_not_panic() {
    // 3 bytes of all-ones — not a valid deflate stream.
    let garbage = vec![0xFFu8; 3];
    let result = decode_deflate(&garbage);
    assert!(result.is_err(), "expected an error from corrupt deflate");
}

/// Save round-trip: rebuild a PNG by replacing IDAT with a freshly
/// constructed zlib stream and re-emitting every chunk via
/// `write_chunks`. The saved bytes must re-read to the same decoded
/// output as the original. Covers the end-to-end save path
/// (`build_zlib_stream` + IDAT replacement + `write_chunks` + CRC) that
/// the unit tests check piecewise but no other integration test
/// exercises as a whole.
#[test]
fn save_and_reread_unedited_png_round_trips() {
    use pngbend::png::{Chunk, build_zlib_stream, write_chunks};

    let Some(raw) = read_sample_or_skip("save_and_reread_unedited_png_round_trips") else {
        return;
    };
    let chunks_orig = read_chunks(&raw);
    let info = parse_ihdr(&chunks_orig).expect("parse IHDR");
    let idat = concat_idat(&chunks_orig);
    let zlib_header = [idat[0], idat[1]];
    let deflate_buf = idat[2..idat.len() - 4].to_vec();
    let decoded = decode_deflate(&deflate_buf).expect("decode");

    // Same shape as `app::edit::PngBendApp::assemble_png_bytes`: collapse
    // every IDAT into one rebuilt entry, copy other chunks through.
    let zlib = build_zlib_stream(&deflate_buf, &zlib_header, &decoded.output);
    let out_chunks: Vec<Chunk> = chunks_orig
        .iter()
        .scan(false, |seen, c| {
            if &c.typ == b"IDAT" {
                if *seen {
                    return Some(None);
                }
                *seen = true;
                Some(Some(Chunk {
                    typ: *b"IDAT",
                    data: zlib.clone(),
                }))
            } else {
                Some(Some(Chunk {
                    typ: c.typ,
                    data: c.data.clone(),
                }))
            }
        })
        .flatten()
        .collect();
    let saved = write_chunks(&out_chunks);

    // Re-read the saved bytes and decode again. The output bytes must
    // match the original — anything else means the save chain dropped
    // or corrupted something.
    let chunks_re = read_chunks(&saved);
    let info_re = parse_ihdr(&chunks_re).expect("re-parse IHDR");
    assert_eq!(info_re.width, info.width);
    assert_eq!(info_re.height, info.height);

    let idat_re = concat_idat(&chunks_re);
    assert!(idat_re.len() > 6, "saved IDAT too short");
    let deflate_re = &idat_re[2..idat_re.len() - 4];
    let decoded_re = decode_deflate(deflate_re).expect("re-decode saved bytes");
    assert_eq!(
        decoded_re.output, decoded.output,
        "saved-and-reread output must match original"
    );
}

/// The fast path in `app::edit::apply_literal_swap_incremental` assumes
/// that patching `output[lit.out_pos]` to the new symbol and propagating
/// that byte through every `reverse_graph` descendant yields the same
/// bytes as re-running `decode_deflate` on the patched bit stream.
///
/// This is load-bearing — if the assumption ever breaks, edits silently
/// corrupt the rendered image. So we build the full incremental state
/// and compare it byte-for-byte against a fresh decode.
#[test]
fn incremental_literal_swap_matches_full_decode() {
    let Some(raw) = read_sample_or_skip("incremental_literal_swap_matches_full_decode") else {
        return;
    };
    let chunks = read_chunks(&raw);
    let idat = concat_idat(&chunks);
    let mut deflate_buf = idat[2..idat.len() - 4].to_vec();

    let decoded = decode_deflate(&deflate_buf).expect("initial decode");
    let reverse_graph = build_reverse_graph(&decoded.events, decoded.output.len());

    // Find a literal with a same-length swap alternative we can apply.
    let (swap_bit, swap_len, new_code, out_pos, new_symbol) = decoded
        .events
        .iter()
        .find_map(|e| match e {
            Event::Lit(lit) => {
                let le = &decoded.lit_encs[lit.block as usize];
                let (_, cur_clen) = le.get(lit.symbol as u16)?;
                le.iter()
                    .find(|(s, _, clen)| *s < 256 && *s != lit.symbol as u16 && *clen == cur_clen)
                    .map(|(sym, code, _)| (lit.bit_start, cur_clen, code, lit.out_pos, sym as u8))
            }
            _ => None,
        })
        .expect("expected at least one swappable literal in the sample");

    // Patch the bit stream exactly as `apply_patches_capturing_prior` would.
    write_bits(
        &mut deflate_buf,
        swap_bit as usize,
        new_code as u32,
        swap_len,
    );

    // Incremental path: patch the one output byte, then propagate via
    // reverse_graph just like `apply_literal_swap_incremental` does.
    let mut incremental = decoded.output.clone();
    incremental[out_pos as usize] = new_symbol;
    let mut frontier: VecDeque<u32> = VecDeque::new();
    frontier.push_back(out_pos);
    while let Some(pos) = frontier.pop_front() {
        for &dst in reverse_graph.neighbors(pos) {
            incremental[dst as usize] = new_symbol;
            frontier.push_back(dst);
        }
    }

    // Authoritative: decode the patched stream from scratch.
    let reference = decode_deflate(&deflate_buf)
        .expect("patched stream should still decode (same-length Huffman swap)")
        .output;

    assert_eq!(
        incremental.len(),
        reference.len(),
        "lengths must match after a same-length literal swap"
    );
    // Find the first mismatch if any, for a useful failure message.
    for (i, (a, b)) in incremental.iter().zip(reference.iter()).enumerate() {
        assert_eq!(
            a, b,
            "byte {i} diverges: incremental={a:#04x} reference={b:#04x}"
        );
    }
}

/// The row-scoped unfilter that the redirect path runs (in
/// `app::edit::apply_dist_redirect_incremental`'s step 6) reuses the
/// pre-edit `unfiltered` buffer for every row outside the diff range.
/// That's correct iff `unfilter_rows_into` chains through filter-2/3/4
/// downstream rows — those filter types read the prior row, so a
/// silently-skipped row leaves every later pixel drifted. This test
/// compares the row-scoped result against a full from-scratch unfilter
/// of the post-edit output.
#[test]
fn incremental_redirect_unfilter_matches_full_decode() {
    let Some(raw) = read_sample_or_skip("incremental_redirect_unfilter_matches_full_decode") else {
        return;
    };
    let chunks = read_chunks(&raw);
    let info = parse_ihdr(&chunks).expect("ihdr");
    let idat = concat_idat(&chunks);
    let mut deflate_buf = idat[2..idat.len() - 4].to_vec();

    let decoded = decode_deflate(&deflate_buf).expect("initial decode");

    // Find a back-ref with at least one valid redirect target via the
    // same helper the app uses, then grab the new Huffman code out of
    // the dist EncTable.
    let (swap_bit, swap_len, new_code, affected_from) = decoded
        .events
        .iter()
        .find_map(|e| match e {
            Event::Ref(r) => {
                let alts = valid_dist_alts(
                    r.block,
                    r.dist_sym,
                    r.out_pos,
                    r.src_out_pos,
                    &decoded.dist_encs,
                );
                let &(new_sym, _new_src, _new_dist) = alts.first()?;
                let de = &decoded.dist_encs[r.block as usize];
                let (new_code, new_clen) = de.get(new_sym as u16)?;
                Some((r.dist_bit_start, new_clen, new_code, r.out_pos))
            }
            _ => None,
        })
        .expect("no redirect candidate in sample");

    // Full unfilter of the *pre-edit* output — this is what `CoreData`
    // holds across edits, and what the redirect path reuses.
    let mut unfiltered_cache = unfilter(&decoded.output, &info).expect("initial unfilter");

    // Apply the redirect patch, re-decode — this is the new reference.
    write_bits(
        &mut deflate_buf,
        swap_bit as usize,
        new_code as u32,
        swap_len,
    );
    let decoded_after = decode_deflate(&deflate_buf).expect("decode after redirect");
    let reference = unfilter(&decoded_after.output, &info).expect("reference unfilter");

    // Incremental: diff old vs new output starting at `affected_from`,
    // then call `unfilter_rows_into` with the diffed row range — the
    // same shape `apply_dist_redirect_incremental` produces.
    let old_output = decoded.output;
    let new_output = &decoded_after.output;
    let row_stride = info.row_stride;
    let h = info.height as usize;

    let mut first = affected_from as usize;
    while first < old_output.len() && old_output[first] == new_output[first] {
        first += 1;
    }
    assert!(first < old_output.len(), "redirect must change some byte");
    let mut last = old_output.len();
    while last > first && old_output[last - 1] == new_output[last - 1] {
        last -= 1;
    }
    let first_row = (first / row_stride).min(h - 1);
    let last_row = ((last - 1) / row_stride).min(h - 1);
    let mut row_touched = vec![false; h];
    for touched in row_touched.iter_mut().take(last_row + 1).skip(first_row) {
        *touched = true;
    }
    unfilter_rows_into(
        new_output,
        &info,
        &mut unfiltered_cache,
        first_row,
        |y| row_touched[y],
        |_| {},
    )
    .expect("unfilter_rows redirect");

    assert_eq!(
        unfiltered_cache.len(),
        reference.len(),
        "row-scoped unfilter must preserve buffer length"
    );
    for (i, (a, b)) in unfiltered_cache.iter().zip(reference.iter()).enumerate() {
        assert_eq!(
            a, b,
            "row-scoped unfilter diverges at byte {i}: incremental={a:#04x} full={b:#04x}"
        );
    }
}

/// The undo invariant the entire history stack rests on: applying N edits
/// to `deflate_buf` then undoing them in reverse must restore it byte-for-
/// byte. The proptest in `app::edit::tests::forward_then_inverse_restores_buffer`
/// covers the bit-level primitive on synthetic inputs; this test exercises
/// the same primitive on a real PNG's deflate stream against patches built
/// from real `EncTable`s — the path that actually runs in the editor.
#[test]
fn apply_then_undo_n_literal_swaps_restores_deflate_buf() {
    let Some(raw) = read_sample_or_skip("apply_then_undo_n_literal_swaps_restores_deflate_buf")
    else {
        return;
    };
    let chunks = read_chunks(&raw);
    let idat = concat_idat(&chunks);
    let original_deflate = idat[2..idat.len() - 4].to_vec();
    let mut deflate_buf = original_deflate.clone();

    let decoded = decode_deflate(&deflate_buf).expect("initial decode");

    // Pick the first 4 swappable literals — distinct events with same-
    // length Huffman alternatives. Mirrors `find_literal_swaps` in the
    // profile binary.
    let swaps: Vec<(u32, u8, u16)> = decoded
        .events
        .iter()
        .filter_map(|e| {
            let Event::Lit(lit) = e else { return None };
            let le = &decoded.lit_encs[lit.block as usize];
            let (_, cur_clen) = le.get(lit.symbol as u16)?;
            let (_, new_code, _) = le
                .iter()
                .find(|(s, _, clen)| *s < 256 && *s != lit.symbol as u16 && *clen == cur_clen)?;
            Some((lit.bit_start, cur_clen, new_code))
        })
        .take(4)
        .collect();
    assert_eq!(swaps.len(), 4, "expected ≥ 4 swappable literals in sample");

    // Forward: apply each swap, capturing the bits we overwrote so we can
    // put them back. Order matters when patches overlap, but consecutive
    // literal swaps in the same stream don't (each has a unique
    // `bit_start`); we still record + replay LIFO to mirror the undo
    // stack's invariant.
    let mut undo_stack: Vec<(u32, u32, u8)> = Vec::with_capacity(swaps.len());
    for &(bit, code_len, new_code) in &swaps {
        let bs = bit as usize;
        let prev = read_bits_at(&deflate_buf, bs, code_len);
        write_bits(&mut deflate_buf, bs, new_code as u32, code_len);
        undo_stack.push((bit, prev, code_len));
    }
    assert_ne!(
        deflate_buf, original_deflate,
        "edits must change the stream"
    );

    // Undo LIFO.
    while let Some((bit, prev_value, code_len)) = undo_stack.pop() {
        write_bits(&mut deflate_buf, bit as usize, prev_value, code_len);
    }

    assert_eq!(
        deflate_buf, original_deflate,
        "deflate_buf must be byte-identical to the original after a full undo"
    );
}

/// State-equivalence test for the new surgical-redirect path.
///
/// `apply_dist_redirect_incremental` skips `decode_deflate` and updates
/// `output` via "recopy from new src + propagate via reverse_graph". This
/// must produce the same `output` bytes that a fresh `decode_deflate`
/// would on the patched bit stream — otherwise downstream renders silently
/// drift. The redirect is the surgical analogue of the literal-swap path
/// that `incremental_literal_swap_matches_full_decode` already covers.
#[test]
fn surgical_redirect_output_matches_full_decode() {
    let Some(raw) = read_sample_or_skip("surgical_redirect_output_matches_full_decode") else {
        return;
    };
    let chunks = read_chunks(&raw);
    let idat = concat_idat(&chunks);
    let mut deflate_buf = idat[2..idat.len() - 4].to_vec();

    let decoded = decode_deflate(&deflate_buf).expect("initial decode");
    let reverse_graph = build_reverse_graph(&decoded.events, decoded.output.len());

    // Find a back-ref with a redirect target. Same pattern as the
    // existing redirect test, with the additional payload we need to
    // mimic the surgical path (out_pos, copy_len, new_src).
    let (swap_bit, swap_len, new_code, out_pos, copy_len, new_src) = decoded
        .events
        .iter()
        .find_map(|e| match e {
            Event::Ref(r) => {
                let alts = valid_dist_alts(
                    r.block,
                    r.dist_sym,
                    r.out_pos,
                    r.src_out_pos,
                    &decoded.dist_encs,
                );
                let &(new_sym, new_src, _new_dist) = alts.first()?;
                let de = &decoded.dist_encs[r.block as usize];
                let (new_code, new_clen) = de.get(new_sym as u16)?;
                Some((
                    r.dist_bit_start as usize,
                    new_clen,
                    new_code,
                    r.out_pos as usize,
                    r.copy_len as usize,
                    new_src as usize,
                ))
            }
            _ => None,
        })
        .expect("no redirect candidate in sample");

    // Surgical: recopy output[out_pos..+copy_len] from new_src, then BFS
    // through the *existing* reverse_graph from each of those positions.
    let mut surgical = decoded.output.clone();
    for off in 0..copy_len {
        surgical[out_pos + off] = surgical[new_src + off];
    }
    let mut frontier: VecDeque<u32> = VecDeque::new();
    for off in 0..copy_len {
        frontier.push_back((out_pos + off) as u32);
    }
    while let Some(pos) = frontier.pop_front() {
        let val = surgical[pos as usize];
        for &dst in reverse_graph.neighbors(pos) {
            surgical[dst as usize] = val;
            frontier.push_back(dst);
        }
    }

    // Authoritative: patch the bitstream + decode from scratch.
    write_bits(&mut deflate_buf, swap_bit, new_code as u32, swap_len);
    let reference = decode_deflate(&deflate_buf)
        .expect("patched stream should still decode (same-clen redirect)")
        .output;

    assert_eq!(
        surgical.len(),
        reference.len(),
        "lengths must match after a same-clen redirect"
    );
    for (i, (a, b)) in surgical.iter().zip(reference.iter()).enumerate() {
        assert_eq!(
            a, b,
            "byte {i} diverges: surgical={a:#04x} reference={b:#04x}"
        );
    }
}

/// Regression test for the surgical-redirect overlap bug.
///
/// LZ77 allows refs whose source range OVERLAPS the destination
/// (`src + copy_len > out_pos`) — that's how the format encodes
/// run-length patterns. For an overlap ref, `events[i]`'s reverse-graph
/// edges include positions inside `[out_pos, out_pos+copy_len)` as
/// *sources*. When a redirect moves `src`, those intra-destination
/// edges shift; a BFS that walks them on the **pre-rebuild** graph
/// overwrites correctly recopied bytes with stale values.
///
/// This was caught by a user noticing `apply → undo` produced a
/// different glitch than the original edit. The non-overlap test
/// `surgical_redirect_output_matches_full_decode` missed it because
/// its first-redirect-candidate happened to be non-overlap.
#[test]
fn surgical_redirect_apply_undo_round_trip_with_overlap() {
    let Some(raw) = read_sample_or_skip("surgical_redirect_apply_undo_round_trip_with_overlap")
    else {
        return;
    };
    let chunks = read_chunks(&raw);
    let idat = concat_idat(&chunks);
    let deflate_buf = idat[2..idat.len() - 4].to_vec();
    let decoded = decode_deflate(&deflate_buf).expect("initial decode");
    let original_output = decoded.output.clone();

    // Find a redirect target where the ref OVERLAPS its destination —
    // i.e. `src_out_pos + copy_len > out_pos`. Those are the cases the
    // bug shows up on.
    let target = decoded
        .events
        .iter()
        .enumerate()
        .find_map(|(idx, e)| match e {
            Event::Ref(r) if r.src_out_pos as usize + r.copy_len as usize > r.out_pos as usize => {
                let alts = valid_dist_alts(
                    r.block,
                    r.dist_sym,
                    r.out_pos,
                    r.src_out_pos,
                    &decoded.dist_encs,
                );
                let &(_, new_src, _) = alts.first()?;
                Some((
                    idx,
                    r.out_pos as usize,
                    r.copy_len as usize,
                    r.src_out_pos,
                    new_src,
                ))
            }
            _ => None,
        });
    let Some((event_idx, out_pos, copy_len, src_orig, new_src)) = target else {
        // No overlap candidate in this sample — test trivially passes.
        // The non-overlap path is covered by the other test.
        return;
    };

    let mut output = decoded.output.clone();
    let mut events = decoded.events.clone();
    let mut rev = build_reverse_graph(&events, output.len());

    // Mirrors the fixed `apply_dist_redirect_incremental` order:
    //   mutate event → rebuild rev → recopy → BFS over fresh rev.
    // The rebuild has to happen before the BFS for overlap refs (where
    // some of the ref's edges land within its destination range).
    let redirect_step = |events: &mut Vec<Event>,
                         output: &mut Vec<u8>,
                         rev: &mut pngbend::index::ReverseGraph,
                         new_src: u32| {
        if let Event::Ref(r) = &mut events[event_idx] {
            r.src_out_pos = new_src;
        }
        *rev = build_reverse_graph(events, output.len());
        let new_src_us = new_src as usize;
        for off in 0..copy_len {
            output[out_pos + off] = output[new_src_us + off];
        }
        let mut frontier: VecDeque<u32> = VecDeque::new();
        for off in 0..copy_len {
            frontier.push_back((out_pos + off) as u32);
        }
        while let Some(pos) = frontier.pop_front() {
            let val = output[pos as usize];
            for &dst in rev.neighbors(pos) {
                output[dst as usize] = val;
                frontier.push_back(dst);
            }
        }
    };

    redirect_step(&mut events, &mut output, &mut rev, new_src);
    let post_forward_output = output.clone();
    redirect_step(&mut events, &mut output, &mut rev, src_orig);

    assert_eq!(
        output.len(),
        original_output.len(),
        "lengths must match after a same-clen forward+undo round trip"
    );
    for (i, (a, b)) in output.iter().zip(original_output.iter()).enumerate() {
        assert_eq!(
            a, b,
            "byte {i} drifted: round-trip={a:#04x} original={b:#04x} (post-forward was {:#04x}); event_idx={event_idx} out_pos={out_pos} copy_len={copy_len} src_orig={src_orig} new_src={new_src}",
            post_forward_output[i],
        );
    }
}

/// Synthetic overlap regression. Build an `events` list by hand with a
/// ref that overlaps its destination (`src + len > out_pos`) and where
/// the period of the resulting run-length pattern *changes* under the
/// redirect — i.e. the pre-edit overlap produces an `a,b,c,a,b,c,…`
/// pattern (period 3) and the post-edit produces something different
/// (period 4). On those, the BFS using a stale `rev_graph` writes wrong
/// values into the destination range; period-2-in-both-directions
/// candidates from real photos can mask the bug because every wrong-edge
/// write coincidentally lands on a correct value.
///
/// This test bypasses the deflate decoder and constructs the events +
/// output state directly so we can exercise an exact period mismatch.
#[test]
fn surgical_redirect_overlap_period_mismatch_round_trip() {
    use pngbend::deflate::{LitEvent, RefEvent};
    use pngbend::index::ReverseGraph;

    // Output layout (32 bytes for headroom):
    //   indices 0..6 are literal events (a, b, c, p, q, r, ...)
    //   index   6..14: a single ref. src_orig=3, copy_len=8 → period 3
    //                  pattern p,q,r,p,q,r,p,q.
    //   index   14..18: a downstream ref of length 4 sourced from
    //                   indices 6..10, exercising "external descendants
    //                   that depend on the redirected ref's bytes".
    let n_lit: u32 = 6;
    let mut events: Vec<Event> = (0..n_lit)
        .map(|i| {
            Event::Lit(LitEvent {
                out_pos: i,
                symbol: 0,
                bit_start: i * 8,
                block: 0,
            })
        })
        .collect();
    // Ref event 6: src=3 (=> period 3), copy_len=8, out_pos=6.
    events.push(Event::Ref(RefEvent {
        out_pos: n_lit,
        src_out_pos: 3,
        copy_len: 8,
        dist_sym: 0,
        dist_bit_start: n_lit * 8,
        block: 0,
    }));
    // Ref event 7: downstream, src=6 (= start of dest range), len=4, out_pos=14.
    //   This makes positions 14..18 transitive descendants of 6..10
    //   *outside* the destination range, so any wrong write inside
    //   [6..14] propagates here.
    events.push(Event::Ref(RefEvent {
        out_pos: n_lit + 8,
        src_out_pos: n_lit,
        copy_len: 4,
        dist_sym: 0,
        dist_bit_start: 0,
        block: 0,
    }));

    // Pre-edit output (constructed from the event chain by hand).
    // Literals at 0..6: distinct values so period mismatches show.
    let mut output: Vec<u8> = vec![10, 20, 30, 40, 50, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    // Ref at 6: src=3, len=8, period 3 (out_pos - src = 3).
    //   o[6]=o[3]=40, o[7]=o[4]=50, o[8]=o[5]=60,
    //   o[9]=o[6]=40, o[10]=o[7]=50, o[11]=o[8]=60,
    //   o[12]=o[9]=40, o[13]=o[10]=50.
    for off in 0..8 {
        output[6 + off] = output[3 + off];
    }
    // Ref at 14: src=6, len=4 → o[14]=o[6]=40, o[15]=o[7]=50, o[16]=o[8]=60, o[17]=o[9]=40.
    for off in 0..4 {
        output[14 + off] = output[6 + off];
    }
    let original = output.clone();

    let mut rev = build_reverse_graph(&events, output.len());
    let event_idx: usize = n_lit as usize; // index of the redirected ref
    let out_pos: usize = n_lit as usize;
    let copy_len = 8;
    let src_orig: u32 = 3;
    // New src=2 → period 4 (out_pos - new_src = 4).
    let new_src: u32 = 2;

    let redirect_step =
        |events: &mut Vec<Event>, output: &mut Vec<u8>, rev: &mut ReverseGraph, new_src: u32| {
            if let Event::Ref(r) = &mut events[event_idx] {
                r.src_out_pos = new_src;
            }
            // Mirrors the fixed production order in
            // `apply_dist_redirect_incremental`: rebuild rev BEFORE the
            // recopy + BFS so the BFS sees post-edit topology. With a
            // stale rev, an overlap ref's intra-destination edges differ
            // from the post-edit ones and the BFS overwrites correctly
            // recopied bytes — the bug this test was added to catch.
            *rev = build_reverse_graph(events, output.len());
            let new_src_us = new_src as usize;
            for off in 0..copy_len {
                output[out_pos + off] = output[new_src_us + off];
            }
            let mut frontier: VecDeque<u32> = VecDeque::new();
            for off in 0..copy_len {
                frontier.push_back((out_pos + off) as u32);
            }
            while let Some(pos) = frontier.pop_front() {
                let val = output[pos as usize];
                for &dst in rev.neighbors(pos) {
                    output[dst as usize] = val;
                    frontier.push_back(dst);
                }
            }
        };

    redirect_step(&mut events, &mut output, &mut rev, new_src);
    redirect_step(&mut events, &mut output, &mut rev, src_orig);

    assert_eq!(
        output,
        original,
        "apply→undo round trip must restore output. \
         Diverges at: {:?} vs {:?}",
        &output[6..18],
        &original[6..18],
    );
}
