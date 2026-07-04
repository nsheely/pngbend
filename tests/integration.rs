//! End-to-end tests: load a real PNG, decode its IDAT stream, verify the
//! library's public surface produces sensible output.

use std::collections::VecDeque;
use std::path::PathBuf;

use pngbend::bitstream::{read_bits_at, write_bits};
use pngbend::deflate::{Event, block_of, decode_deflate, inflate};
use pngbend::index::{build_pos_to_ev, build_reverse_graph, valid_dist_alts};
use pngbend::png::{
    ChunkType, Warning, concat_idat, parse_ihdr, read_chunks, unfilter, unfilter_rows_into,
};

fn sample_path() -> PathBuf {
    let manifest =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set under cargo test");
    PathBuf::from(manifest)
        .join("samples")
        .join("checksum_sunset.png")
}

/// Read the integration-test sample, or skip cleanly if it isn't on
/// disk, so `cargo test` passes on a fresh clone without samples.
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

    let chunks = read_chunks(&raw).expect("read chunks").chunks;
    assert!(!chunks.is_empty(), "no PNG chunks parsed");
    assert!(
        chunks.iter().any(|c| c.typ == ChunkType::IHDR),
        "no IHDR chunk found"
    );
    assert!(
        chunks.iter().any(|c| c.typ == ChunkType::IDAT),
        "no IDAT chunk found"
    );

    let info = parse_ihdr(&chunks).expect("parse IHDR");
    assert!(info.width > 0 && info.height > 0);
    assert!(info.bpp >= 1 && info.bpp <= 4);

    // Concatenate IDAT, strip 2-byte zlib header + 4-byte Adler trailer.
    let idat = concat_idat(&chunks);
    assert!(idat.len() > 6, "IDAT too short");
    let deflate = &idat[2..idat.len() - 4];

    let decoded = decode_deflate(deflate, None).expect("decode deflate");
    let expected_min = info.height as usize * info.row_stride;
    assert_eq!(
        decoded.output.len(),
        expected_min,
        "output length should equal h * (1 + w*bpp)"
    );
    assert!(decoded.num_blocks() >= 1);
    // One lit/dist encoder table per block, so the two lengths agree
    // with each other and num_blocks() by construction.
    assert_eq!(decoded.lit_encs.len(), decoded.dist_encs.len());

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
    // A photo PNG is heavily back-referenced; ≥1 Ref is a sane lower
    // bound for a real sample.
    assert!(ref_count > 0, "expected at least one back-ref event");

    // pos_to_ev and reverse_graph should also build cleanly.
    let pos_to_ev = build_pos_to_ev(&decoded.events, decoded.output.len());
    assert_eq!(pos_to_ev.len(), decoded.output.len());

    let rev = build_reverse_graph(&decoded.events, decoded.output.len());
    assert_eq!(rev.len(), decoded.output.len());
}

/// The event-free `inflate` fast path and the recording `decode_deflate`
/// share one decode core, so on a real dynamic-Huffman + back-reference
/// stream they must produce byte-identical output. Guards the two paths
/// against drift.
#[test]
fn inflate_matches_decode_deflate_on_real_stream() {
    let Some(raw) = read_sample_or_skip("inflate_matches_decode_deflate_on_real_stream") else {
        return;
    };
    let chunks = read_chunks(&raw).expect("read chunks").chunks;
    let idat = concat_idat(&chunks);
    let deflate = &idat[2..idat.len() - 4];

    let full = decode_deflate(deflate, None).expect("decode_deflate");
    let lean = inflate(deflate, None).expect("inflate");
    assert_eq!(
        lean, full.output,
        "event-free inflate must byte-match the recording decoder"
    );
}

#[test]
fn malformed_deflate_returns_error_not_panic() {
    // 3 bytes of all-ones, not a valid deflate stream.
    let garbage = vec![0xFFu8; 3];
    let result = decode_deflate(&garbage, None);
    assert!(result.is_err(), "expected an error from corrupt deflate");
}

#[test]
fn crc_corrupted_png_loads_with_warning() {
    // Glitcher's contract: a PNG whose chunk CRCs aren't refreshed must
    // still load. Flip a bit in a real sample's IHDR CRC; read_chunks
    // must warn, not fail.
    let Some(mut raw) = read_sample_or_skip("crc_corrupted_png_loads_with_warning") else {
        return;
    };
    // Bytes 8..12 are IHDR length (always 13), 12..16 are 'IHDR', 16..29
    // are the 13 IHDR data bytes, 29..33 are the IHDR CRC. Flip a bit.
    raw[29] ^= 0x80;
    let parsed = read_chunks(&raw).expect("CRC mismatch must not fail the parse");
    assert!(
        parsed
            .warnings
            .iter()
            .any(|w| matches!(w, Warning::ChunkCrc { typ } if *typ == ChunkType::IHDR)),
        "expected an IHDR CRC warning, got {:?}",
        parsed.warnings
    );
    let info = parse_ihdr(&parsed.chunks).expect("IHDR data still parses");
    assert!(info.width > 0 && info.height > 0);
}

/// Save round-trip: rebuild a PNG by replacing IDAT with a fresh zlib
/// stream and re-emitting every chunk via `write_chunks`. The saved
/// bytes must re-read to the same decoded output as the original. Covers
/// the end-to-end save path (`build_zlib_stream` + IDAT replacement +
/// `write_chunks` + CRC) that unit tests check only piecewise.
#[test]
fn save_and_reread_unedited_png_round_trips() {
    use pngbend::png::{Chunk, build_zlib_stream, write_chunks};

    let Some(raw) = read_sample_or_skip("save_and_reread_unedited_png_round_trips") else {
        return;
    };
    let chunks_orig = read_chunks(&raw).expect("read chunks").chunks;
    let info = parse_ihdr(&chunks_orig).expect("parse IHDR");
    let idat = concat_idat(&chunks_orig);
    let zlib_header = [idat[0], idat[1]];
    let deflate_buf = idat[2..idat.len() - 4].to_vec();
    let decoded = decode_deflate(&deflate_buf, None).expect("decode");

    // Same shape as `app::edit::PngBendApp::assemble_png_bytes`: collapse
    // every IDAT into one rebuilt entry, copy other chunks through.
    let zlib = build_zlib_stream(&deflate_buf, zlib_header, &decoded.output);
    let out_chunks: Vec<Chunk> = chunks_orig
        .iter()
        .scan(false, |seen, c| {
            if c.typ == ChunkType::IDAT {
                if *seen {
                    return Some(None);
                }
                *seen = true;
                Some(Some(Chunk {
                    typ: ChunkType::IDAT,
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

    // Re-read and decode the saved bytes. Output must match the
    // original; anything else means the save chain corrupted something.
    let chunks_re = read_chunks(&saved).expect("read saved chunks").chunks;
    let info_re = parse_ihdr(&chunks_re).expect("re-parse IHDR");
    assert_eq!(info_re.width, info.width);
    assert_eq!(info_re.height, info.height);

    let idat_re = concat_idat(&chunks_re);
    assert!(idat_re.len() > 6, "saved IDAT too short");
    let deflate_re = &idat_re[2..idat_re.len() - 4];
    let decoded_re = decode_deflate(deflate_re, None).expect("re-decode saved bytes");
    assert_eq!(
        decoded_re.output, decoded.output,
        "saved-and-reread output must match original"
    );
}

/// The fast path in `app::edit::apply_literal_swap_incremental` assumes
/// that patching `output[lit.out_pos]` to the new symbol and propagating
/// it through every `reverse_graph` descendant yields the same bytes as
/// re-decoding the patched bit stream.
///
/// Load-bearing: if the assumption breaks, edits silently corrupt the
/// render. Build the full incremental state and compare byte-for-byte
/// against a fresh decode.
#[test]
fn incremental_literal_swap_matches_full_decode() {
    let Some(raw) = read_sample_or_skip("incremental_literal_swap_matches_full_decode") else {
        return;
    };
    let chunks = read_chunks(&raw).expect("read chunks").chunks;
    let idat = concat_idat(&chunks);
    let mut deflate_buf = idat[2..idat.len() - 4].to_vec();

    let decoded = decode_deflate(&deflate_buf, None).expect("initial decode");
    let reverse_graph = build_reverse_graph(&decoded.events, decoded.output.len());

    // Find a literal with a same-length swap alternative we can apply.
    let (swap_bit, swap_len, new_code, out_pos, new_symbol) = decoded
        .events
        .iter()
        .enumerate()
        .find_map(|(i, e)| match e {
            Event::Lit(lit) => {
                let block = block_of(&decoded.block_starts, i as u32);
                let le = &decoded.lit_encs[block as usize];
                let cur_clen = le.get(lit.symbol as u16)?.len;
                le.iter()
                    .find(|(s, sc)| *s < 256 && *s != lit.symbol as u16 && sc.len == cur_clen)
                    .map(|(sym, sc)| (lit.bit_start, cur_clen, sc.code, lit.out_pos, sym as u8))
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

    // Incremental path: patch the one output byte, propagate via
    // reverse_graph as `apply_literal_swap_incremental` does.
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
    let reference = decode_deflate(&deflate_buf, None)
        .expect("patched stream should still decode (same-length Huffman swap)")
        .output;

    assert_eq!(
        incremental.len(),
        reference.len(),
        "lengths must match after a same-length literal swap"
    );
    // Report the first mismatch for a useful failure message.
    for (i, (a, b)) in incremental.iter().zip(reference.iter()).enumerate() {
        assert_eq!(
            a, b,
            "byte {i} diverges: incremental={a:#04x} reference={b:#04x}"
        );
    }
}

/// The row-scoped unfilter in `apply_dist_redirect_incremental`'s step 6
/// reuses the pre-edit `unfiltered` buffer for every row outside the
/// diff range. Correct iff `unfilter_rows_into` chains through
/// filter-2/3/4 downstream rows, which read the prior row: a skipped row
/// would drift every later pixel. Compares the row-scoped result against
/// a full from-scratch unfilter of the post-edit output.
#[test]
fn incremental_redirect_unfilter_matches_full_decode() {
    let Some(raw) = read_sample_or_skip("incremental_redirect_unfilter_matches_full_decode") else {
        return;
    };
    let chunks = read_chunks(&raw).expect("read chunks").chunks;
    let info = parse_ihdr(&chunks).expect("ihdr");
    let idat = concat_idat(&chunks);
    let mut deflate_buf = idat[2..idat.len() - 4].to_vec();

    let decoded = decode_deflate(&deflate_buf, None).expect("initial decode");

    // Find a back-ref with a valid redirect target via the app's helper,
    // then grab the new Huffman code from the dist EncTable.
    let (swap_bit, swap_len, new_code, affected_from) = decoded
        .events
        .iter()
        .enumerate()
        .find_map(|(i, e)| match e {
            Event::Ref(r) => {
                let block = block_of(&decoded.block_starts, i as u32);
                let alts = valid_dist_alts(
                    block,
                    r.dist_sym,
                    r.out_pos,
                    r.src_out_pos,
                    &decoded.dist_encs,
                );
                let &(new_sym, _new_src, _new_dist) = alts.first()?;
                let de = &decoded.dist_encs[block as usize];
                let sc = de.get(new_sym as u16)?;
                let (new_code, new_clen) = (sc.code, sc.len);
                Some((r.dist_bit_start, new_clen, new_code, r.out_pos))
            }
            _ => None,
        })
        .expect("no redirect candidate in sample");

    // Full unfilter of the *pre-edit* output: what `CoreData` holds
    // across edits and the redirect path reuses.
    let mut unfiltered_cache = unfilter(&decoded.output, &info).expect("initial unfilter");

    // Apply the redirect patch and re-decode: the new reference.
    write_bits(
        &mut deflate_buf,
        swap_bit as usize,
        new_code as u32,
        swap_len,
    );
    let decoded_after = decode_deflate(&deflate_buf, None).expect("decode after redirect");
    let reference = unfilter(&decoded_after.output, &info).expect("reference unfilter");

    // Incremental: diff old vs new output from `affected_from`, then
    // `unfilter_rows_into` over the diffed row range, the shape
    // `apply_dist_redirect_incremental` produces.
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

/// The undo invariant the history stack rests on: applying N edits to
/// `deflate_buf` then undoing in reverse must restore it byte-for-byte.
/// `app::edit::tests::forward_then_inverse_restores_buffer` covers the
/// bit-level primitive on synthetic inputs; this exercises it on a real
/// PNG's deflate stream with patches built from real `EncTable`s, the
/// path that runs in the editor.
#[test]
fn apply_then_undo_n_literal_swaps_restores_deflate_buf() {
    let Some(raw) = read_sample_or_skip("apply_then_undo_n_literal_swaps_restores_deflate_buf")
    else {
        return;
    };
    let chunks = read_chunks(&raw).expect("read chunks").chunks;
    let idat = concat_idat(&chunks);
    let original_deflate = idat[2..idat.len() - 4].to_vec();
    let mut deflate_buf = original_deflate.clone();

    let decoded = decode_deflate(&deflate_buf, None).expect("initial decode");

    // First 4 swappable literals: distinct events with same-length
    // Huffman alternatives. Mirrors `find_literal_swaps` in profile.rs.
    let swaps: Vec<(u32, u8, u16)> = decoded
        .events
        .iter()
        .enumerate()
        .filter_map(|(i, e)| {
            let Event::Lit(lit) = e else { return None };
            let block = block_of(&decoded.block_starts, i as u32);
            let le = &decoded.lit_encs[block as usize];
            let cur_clen = le.get(lit.symbol as u16)?.len;
            let (_, new) = le
                .iter()
                .find(|(s, sc)| *s < 256 && *s != lit.symbol as u16 && sc.len == cur_clen)?;
            Some((lit.bit_start, cur_clen, new.code))
        })
        .take(4)
        .collect();
    assert_eq!(swaps.len(), 4, "expected ≥ 4 swappable literals in sample");

    // Forward: apply each swap, capturing overwritten bits to restore
    // later. Order matters only for overlapping patches; consecutive
    // literal swaps don't overlap (unique `bit_start`), but we still
    // record + replay LIFO to mirror the undo stack's invariant.
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

/// State-equivalence test for the surgical-redirect path.
///
/// `apply_dist_redirect_incremental` skips `decode_deflate` and updates
/// `output` via "recopy from new src + propagate via reverse_graph". It
/// must produce the same `output` bytes a fresh `decode_deflate` would on
/// the patched bit stream, else downstream renders drift. The surgical
/// analogue of the literal-swap path
/// `incremental_literal_swap_matches_full_decode` covers.
#[test]
fn surgical_redirect_output_matches_full_decode() {
    let Some(raw) = read_sample_or_skip("surgical_redirect_output_matches_full_decode") else {
        return;
    };
    let chunks = read_chunks(&raw).expect("read chunks").chunks;
    let idat = concat_idat(&chunks);
    let mut deflate_buf = idat[2..idat.len() - 4].to_vec();

    let decoded = decode_deflate(&deflate_buf, None).expect("initial decode");
    let reverse_graph = build_reverse_graph(&decoded.events, decoded.output.len());

    // Find a back-ref with a redirect target, plus the payload the
    // surgical path needs (out_pos, copy_len, new_src).
    let (swap_bit, swap_len, new_code, out_pos, copy_len, new_src) = decoded
        .events
        .iter()
        .enumerate()
        .find_map(|(i, e)| match e {
            Event::Ref(r) => {
                let block = block_of(&decoded.block_starts, i as u32);
                let alts = valid_dist_alts(
                    block,
                    r.dist_sym,
                    r.out_pos,
                    r.src_out_pos,
                    &decoded.dist_encs,
                );
                let &(new_sym, new_src, _new_dist) = alts.first()?;
                let de = &decoded.dist_encs[block as usize];
                let sc = de.get(new_sym as u16)?;
                let (new_code, new_clen) = (sc.code, sc.len);
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

    // Authoritative: patch the bitstream and decode from scratch.
    write_bits(&mut deflate_buf, swap_bit, new_code as u32, swap_len);
    let reference = decode_deflate(&deflate_buf, None)
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
/// (`src + copy_len > out_pos`), encoding run-length patterns. For an
/// overlap ref, `events[i]`'s reverse-graph edges include positions
/// inside `[out_pos, out_pos+copy_len)` as *sources*. When a redirect
/// moves `src`, those intra-destination edges shift; a BFS walking them
/// on the **pre-rebuild** graph overwrites correctly recopied bytes with
/// stale values.
///
/// Surfaces as `apply`/`undo` producing a different glitch than the
/// original edit. `surgical_redirect_output_matches_full_decode` misses
/// it because its first redirect candidate is non-overlap.
#[test]
fn surgical_redirect_apply_undo_round_trip_with_overlap() {
    let Some(raw) = read_sample_or_skip("surgical_redirect_apply_undo_round_trip_with_overlap")
    else {
        return;
    };
    let chunks = read_chunks(&raw).expect("read chunks").chunks;
    let idat = concat_idat(&chunks);
    let deflate_buf = idat[2..idat.len() - 4].to_vec();
    let decoded = decode_deflate(&deflate_buf, None).expect("initial decode");
    let original_output = decoded.output.clone();

    // Find a redirect target where the ref OVERLAPS its destination
    // (`src_out_pos + copy_len > out_pos`): the cases the bug shows on.
    let target = decoded
        .events
        .iter()
        .enumerate()
        .find_map(|(idx, e)| match e {
            Event::Ref(r) if r.src_out_pos as usize + r.copy_len as usize > r.out_pos as usize => {
                let block = block_of(&decoded.block_starts, idx as u32);
                let alts = valid_dist_alts(
                    block,
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
        // No overlap candidate in this sample: trivially pass. The
        // non-overlap path is covered by the other test.
        return;
    };

    let mut output = decoded.output.clone();
    let mut events = decoded.events.clone();
    let mut rev = build_reverse_graph(&events, output.len());

    // Mirrors the `apply_dist_redirect_incremental` order: mutate event,
    // rebuild rev, recopy, BFS over fresh rev. The rebuild must precede
    // the BFS for overlap refs (some ref edges land within the
    // destination range).
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

/// Synthetic overlap regression. Hand-build an `events` list with a ref
/// that overlaps its destination (`src + len > out_pos`) where the
/// run-length period *changes* under the redirect: pre-edit period 3
/// (`a,b,c,a,b,c,...`), post-edit period 4. There, a BFS on a stale
/// `rev_graph` writes wrong values into the destination range;
/// period-2-both-ways candidates from real photos mask the bug because
/// every wrong-edge write coincidentally lands on a correct value.
///
/// Bypasses the deflate decoder and constructs events + output directly
/// to exercise an exact period mismatch.
#[test]
fn surgical_redirect_overlap_period_mismatch_round_trip() {
    use pngbend::deflate::{LitEvent, RefEvent};
    use pngbend::index::ReverseGraph;

    // Output layout (32 bytes for headroom):
    //   0..6:   literal events (a, b, c, p, q, r, ...)
    //   6..14:  single ref, src_orig=3, copy_len=8, period 3:
    //           pattern p,q,r,p,q,r,p,q.
    //   14..18: downstream ref, len 4, sourced from 6..10, exercising
    //           external descendants that depend on the redirected ref's
    //           bytes.
    let n_lit: u32 = 6;
    let mut events: Vec<Event> = (0..n_lit)
        .map(|i| {
            Event::Lit(LitEvent {
                out_pos: i,
                symbol: 0,
                bit_start: i * 8,
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
    // Ref at 14: src=6, len=4: o[14]=o[6]=40, o[15]=o[7]=50, o[16]=o[8]=60, o[17]=o[9]=40.
    for off in 0..4 {
        output[14 + off] = output[6 + off];
    }
    let original = output.clone();

    let mut rev = build_reverse_graph(&events, output.len());
    let event_idx: usize = n_lit as usize; // index of the redirected ref
    let out_pos: usize = n_lit as usize;
    let copy_len = 8;
    let src_orig: u32 = 3;
    // New src=2, period 4 (out_pos - new_src = 4).
    let new_src: u32 = 2;

    let redirect_step =
        |events: &mut Vec<Event>, output: &mut Vec<u8>, rev: &mut ReverseGraph, new_src: u32| {
            if let Event::Ref(r) = &mut events[event_idx] {
                r.src_out_pos = new_src;
            }
            // Mirrors production order in
            // `apply_dist_redirect_incremental`: rebuild rev BEFORE
            // recopy + BFS so the BFS sees post-edit topology. With a
            // stale rev, an overlap ref's intra-destination edges differ
            // and the BFS overwrites correctly recopied bytes, the bug
            // this test guards against.
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

// Hand-built sub-byte PNG fixtures
//
// Build a minimal valid PNG in memory (IHDR + optional PLTE +
// IDAT-as-stored-deflate-block + IEND) and round-trip it through
// pngbend's full loader. Covers the 1/2/4-bit greyscale and indexed
// paths without committing binary fixtures.

use pngbend::png::{Chunk, build_zlib_stream, parse_zlib_stream, to_rgba8, write_chunks};

/// Wrap raw bytes in a single DEFLATE stored block (BFINAL=1, BTYPE=00).
/// `LEN`/`NLEN` are little-endian per RFC 1951 §3.2.4.
fn deflate_stored(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + data.len());
    out.push(0x01); // BFINAL=1 (bit 0), BTYPE=00 (bits 1-2), pad to byte.
    let len = data.len() as u16;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&(!len).to_le_bytes());
    out.extend_from_slice(data);
    out
}

/// Assemble a complete PNG from raw scanlines (each row a filter byte +
/// pixel data) plus an optional palette. Uses pngbend's own
/// `build_zlib_stream` and `write_chunks` so the round-trip exercises the
/// matching read paths.
fn build_png(
    width: u32,
    height: u32,
    bit_depth: u8,
    color_type: u8,
    scanlines: &[u8],
    palette: Option<&[u8]>,
) -> Vec<u8> {
    let mut ihdr_data = vec![0u8; 13];
    ihdr_data[0..4].copy_from_slice(&width.to_be_bytes());
    ihdr_data[4..8].copy_from_slice(&height.to_be_bytes());
    ihdr_data[8] = bit_depth;
    ihdr_data[9] = color_type;
    // compression_method=0, filter_method=0, interlace_method=0 already zero.

    let deflate = deflate_stored(scanlines);
    let idat_payload = build_zlib_stream(&deflate, [0x78, 0x9C], scanlines);

    let mut chunks = vec![Chunk {
        typ: ChunkType::IHDR,
        data: ihdr_data,
    }];
    if let Some(pal) = palette {
        chunks.push(Chunk {
            typ: ChunkType::PLTE,
            data: pal.to_vec(),
        });
    }
    chunks.push(Chunk {
        typ: ChunkType::IDAT,
        data: idat_payload,
    });
    chunks.push(Chunk {
        typ: ChunkType::IEND,
        data: vec![],
    });
    write_chunks(&chunks)
}

/// Decode a synthetic PNG through pngbend's full loader stack to RGBA8.
/// Palette discovered from the PLTE chunk if present, so test sites pass
/// only the PNG bytes.
fn round_trip(png: &[u8]) -> Vec<u8> {
    let chunks = read_chunks(png).expect("chunks").chunks;
    let info = parse_ihdr(&chunks).expect("ihdr");
    let palette = chunks
        .iter()
        .find(|c| c.typ == ChunkType::PLTE)
        .map(|p| pngbend::png::decode_palette(&p.data, None));
    let idat = concat_idat(&chunks);
    let zlib = parse_zlib_stream(&idat).expect("zlib");
    let decoded = decode_deflate(zlib.deflate_buf, None).expect("deflate");
    let unfiltered = unfilter(&decoded.output, &info).expect("unfilter");
    to_rgba8(&unfiltered, &info, palette.as_deref()).expect("rgba")
}

#[test]
fn round_trip_8x1_1bit_greyscale() {
    // Scanline: filter=None (0), 1 data byte 0b1011_0001.
    // Pixels 1, 0, 1, 1, 0, 0, 0, 1; lumas 255 0 255 255 0 0 0 255.
    let scanlines = vec![0u8, 0b1011_0001];
    let png = build_png(8, 1, 1, 0, &scanlines, None);
    let rgba = round_trip(&png);
    let lumas: Vec<u8> = rgba.chunks_exact(4).map(|p| p[0]).collect();
    assert_eq!(lumas, vec![255, 0, 255, 255, 0, 0, 0, 255]);
}

#[test]
fn round_trip_4x1_2bit_greyscale() {
    // 2-bit samples 0, 1, 2, 3 packed MSB-first: byte 0b00_01_10_11 = 0x1B.
    // Scaled lumas: 0, 85, 170, 255.
    let scanlines = vec![0u8, 0b00_01_10_11];
    let png = build_png(4, 1, 2, 0, &scanlines, None);
    let rgba = round_trip(&png);
    let lumas: Vec<u8> = rgba.chunks_exact(4).map(|p| p[0]).collect();
    assert_eq!(lumas, vec![0, 85, 170, 255]);
}

#[test]
fn round_trip_2x1_4bit_greyscale() {
    // 4-bit samples 0xA (high nibble), 0x5 (low nibble).
    // Scaled lumas: 0xAA, 0x55.
    let scanlines = vec![0u8, 0xA5];
    let png = build_png(2, 1, 4, 0, &scanlines, None);
    let rgba = round_trip(&png);
    let lumas: Vec<u8> = rgba.chunks_exact(4).map(|p| p[0]).collect();
    assert_eq!(lumas, vec![0xAA, 0x55]);
}

#[test]
fn round_trip_8x1_1bit_indexed() {
    // 2-colour palette: index 0 = red, index 1 = blue.
    // Byte 0b1010_0000: indices 1, 0, 1, 0, 0, 0, 0, 0.
    let palette = vec![255, 0, 0, 0, 0, 255]; // PLTE is RGB triples.
    let scanlines = vec![0u8, 0b1010_0000];
    let png = build_png(8, 1, 1, 3, &scanlines, Some(&palette));
    let rgba = round_trip(&png);
    let expected: Vec<u8> = [1u8, 0, 1, 0, 0, 0, 0, 0]
        .iter()
        .flat_map(|&i| {
            if i == 0 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            }
        })
        .collect();
    assert_eq!(rgba, expected);
}

#[test]
fn round_trip_5x1_1bit_greyscale_width_not_multiple_of_eight() {
    // 5 pixels at 1 bit each = 5 used bits in 1 byte (3 padding bits).
    // Bits 1, 0, 1, 0, 1, _, _, _: byte 0b1010_1000.
    let scanlines = vec![0u8, 0b1010_1000];
    let png = build_png(5, 1, 1, 0, &scanlines, None);
    let rgba = round_trip(&png);
    let lumas: Vec<u8> = rgba.chunks_exact(4).map(|p| p[0]).collect();
    assert_eq!(lumas, vec![255, 0, 255, 0, 255]);
    assert_eq!(rgba.len(), 5 * 4);
}

#[test]
fn round_trip_5x1_4bit_indexed_odd_width() {
    // 5 pixels at 4 bits each = 5 nibbles in 3 bytes (1 padding nibble).
    // Indices 1, 2, 3, 4, 5: bytes 0x12, 0x34, 0x5_ (low nibble padding).
    let scanlines = vec![0u8, 0x12, 0x34, 0x50];
    // 6 palette entries (indices 0..5): black, then five distinct colours.
    let palette: Vec<u8> = vec![
        0, 0, 0, // 0: black
        10, 0, 0, // 1
        0, 20, 0, // 2
        0, 0, 30, // 3
        40, 40, 0, // 4
        0, 40, 40, // 5
    ];
    let png = build_png(5, 1, 4, 3, &scanlines, Some(&palette));
    let rgba = round_trip(&png);
    assert_eq!(rgba[0..4], [10, 0, 0, 255]); // index 1
    assert_eq!(rgba[4..8], [0, 20, 0, 255]); // index 2
    assert_eq!(rgba[8..12], [0, 0, 30, 255]); // index 3
    assert_eq!(rgba[12..16], [40, 40, 0, 255]); // index 4
    assert_eq!(rgba[16..20], [0, 40, 40, 255]); // index 5
}

#[test]
fn round_trip_3x2_1bit_greyscale_multi_row() {
    // Two rows, 3 pixels each = 1 data byte per row (5 padding bits).
    // Row 0: 1, 0, 1, _, _, _, _, _ = 0b1010_0000
    // Row 1: 0, 1, 0, _, _, _, _, _ = 0b0100_0000
    let scanlines = vec![0u8, 0b1010_0000, 0u8, 0b0100_0000];
    let png = build_png(3, 2, 1, 0, &scanlines, None);
    let rgba = round_trip(&png);
    let lumas: Vec<u8> = rgba.chunks_exact(4).map(|p| p[0]).collect();
    assert_eq!(lumas, vec![255, 0, 255, 0, 255, 0]);
}
