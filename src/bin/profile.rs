//! Workload driver for profiling. Built under the `profiling` cargo
//! profile (release + debug symbols) so `perf` and `flamegraph` see
//! real symbol names.
//!
//! Run:
//! ```text
//!   cargo build --profile profiling --bin profile
//!   flamegraph --bin profile   # uses perf; needs debug syms
//!   perf record -g ./target/profiling/profile && perf report --stdio -g
//! ```
//!
//! Mirrors an interactive session on a large image: open, select a
//! pixel (cascade BFS), apply a literal swap, undo, repeat. Also
//! exercises the filter rebuild over a typing sequence and the three
//! event-driven overlays. Each phase prints its own wall-clock.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use pngbend::bitstream::{read_bits_at, write_bits};
use pngbend::composite::{composite_rgba, composite_rows_into};
use pngbend::deflate::{EncTable, Event, block_of, decode_deflate};
use pngbend::index::{
    CascadeScratch, PixelIndex, PixelRow, build_pixel_index, build_pos_to_ev, build_reverse_graph,
    valid_dist_alts,
};
use pngbend::overlays::{
    compute_filter_expansion, make_block_overlay_bytes, make_cascade_overlay_bytes,
    make_distance_overlay_bytes, make_literal_overlay_bytes,
};
use pngbend::png::{
    concat_idat, parse_ihdr, read_chunks, to_rgba8, to_rgba8_rows_into, unfilter,
    unfilter_rows_into,
};

const EDIT_ITERATIONS: usize = 5;
const FILTER_KEYSTROKES: &[&str] = &["", "a", "ab", "ab1", "ab12", "ab", "a", ""];

fn main() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("samples")
        .join("checksum_sunset.png");
    let raw = std::fs::read(&path).expect("read sample png");

    println!("# Phases (wall clock). Same binary is what perf / flamegraph profile.");
    println!("# sample: {}", path.display());
    println!("# bytes:  {}", raw.len());
    println!();

    // Phase 1: initial load
    let t = Instant::now();
    let chunks = read_chunks(&raw).expect("chunks").chunks;
    let info = parse_ihdr(&chunks).expect("ihdr");
    let idat = concat_idat(&chunks);
    let mut deflate_buf = idat[2..idat.len() - 4].to_vec();
    let phase_chunks = t.elapsed();

    let t = Instant::now();
    let decoded = decode_deflate(&deflate_buf, None).expect("decode");
    let phase_decode = t.elapsed();

    let t = Instant::now();
    let geom = &info;
    let raster = pngbend::Raster::new(info);
    let pos_to_ev = build_pos_to_ev(&decoded.events, decoded.output.len());
    let phase_pos = t.elapsed();

    let t = Instant::now();
    let reverse_graph = build_reverse_graph(&decoded.events, decoded.output.len());
    let phase_rev = t.elapsed();

    let t = Instant::now();
    let pixel_index = build_pixel_index(
        &decoded.events,
        &decoded.output,
        &pos_to_ev,
        &decoded.lit_encs,
        &decoded.dist_encs,
        &decoded.block_starts,
        &raster,
    );
    let phase_pi = t.elapsed();

    let t = Instant::now();
    let unfiltered = unfilter(&decoded.output, &info).expect("unfilter");
    let phase_unfilter = t.elapsed();

    let t = Instant::now();
    let base_rgba = to_rgba8(&unfiltered, &info, None).expect("rgba");
    let phase_rgba = t.elapsed();

    let total_load = phase_chunks
        + phase_decode
        + phase_pos
        + phase_rev
        + phase_pi
        + phase_unfilter
        + phase_rgba;

    println!("phase 1: initial load");
    println!(
        "  {:5} {:6} events, {:6}x{:4}, bpp={}",
        "",
        decoded.events.len(),
        info.width,
        info.height,
        info.bpp,
    );
    println!("  chunks  + parse_ihdr          {phase_chunks:?}");
    println!("  decode_deflate                {phase_decode:?}");
    println!("  build_pos_to_ev               {phase_pos:?}");
    println!("  build_reverse_graph           {phase_rev:?}");
    println!("  build_pixel_index             {phase_pi:?}");
    println!("  unfilter                      {phase_unfilter:?}");
    println!("  to_rgba8                      {phase_rgba:?}");
    println!("  ----------------------------------------");
    println!("  load total                    {total_load:?}");
    println!();

    // Phase 2: overlay generation
    let t = Instant::now();
    let lit_overlay = make_literal_overlay_bytes(&decoded.events, geom);
    let phase_lit_ov = t.elapsed();

    let t = Instant::now();
    let dist_overlay = make_distance_overlay_bytes(&decoded.events, geom, decoded.max_distance);
    let phase_dist_ov = t.elapsed();

    let t = Instant::now();
    let blk_overlay = make_block_overlay_bytes(
        &decoded.events,
        geom,
        &decoded.block_starts,
        decoded.num_blocks(),
    );
    let phase_blk_ov = t.elapsed();

    println!("phase 2: overlay generation (one per mode switch)");
    println!("  make_literal_overlay_bytes    {phase_lit_ov:?}");
    println!("  make_distance_overlay_bytes   {phase_dist_ov:?}");
    println!("  make_block_overlay_bytes      {phase_blk_ov:?}");
    println!();

    // Phase 3: composite (per texture rebuild)
    let t = Instant::now();
    let mut composited = Vec::with_capacity(base_rgba.len());
    // Mirror the app's per-edit texture rebuild via composite_rgba.
    for _ in 0..4 {
        composited = composite_rgba(&base_rgba, &lit_overlay);
    }
    let phase_compose = t.elapsed();
    println!("phase 3: composite_rgba × 4    {phase_compose:?}");
    println!();

    // Phase 4: filter rebuild (per keystroke)
    let mut total_filter = Duration::ZERO;
    let mut scratch = String::with_capacity(48);
    let mut last_count = 0;
    for &text in FILTER_KEYSTROKES {
        let t = Instant::now();
        let n = run_filter(&pixel_index, text, false, &mut scratch);
        total_filter += t.elapsed();
        last_count = n;
    }
    println!(
        "phase 4: rebuild_filter × {}        {total_filter:?}  (last: {} rows)",
        FILTER_KEYSTROKES.len(),
        last_count,
    );

    // Phase 4b: same keystroke sequence, with narrowing.
    //
    // When the new needle extends the previous, re-test only rows in the
    // prior filtered view instead of rescanning `pi.lit + pi.refs`.
    // Mirrors `FilterSpec::is_refinement_of` + the narrowing branch in
    // `rebuild_filter`.
    let mut total_narrow = Duration::ZERO;
    let mut prev_needle: Option<String> = None;
    let mut view: Vec<(bool, u32)> = Vec::new(); // (is_lit, index)
    let mut per_keystroke: Vec<Duration> = Vec::new();
    for &text in FILTER_KEYSTROKES {
        let t = Instant::now();
        let needle = text.to_ascii_lowercase();
        let can_narrow = prev_needle
            .as_ref()
            .is_some_and(|old| needle.contains(old.as_str()));
        if can_narrow {
            view.retain(|&(is_lit, i)| {
                let row = if is_lit {
                    &pixel_index.lit[i as usize]
                } else {
                    &pixel_index.refs[i as usize]
                };
                row_matches_generic(row, i, is_lit, &needle, &mut scratch)
            });
        } else {
            view.clear();
            for (i, row) in pixel_index.lit.iter().enumerate() {
                if row_matches_generic(row, i as u32, true, &needle, &mut scratch) {
                    view.push((true, i as u32));
                }
            }
            for (i, row) in pixel_index.refs.iter().enumerate() {
                if row_matches_generic(row, i as u32, false, &needle, &mut scratch) {
                    view.push((false, i as u32));
                }
            }
        }
        let elapsed = t.elapsed();
        total_narrow += elapsed;
        per_keystroke.push(elapsed);
        prev_needle = Some(needle);
    }
    let last_count_narrow = view.len();
    println!(
        "phase 4b: rebuild_filter × {} w/narrow {total_narrow:?}  (last: {} rows)",
        FILTER_KEYSTROKES.len(),
        last_count_narrow,
    );
    for (text, dur) in FILTER_KEYSTROKES.iter().zip(per_keystroke.iter()) {
        println!("  \"{text}\": {dur:?}");
    }
    println!();

    // Phase 5: edit, FULL reload baseline.
    //
    // Re-decode the patched stream and rebuild every index from scratch
    // per edit. Upper bound on edit cost without the incremental paths
    // in `app::edit::apply_*_incremental`.
    let mut total_full = Duration::ZERO;
    let mut scratch_cascade = CascadeScratch::default();
    let swaps = find_literal_swaps(
        &decoded.events,
        &decoded.lit_encs,
        &decoded.block_starts,
        EDIT_ITERATIONS,
    );
    println!(
        "phase 5: {} FULL-reload edit cycles (baseline, no incremental path)",
        swaps.len()
    );
    for (i, swap) in swaps.iter().enumerate() {
        let t = Instant::now();
        let bit_start = swap.bit_start as usize;
        let prior_code = read_bits_at(&deflate_buf, bit_start, swap.code_len);
        write_bits(
            &mut deflate_buf,
            bit_start,
            swap.new_code as u32,
            swap.code_len,
        );
        let decoded = decode_deflate(&deflate_buf, None).expect("decode after edit");
        let pos_to_ev = build_pos_to_ev(&decoded.events, decoded.output.len());
        let reverse_graph = build_reverse_graph(&decoded.events, decoded.output.len());
        let _pi = build_pixel_index(
            &decoded.events,
            &decoded.output,
            &pos_to_ev,
            &decoded.lit_encs,
            &decoded.dist_encs,
            &decoded.block_starts,
            &raster,
        );
        let unfiltered = unfilter(&decoded.output, &info).expect("unfilter");
        let _rgba = to_rgba8(&unfiltered, &info, None).expect("rgba");
        let _c = scratch_cascade.run(&[swap.out_pos], &reverse_graph);
        let _fe = compute_filter_expansion(&[swap.out_pos], &decoded.output, geom);
        write_bits(&mut deflate_buf, bit_start, prior_code, swap.code_len);
        total_full += t.elapsed();
        println!("  cycle {i}: {:?}", total_full / (i as u32 + 1));
    }
    println!();

    // Phase 5b: edit, INCREMENTAL reload (literal-swap path).
    //
    // Mirrors `app::edit::apply_literal_swap_incremental`: patch output,
    // BFS through reverse_graph, re-run unfilter+rgba. No decode, index
    // rebuild, or reverse-graph reconstruction; existing indices stay
    // valid because LZ77 topology is unchanged.
    let mut total_incr = Duration::ZERO;
    // Keep `output` mutable across cycles so successive swaps see their
    // predecessors, as the app does.
    let mut output_incr = decoded.output.clone();
    let h = info.height as usize;
    let mut unfiltered_cache = unfilter(&output_incr, &info).expect("unfilter");
    let mut base_rgba_cache = to_rgba8(&unfiltered_cache, &info, None).expect("rgba");
    // Reusable row-touched scratch: `bool`-per-row, h=1600 is 1.6 KB.
    let mut row_touched = vec![false; h];
    let mut composite_scratch = Vec::with_capacity(base_rgba_cache.len());
    // Pre-build the cascade overlay once. In the app it comes from the
    // most recent `select_pixel`, reused across apply/undo since LZ77
    // topology is stable for literal swaps.
    let warm_cascade = scratch_cascade.run(&[swaps[0].out_pos], &reverse_graph);
    let warm_fe = compute_filter_expansion(warm_cascade.affected, &output_incr, geom);
    let cached_cascade_overlay = make_cascade_overlay_bytes(&warm_cascade, &warm_fe, geom);
    println!(
        "phase 5b: {} INCREMENTAL-reload edit cycles + UI follow-up (row-scoped)",
        swaps.len()
    );
    for (i, swap) in swaps.iter().enumerate() {
        let t_total = Instant::now();

        let t_stage = Instant::now();
        let bit_start = swap.bit_start as usize;
        let out_pos = swap.out_pos as usize;
        let prior_code = read_bits_at(&deflate_buf, bit_start, swap.code_len);
        write_bits(
            &mut deflate_buf,
            bit_start,
            swap.new_code as u32,
            swap.code_len,
        );
        let t_bits = t_stage.elapsed();

        let t_stage = Instant::now();
        output_incr[out_pos] = swap.new_symbol;
        for touched in row_touched.iter_mut() {
            *touched = false;
        }
        let mut first_affected = usize::MAX;
        let touch = |pos: usize, first: &mut usize, rows: &mut [bool]| {
            let r = pos / info.row_stride;
            if r < rows.len() && !rows[r] {
                rows[r] = true;
                if r < *first {
                    *first = r;
                }
            }
        };
        touch(out_pos, &mut first_affected, &mut row_touched);
        let mut frontier: VecDeque<u32> = VecDeque::new();
        frontier.push_back(swap.out_pos);
        while let Some(pos) = frontier.pop_front() {
            for &dst in reverse_graph.neighbors(pos) {
                output_incr[dst as usize] = swap.new_symbol;
                touch(dst as usize, &mut first_affected, &mut row_touched);
                frontier.push_back(dst);
            }
        }
        let t_propagate = t_stage.elapsed();

        // row-scoped unfilter
        let t_stage = Instant::now();
        let mut rebuilt: Vec<usize> = Vec::new();
        if first_affected != usize::MAX {
            unfilter_rows_into(
                &output_incr,
                &info,
                &mut unfiltered_cache,
                first_affected,
                |y| row_touched[y],
                |y| rebuilt.push(y),
            )
            .expect("unfilter_rows");
        }
        let t_unfilter = t_stage.elapsed();

        // row-scoped rgba
        let t_stage = Instant::now();
        to_rgba8_rows_into(
            &unfiltered_cache,
            &info,
            None,
            &mut base_rgba_cache,
            rebuilt.iter().copied(),
        )
        .expect("rgba rows");
        let t_rgba = t_stage.elapsed();

        // A literal swap doesn't move LZ77 topology, so the app keeps
        // the cascade overlay from the most recent select_pixel: no BFS,
        // filter expansion, or per-edit overlay alloc on apply/undo.
        // Overlay materialised once outside the loop.

        // Row-scoped composite: only rewritten rows need re-blending.
        // The rest of `composite_scratch` keeps last frame's composite,
        // still correct (base_rgba unchanged outside `rebuilt`, overlay
        // unchanged).
        let t_stage = Instant::now();
        if composite_scratch.len() != base_rgba_cache.len() {
            composite_scratch.clear();
            composite_scratch.extend_from_slice(&base_rgba_cache);
        }
        composite_rows_into(
            &base_rgba_cache,
            &cached_cascade_overlay,
            &mut composite_scratch,
            info.width,
            rebuilt.iter().copied(),
        );
        let t_composite = t_stage.elapsed();

        // Revert bit stream + data for the next iteration.
        write_bits(&mut deflate_buf, bit_start, prior_code, swap.code_len);
        output_incr[out_pos] = decoded.output[out_pos];

        let elapsed = t_total.elapsed();
        total_incr += elapsed;
        println!(
            "  cycle {i}: {elapsed:?}  rows={} bits={t_bits:?} prop={t_propagate:?} unfilter={t_unfilter:?} rgba={t_rgba:?} composite={t_composite:?}",
            rebuilt.len(),
        );
    }
    println!();
    println!(
        "  speedup: {:.1}× ({:.1} ms → {:.1} ms per edit)",
        total_full.as_secs_f64() / total_incr.as_secs_f64(),
        total_full.as_secs_f64() * 1000.0 / swaps.len() as f64,
        total_incr.as_secs_f64() * 1000.0 / swaps.len() as f64,
    );
    println!();

    // Phase 5c: redirect baseline (decode + rebuild + row-scoped render).
    //
    // Re-decodes the patched stream, rebuilds every derived index, diffs
    // old-vs-new output, and re-unfilters/re-converts only changed rows.
    // The "decode + full rebuild" floor that the surgical redirect path
    // [`app::edit::apply_dist_redirect_incremental`] avoids; running both
    // makes the surgical-vs-baseline ratio explicit.
    let mut total_redir = Duration::ZERO;
    let mut prior_output = decoded.output.clone();
    let mut redir_unfiltered = unfilter(&prior_output, &info).expect("unfilter");
    let mut redir_rgba = to_rgba8(&redir_unfiltered, &info, None).expect("rgba");
    println!(
        "phase 5c: {} redirect-flavoured edit cycles (full rebuild + row-scoped render)",
        swaps.len()
    );
    for (i, swap) in swaps.iter().enumerate() {
        let t_total = Instant::now();

        let t_stage = Instant::now();
        let bit_start = swap.bit_start as usize;
        let out_pos = swap.out_pos as usize;
        let prior_code = read_bits_at(&deflate_buf, bit_start, swap.code_len);
        write_bits(
            &mut deflate_buf,
            bit_start,
            swap.new_code as u32,
            swap.code_len,
        );
        let t_bits = t_stage.elapsed();

        let t_stage = Instant::now();
        let decoded_after = decode_deflate(&deflate_buf, None).expect("decode after edit");
        let t_decode = t_stage.elapsed();

        // Rebuild the three index structures, required when LZ77
        // topology changes.
        let t_stage = Instant::now();
        let pos_to_ev_after = build_pos_to_ev(&decoded_after.events, decoded_after.output.len());
        let t_pte = t_stage.elapsed();
        let t_stage = Instant::now();
        let _rev_after = build_reverse_graph(&decoded_after.events, decoded_after.output.len());
        let t_rev = t_stage.elapsed();
        let t_stage = Instant::now();
        let _pi_after = build_pixel_index(
            &decoded_after.events,
            &decoded_after.output,
            &pos_to_ev_after,
            &decoded_after.lit_encs,
            &decoded_after.dist_encs,
            &decoded_after.block_starts,
            &raster,
        );
        let t_pi = t_stage.elapsed();

        // Diff old vs new output from the redirected ref's out_pos.
        // Literal-swap's `out_pos` is a fine stand-in.
        let t_stage = Instant::now();
        let diff = {
            let old = &prior_output;
            let new = &decoded_after.output;
            let mut first = out_pos;
            const CHUNK: usize = 8;
            while first + CHUNK <= old.len()
                && old[first..first + CHUNK] == new[first..first + CHUNK]
            {
                first += CHUNK;
            }
            while first < old.len() && old[first] == new[first] {
                first += 1;
            }
            if first == old.len() {
                None
            } else {
                let mut last = old.len();
                while last >= first + CHUNK && old[last - CHUNK..last] == new[last - CHUNK..last] {
                    last -= CHUNK;
                }
                while last > first && old[last - 1] == new[last - 1] {
                    last -= 1;
                }
                Some((first, last))
            }
        };
        let t_diff = t_stage.elapsed();

        let t_stage = Instant::now();
        let mut rebuilt: Vec<usize> = Vec::new();
        if let Some((first_byte, last_byte)) = diff {
            let first_row = first_byte / info.row_stride;
            let last_row = (last_byte.saturating_sub(1)) / info.row_stride;
            let first_row = first_row.min(h.saturating_sub(1));
            let last_row = last_row.min(h.saturating_sub(1));
            // Copy new output to scratch; `prior_output` is the
            // previous-frame state.
            prior_output.copy_from_slice(&decoded_after.output);
            let mut redir_row_touched = vec![false; h];
            for touched in redir_row_touched
                .iter_mut()
                .take(last_row + 1)
                .skip(first_row)
            {
                *touched = true;
            }
            unfilter_rows_into(
                &prior_output,
                &info,
                &mut redir_unfiltered,
                first_row,
                |y| redir_row_touched[y],
                |y| rebuilt.push(y),
            )
            .expect("unfilter_rows redirect");
            to_rgba8_rows_into(
                &redir_unfiltered,
                &info,
                None,
                &mut redir_rgba,
                rebuilt.iter().copied(),
            )
            .expect("rgba rows redirect");
        }
        let t_render = t_stage.elapsed();

        // Revert for next iteration.
        write_bits(&mut deflate_buf, bit_start, prior_code, swap.code_len);

        let elapsed = t_total.elapsed();
        total_redir += elapsed;
        println!(
            "  cycle {i}: {elapsed:?}  rows={} bits={t_bits:?} decode={t_decode:?} pos_to_ev={t_pte:?} rev_graph={t_rev:?} pixel_idx={t_pi:?} diff={t_diff:?} render={t_render:?}",
            rebuilt.len(),
        );
    }
    println!(
        "  redirect vs full reload: {:.1}× ({:.1} ms → {:.1} ms per edit)",
        total_full.as_secs_f64() / total_redir.as_secs_f64(),
        total_full.as_secs_f64() * 1000.0 / swaps.len() as f64,
        total_redir.as_secs_f64() * 1000.0 / swaps.len() as f64,
    );
    println!();

    // Phase 5d: SURGICAL redirect (mirror of `apply_dist_redirect_incremental`).
    //
    // Skips `decode_deflate` + `build_pos_to_ev` + `build_pixel_index`.
    // Patches the bit, recopies the ref's bytes from the new src,
    // BFS-propagates through the existing reverse_graph, then rebuilds
    // reverse_graph (one ref's edges moved; CSR can't be mutated in
    // place). max_distance refreshed by an O(events) scan. Render stays
    // row-scoped via the rows propagation touched.
    let redirect_targets: Vec<RedirectSwap> = find_redirect_swaps(
        &decoded.events,
        &decoded.dist_encs,
        &decoded.block_starts,
        EDIT_ITERATIONS,
    );
    let mut total_surgical_redir = Duration::ZERO;
    let mut surg_output = decoded.output.clone();
    let mut surg_events = decoded.events.clone();
    let mut surg_rev = build_reverse_graph(&surg_events, surg_output.len());
    let mut surg_unfiltered = unfilter(&surg_output, &info).expect("surg unfilter init");
    let mut surg_rgba = to_rgba8(&surg_unfiltered, &info, None).expect("surg rgba init");
    println!(
        "phase 5d: {} SURGICAL redirect cycles (no decode, no pos_to_ev/pixel_index rebuild)",
        redirect_targets.len()
    );
    for (i, target) in redirect_targets.iter().enumerate() {
        let t_total = Instant::now();

        let t_stage = Instant::now();
        let bit_start = target.bit_start as usize;
        let out_pos = target.out_pos as usize;
        let new_src = target.new_src as usize;
        let prior_code = read_bits_at(&deflate_buf, bit_start, target.code_len);
        write_bits(
            &mut deflate_buf,
            bit_start,
            target.new_code as u32,
            target.code_len,
        );
        // Patch the ref event in place.
        if let Event::Ref(r) = &mut surg_events[target.event_idx] {
            r.src_out_pos = target.new_src;
            r.dist_sym = target.new_dist_sym;
        }
        let t_bits = t_stage.elapsed();

        // Recopy bytes from new src.
        let t_stage = Instant::now();
        for off in 0..target.copy_len {
            surg_output[out_pos + off] = surg_output[new_src + off];
        }
        let t_recopy = t_stage.elapsed();

        // BFS-propagate through the (still-old) reverse_graph from each
        // newly-rewritten output position.
        let t_stage = Instant::now();
        let mut row_touched = vec![false; h];
        let mut first_affected = usize::MAX;
        let mut frontier: VecDeque<u32> = VecDeque::new();
        let touch = |pos: usize, first: &mut usize, rows: &mut [bool]| {
            let r = pos / info.row_stride;
            if r < rows.len() && !rows[r] {
                rows[r] = true;
                if r < *first {
                    *first = r;
                }
            }
        };
        for off in 0..target.copy_len {
            let pos = out_pos + off;
            touch(pos, &mut first_affected, &mut row_touched);
            frontier.push_back(pos as u32);
        }
        while let Some(pos) = frontier.pop_front() {
            let val = surg_output[pos as usize];
            for &dst in surg_rev.neighbors(pos) {
                surg_output[dst as usize] = val;
                touch(dst as usize, &mut first_affected, &mut row_touched);
                frontier.push_back(dst);
            }
        }
        let t_propagate = t_stage.elapsed();

        // Rebuild reverse_graph (one ref's outgoing edges moved; CSR has
        // no cheap edge-set mutation). Only structural index that needs
        // it. Result discarded: the teardown below rebuilds from the
        // un-edited `surg_events`; we only want the wall-clock here.
        let t_stage = Instant::now();
        let _new_rev = build_reverse_graph(&surg_events, surg_output.len());
        std::hint::black_box(&_new_rev);
        let t_rev = t_stage.elapsed();

        // Refresh `max_distance` with a cheap O(events) max.
        let t_stage = Instant::now();
        let _max_d: u32 = surg_events
            .iter()
            .filter_map(|e| match e {
                Event::Ref(r) => Some(r.out_pos - r.src_out_pos),
                _ => None,
            })
            .max()
            .unwrap_or(1);
        let t_max = t_stage.elapsed();

        // Row-scoped unfilter + RGBA on touched rows.
        let t_stage = Instant::now();
        let mut rebuilt: Vec<usize> = Vec::new();
        if first_affected != usize::MAX {
            unfilter_rows_into(
                &surg_output,
                &info,
                &mut surg_unfiltered,
                first_affected,
                |y| row_touched[y],
                |y| rebuilt.push(y),
            )
            .expect("surg unfilter rows");
            to_rgba8_rows_into(
                &surg_unfiltered,
                &info,
                None,
                &mut surg_rgba,
                rebuilt.iter().copied(),
            )
            .expect("surg rgba rows");
        }
        let t_render = t_stage.elapsed();

        // Stop the per-cycle timer BEFORE the test-only undo. It's not
        // part of `apply_dist_redirect_incremental` and would inflate
        // the per-edit number.
        let elapsed = t_total.elapsed();
        total_surgical_redir += elapsed;
        println!(
            "  cycle {i}: {elapsed:?}  rows={} bits={t_bits:?} recopy={t_recopy:?} prop={t_propagate:?} rev={t_rev:?} max={t_max:?} render={t_render:?}",
            rebuilt.len(),
        );

        // Reset to the pre-edit state for the next iteration. Not timed.
        write_bits(&mut deflate_buf, bit_start, prior_code, target.code_len);
        if let Event::Ref(r) = &mut surg_events[target.event_idx] {
            r.src_out_pos = target.old_src;
            r.dist_sym = target.old_dist_sym;
        }
        surg_output[out_pos..out_pos + target.copy_len]
            .copy_from_slice(&decoded.output[out_pos..out_pos + target.copy_len]);
        surg_rev = build_reverse_graph(&surg_events, surg_output.len());
        let mut undo_frontier: VecDeque<u32> = VecDeque::new();
        for off in 0..target.copy_len {
            undo_frontier.push_back((out_pos + off) as u32);
        }
        while let Some(pos) = undo_frontier.pop_front() {
            let val = surg_output[pos as usize];
            for &dst in surg_rev.neighbors(pos) {
                surg_output[dst as usize] = val;
                undo_frontier.push_back(dst);
            }
        }
    }
    println!(
        "  surgical vs full-reload redirect: {:.1}× ({:.1} ms → {:.1} ms per edit)",
        total_redir.as_secs_f64() / total_surgical_redir.as_secs_f64().max(1e-9),
        total_redir.as_secs_f64() * 1000.0 / redirect_targets.len().max(1) as f64,
        total_surgical_redir.as_secs_f64() * 1000.0 / redirect_targets.len().max(1) as f64,
    );
    // Keep the optimizer from dropping the final rebuilt surg_rev (last
    // iteration writes it, nothing reads it).
    std::hint::black_box(&surg_rev);
    println!();

    // Phase 6: cascade BFS only (per-click hot path)
    let t = Instant::now();
    let seeds: Vec<u32> = (0..256).map(|i| (i * 4093) as u32).collect();
    for seed in &seeds {
        let _c = scratch_cascade.run(&[*seed], &reverse_graph);
    }
    let phase_cascade = t.elapsed();
    println!(
        "phase 6: cascade_bfs × {}          {phase_cascade:?}",
        seeds.len()
    );

    // Prevent the optimizer from discarding work.
    std::hint::black_box(&composited);
    std::hint::black_box(&dist_overlay);
    std::hint::black_box(&blk_overlay);
}

// helpers

struct LiteralSwap {
    bit_start: u32,
    code_len: u8,
    new_code: u16,
    out_pos: u32,
    new_symbol: u8,
}

/// Find up to `n` literal events whose Huffman code has a same-length
/// alternative: the pixels the app lists as "editable literal".
fn find_literal_swaps(
    events: &[Event],
    lit_encs: &[EncTable],
    block_starts: &[u32],
    n: usize,
) -> Vec<LiteralSwap> {
    let mut out = Vec::with_capacity(n);
    for (i, e) in events.iter().enumerate() {
        if out.len() >= n {
            break;
        }
        let Event::Lit(lit) = e else {
            continue;
        };
        let le = &lit_encs[block_of(block_starts, i as u32) as usize];
        let Some(cur) = le.get(lit.symbol as u16) else {
            continue;
        };
        let cur_clen = cur.len;
        // Pick the first alternative with matching clen (skip the current).
        if let Some((alt_sym, alt)) = le
            .iter()
            .find(|(s, sc)| *s < 256 && *s != lit.symbol as u16 && sc.len == cur_clen)
        {
            out.push(LiteralSwap {
                bit_start: lit.bit_start,
                code_len: cur_clen,
                new_code: alt.code,
                out_pos: lit.out_pos,
                new_symbol: alt_sym as u8,
            });
        }
    }
    out
}

/// Simulate one filter-rebuild keystroke. Mirrors `app::rebuild_filter`:
/// iterate both PixelIndex arrays, format each row's display text into a
/// reused scratch `String`, test for substring match.
fn run_filter(pi: &PixelIndex, text: &str, editable_only: bool, scratch: &mut String) -> usize {
    let needle: String = text.to_ascii_lowercase();
    let mut count = 0;
    for (i, row) in pi.lit.iter().enumerate() {
        if editable_only && !row.has_edit {
            continue;
        }
        if !needle.is_empty() {
            scratch.clear();
            use std::fmt::Write;
            let [r, g, b] = row.rgb;
            let _ = write!(
                scratch,
                "{:5}  ({:4},{:4})  #{r:02x}{g:02x}{b:02x}",
                i + 1,
                row.x(),
                row.y(),
            );
            scratch.make_ascii_lowercase();
            if !scratch.contains(&needle) {
                continue;
            }
        }
        count += 1;
    }
    for (i, row) in pi.refs.iter().enumerate() {
        if editable_only && !row.has_edit {
            continue;
        }
        if !needle.is_empty() {
            scratch.clear();
            use std::fmt::Write;
            let _ = write!(
                scratch,
                "{:5}  ({:4},{:4})  d=    0 len=0",
                i + 1,
                row.x(),
                row.y(),
            );
            scratch.make_ascii_lowercase();
            if !scratch.contains(&needle) {
                continue;
            }
        }
        count += 1;
    }
    count
}

/// Test a single row against an already-lower-cased needle. True if the
/// formatted row text contains it. Shared between phase 4 and the
/// narrowing path so both measure the same per-row cost.
fn row_matches_generic(
    row: &PixelRow,
    i: u32,
    is_lit: bool,
    needle: &str,
    scratch: &mut String,
) -> bool {
    if needle.is_empty() {
        return true;
    }
    scratch.clear();
    use std::fmt::Write;
    let [r, g, b] = row.rgb;
    if is_lit {
        let _ = write!(
            scratch,
            "{:5}  ({:4},{:4})  #{r:02x}{g:02x}{b:02x}",
            i + 1,
            row.x(),
            row.y(),
        );
    } else {
        let _ = write!(
            scratch,
            "{:5}  ({:4},{:4})  d=    0 len=0",
            i + 1,
            row.x(),
            row.y(),
        );
    }
    scratch.make_ascii_lowercase();
    scratch.contains(needle)
}

/// A redirect target: bits to flip + the structural payload the surgical
/// apply needs (event index, new src, new dist symbol). Mirrors
/// `app::edit::EditKind::DistRedirect`.
struct RedirectSwap {
    event_idx: usize,
    bit_start: u32,
    code_len: u8,
    new_code: u16,
    out_pos: u32,
    copy_len: usize,
    old_src: u32,
    old_dist_sym: u8,
    new_src: u32,
    new_dist_sym: u8,
}

/// Find up to `n` ref events with a valid redirect target, the predicate
/// `app::select::build_ref_redirect_edits` uses to list UI alternatives.
fn find_redirect_swaps(
    events: &[Event],
    dist_encs: &[EncTable],
    block_starts: &[u32],
    n: usize,
) -> Vec<RedirectSwap> {
    let mut out = Vec::with_capacity(n);
    for (i, e) in events.iter().enumerate() {
        if out.len() >= n {
            break;
        }
        let Event::Ref(r) = e else { continue };
        let block = block_of(block_starts, i as u32);
        let alts = valid_dist_alts(block, r.dist_sym, r.out_pos, r.src_out_pos, dist_encs);
        let &(new_dsym, new_src, _new_dist) = alts.first().unwrap_or(&(0, 0, 0));
        if alts.is_empty() {
            continue;
        }
        let de = &dist_encs[block as usize];
        let Some(sc) = de.get(new_dsym as u16) else {
            continue;
        };
        out.push(RedirectSwap {
            event_idx: i,
            bit_start: r.dist_bit_start,
            code_len: sc.len,
            new_code: sc.code,
            out_pos: r.out_pos,
            copy_len: r.copy_len as usize,
            old_src: r.src_out_pos,
            old_dist_sym: r.dist_sym,
            new_src,
            new_dist_sym: new_dsym,
        });
    }
    out
}
