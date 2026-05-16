//! Criterion benchmarks for the hot paths: deflate decode, reverse-graph
//! construction, cascade BFS, and the integer alpha compositor.

use std::path::PathBuf;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use pngbend::composite::composite_rgba;
use pngbend::coords::ImgGeom;
use pngbend::deflate::decode_deflate;
use pngbend::index::{CascadeScratch, build_pixel_index, build_pos_to_ev, build_reverse_graph};
use pngbend::png::{PngInfo, concat_idat, parse_ihdr, read_chunks};

fn sample_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("samples")
        .join("checksum_sunset.png")
}

fn sample_raw() -> Vec<u8> {
    std::fs::read(sample_path()).expect("read sample png")
}

fn sample_deflate() -> Vec<u8> {
    let raw = sample_raw();
    let chunks = read_chunks(&raw).expect("chunks");
    let idat = concat_idat(&chunks);
    idat[2..idat.len() - 4].to_vec()
}

fn sample_info() -> PngInfo {
    let raw = sample_raw();
    let chunks = read_chunks(&raw).expect("chunks");
    parse_ihdr(&chunks).expect("parse ihdr")
}

fn bench_decode_deflate(c: &mut Criterion) {
    let deflate = sample_deflate();
    c.bench_function("decode_deflate/sample", |b| {
        b.iter(|| {
            let decoded = decode_deflate(black_box(&deflate), None).expect("decode");
            black_box(decoded.output.len());
        });
    });
}

fn bench_build_reverse_graph(c: &mut Criterion) {
    let deflate = sample_deflate();
    let decoded = decode_deflate(&deflate, None).expect("decode");
    c.bench_function("build_reverse_graph/sample", |b| {
        b.iter(|| {
            let rg = build_reverse_graph(black_box(&decoded.events), decoded.output.len());
            black_box(rg.len());
        });
    });
}

fn bench_cascade_bfs(c: &mut Criterion) {
    let deflate = sample_deflate();
    let decoded = decode_deflate(&deflate, None).expect("decode");
    let rev = build_reverse_graph(&decoded.events, decoded.output.len());

    // Seed the BFS with a handful of positions near the start — this
    // reaches a substantial fraction of the output via forward propagation.
    let seeds: Vec<u32> = (0..16).map(|i| (i * 257) as u32).collect();

    c.bench_function("cascade_bfs/sample", |b| {
        // Reuse the scratch across iterations to mirror the production
        // path — the GUI keeps one scratch per loaded file and reuses
        // it on every click via the epoch-versioned invalidation.
        let mut scratch = CascadeScratch::new();
        b.iter(|| {
            let cascade = scratch.run(black_box(&seeds), black_box(&rev));
            black_box(cascade.affected.len());
        });
    });
}

fn bench_composite_rgba(c: &mut Criterion) {
    // 512x512 synthetic image and overlay.
    let n = 512 * 512 * 4;
    let base: Vec<u8> = (0..n).map(|i| (i & 0xFF) as u8).collect();
    let overlay: Vec<u8> = (0..n).map(|i| ((i * 3) & 0xFF) as u8).collect();
    c.bench_function("composite_rgba/512x512", |b| {
        b.iter(|| {
            let out = composite_rgba(black_box(&base), black_box(&overlay));
            black_box(out.len());
        });
    });
}

fn bench_build_pos_to_ev(c: &mut Criterion) {
    let deflate = sample_deflate();
    let decoded = decode_deflate(&deflate, None).expect("decode");
    c.bench_function("build_pos_to_ev/sample", |b| {
        b.iter(|| {
            let p = build_pos_to_ev(black_box(&decoded.events), decoded.output.len());
            black_box(p.len());
        });
    });
}

fn bench_parse_ihdr(c: &mut Criterion) {
    let raw = sample_raw();
    let chunks = read_chunks(&raw).expect("chunks");
    c.bench_function("parse_ihdr/sample", |b| {
        b.iter(|| {
            let info = parse_ihdr(black_box(&chunks)).expect("parse");
            black_box(ImgGeom::new(info.width, info.height, info.bits_per_pixel()));
        });
    });
}

/// `build_pixel_index` runs once per file load and dominates wall-clock
/// time for that load. It also scales superlinearly with pixel count
/// thanks to the per-block alphabet precomputes — instrumenting it
/// here gives the per-pixel coefficient most pinches show up in.
fn bench_build_pixel_index(c: &mut Criterion) {
    let deflate = sample_deflate();
    let decoded = decode_deflate(&deflate, None).expect("decode");
    let info = sample_info();
    let geom = ImgGeom::new(info.width, info.height, info.bits_per_pixel());
    let pos_to_ev = build_pos_to_ev(&decoded.events, decoded.output.len());

    c.bench_function("build_pixel_index/sample", |b| {
        b.iter(|| {
            let pi = build_pixel_index(
                black_box(&decoded.events),
                black_box(&decoded.output),
                black_box(&pos_to_ev),
                black_box(&decoded.lit_encs),
                black_box(&decoded.dist_encs),
                black_box(&geom),
            );
            black_box(pi.lit.len() + pi.refs.len());
        });
    });
}

criterion_group!(
    benches,
    bench_decode_deflate,
    bench_build_reverse_graph,
    bench_cascade_bfs,
    bench_composite_rgba,
    bench_build_pos_to_ev,
    bench_parse_ihdr,
    bench_build_pixel_index,
);
criterion_main!(benches);
