# pngbend

![CI](https://github.com/nsheely/pngbend/actions/workflows/build.yml/badge.svg)
![License: MIT/Apache 2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-yellow.svg)
![Rust](https://img.shields.io/badge/rust-2024-orange.svg)

An interactive editor for the DEFLATE bits inside a PNG file. Click a pixel to see which literal or LZ77 back-reference produced each byte; swap a literal or redirect a back-reference; save the result as a valid PNG.

## Background

UCNV's [*The Art of PNG Glitch*](https://ucnv.github.io/pnglitch/) covers PNG glitching across multiple stages of the encode pipeline. pngbend works on the compressed DEFLATE data inside IDAT chunks, replacing random byte hits with same-width Huffman codeword swaps that keep the parser aligned.

## What It Does

- **Inspect** any pixel — see which literal or back-reference produced it.
- **Swap a literal** for another with the same Huffman code length. The change cascades through every back-reference that copies from it.
- **Redirect a back-reference** to a different distance symbol of the same code length and extra-bit count.
- **Visualise** literal/back-ref pixels, distance heatmap, block boundaries, cascade BFS overlay.
- **Save** as a valid PNG.

## Download

Grab the latest binary for your platform from the [Releases page](https://github.com/nsheely/pngbend/releases).

## Build from Source

```sh
cargo run --release                  # GUI
cargo run --release -- path/to.png   # GUI with file pre-loaded
cargo test                           # unit + integration + proptest
cargo bench                          # criterion benches
cargo clippy --all-targets -- -D warnings
```

Drag a PNG onto the window or use **File → Open**. **Ctrl+Z / Ctrl+Y** undo / redo. **Ctrl+Shift+S** save.

## Project Structure

```
src/
├── main.rs              # GUI entry
├── lib.rs               # library root
├── bitstream.rs         # LSB-first bit reader/writer + bit patcher
├── coords.rs            # OutPos / PixelXY / BitPos newtypes
├── composite.rs         # SIMD alpha compositor
├── deflate/             # RFC 1951 decoder + per-event tracking
├── png/                 # chunks, CRC, zlib wrap, IHDR, row filters
├── index/               # pos_to_ev, reverse_graph, cascade, pixel
├── overlays/            # event / cascade / composite overlays
├── app/                 # GUI (edit, history, panels, IO, ...)
└── bin/profile.rs       # perf-driven profiling binary
```

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache 2.0](LICENSE-APACHE).
