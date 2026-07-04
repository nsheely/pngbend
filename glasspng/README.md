# glasspng

A glass-box PNG codec: decode to pixels like any codec, or open the file up
and read the DEFLATE stream behind every pixel.

Most PNG libraries hand you a pixel buffer and throw away how it was
compressed. `glasspng` keeps that layer (every literal and LZ77
back-reference the decoder emitted, with bit offsets, plus the per-block
Huffman tables) so you can edit the compressed representation and re-emit a
valid PNG *without recompressing*. That's what makes bit-level databending and
lossless surgical edits possible on data a normal decode discards.

It decodes every colour type, bit depth, and Adam7 interlacing the format
defines. Zero dependencies, `std` only.

```rust
let img = glasspng::decode(&bytes)?;              // standard: bytes -> RGBA8
let out = glasspng::encode(&img, &Default::default())?; // pixels -> PNG
let gb  = glasspng::decode_with_events(&bytes)?;  // glass-box: + the DEFLATE event stream
```

Edit `gb.deflate.output`, then `gb.deflate.to_deflate()` (or `png::build_zlib_stream`
+ `png::write_chunks`) to save. The `bitstream` / `deflate` / `png` / `coords`
modules are public for stage-by-stage use.

The encoder writes the byte-aligned non-indexed colour types (grey,
grey+alpha, RGB, RGBA, 8 or 16-bit), compressing with greedy LZ77 and the
smallest of stored, fixed-Huffman, and dynamic-Huffman blocks. Indexed,
sub-byte, and interlaced output aren't emitted yet.

## License

MIT OR Apache-2.0
