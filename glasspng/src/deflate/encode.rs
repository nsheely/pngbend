//! DEFLATE serialization, the inverse of [`super::decode_deflate`].
//!
//! Emits a raw DEFLATE stream (no zlib wrapper) three ways:
//! - [`serialize_stored`]: BTYPE=00 uncompressed blocks from raw bytes, one
//!   per 65535-byte run.
//! - [`serialize_fixed`]: single BTYPE=01 fixed-Huffman block from an
//!   [`Event`] stream, preserving its back-references.
//! - [`serialize_dynamic`]: single BTYPE=10 block with package-merge-optimal
//!   code lengths for the event stream's symbol frequencies.
//!
//! [`compress`] runs greedy LZ77 ([`lz77`]) over raw bytes and picks the
//! smallest encoding: dynamic vs stored always, fixed only on small input.
//! [`super::DecodedDeflate::to_deflate`] re-serializes an edited event stream
//! as fixed Huffman.
//!
//! Bit-ordering hazard, contained in [`DeflateWriter`]: Huffman codes go out
//! MSB-first (matching the decoder's reversed-peek LUT); every other field
//! (block header, extra bits, stored LEN/NLEN) goes out LSB-first so
//! `BitReader::read_bits` recovers it.

use crate::bitstream::write_bits;

use super::DecodedDeflate;
use super::constants::{CLORDER, DBASE, DEXT, LBASE, LEXT, symbol_index};
use super::events::{EncTable, Event, SymCode};
use super::huffman::{build_tree, enc_from_lengths, fixed_code_lengths, package_merge};
use super::lz77::lz77;

/// Growable bit sink for a DEFLATE stream.
struct DeflateWriter {
    buf: Vec<u8>,
    bit_pos: usize,
}

impl DeflateWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            bit_pos: 0,
        }
    }

    #[inline]
    fn ensure(&mut self, extra_bits: usize) {
        let need = (self.bit_pos + extra_bits).div_ceil(8);
        if self.buf.len() < need {
            self.buf.resize(need, 0);
        }
    }

    /// Emit a Huffman codeword MSB-first (the decoder LUT is keyed on the
    /// reversed peek).
    #[inline]
    fn write_code(&mut self, sc: SymCode) {
        self.ensure(sc.len as usize);
        write_bits(&mut self.buf, self.bit_pos, sc.code as u32, sc.len);
        self.bit_pos += sc.len as usize;
    }

    /// Emit `n` low bits of `value` LSB-first, so `read_bits(n)` recovers it.
    /// `write_bits` writes MSB-first, so the value is pre-reversed.
    #[inline]
    fn write_lsb(&mut self, value: u32, n: u32) {
        if n == 0 {
            return;
        }
        self.ensure(n as usize);
        write_bits(&mut self.buf, self.bit_pos, reverse_bits(value, n), n as u8);
        self.bit_pos += n as usize;
    }

    fn align_to_byte(&mut self) {
        let rem = self.bit_pos % 8;
        if rem != 0 {
            self.bit_pos += 8 - rem;
        }
        self.ensure(0);
    }

    /// Append whole bytes; only valid when byte-aligned (after
    /// [`align_to_byte`]).
    fn write_bytes(&mut self, bytes: &[u8]) {
        debug_assert_eq!(self.bit_pos % 8, 0);
        self.buf.truncate(self.bit_pos / 8);
        self.buf.extend_from_slice(bytes);
        self.bit_pos += bytes.len() * 8;
    }

    fn finish(mut self) -> Vec<u8> {
        self.ensure(0);
        self.buf
    }
}

#[inline]
fn reverse_bits(v: u32, n: u32) -> u32 {
    let mut r = 0u32;
    for i in 0..n {
        r |= ((v >> i) & 1) << (n - 1 - i);
    }
    r
}

/// Map a copy length (3..=258) to its `(length symbol, extra value, extra
/// bit count)`. The on-wire literal-alphabet symbol is `257 + i`.
fn length_to_sym(len: u16) -> (u16, u32, u32) {
    let len = len as u32;
    let i = symbol_index(&LBASE, len);
    (257 + i as u16, len - LBASE[i], LEXT[i])
}

/// The fixed lit/dist Huffman encoder tables (RFC 1951 §3.2.6), built once
/// per call from the shared [`fixed_code_lengths`].
fn fixed_tables() -> (EncTable, EncTable) {
    let (ll, dl) = fixed_code_lengths();
    let (_, lit_enc) = build_tree(&ll).expect("fixed literal lengths are valid");
    let (_, dist_enc) = build_tree(&dl).expect("fixed distance lengths are valid");
    (lit_enc, dist_enc)
}

/// Emit an [`Event`] stream followed by the end-of-block symbol, using
/// `lit_enc` for literals and length symbols and `dist_enc` for distances.
/// Shared by the fixed- and dynamic-Huffman serializers.
fn write_events(w: &mut DeflateWriter, events: &[Event], lit_enc: &EncTable, dist_enc: &EncTable) {
    for e in events {
        match e {
            Event::Lit(l) => w.write_code(lit_enc.get(l.symbol as u16).expect("literal in table")),
            Event::Ref(r) => {
                let (lsym, lextra, lnbits) = length_to_sym(r.copy_len);
                w.write_code(lit_enc.get(lsym).expect("length symbol in table"));
                w.write_lsb(lextra, lnbits);
                let dsym = r.dist_sym as u16;
                w.write_code(dist_enc.get(dsym).expect("distance symbol in table"));
                let dist = r.out_pos - r.src_out_pos;
                w.write_lsb(dist - DBASE[dsym as usize], DEXT[dsym as usize]);
            }
        }
    }
    w.write_code(lit_enc.get(256).expect("end-of-block symbol")); // 256 = EOB
}

/// Serialize `output` as a chain of uncompressed (BTYPE=00) blocks, one per
/// 65535-byte run. Correct for any bytes; no compression.
pub fn serialize_stored(output: &[u8]) -> Vec<u8> {
    let mut w = DeflateWriter::new();
    if output.is_empty() {
        w.write_lsb(1, 1); // BFINAL
        w.write_lsb(0, 2); // BTYPE = 00 (stored)
        w.align_to_byte();
        w.write_bytes(&[0, 0, 0xFF, 0xFF]); // LEN = 0, NLEN = ~0
        return w.finish();
    }
    let mut chunks = output.chunks(0xFFFF).peekable();
    while let Some(chunk) = chunks.next() {
        let is_final = chunks.peek().is_none();
        w.write_lsb(is_final as u32, 1);
        w.write_lsb(0, 2);
        w.align_to_byte();
        let len = chunk.len() as u16;
        w.write_bytes(&len.to_le_bytes());
        w.write_bytes(&(!len).to_le_bytes());
        w.write_bytes(chunk);
    }
    w.finish()
}

/// Serialize an [`Event`] stream as a single fixed-Huffman (BTYPE=01)
/// block, preserving its literals and back-references.
pub fn serialize_fixed(events: &[Event]) -> Vec<u8> {
    let (lit_enc, dist_enc) = fixed_tables();
    let mut w = DeflateWriter::new();
    w.write_lsb(1, 1); // BFINAL = 1 (single block)
    w.write_lsb(1, 2); // BTYPE = 01 (fixed Huffman)
    write_events(&mut w, events, &lit_enc, &dist_enc);
    w.finish()
}

impl DecodedDeflate {
    /// Re-serialize this decode's [`Event`] stream to a fixed-Huffman
    /// DEFLATE stream. `decode_deflate(x.to_deflate()).output == x.output`.
    pub fn to_deflate(&self) -> Vec<u8> {
        serialize_fixed(&self.events)
    }
}

/// Threshold below which the fixed-Huffman encoding is also trialled.
/// Dynamic Huffman is never worse than fixed on larger input (its optimal
/// codes dominate once the header amortizes), so fixed only competes on
/// small input where it avoids the per-block header table.
const FIXED_TRIAL_LIMIT: usize = 4096;

/// Compress `data` to a DEFLATE stream via greedy LZ77, choosing the
/// smallest valid encoding. Dynamic Huffman wins on real images; stored
/// guards incompressible input, and fixed Huffman is trialled only on small
/// input (its full re-serialization isn't worth it once dynamic dominates).
pub fn compress(data: &[u8]) -> Vec<u8> {
    let events = lz77(data);
    let dynamic = serialize_dynamic(&events);
    let stored = serialize_stored(data);
    let mut best = if dynamic.len() <= stored.len() {
        dynamic
    } else {
        stored
    };
    if data.len() < FIXED_TRIAL_LIMIT {
        let fixed = serialize_fixed(&events);
        if fixed.len() < best.len() {
            best = fixed;
        }
    }
    best
}

/// Run-length encode a code-length sequence into `(symbol, extra)` pairs
/// over the code-length alphabet (RFC 1951 §3.2.7): 16 repeats the previous
/// length 3..=6 times, 17 repeats zero 3..=10, 18 repeats zero 11..=138.
fn rle_code_lengths(lengths: &[u8]) -> Vec<(u8, u8)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < lengths.len() {
        let v = lengths[i];
        let mut run = 1;
        while i + run < lengths.len() && lengths[i + run] == v {
            run += 1;
        }
        i += run;
        if v == 0 {
            while run >= 11 {
                let take = run.min(138);
                out.push((18, (take - 11) as u8));
                run -= take;
            }
            while run >= 3 {
                let take = run.min(10);
                out.push((17, (take - 3) as u8));
                run -= take;
            }
            for _ in 0..run {
                out.push((0, 0));
            }
        } else {
            out.push((v, 0));
            run -= 1;
            while run >= 3 {
                let take = run.min(6);
                out.push((16, (take - 3) as u8));
                run -= take;
            }
            for _ in 0..run {
                out.push((v, 0));
            }
        }
    }
    out
}

/// Count of `lengths` after trimming trailing zeros, floored at `floor`.
/// HLIT / HDIST are this over the lit / dist code-length arrays. Their
/// floors are the RFC 1951 minimums: 257 literal/length codes (HLIT is
/// transmitted as count - 257) and 1 distance code.
fn used_len(lengths: &[u8], floor: usize) -> usize {
    lengths
        .iter()
        .rposition(|&l| l != 0)
        .map_or(0, |i| i + 1)
        .max(floor)
}

/// Serialize an [`Event`] stream as a single dynamic-Huffman (BTYPE=10)
/// block with optimal code lengths for its symbol frequencies.
pub fn serialize_dynamic(events: &[Event]) -> Vec<u8> {
    let mut lit_freq = [0u32; 286];
    let mut dist_freq = [0u32; 30];
    for e in events {
        match e {
            Event::Lit(l) => lit_freq[l.symbol as usize] += 1,
            Event::Ref(r) => {
                let (lsym, _, _) = length_to_sym(r.copy_len);
                lit_freq[lsym as usize] += 1;
                dist_freq[r.dist_sym as usize] += 1;
            }
        }
    }
    lit_freq[256] += 1; // end-of-block, emitted once

    let lit_lengths = package_merge(&lit_freq, 15);
    let mut dist_lengths = package_merge(&dist_freq, 15);
    if dist_lengths.iter().all(|&l| l == 0) {
        dist_lengths[0] = 1; // DEFLATE requires at least one distance code
    }

    let hlit = used_len(&lit_lengths, 257);
    let hdist = used_len(&dist_lengths, 1);

    let combined: Vec<u8> = lit_lengths[..hlit]
        .iter()
        .chain(&dist_lengths[..hdist])
        .copied()
        .collect();
    let rle = rle_code_lengths(&combined);

    let mut cl_freq = [0u32; 19];
    for &(sym, _) in &rle {
        cl_freq[sym as usize] += 1;
    }
    let cl_lengths = package_merge(&cl_freq, 7);
    let hclen = (0..19)
        .rev()
        .find(|&p| cl_lengths[CLORDER[p]] != 0)
        .map_or(4, |p| (p + 1).max(4));

    let lit_enc = enc_from_lengths(&lit_lengths);
    let dist_enc = enc_from_lengths(&dist_lengths);
    let cl_enc = enc_from_lengths(&cl_lengths);

    let mut w = DeflateWriter::new();
    w.write_lsb(1, 1); // BFINAL
    w.write_lsb(2, 2); // BTYPE = 10 (dynamic)
    w.write_lsb(hlit as u32 - 257, 5);
    w.write_lsb(hdist as u32 - 1, 5);
    w.write_lsb(hclen as u32 - 4, 4);
    for &clorder in CLORDER.iter().take(hclen) {
        w.write_lsb(cl_lengths[clorder] as u32, 3);
    }
    for &(sym, extra) in &rle {
        w.write_code(cl_enc.get(sym as u16).expect("code-length symbol present"));
        match sym {
            16 => w.write_lsb(extra as u32, 2),
            17 => w.write_lsb(extra as u32, 3),
            18 => w.write_lsb(extra as u32, 7),
            _ => {}
        }
    }
    write_events(&mut w, events, &lit_enc, &dist_enc);
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::super::lz77::lz77;
    use super::super::{LitEvent, RefEvent, decode_deflate};
    use super::*;

    #[test]
    fn stored_round_trips_including_empty_and_multiblock() {
        for data in [
            vec![],
            vec![42u8],
            b"hello deflate".to_vec(),
            (0..70_000u32).map(|i| (i * 7) as u8).collect(), // > 65535: multi-block
        ] {
            let deflate = serialize_stored(&data);
            let decoded = decode_deflate(&deflate, None).expect("decode stored");
            assert_eq!(decoded.output, data, "len {}", data.len());
        }
    }

    #[test]
    fn fixed_round_trips_literals_and_a_backref() {
        // "a" then copy 3 bytes from distance 1 -> "aaaa".
        let events = vec![
            Event::Lit(LitEvent {
                out_pos: 0,
                bit_start: 0,
                symbol: b'a',
            }),
            Event::Ref(RefEvent {
                out_pos: 1,
                src_out_pos: 0,
                dist_bit_start: 0,
                copy_len: 3,
                dist_sym: 0, // DBASE[0] = 1
            }),
        ];
        let deflate = serialize_fixed(&events);
        let decoded = decode_deflate(&deflate, None).expect("decode fixed");
        assert_eq!(decoded.output, b"aaaa");
    }

    #[test]
    fn to_deflate_round_trips_a_real_stream() {
        // Decode a stored stream to get an event list, re-serialize it as
        // fixed Huffman, and confirm the output survives.
        let original: Vec<u8> = b"the quick brown fox the quick brown fox".to_vec();
        let decoded = decode_deflate(&serialize_stored(&original), None).unwrap();
        let refixed = decode_deflate(&decoded.to_deflate(), None).unwrap();
        assert_eq!(refixed.output, original);
    }

    #[test]
    fn length_symbol_boundaries() {
        assert_eq!(length_to_sym(3), (257, 0, 0));
        assert_eq!(length_to_sym(258), (285, 0, 0));
        // Length 11 falls in symbol 265 (base 11, 1 extra bit).
        let (sym, extra, nbits) = length_to_sym(12);
        assert_eq!((sym, extra, nbits), (265, 1, 1));
    }

    #[test]
    fn compress_round_trips_and_shrinks_repetitive_data() {
        // Highly repetitive input: LZ77 should beat a stored block by a lot.
        let repetitive: Vec<u8> = b"abcabcabc".iter().cycle().take(4096).copied().collect();
        let out = compress(&repetitive);
        assert_eq!(decode_deflate(&out, None).unwrap().output, repetitive);
        assert!(
            out.len() < repetitive.len() / 4,
            "compressed {} of {}",
            out.len(),
            repetitive.len()
        );

        // Overlapping run (distance 1) and arbitrary bytes still round-trip.
        let run = vec![7u8; 300];
        assert_eq!(decode_deflate(&compress(&run), None).unwrap().output, run);
        let arbitrary: Vec<u8> = (0..1000u32).map(|i| (i * 37 + 11) as u8).collect();
        assert_eq!(
            decode_deflate(&compress(&arbitrary), None).unwrap().output,
            arbitrary
        );
    }

    #[test]
    fn dynamic_round_trips_and_beats_fixed_on_varied_data() {
        // Pseudo-random bytes over 32 symbols: LZ77 finds few matches, so
        // ~all literals. Dynamic's 5-bit codes beat fixed's 8-bit literals,
        // and the volume amortizes the header.
        let data: Vec<u8> = (0..20_000u32)
            .map(|i| ((i.wrapping_mul(2_654_435_761) >> 24) % 32) as u8)
            .collect();
        let events = lz77(&data);
        let dynamic = serialize_dynamic(&events);
        assert_eq!(decode_deflate(&dynamic, None).unwrap().output, data);
        assert!(
            dynamic.len() < serialize_fixed(&events).len(),
            "dynamic {} vs fixed {}",
            dynamic.len(),
            serialize_fixed(&events).len()
        );
        // compress() picks the smallest encoding and still round-trips.
        let best = compress(&data);
        assert!(best.len() <= dynamic.len());
        assert_eq!(decode_deflate(&best, None).unwrap().output, data);
    }
}
