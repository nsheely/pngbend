//! zlib wrapping (RFC 1950) around a raw DEFLATE stream.

use crate::deflate::{DecodeError, decode_deflate};

/// Wrap `deflate_buf` in a zlib stream: 2-byte `header` prefix +
/// `deflate_buf` + 4-byte Adler-32 trailer computed over `raw`. The
/// caller passes the already-decoded raw bytes so saving doesn't have
/// to re-inflate the stream just to compute the Adler-32.
pub fn build_zlib_stream(deflate_buf: &[u8], header: &[u8], raw: &[u8]) -> Vec<u8> {
    let adler = adler32(raw);
    let mut out = Vec::with_capacity(2 + deflate_buf.len() + 4);
    out.extend_from_slice(&header[..2.min(header.len())]);
    out.extend_from_slice(deflate_buf);
    out.extend_from_slice(&adler.to_be_bytes());
    out
}

/// Decompress a raw deflate stream (no zlib header/trailer).
pub fn inflate_raw(deflate: &[u8]) -> Result<Vec<u8>, DecodeError> {
    Ok(decode_deflate(deflate)?.output)
}

// ── Adler-32 (RFC 1950) ──────────────────────────────────────────────────────

fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let mut s1: u32 = 1;
    let mut s2: u32 = 0;
    // 5552 is the largest n such that 255*n*(n+1)/2 + (n+1)*(BASE-1) < 2^32
    // (RFC 1950) — batching keeps the accumulators below u32 max.
    for chunk in data.chunks(5552) {
        for &b in chunk {
            s1 = s1.wrapping_add(b as u32);
            s2 = s2.wrapping_add(s1);
        }
        s1 %= MOD;
        s2 %= MOD;
    }
    (s2 << 16) | s1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adler32_known_vectors() {
        assert_eq!(adler32(b""), 1);
        assert_eq!(adler32(b"a"), 0x00620062);
        assert_eq!(adler32(b"Wikipedia"), 0x11E60398);
    }

    #[test]
    fn adler32_handles_long_input() {
        // > 5552 bytes — exercise the batch boundary
        let data = vec![0xAAu8; 6000];
        let computed = adler32(&data);
        const MOD: u32 = 65521;
        let mut s1: u32 = 1;
        let mut s2: u32 = 0;
        for &b in &data {
            s1 = (s1 + b as u32) % MOD;
            s2 = (s2 + s1) % MOD;
        }
        assert_eq!(computed, (s2 << 16) | s1);
    }
}
