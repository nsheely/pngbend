//! zlib wrapping (RFC 1950) around a raw DEFLATE stream.

use super::chunks::Warning;

/// Fatal errors from [`parse_zlib_stream`]: cases where the wrapper
/// bytes won't slice cleanly into a deflate buffer. FCHECK and the
/// trailing Adler-32 are checksums that surface as warnings instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZlibError {
    Truncated { actual: usize },
    BadCompressionMethod { cmf: u8 },
    FdictSet,
}

impl std::fmt::Display for ZlibError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated { actual } => {
                write!(f, "zlib stream truncated ({actual} bytes, need ≥ 6)")
            }
            Self::BadCompressionMethod { cmf } => {
                write!(f, "zlib CMF {cmf:#04x}: PNG requires CM=8 and CINFO≤7")
            }
            Self::FdictSet => write!(f, "zlib FDICT bit set (forbidden in PNG)"),
        }
    }
}

impl std::error::Error for ZlibError {}

/// Output of [`parse_zlib_stream`]: the three pieces of the wrapped
/// deflate stream plus any header-level warnings (currently FCHECK).
#[derive(Debug)]
pub struct ParsedZlib<'a> {
    pub header: [u8; 2],
    pub deflate_buf: &'a [u8],
    pub stored_adler: u32,
    pub warnings: Vec<Warning>,
}

/// Split an IDAT-concatenated zlib stream. Fails when the slice itself
/// would be wrong (truncation, non-deflate CM, FDICT); a non-validating
/// FCHECK is returned as a warning so the file still loads.
pub fn parse_zlib_stream(idat: &[u8]) -> Result<ParsedZlib<'_>, ZlibError> {
    if idat.len() < 6 {
        return Err(ZlibError::Truncated { actual: idat.len() });
    }
    let cmf = idat[0];
    let flg = idat[1];
    // CM (low nibble) must be 8 (deflate); CINFO (high nibble) <= 7 caps the
    // window at 32 KiB. Anything else is not a PNG zlib stream.
    if cmf & 0x0F != 8 || cmf >> 4 > 7 {
        return Err(ZlibError::BadCompressionMethod { cmf });
    }
    if (flg >> 5) & 1 == 1 {
        return Err(ZlibError::FdictSet);
    }
    let mut warnings = Vec::new();
    if (u16::from(cmf) * 256 + u16::from(flg)) % 31 != 0 {
        warnings.push(Warning::ZlibHeaderChecksum);
    }
    let deflate_buf = &idat[2..idat.len() - 4];
    let trailer = &idat[idat.len() - 4..];
    let stored_adler = u32::from_be_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
    Ok(ParsedZlib {
        header: [cmf, flg],
        deflate_buf,
        stored_adler,
        warnings,
    })
}

/// Default zlib header the encoder writes: CMF=0x78 (deflate, 32 KiB window),
/// FLG=0x9c (default compression level, FDICT clear, FCHECK chosen so the two
/// bytes are a multiple of 31). The re-emit path instead reuses a file's own
/// header bytes, so this is only the fresh-encode default.
pub const ZLIB_DEFAULT_HEADER: [u8; 2] = [0x78, 0x9c];

/// Wrap `deflate_buf` in a zlib stream: 2-byte `header` prefix +
/// `deflate_buf` + 4-byte Adler-32 trailer computed over `raw`. The
/// caller passes the already-decoded raw bytes so saving doesn't have
/// to re-inflate the stream just to compute the Adler-32.
pub fn build_zlib_stream(deflate_buf: &[u8], header: [u8; 2], raw: &[u8]) -> Vec<u8> {
    let adler = adler32(raw);
    let mut out = Vec::with_capacity(2 + deflate_buf.len() + 4);
    out.extend_from_slice(&header);
    out.extend_from_slice(deflate_buf);
    out.extend_from_slice(&adler.to_be_bytes());
    out
}

// Adler-32 (RFC 1950)

pub fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let mut s1: u32 = 1;
    let mut s2: u32 = 0;
    // 5552 is the largest n such that 255*n*(n+1)/2 + (n+1)*(BASE-1) < 2^32
    // (RFC 1950); batching keeps the accumulators below u32 max.
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
        // > 5552 bytes: exercise the batch boundary
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

    /// A zlib IDAT with the default header and a correct Adler-32 over `decoded`.
    fn build_idat(decoded: &[u8], deflate: &[u8]) -> Vec<u8> {
        build_zlib_stream(deflate, ZLIB_DEFAULT_HEADER, decoded)
    }

    #[test]
    fn parse_zlib_stream_accepts_canonical_header() {
        let idat = build_idat(b"hello", b"\x00\x01\x02");
        let parsed = parse_zlib_stream(&idat).expect("valid header");
        assert_eq!(parsed.header, [0x78, 0x9C]);
        assert_eq!(parsed.deflate_buf, b"\x00\x01\x02");
        assert_eq!(parsed.stored_adler, adler32(b"hello"));
        assert!(parsed.warnings.is_empty());
    }

    #[test]
    fn parse_zlib_stream_rejects_truncated() {
        let err = parse_zlib_stream(&[0x78]).unwrap_err();
        assert!(matches!(err, ZlibError::Truncated { actual: 1 }));
    }

    #[test]
    fn parse_zlib_stream_rejects_non_deflate_cm() {
        // CM=1 (reserved), CINFO=7. CMF = 0x71. Any FLG making FCHECK pass.
        let cmf = 0x71u8;
        let flg = ((31 - (u16::from(cmf) * 256) % 31) % 31) as u8;
        let mut idat = vec![cmf, flg];
        idat.extend_from_slice(&[0u8; 8]);
        let err = parse_zlib_stream(&idat).unwrap_err();
        assert!(
            matches!(err, ZlibError::BadCompressionMethod { cmf: 0x71 }),
            "got {err:?}"
        );
    }

    #[test]
    fn parse_zlib_stream_surfaces_bad_fcheck_as_warning() {
        // CMF=0x78 valid, FLG=0x00 makes (0x78*256+0)%31 != 0. The
        // stream should still parse: FCHECK is an integrity check on
        // header bytes whose values are otherwise sane.
        let mut idat = vec![0x78, 0x00];
        idat.extend_from_slice(&[0u8; 8]);
        let parsed = parse_zlib_stream(&idat).expect("FCHECK is a warning, not an error");
        assert_eq!(parsed.warnings.len(), 1);
        assert_eq!(parsed.warnings[0], Warning::ZlibHeaderChecksum);
    }

    #[test]
    fn parse_zlib_stream_rejects_fdict_set() {
        // CMF=0x78, FLG with FDICT bit (1<<5) set, FCHECK adjusted to pass.
        let cmf = 0x78u8;
        let mut flg = 0x20u8; // FDICT
        let target = 31 - (u16::from(cmf) * 256 + u16::from(flg & 0xE0)) % 31;
        flg = (flg & 0xE0) | (target % 31) as u8;
        assert_eq!((u16::from(cmf) * 256 + u16::from(flg)) % 31, 0);
        let mut idat = vec![cmf, flg];
        idat.extend_from_slice(&[0u8; 8]);
        let err = parse_zlib_stream(&idat).unwrap_err();
        assert!(matches!(err, ZlibError::FdictSet), "got {err:?}");
    }
}
