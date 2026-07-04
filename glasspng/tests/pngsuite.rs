//! Differential conformance against the reference `png` crate over the
//! official PngSuite corpus (every colour type, bit depth, interlacing,
//! filter, and a set of deliberately corrupt files).
//!
//! For each valid image both decoders must agree on dimensions and RGBA8
//! pixels; for each corrupt `x*` image `glasspng` must error, not panic.
//! The corpus lives in `tests/pngsuite/`; the test skips cleanly if absent.

use std::path::{Path, PathBuf};

/// Decode `bytes` with the `png` crate, normalised to RGBA8 the same way
/// `glasspng` produces it: EXPAND sub-byte/palette/tRNS, replicate luma,
/// and take the high byte of 16-bit samples. `None` if `png` can't decode.
fn reference_rgba8(bytes: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let mut decoder = png::Decoder::new(bytes);
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let data = &buf[..info.buffer_size()];

    let (w, h) = (info.width, info.height);
    let n = (w as usize) * (h as usize);
    let step = if info.bit_depth == png::BitDepth::Sixteen {
        2
    } else {
        1
    };
    // High byte of global sample index `s` (the byte itself at 8-bit).
    let hi = |s: usize| data[s * step];

    let mut rgba = vec![0u8; n * 4];
    match info.color_type {
        png::ColorType::Grayscale => {
            for i in 0..n {
                let v = hi(i);
                rgba[i * 4..i * 4 + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        png::ColorType::GrayscaleAlpha => {
            for i in 0..n {
                let (v, a) = (hi(i * 2), hi(i * 2 + 1));
                rgba[i * 4..i * 4 + 4].copy_from_slice(&[v, v, v, a]);
            }
        }
        png::ColorType::Rgb => {
            for i in 0..n {
                let (r, g, b) = (hi(i * 3), hi(i * 3 + 1), hi(i * 3 + 2));
                rgba[i * 4..i * 4 + 4].copy_from_slice(&[r, g, b, 255]);
            }
        }
        png::ColorType::Rgba => {
            for i in 0..n {
                for c in 0..4 {
                    rgba[i * 4 + c] = hi(i * 4 + c);
                }
            }
        }
        // EXPAND turns Indexed into Rgb/Rgba, so this shouldn't occur.
        png::ColorType::Indexed => return None,
    }
    Some((w, h, rgba))
}

fn corpus_dir() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pngsuite");
    dir.is_dir().then_some(dir)
}

/// Human-readable colour-type / depth / interlacing decoded from a PngSuite
/// filename (spec `...[n|i][N][c][DD]`), for readable failure output.
fn describe(name: &str) -> String {
    let stem = name.strip_suffix(".png").unwrap_or(name);
    let b = stem.as_bytes();
    if b.len() < 5 {
        return stem.to_string();
    }
    let colour = match b[b.len() - 4] {
        b'0' => "grayscale",
        b'2' => "rgb",
        b'3' => "palette",
        b'4' => "gray+alpha",
        b'6' => "rgba",
        _ => "?",
    };
    let depth = stem[stem.len() - 2..].trim_start_matches('0');
    let interlace = if b[b.len() - 5] == b'i' {
        "interlaced"
    } else {
        "non-interlaced"
    };
    format!("{colour} {depth}-bit {interlace}")
}

#[test]
fn glasspng_matches_png_crate_over_pngsuite() {
    let Some(dir) = corpus_dir() else {
        eprintln!("skipping: no tests/pngsuite corpus present");
        return;
    };

    let mut checked = 0usize;
    let mut corrupt_checked = 0usize;
    let mut failures: Vec<String> = Vec::new();

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read corpus dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    paths.sort();

    for path in paths {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let bytes = std::fs::read(&path).unwrap();
        let ours = glasspng::decode(&bytes);

        // PngSuite names corrupt files with a leading `x`. glasspng
        // deliberately tolerates stale checksums (the glitch premise), so
        // the bad-IDAT-CRC (`xcs`) and bad-IHDR-CRC (`xhd`) files decode
        // with a warning; every structurally-corrupt file must error.
        if name.starts_with('x') {
            corrupt_checked += 1;
            let checksum_only = name.starts_with("xcs") || name.starts_with("xhd");
            match (checksum_only, &ours) {
                (true, Ok(img)) if !img.warnings.is_empty() => {
                    // The tolerant path warns; the strict path must reject.
                    if !matches!(
                        glasspng::decode_strict(&bytes),
                        Err(glasspng::PngError::BadChecksum(_))
                    ) {
                        failures.push(format!(
                            "{name}: decode_strict should reject a bad checksum"
                        ));
                    }
                }
                (true, _) => failures.push(format!(
                    "{name}: checksum-corrupt should decode with a warning"
                )),
                (false, Ok(_)) => failures.push(format!(
                    "{name}: structurally corrupt, decoded Ok (want error)"
                )),
                (false, Err(_)) => {}
            }
            continue;
        }

        let Some((w, h, expected)) = reference_rgba8(&bytes) else {
            continue; // png crate couldn't decode it either; nothing to compare
        };
        let kind = describe(&name);
        match ours {
            Err(e) => failures.push(format!("{name} ({kind}): glasspng errored: {e}")),
            Ok(img) => {
                if (img.info.width, img.info.height) != (w, h) {
                    failures.push(format!(
                        "{name} ({kind}): dims {}x{} vs png {w}x{h}",
                        img.info.width, img.info.height
                    ));
                } else if img.pixels != expected {
                    let at = img
                        .pixels
                        .iter()
                        .zip(&expected)
                        .position(|(a, b)| a != b)
                        .unwrap_or(0);
                    failures.push(format!(
                        "{name} ({kind}): pixel mismatch at byte {at} (ours={} png={})",
                        img.pixels[at], expected[at]
                    ));
                } else {
                    checked += 1;
                    // A valid image must also pass the strict decoder.
                    if let Err(e) = glasspng::decode_strict(&bytes) {
                        failures.push(format!(
                            "{name} ({kind}): decode_strict false-rejected: {e}"
                        ));
                    }
                }
            }
        }
    }

    eprintln!(
        "PngSuite: {checked} valid images matched the png crate, {corrupt_checked} corrupt files handled"
    );

    assert!(
        failures.is_empty(),
        "{} PngSuite mismatch(es):\n{}",
        failures.len(),
        failures.join("\n")
    );
    // Sanity: we actually exercised the corpus, not an empty directory.
    assert!(
        checked > 100 && corrupt_checked >= 10,
        "unexpectedly few images checked ({checked} valid, {corrupt_checked} corrupt)"
    );
}
