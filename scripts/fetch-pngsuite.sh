#!/usr/bin/env bash
# Fetch the PngSuite conformance corpus for glasspng/tests/pngsuite.rs.
#
# Willem van Schaik's freeware PngSuite, sourced from image-rs/image-png (the
# same `png` crate the test differentially compares against) pinned to a
# commit so the set is reproducible. This mirror is the 2017jul19 set minus
# exif2c08.png (175 images). Fetched from GitHub rather than schaik.com, whose
# TLS is unreliable from CI. Not vendored in git; the test skips cleanly when
# the corpus is absent.
set -euo pipefail

repo="image-rs/image-png"
sha="4ab5484248dc04e6b65ef52fb58b506f88e734f4"
dest="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/glasspng/tests/pngsuite"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Fetching ${repo}@${sha:0:12} tests/pngsuite"
curl -fsSL "https://codeload.github.com/${repo}/tar.gz/${sha}" -o "$tmp/src.tar.gz"
mkdir -p "$dest"
# Extract just the pngsuite dir, stripping the archive's `<repo>-<sha>/tests/
# pngsuite/` prefix so files land flat in $dest.
tar -xzf "$tmp/src.tar.gz" -C "$dest" --strip-components=3 "image-png-${sha}/tests/pngsuite"

count="$(find "$dest" -name '*.png' | wc -l | tr -d ' ')"
echo "Extracted $count PNGs to $dest"
