#!/usr/bin/env bash
# Fetch Willem van Schaik's PngSuite corpus for the glasspng differential
# conformance test (glasspng/tests/pngsuite.rs). The images are freeware but
# not vendored in git; this pulls them into the path the test reads. The test
# skips cleanly when the corpus is absent, so this is only needed to actually
# run the conformance check locally or in CI.
set -euo pipefail

version="PngSuite-2017jul19"
url="https://www.schaik.com/pngsuite/${version}.zip"
dest="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/glasspng/tests/pngsuite"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Fetching $url"
curl -fsSL "$url" -o "$tmp/pngsuite.zip"
mkdir -p "$dest"
unzip -oq "$tmp/pngsuite.zip" -d "$dest"

count="$(find "$dest" -name '*.png' | wc -l | tr -d ' ')"
echo "Extracted $count PNGs to $dest"
