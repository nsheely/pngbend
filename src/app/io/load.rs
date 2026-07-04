//! Background PNG loader and display-buffer renderer.
//!
//! The worker thread runs [`load_file`], which does all the expensive work
//! (deflate decode, pixel index, reverse graph, initial RGBA render) off
//! the UI thread. The app polls the result channel from its frame loop in
//! [`super::super::PngBendApp::try_recv_load`].

use std::path::PathBuf;
use std::sync::mpsc::{self, TryRecvError};
use std::thread;

use crate::deflate::{DecodeError, DecodedDeflate, decode_deflate};
use crate::index::{build_pixel_index, build_pos_to_ev, build_reverse_graph};
use crate::png::{
    Chunk, ChunkType, ChunksError, PaletteEntry, PngInfo, ZlibError, adler32, concat_idat,
    decode_palette, parse_ihdr, parse_zlib_stream, read_chunks,
};

use super::super::PngBendApp;
use super::super::overlay_cache::OverlayMode;
use super::CoreData;

/// Full bundle produced by the background loader. Consumed by
/// [`PngBendApp::on_load_done`] to replace the app's current file state.
pub(in crate::app) struct LoadedFile {
    pub path: PathBuf,
    pub chunks: Vec<Chunk>,
    pub zlib_header: [u8; 2],
    pub deflate_buf: Vec<u8>,
    pub core: CoreData,
    pub base_rgba: Vec<u8>,
    /// Integrity warnings collected during load (stale CRCs, FCHECK,
    /// Adler-32). Shown in the status bar; not a load failure.
    pub warnings: Vec<String>,
}

/// Structured errors surfaced by the background load thread.
#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    MissingIhdr,
    MissingIdat,
    Chunks(ChunksError),
    Zlib(ZlibError),
    Deflate(DecodeError),
    /// Native unfilter or RGBA conversion failed. Distinct from
    /// `Deflate` / `Chunks` errors so the loader can fail cleanly
    /// without claiming the input is malformed when really it just
    /// hit a corner of the spec the converter doesn't model. Named
    /// `Render` (not `Display`) to avoid colliding with the `Display`
    /// trait this enum also implements.
    Render,
    /// Width/height past `u16::MAX` or unfiltered output past `u32::MAX`,
    /// the limits set by `PixelRow.xy: (u16, u16)` and `u32` event positions.
    Unsupported {
        width: u32,
        height: u32,
        reason: &'static str,
    },
}

/// Hard upper bound on width or height. `PixelRow.xy` packs each
/// coordinate as `u16`, so any image with a dimension past this would
/// silently overflow into another pixel's slot.
pub const MAX_DIMENSION: u32 = u16::MAX as u32;

/// User-facing message: this Display is what the status bar shows.
/// The inner [`ChunksError`] / [`ZlibError`] / [`DecodeError`] keep
/// their technical Display impls for library users and logs; this layer
/// translates them into something a person opening a PNG can act on.
impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "Couldn't read file: {e}"),
            Self::MissingIhdr => write!(f, "This file isn't a valid PNG (no header)."),
            Self::MissingIdat => write!(f, "This PNG has no image data."),
            Self::Chunks(e) => match e {
                ChunksError::MissingSignature => write!(f, "This file isn't a PNG."),
                ChunksError::Truncated => write!(f, "This PNG is truncated."),
            },
            Self::Zlib(e) => match e {
                ZlibError::Truncated { .. } => write!(f, "This PNG's image data is truncated."),
                ZlibError::BadCompressionMethod { .. } => {
                    write!(f, "This PNG uses an unrecognised compression method.")
                }
                ZlibError::FdictSet => write!(
                    f,
                    "This PNG uses a feature pngbend doesn't support (preset dictionary)."
                ),
            },
            Self::Deflate(e) => match e {
                DecodeError::OutputTooLarge { .. } => write!(
                    f,
                    "This PNG decompresses to more data than its dimensions claim, a possible decompression bomb."
                ),
                _ => write!(f, "This PNG's compressed image data is corrupted."),
            },
            Self::Render => write!(f, "Couldn't render this PNG."),
            Self::Unsupported {
                width,
                height,
                reason,
            } => write!(
                f,
                "This PNG is too big for pngbend ({width}×{height}: {reason})."
            ),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Chunks(e) => Some(e),
            Self::Zlib(e) => Some(e),
            Self::Deflate(e) => Some(e),
            _ => None,
        }
    }
}

/// Estimate of the resident working set the loader will produce, in
/// bytes. Surfaces a "this image will use ~X GB" hint in the status bar
/// before the load starts so large images stay the user's call.
///
/// Three components:
/// - **per output byte**: `events`, `output`, `reverse_graph`,
///   `unfiltered`, and the cascade depth map. ~18 B/byte covers typical
///   photo content with margin for highly back-referenced inputs.
/// - **per pixel**: `pixel_index` + `filtered_idx` + three `w × h × 4`
///   RGBA buffers (`base_rgba`, `composite_scratch`, one LRU overlay).
///   ~24 B/pixel.
/// - **constant**: egui state, font atlas, Huffman tables, etc. ~16 MB.
pub fn estimate_working_set_bytes(width: u32, height: u32, bpp: usize) -> u64 {
    let pixels = u64::from(width) * u64::from(height);
    let output_bytes = u64::from(height) * (1 + u64::from(width) * bpp as u64);
    let per_byte = output_bytes * 18;
    let per_pixel = pixels * 24;
    let constant = 16 * 1024 * 1024;
    per_byte + per_pixel + constant
}

/// Render a byte count in the closest power-of-2 binary unit (KB / MB /
/// GB), one decimal place, e.g. `"~2.7 GB"` for the status bar.
pub(in crate::app) fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Read just enough of `path` to parse the IHDR chunk. Used by the GUI
/// to surface dimensions + working-set estimate in the status bar
/// *before* the worker thread commits to a multi-second full load.
/// Reads ~64 bytes; returns `None` if the file isn't a PNG and the
/// caller falls through to the worker thread for a more specific error.
pub(in crate::app) fn peek_ihdr(path: &std::path::Path) -> Option<PngInfo> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf = [0u8; 64];
    let n = f.read(&mut buf).ok()?;
    // 64-byte sniffs may straddle the IHDR CRC; if `read_chunks` rejects
    // a truncated tail that's a peek-only failure, not a load failure, so
    // treat any chunks error as "no preview, fall through to full load."
    parse_ihdr(&read_chunks(&buf[..n]).ok()?.chunks)
}

impl PngBendApp {
    pub(in crate::app) fn open_path(&mut self, path: PathBuf) {
        let est = peek_ihdr(&path).map(|info| {
            (
                info.width,
                info.height,
                estimate_working_set_bytes(info.width, info.height, info.bpp),
            )
        });
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        self.status = match est {
            Some((w, h, bytes)) => format!(
                "Loading {name}  ·  {w}×{h}, ~{} working set…",
                format_bytes(bytes)
            ),
            None => format!("Loading {name}…"),
        };
        let p = path.clone();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            tx.send(load_file(p)).ok();
        });
        self.async_ops.load_rx = Some(rx);
        self.reset_for_new_file();
    }

    pub(in crate::app) fn try_recv_load(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.async_ops.load_rx.take() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(loaded)) => {
                self.on_load_done(loaded);
                ctx.request_repaint();
            }
            Ok(Err(e)) => {
                self.status = format!("Error: {e}");
            }
            Err(TryRecvError::Empty) => {
                self.async_ops.load_rx = Some(rx);
                ctx.request_repaint_after(std::time::Duration::from_millis(100));
            }
            Err(TryRecvError::Disconnected) => {
                self.status = "Load thread disconnected".to_string();
            }
        }
    }

    fn on_load_done(&mut self, loaded: LoadedFile) {
        self.doc.path = Some(loaded.path.clone());
        self.doc.chunks = loaded.chunks;
        self.doc.zlib_header = loaded.zlib_header;
        self.doc.deflate_buf = loaded.deflate_buf;
        self.view.base_rgba = loaded.base_rgba;
        self.view.texture_dirty = true;
        // CoreData holds a CSR reverse graph (two contiguous Vecs), so
        // dropping it in-place is O(1); no background-thread drop needed.
        self.doc.core = Some(loaded.core);
        self.reset_for_new_file();
        // Post-load display defaults, beyond the generic reset.
        self.sel.info_text = "Click a pixel to inspect it.".to_string();
        self.view.overlay_mode = OverlayMode::None;
        self.list.pixel_type = super::super::PixelType::Lit;
        self.list.filter_text.clear();
        self.list.editable_only = false;
        self.rebuild_filter();

        if let Some(ref c) = self.doc.core {
            // Status names the file and its broad shape; per-class
            // counts (Literals / Backrefs / All) live in the left
            // panel so the status bar doesn't duplicate them.
            let name = loaded
                .path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| loaded.path.display().to_string());
            let warn = if loaded.warnings.is_empty() {
                String::new()
            } else {
                format!("  ·  warning: {}", loaded.warnings.join("; "))
            };
            self.status = format!(
                "Loaded {name}  ·  {}×{}, bpp={}, {} DEFLATE block{}{warn}",
                c.info.width,
                c.info.height,
                c.info.bpp,
                c.num_blocks(),
                if c.num_blocks() == 1 { "" } else { "s" },
            );
        }
    }
}

fn load_file(path: PathBuf) -> Result<LoadedFile, LoadError> {
    let raw = std::fs::read(&path).map_err(LoadError::Io)?;
    let mut parsed = read_chunks(&raw).map_err(LoadError::Chunks)?;
    // glasspng reports integrity issues as typed `Warning`s; the loader only
    // displays them, so flatten to strings via `Display` at the boundary.
    let mut warnings: Vec<String> = std::mem::take(&mut parsed.warnings)
        .iter()
        .map(ToString::to_string)
        .collect();
    let chunks = parsed.chunks;
    let info = parse_ihdr(&chunks).ok_or(LoadError::MissingIhdr)?;
    if info.width > MAX_DIMENSION || info.height > MAX_DIMENSION {
        return Err(LoadError::Unsupported {
            width: info.width,
            height: info.height,
            reason: "width and height must each fit in 16 bits (≤ 65535)",
        });
    }
    // `output_len` must fit in u32 so event position fields can address
    // every byte. The dimension check above caps the multiplier; this
    // catches the case where both dims sit near 65535 and bpp is high.
    // Interlaced output is the sum of the seven Adam7 pass sizes, which
    // has more per-row filter bytes than the progressive raster.
    let output_bytes = if info.interlaced {
        crate::png::interlaced_output_len(&info) as u64
    } else {
        u64::from(info.height) * (1 + u64::from(info.width) * info.bpp as u64)
    };
    if output_bytes > u32::MAX as u64 {
        return Err(LoadError::Unsupported {
            width: info.width,
            height: info.height,
            reason: "unfiltered output exceeds 4 GiB (event positions are u32)",
        });
    }
    // Palette (only present for indexed PNGs; harmless for others).
    let palette = chunks
        .iter()
        .find(|c| c.typ == ChunkType::PLTE)
        .map(|plte| {
            let trns = chunks
                .iter()
                .find(|c| c.typ == ChunkType::TRNS)
                .map(|c| c.data.as_slice());
            decode_palette(&plte.data, trns)
        });

    let idat = concat_idat(&chunks);
    if idat.is_empty() {
        return Err(LoadError::MissingIdat);
    }
    // Hard errors here are the cases where slicing would land on the
    // wrong bytes (truncation, non-deflate CM, FDICT). FCHECK and the
    // Adler-32 trailer are checksums and surface as warnings instead.
    let zlib = parse_zlib_stream(&idat).map_err(LoadError::Zlib)?;
    warnings.extend(zlib.warnings.iter().map(ToString::to_string));
    let zlib_header = zlib.header;
    let stored_adler = zlib.stored_adler;
    let deflate_buf = zlib.deflate_buf.to_vec();

    // Cap inflated size at the IHDR-derived expected output. RFC 2083:
    // a well-formed IDAT decodes to exactly `h * (1 + w * bpp)` bytes.
    // Rejecting anything past that defends against decompression-bomb
    // PNGs whose tiny IDATs would expand to gigabytes.
    let decoded =
        decode_deflate(&deflate_buf, Some(output_bytes as usize)).map_err(LoadError::Deflate)?;
    if stored_adler != adler32(&decoded.output) {
        warnings.push(crate::png::Warning::StaleImageAdler.to_string());
    }

    // Unfilter and convert. Both run natively for every PNG colour mode
    // the spec defines, including sub-byte greyscale and indexed depths,
    // so a failure here is a genuine decode error rather than an
    // "unsupported format" signal. Interlaced images reassemble their
    // seven passes; `unfiltered` is then the per-pass raw bytes.
    let (unfiltered, base_rgba) = if info.interlaced {
        let unfiltered = crate::png::deinterlace_unfilter(&decoded.output, &info)
            .map_err(|_| LoadError::Render)?;
        let base_rgba =
            crate::png::deinterlace_to_rgba8(&decoded.output, &info, palette.as_deref())
                .map_err(|_| LoadError::Render)?;
        (unfiltered, base_rgba)
    } else {
        let unfiltered =
            crate::png::unfilter(&decoded.output, &info).map_err(|_| LoadError::Render)?;
        let base_rgba = crate::png::to_rgba8(&unfiltered, &info, palette.as_deref())
            .map_err(|_| LoadError::Render)?;
        (unfiltered, base_rgba)
    };

    let mut core = build_core_from_decoded(decoded, info, palette);
    core.unfiltered = unfiltered;

    // Drop the IDAT bytes inside `chunks`: `deflate_buf` already holds the
    // decompressed source of truth, and save_png re-emits a fresh IDAT
    // built from the (possibly edited) `deflate_buf`. Holding both costs
    // ~3-4 MB on a typical 4 MP photo and scales linearly with input size.
    let mut chunks = chunks;
    for c in chunks.iter_mut().filter(|c| c.typ == ChunkType::IDAT) {
        c.data = Vec::new();
    }

    Ok(LoadedFile {
        path,
        chunks,
        warnings,
        zlib_header,
        deflate_buf,
        core,
        base_rgba,
    })
}

/// Build a fully-indexed [`CoreData`] from a freshly decoded DEFLATE
/// stream plus IHDR info and palette. Leaves [`CoreData::unfiltered`]
/// empty; the caller fills it after running the row-filter inverse.
pub(in crate::app) fn build_core_from_decoded(
    decoded: DecodedDeflate,
    info: PngInfo,
    palette: Option<Vec<PaletteEntry>>,
) -> CoreData {
    let DecodedDeflate {
        output,
        events,
        lit_encs,
        dist_encs,
        block_starts,
        max_distance,
    } = decoded;
    // `pos_to_ev` lives only for the duration of this function. The
    // pixel-index build does `bpp` lookups per pixel (12 M on a 4 MP
    // image), where the dense `Vec<u32>` is ~100× faster than the
    // `O(log events)` `index::event_at` runtime queries use. Once the
    // pixel index is built nothing else needs the dense map, so the
    // explicit `drop` releases the multi-MB buffer before storing the
    // long-lived `CoreData`.
    let raster = crate::Raster::new(info);
    let pos_to_ev = build_pos_to_ev(&events, output.len());
    let reverse_graph = build_reverse_graph(&events, output.len());
    let pixel_index = build_pixel_index(
        &events,
        &output,
        &pos_to_ev,
        &lit_encs,
        &dist_encs,
        &block_starts,
        &raster,
    );
    drop(pos_to_ev);

    CoreData {
        info,
        raster,
        palette,
        output,
        events,
        lit_encs,
        dist_encs,
        block_starts,
        pixel_index,
        reverse_graph,
        unfiltered: Vec::new(),
        max_distance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate::serialize_stored;
    use crate::png::ColorType;

    #[test]
    fn build_core_maps_interlaced_pixels() {
        // 2x2 greyscale, Adam7. Passes 1, 6, 7 place the four pixels; the
        // output is filter-byte-0 rows [pass1, pass6, pass7].
        let output = vec![0u8, 10, 0, 20, 0, 30, 40];
        let decoded = decode_deflate(&serialize_stored(&output), None).expect("decode");
        let mut info = PngInfo::new(2, 2, 8, ColorType::Greyscale);
        info.interlaced = true;
        let core = build_core_from_decoded(decoded, info, None);
        assert!(core.raster.info().interlaced);
        // build_pixel_index walked the interlaced raster; every screen pixel
        // appears as a literal row at its reassembled position.
        let mut coords: Vec<(u32, u32)> = core
            .pixel_index
            .lit
            .iter()
            .map(|r| (r.x(), r.y()))
            .collect();
        coords.sort_unstable();
        assert_eq!(coords, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }
}
