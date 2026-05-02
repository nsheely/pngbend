//! Background PNG loader and display-buffer renderer.
//!
//! The worker thread runs [`load_file`], which does all the expensive work
//! (deflate decode, pixel index, reverse graph, initial RGBA render) off
//! the UI thread. The app polls the result channel from its frame loop in
//! [`super::super::PngBendApp::try_recv_load`].

use std::path::PathBuf;
use std::sync::mpsc::{self, TryRecvError};
use std::thread;

use crate::coords::ImgGeom;
use crate::deflate::{DecodeError, DecodedDeflate, decode_deflate};
use crate::index::{build_pixel_index, build_pos_to_ev, build_reverse_graph};
use crate::png::{
    Chunk, ChunksError, PaletteEntry, PngInfo, ZlibError, adler32, concat_idat, decode_palette,
    parse_ihdr, parse_zlib_stream, read_chunks,
};

use super::super::PngBendApp;
use super::super::overlay_cache::OverlayMode;
use super::CoreData;

/// Full bundle produced by the background loader. Consumed by
/// [`PngBendApp::on_load_done`] to replace the app's current file state.
pub(crate) struct LoadedFile {
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
    /// Display pipeline failed — both the in-app converter and the
    /// `image`-crate fallback couldn't produce an RGBA buffer. The
    /// underlying error text isn't surfaced to the user (it's not
    /// actionable for them); the variant exists so the loader can fail
    /// distinctly from `Deflate` / `Chunks` etc.
    Display,
    /// Width/height past `u16::MAX` or unfiltered output past `u32::MAX`
    /// — limits set by `PixelRow.xy: (u16, u16)` and `u32` event positions.
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

/// User-facing message — this Display is what the status bar shows.
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
                    "This PNG decompresses to more data than its dimensions claim — possibly a decompression bomb."
                ),
                _ => write!(f, "This PNG's compressed image data is corrupted."),
            },
            Self::Display => write!(f, "Couldn't render this PNG."),
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
/// - **per output byte** — `events`, `output`, `reverse_graph`,
///   `unfiltered`, and the cascade depth map. ~18 B/byte covers typical
///   photo content with margin for highly back-referenced inputs.
/// - **per pixel** — `pixel_index` + `filtered_idx` + three `w × h × 4`
///   RGBA buffers (`base_rgba`, `composite_scratch`, one LRU overlay).
///   ~24 B/pixel.
/// - **constant** — egui state, font atlas, Huffman tables, etc. ~16 MB.
pub fn estimate_working_set_bytes(width: u32, height: u32, bpp: usize) -> u64 {
    let pixels = u64::from(width) * u64::from(height);
    let output_bytes = u64::from(height) * (1 + u64::from(width) * bpp as u64);
    let per_byte = output_bytes * 18;
    let per_pixel = pixels * 24;
    let constant = 16 * 1024 * 1024;
    per_byte + per_pixel + constant
}

/// Render a byte count in the closest power-of-2 binary unit (KB / MB /
/// GB), one decimal place — e.g. `"~2.7 GB"` for the status bar.
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
    parse_ihdr(&read_chunks(&buf[..n]).ok()?)
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
        self.status = match est {
            Some((w, h, bytes)) => format!(
                "Loading {} ({}×{}, ~{} working set)…",
                path.display(),
                w,
                h,
                format_bytes(bytes),
            ),
            None => format!("Loading {}…", path.display()),
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
        // dropping it in-place is O(1) — no background-thread drop needed.
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
            let mode = if c.editable { "" } else { "  |  read-only" };
            let warn = if loaded.warnings.is_empty() {
                String::new()
            } else {
                format!("  |  warning: {}", loaded.warnings.join("; "))
            };
            self.status = format!(
                "Loaded {}  |  {}×{}  bpp={}  blocks={}  literals={}  backrefs={} w/redirects{mode}{warn}",
                loaded.path.display(),
                c.geom.w,
                c.geom.h,
                c.geom.bpp,
                c.num_blocks,
                c.pixel_index.lit.len(),
                c.pixel_index.refs.len(),
            );
        }
    }
}

fn load_file(path: PathBuf) -> Result<LoadedFile, LoadError> {
    let raw = std::fs::read(&path).map_err(LoadError::Io)?;
    let mut parsed = read_chunks(&raw).map_err(LoadError::Chunks)?;
    let mut warnings = std::mem::take(&mut parsed.warnings);
    let chunks = parsed.chunks;
    let info = parse_ihdr(&chunks).ok_or(LoadError::MissingIhdr)?;
    if info.width > MAX_DIMENSION || info.height > MAX_DIMENSION {
        return Err(LoadError::Unsupported {
            width: info.width,
            height: info.height,
            reason: "width and height must each fit in 16 bits (≤ 65535)",
        });
    }
    // `output_len = h × (1 + w × bpp)` must fit in u32 so event position
    // fields can address every byte. The dimension check above caps the
    // multiplier; this catches the case where both dims sit near 65535
    // and bpp is high.
    let output_bytes = u64::from(info.height) * (1 + u64::from(info.width) * info.bpp as u64);
    if output_bytes > u32::MAX as u64 {
        return Err(LoadError::Unsupported {
            width: info.width,
            height: info.height,
            reason: "unfiltered output exceeds 4 GiB (event positions are u32)",
        });
    }
    let w = info.width as usize;
    let h = info.height as usize;
    let bpp = info.bpp;

    // Palette (only present for indexed PNGs; harmless for others).
    let palette = chunks.iter().find(|c| &c.typ == b"PLTE").map(|plte| {
        let trns = chunks
            .iter()
            .find(|c| &c.typ == b"tRNS")
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
    let mut zlib = parse_zlib_stream(&idat).map_err(LoadError::Zlib)?;
    warnings.append(&mut zlib.warnings);
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
        warnings.push("stale checksum on PNG image data".to_string());
    }
    let geom = ImgGeom::new(w as u32, h as u32, bpp as u32);

    // Try the in-app filter unfilter + RGBA converter first; fall back to
    // the image crate for sub-byte depths or other unsupported color types.
    // The unfiltered buffer is kept on `CoreData` across the session so
    // incremental edits can update only the rows they touched. Track
    // whether the in-app pipeline owned the display: if the fallback ran
    // the file is read-only (the row-scoped re-render path asserts
    // `unfiltered.len() == h * row_bytes`, which the fallback can't
    // satisfy without re-implementing every PNG colour mode).
    let unfiltered = crate::png::unfilter(&decoded.output, &info).unwrap_or_default();
    let (base_rgba, editable) = if unfiltered.is_empty() {
        (fallback_rgba(&raw)?, false)
    } else {
        match crate::png::to_rgba8(&unfiltered, &info, palette.as_deref()) {
            Ok(r) => (r, true),
            Err(_) => (fallback_rgba(&raw)?, false),
        }
    };

    let mut core = build_core_from_decoded(decoded, info, palette, geom);
    core.unfiltered = unfiltered;
    core.editable = editable;

    // Drop the IDAT bytes inside `chunks`: `deflate_buf` already holds the
    // decompressed source of truth, and save_png re-emits a fresh IDAT
    // built from the (possibly edited) `deflate_buf`. Holding both costs
    // ~3–4 MB on a typical 4 MP photo and scales linearly with input size.
    let mut chunks = chunks;
    for c in chunks.iter_mut().filter(|c| &c.typ == b"IDAT") {
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
/// stream plus geometry and palette. Leaves [`CoreData::unfiltered`]
/// empty — the caller fills it after deciding whether the in-app
/// unfilter or the `image`-crate fallback drives the display.
pub(in crate::app) fn build_core_from_decoded(
    decoded: DecodedDeflate,
    info: PngInfo,
    palette: Option<Vec<PaletteEntry>>,
    geom: ImgGeom,
) -> CoreData {
    let DecodedDeflate {
        output,
        events,
        lit_encs,
        dist_encs,
        num_blocks,
        max_distance,
    } = decoded;
    // `pos_to_ev` lives only for the duration of this function. The
    // pixel-index build does `bpp` lookups per pixel (12 M on a 4 MP
    // image), where the dense `Vec<u32>` is ~100× faster than the
    // `O(log events)` `index::event_at` runtime queries use. Once the
    // pixel index is built nothing else needs the dense map, so the
    // explicit `drop` releases the multi-MB buffer before storing the
    // long-lived `CoreData`.
    let pos_to_ev = build_pos_to_ev(&events, output.len());
    let reverse_graph = build_reverse_graph(&events, output.len());
    let pixel_index = build_pixel_index(&events, &output, &pos_to_ev, &lit_encs, &dist_encs, &geom);
    drop(pos_to_ev);

    CoreData {
        geom,
        info,
        palette,
        output,
        events,
        lit_encs,
        dist_encs,
        num_blocks,
        pixel_index,
        reverse_graph,
        unfiltered: Vec::new(),
        max_distance,
        // Caller overwrites this once it knows whether the in-app
        // pipeline or the image-crate fallback drove the display.
        editable: true,
    }
}

/// Fallback for PNGs our in-app unfilter+converter can't handle
/// (sub-byte depths, unusual colour types). Defers to the `image` crate.
fn fallback_rgba(raw: &[u8]) -> Result<Vec<u8>, LoadError> {
    image::load_from_memory(raw)
        .map(|img| img.into_rgba8().into_raw())
        .map_err(|_| LoadError::Display)
}
