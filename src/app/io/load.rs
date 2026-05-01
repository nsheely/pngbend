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
    Chunk, PaletteEntry, PngInfo, concat_idat, decode_palette, parse_ihdr, read_chunks,
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
}

/// Structured errors surfaced by the background load thread.
#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    MissingIhdr,
    MissingIdat,
    Deflate(DecodeError),
    Display(String),
    /// Image hits a data-structure limit the editor can't represent —
    /// width or height beyond `u16::MAX`, or unfiltered output size beyond
    /// `u32::MAX`. Both come from in-memory layout choices (PixelRow packs
    /// xy as `(u16, u16)`; event positions are `u32`) so the loader has
    /// to refuse rather than silently truncate.
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

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O: {e}"),
            Self::MissingIhdr => write!(f, "missing IHDR chunk"),
            Self::MissingIdat => write!(f, "no IDAT data"),
            Self::Deflate(e) => write!(f, "deflate: {e}"),
            Self::Display(s) => write!(f, "display: {s}"),
            Self::Unsupported {
                width,
                height,
                reason,
            } => write!(f, "{width}×{height} unsupported: {reason}"),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
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
    parse_ihdr(&read_chunks(&buf[..n]))
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
            self.status = format!(
                "Loaded {}  |  {}×{}  bpp={}  blocks={}  literals={}  backrefs={} w/redirects",
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
    let chunks = read_chunks(&raw);
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
    if idat.len() < 6 {
        return Err(LoadError::MissingIdat);
    }
    let zlib_header = [idat[0], idat[1]];
    let deflate_buf = idat[2..idat.len() - 4].to_vec();

    let decoded = decode_deflate(&deflate_buf).map_err(LoadError::Deflate)?;
    let geom = ImgGeom::new(w as u32, h as u32, bpp as u32);

    // Try the in-app filter unfilter + RGBA converter first; fall back to
    // the image crate for sub-byte depths or other unsupported color types.
    // The unfiltered buffer is kept on `CoreData` across the session so
    // incremental edits can update only the rows they touched.
    let unfiltered = crate::png::unfilter(&decoded.output, &info).unwrap_or_default();
    let base_rgba = if unfiltered.is_empty() {
        fallback_rgba(&raw)?
    } else {
        match crate::png::to_rgba8(&unfiltered, &info, palette.as_deref()) {
            Ok(r) => r,
            Err(_) => fallback_rgba(&raw)?,
        }
    };

    let mut core = build_core_from_decoded(decoded, info, palette, geom);
    core.unfiltered = unfiltered;

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
    }
}

/// Fallback for PNGs our in-app unfilter+converter can't handle
/// (sub-byte depths, unusual colour types). Defers to the `image` crate.
fn fallback_rgba(raw: &[u8]) -> Result<Vec<u8>, LoadError> {
    image::load_from_memory(raw)
        .map(|img| img.into_rgba8().into_raw())
        .map_err(|e| LoadError::Display(e.to_string()))
}
