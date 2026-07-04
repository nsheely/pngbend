#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::sync::Arc;

use pngbend::app::PngBendApp;
use pngbend::deflate::decode_deflate;
use pngbend::png::{
    ChunkType, concat_idat, decode_palette, parse_ihdr, parse_zlib_stream, read_chunks, to_rgba8,
    unfilter,
};

/// Decode the embedded app-icon PNG through pngbend's own pipeline.
/// The icon ships with the binary, so every `expect` is a build-time
/// guarantee: a broken `assets/icon.png` fails CI.
fn load_icon() -> egui::IconData {
    let bytes = include_bytes!("../assets/icon.png");
    let chunks = read_chunks(bytes).expect("icon chunks").chunks;
    let info = parse_ihdr(&chunks).expect("icon IHDR");
    let palette = chunks
        .iter()
        .find(|c| c.typ == ChunkType::PLTE)
        .map(|p| decode_palette(&p.data, None));
    let idat = concat_idat(&chunks);
    let zlib = parse_zlib_stream(&idat).expect("icon zlib header");
    let decoded = decode_deflate(zlib.deflate_buf, None).expect("icon deflate");
    let unfiltered = unfilter(&decoded.output, &info).expect("icon unfilter");
    let rgba = to_rgba8(&unfiltered, &info, palette.as_deref()).expect("icon RGBA");
    egui::IconData {
        rgba,
        width: info.width,
        height: info.height,
    }
}

fn main() -> eframe::Result<()> {
    let path = std::env::args().nth(1).map(PathBuf::from);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1550.0, 750.0])
            .with_title("PNGbend")
            .with_icon(Arc::new(load_icon())),
        ..Default::default()
    };

    eframe::run_native(
        "PNGbend",
        options,
        Box::new(|cc| Ok(Box::new(PngBendApp::new(cc, path)))),
    )
}
