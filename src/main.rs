#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::sync::Arc;

use pngbend::app::PngBendApp;
use pngbend::deflate::decode_deflate;
use pngbend::png::{
    concat_idat, decode_palette, parse_ihdr, parse_zlib_stream, read_chunks, to_rgba8, unfilter,
};

/// Decode the embedded app-icon PNG through pngbend's own pipeline.
/// The icon is shipped with the binary so every `expect` here is a
/// build-time guarantee: a broken `assets/icon.png` would fail CI.
fn load_icon() -> egui::IconData {
    let bytes = include_bytes!("../assets/icon.png");
    let parsed = read_chunks(bytes).expect("icon chunks");
    let info = parse_ihdr(&parsed).expect("icon IHDR");
    let palette = parsed
        .iter()
        .find(|c| &c.typ == b"PLTE")
        .map(|p| decode_palette(&p.data, None));
    let idat = concat_idat(&parsed);
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
