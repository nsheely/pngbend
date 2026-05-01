//! Native file-open / save-as dialogs, hosted on a worker thread so the UI
//! stays responsive while the OS picker is up.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;

use crate::png;

use super::super::PngBendApp;

/// Messages sent from the dialog worker back to the main thread.
pub(in crate::app) enum DialogResult {
    Open(PathBuf),
    SaveDone(PathBuf),
    SaveError(String),
}

impl PngBendApp {
    pub(in crate::app) fn save_png(&mut self, ctx: &egui::Context) {
        if self.async_ops.dialog_rx.is_some() {
            return;
        }
        // Pass the already-decoded `core.output` to the zlib builder so
        // it can compute Adler-32 directly without re-inflating the
        // stream just to checksum it.
        let Some(c) = self.doc.core.as_ref() else {
            self.status = "Save failed: no file loaded".to_string();
            return;
        };
        let zlib = png::build_zlib_stream(&self.doc.deflate_buf, &self.doc.zlib_header, &c.output);
        let png_bytes = self.assemble_png_bytes(&zlib);
        let default_name = self
            .doc
            .path
            .as_deref()
            .and_then(|p| p.file_stem())
            .map(|s| format!("{}_edited.png", s.to_string_lossy()))
            .unwrap_or_else(|| "edited.png".to_string());

        let (tx, rx): (_, Receiver<DialogResult>) = mpsc::channel();
        self.async_ops.dialog_rx = Some(rx);
        let ctx = ctx.clone();
        thread::spawn(move || {
            let Some(dest) = rfd::FileDialog::new()
                .set_file_name(&default_name)
                .add_filter("PNG", &["png"])
                .save_file()
            else {
                return; // cancelled — tx drops, rx disconnects
            };
            let result = match std::fs::write(&dest, &png_bytes) {
                Ok(()) => DialogResult::SaveDone(dest),
                Err(e) => DialogResult::SaveError(e.to_string()),
            };
            let _ = tx.send(result);
            ctx.request_repaint();
        });
    }

    pub(in crate::app) fn open_file_dialog(&mut self, ctx: &egui::Context) {
        if self.async_ops.dialog_rx.is_some() {
            return;
        }
        let (tx, rx): (_, Receiver<DialogResult>) = mpsc::channel();
        self.async_ops.dialog_rx = Some(rx);
        let ctx = ctx.clone();
        thread::spawn(move || {
            if let Some(p) = rfd::FileDialog::new()
                .add_filter("PNG", &["png"])
                .pick_file()
            {
                let _ = tx.send(DialogResult::Open(p));
                ctx.request_repaint();
            }
            // cancelled — tx drops, rx disconnects
        });
    }

    pub(in crate::app) fn poll_dialog(&mut self) {
        let Some(rx) = self.async_ops.dialog_rx.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(DialogResult::Open(p)) => {
                self.async_ops.dialog_rx = None;
                self.open_path(p);
            }
            Ok(DialogResult::SaveDone(dest)) => {
                self.async_ops.dialog_rx = None;
                self.status = format!("Saved: {}", dest.display());
                self.doc.dirty = false;
            }
            Ok(DialogResult::SaveError(e)) => {
                self.async_ops.dialog_rx = None;
                self.status = format!("Save failed: {e}");
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.async_ops.dialog_rx = None; // dialog cancelled
            }
        }
    }
}
