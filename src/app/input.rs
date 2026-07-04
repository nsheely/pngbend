//! Per-frame input handling: keyboard navigation through the pixel list,
//! keyboard shortcuts (open / undo / redo / save), and dropped-file opening.
//! Driven from the `eframe::App::ui` loop before the panels are drawn.

use super::PngBendApp;
use super::select::SelectSource;

impl PngBendApp {
    /// Handle keyboard navigation through the pixel list, plus Esc to clear
    /// the selection. Skipped while a text widget has focus.
    pub(super) fn handle_keyboard_nav(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let n_pixels = self.list.filtered_view.len();
        let (up, down, pgup, pgdn, home, end, esc) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowUp),
                i.key_pressed(egui::Key::ArrowDown),
                i.key_pressed(egui::Key::PageUp),
                i.key_pressed(egui::Key::PageDown),
                i.key_pressed(egui::Key::Home),
                i.key_pressed(egui::Key::End),
                i.key_pressed(egui::Key::Escape),
            )
        });

        if esc {
            self.sel.sel_pixel = None;
            self.reset_edit_state();
            self.view.cascade_rgba = None;
            self.sel.info_text = "Click a pixel to inspect it.".to_string();
            self.view.texture_dirty = true;
        }

        let nav_delta: Option<isize> = if up {
            Some(-1)
        } else if down {
            Some(1)
        } else if pgup {
            Some(-(self.list.list_viewport_rows as isize))
        } else if pgdn {
            Some(self.list.list_viewport_rows as isize)
        } else if home && n_pixels > 0 {
            if let Some(xy) = self.filtered_xy(0) {
                self.select_pixel(xy, SelectSource::ListNav);
            }
            None
        } else if end && n_pixels > 0 {
            if let Some(xy) = self.filtered_xy(n_pixels - 1) {
                self.select_pixel(xy, SelectSource::ListNav);
            }
            None
        } else {
            None
        };

        if let Some(delta) = nav_delta
            && let Some(xy) = self.sel.sel_pixel
            && let Some(row) = self.filtered_pos(xy)
        {
            let next =
                (row as isize + delta).clamp(0, n_pixels.saturating_sub(1) as isize) as usize;
            if next != row
                && let Some(next_xy) = self.filtered_xy(next)
            {
                self.select_pixel(next_xy, SelectSource::ListNav);
            }
        }

        if self.view.texture_dirty {
            ctx.request_repaint();
        }
    }

    /// Ctrl+O / Ctrl+Z / Ctrl+Shift+S shortcuts. Skipped while a text widget
    /// has focus so typing in the filter box doesn't trigger commands.
    pub(super) fn handle_keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::O) && i.modifiers.ctrl) {
            self.open_file_dialog(ctx);
        }
        // Ctrl+Shift+Z and Ctrl+Y both redo; plain Ctrl+Z is undo.
        let (z, y, shift, ctrl) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::Z),
                i.key_pressed(egui::Key::Y),
                i.modifiers.shift,
                i.modifiers.ctrl,
            )
        });
        if ctrl && y {
            self.redo();
        } else if ctrl && z {
            if shift {
                self.redo();
            } else {
                self.undo();
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::S) && i.modifiers.ctrl && i.modifiers.shift)
            && self.doc.core.is_some()
        {
            self.save_png(ctx);
        }
    }

    /// Open a PNG dropped onto the window.
    pub(super) fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.first().cloned());
        if let Some(file) = dropped
            && let Some(path) = file.path
        {
            self.open_path(path);
        }
    }
}
