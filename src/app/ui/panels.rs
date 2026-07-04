//! Side panels, menu, and status bar. The central image view is in
//! [`super::image`].

use egui::{Align, Color32, Layout, RichText};

use super::super::PixelType;
use super::super::PngBendApp;
use super::super::overlay_cache::OverlayMode;
use super::super::select::SelectSource;

const ROW_HEIGHT: f32 = 16.0;

/// Which buttons were clicked in the right panel this frame. Returned from
/// [`PngBendApp::ui_right_panel`] so the caller can act on them after the
/// panel closure has released its borrow of `self`.
#[derive(Default)]
pub(in crate::app) struct RightPanelClicks {
    pub apply: bool,
    pub undo: bool,
    pub save: bool,
}

impl PngBendApp {
    pub(in crate::app) fn ui_top_menu(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::Panel::top("menu").show_inside(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open…  (Ctrl+O)").clicked() {
                        ui.close();
                        self.open_file_dialog(ctx);
                    }
                    if enabled_button(ui, self.doc.core.is_some(), "Save PNG…  (Ctrl+Shift+S)") {
                        ui.close();
                        self.save_png(ctx);
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("Edit", |ui| {
                    if enabled_button(ui, self.doc.history.can_undo(), "Undo  (Ctrl+Z)") {
                        ui.close();
                        self.undo();
                    }
                    if enabled_button(ui, self.doc.history.can_redo(), "Redo  (Ctrl+Y)") {
                        ui.close();
                        self.redo();
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("About PNGbend").clicked() {
                        ui.close();
                        self.show_about = true;
                    }
                });
            });
        });
    }

    pub(in crate::app) fn ui_about_window(&mut self, ctx: &egui::Context) {
        if !self.show_about {
            return;
        }
        egui::Window::new("About PNGbend")
            .open(&mut self.show_about)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(format!("PNGbend {}", env!("CARGO_PKG_VERSION")));
                ui.label(env!("CARGO_PKG_DESCRIPTION"));
                ui.add_space(4.0);
                ui.hyperlink_to(
                    "github.com/nsheely/pngbend",
                    "https://github.com/nsheely/pngbend",
                );
                ui.label("Dual-licensed: MIT or Apache-2.0");
            });
    }

    pub(in crate::app) fn ui_status_bar(&self, ui: &mut egui::Ui) {
        egui::Panel::bottom("status").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(&self.hover_info);
                });
            });
        });
    }

    pub(in crate::app) fn ui_left_panel(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("pixel_list")
            .exact_size(240.0)
            .show_inside(ui, |ui| {
                ui.vertical(|ui| {
                    self.ui_pixel_type_radio(ui);
                    self.ui_pixel_count_label(ui);
                    self.ui_filter_controls(ui);
                    self.ui_pixel_list(ui);
                });
            });
    }

    fn ui_pixel_type_radio(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let mut changed = false;
            for (kind, label) in [
                (PixelType::Lit, "Literals"),
                (PixelType::Ref, "Backrefs"),
                (PixelType::All, "All"),
            ] {
                if ui
                    .selectable_label(self.list.pixel_type == kind, label)
                    .clicked()
                {
                    self.list.pixel_type = kind;
                    changed = true;
                }
            }
            if changed {
                self.rebuild_filter();
            }
        });
    }

    fn ui_pixel_count_label(&self, ui: &mut egui::Ui) {
        let Some(c) = &self.doc.core else {
            return;
        };
        let (total, lbl) = match self.list.pixel_type {
            PixelType::Lit => {
                let n = c.pixel_index.lit.len();
                (
                    n,
                    format!(
                        "Literals: {n}  ({} editable)",
                        c.pixel_index.n_lit_with_edit
                    ),
                )
            }
            PixelType::Ref => {
                // Every entry in `refs` is editable by construction
                // (`build_pixel_index` filters out non-redirectable refs),
                // so an "(N editable)" suffix here would always read "of N".
                let n = c.pixel_index.refs.len();
                (n, format!("Backrefs: {n}"))
            }
            PixelType::All => {
                let n = c.pixel_index.lit.len() + c.pixel_index.refs.len();
                (n, format!("All: {n}"))
            }
        };
        ui.label(lbl);

        let shown = self.list.filtered_view.len();
        let sel_pos = self.sel.sel_pixel.and_then(|xy| self.filtered_pos(xy));
        let detail = match (shown < total, sel_pos) {
            (true, Some(p)) => format!("showing {shown} of {total} • {} of {shown}", p + 1),
            (true, None) => format!("showing {shown} of {total}"),
            (false, Some(p)) => format!("pixel {} of {shown}", p + 1),
            (false, None) => String::new(),
        };
        if !detail.is_empty() {
            ui.label(RichText::new(detail).small().color(Color32::from_gray(160)));
        }
    }

    fn ui_filter_controls(&mut self, ui: &mut egui::Ui) {
        // Hint names every structured shape `FilterSpec::parse` recognises,
        // ordered by how often they're useful: coords, RGB prefix, then
        // back-ref metrics. Free text falls back to a substring match.
        let filter_changed = ui
            .add(
                egui::TextEdit::singleline(&mut self.list.filter_text)
                    .hint_text("filter  x,y · #hex · d=N · len=N"),
            )
            .changed();
        let editable_changed = ui
            .checkbox(&mut self.list.editable_only, "Editable only")
            .changed();
        if filter_changed || editable_changed {
            self.rebuild_filter();
        }
    }

    fn ui_pixel_list(&mut self, ui: &mut egui::Ui) {
        // show_rows() pads each row by item_spacing.y internally; all
        // scroll-offset and viewport-row math uses the same total stride.
        let row_stride = ROW_HEIGHT + ui.spacing().item_spacing.y;
        let total_rows = self.list.filtered_view.len();

        // Measure the visible scroll area before building it.
        let avail_h = ui.available_height();
        if avail_h > 0.0 {
            self.list.list_viewport_rows = (avail_h / row_stride) as usize;
        }

        let mut area = egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible);
        if let Some(target_row) = self.list.list_scroll_to.take() {
            let half = self.list.list_viewport_rows / 2;
            let start_row = target_row.saturating_sub(half);
            area = area.scroll_offset(egui::vec2(0.0, start_row as f32 * row_stride));
        }

        area.show_rows(ui, ROW_HEIGHT, total_rows, |ui, row_range| {
            for i in row_range {
                // Resolve both FilterRef and row upfront (both `Copy`) so
                // the &self borrow drops before the &mut self click handler.
                let Some((fref, row)) = self.filtered_row_full(i) else {
                    break;
                };
                let [r, g, b] = row.rgb;
                let is_selected = Some(row.xy()) == self.sel.sel_pixel;

                // Format the row's display text on demand. Virtual scroll
                // means this runs ~25 times per frame, not once per pixel
                // in the index.
                let line = {
                    let c = self.doc.core.as_ref().expect("row implies core");
                    let mut s = String::with_capacity(48);
                    super::super::row_text::append_row_text(&mut s, fref, &row, c);
                    s
                };

                let bg = Color32::from_rgb(r, g, b);
                let luma = 0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32;
                let fg = if row.has_edit {
                    if luma < 128.0 {
                        Color32::WHITE
                    } else {
                        Color32::BLACK
                    }
                } else if luma < 128.0 {
                    Color32::from_gray(180)
                } else {
                    Color32::from_gray(100)
                };

                let mut frame = egui::Frame::new()
                    .inner_margin(egui::Margin::same(2))
                    .fill(bg);
                if is_selected {
                    frame = frame.stroke(egui::Stroke::new(1.5, Color32::YELLOW));
                }

                let response = frame
                    .show(ui, |ui| {
                        ui.add_sized(
                            [ui.available_width(), ROW_HEIGHT],
                            egui::Label::new(RichText::new(&line).monospace().color(fg).size(10.0))
                                .selectable(false),
                        )
                    })
                    .response
                    .interact(egui::Sense::click());

                if response.clicked() {
                    self.select_pixel(row.xy(), SelectSource::Refocus);
                    self.view.texture_dirty = true;
                }
            }
        });
    }

    /// Returns which buttons were clicked this frame.
    pub(in crate::app) fn ui_right_panel(&mut self, ui: &mut egui::Ui) -> RightPanelClicks {
        let mut clicks = RightPanelClicks::default();
        egui::Panel::right("right_panel")
            .exact_size(340.0)
            .show_inside(ui, |ui| {
                ui.vertical(|ui| {
                    self.ui_overlay_selector(ui);
                    self.ui_pixel_info(ui);
                    self.ui_edit_list(ui);
                    self.ui_action_buttons(ui, &mut clicks);
                });
            });
        clicks
    }

    fn ui_overlay_selector(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.label("Overlay:");
            ui.horizontal_wrapped(|ui| {
                for mode in OverlayMode::ALL {
                    if ui
                        .selectable_label(self.view.overlay_mode == mode, mode.label())
                        .clicked()
                    {
                        self.view.overlay_mode = mode;
                        self.view.texture_dirty = true;
                    }
                }
            });
        });
    }

    fn ui_pixel_info(&self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.label("Pixel info (click image):");
            // Bi-directional scroll: channel-detail lines run wider than
            // the 340 px panel. Vertical-only + wrap would mid-break them
            // and shuffle the columns out of alignment, so we let them
            // extend and scroll horizontally instead.
            egui::ScrollArea::both()
                .id_salt("info_scroll")
                .max_height(150.0)
                .show(ui, |ui| {
                    ui.add(
                        egui::Label::new(RichText::new(&self.sel.info_text).monospace())
                            .wrap_mode(egui::TextWrapMode::Extend)
                            .selectable(true),
                    );
                });
        });
    }

    fn ui_edit_list(&mut self, ui: &mut egui::Ui) {
        // Cap height so the action buttons below stay visible.
        let edits_max_h = (ui.available_height() - 40.0).max(60.0);
        ui.group(|ui| {
            ui.label("Available edits:");
            egui::ScrollArea::vertical()
                .id_salt("edits_scroll")
                .max_height(edits_max_h)
                .show(ui, |ui| {
                    let opts_len = self.sel.edit_options.len();
                    for i in 0..opts_len {
                        let selected = self.sel.selected_edit == Some(i);
                        let bg = self.sel.edit_options[i].bg_color;
                        let fg = self.sel.edit_options[i].fg_color;
                        let label = self.sel.edit_options[i].label.clone();

                        let mut frame = egui::Frame::new()
                            .inner_margin(egui::Margin::same(2))
                            .fill(bg);
                        if selected {
                            frame = frame.stroke(egui::Stroke::new(1.5, Color32::YELLOW));
                        }

                        // `add_sized` centres short labels inside their
                        // rect; min_width + a horizontal layout +
                        // `Label::truncate()` pin the label flush-left
                        // regardless of length.
                        let r = frame
                            .show(ui, |ui| {
                                ui.set_min_width(ui.available_width());
                                ui.set_min_height(16.0);
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::Label::new(
                                            RichText::new(&label).monospace().color(fg).size(9.0),
                                        )
                                        .truncate()
                                        .selectable(false),
                                    );
                                });
                            })
                            .response
                            .interact(egui::Sense::click());

                        if r.clicked() {
                            self.sel.selected_edit = Some(i);
                            self.status = format!("Selected: {label}");
                        }
                    }
                });
        });
    }

    fn ui_action_buttons(&self, ui: &mut egui::Ui, clicks: &mut RightPanelClicks) {
        ui.horizontal(|ui| {
            clicks.apply = enabled_button(ui, self.sel.selected_edit.is_some(), "Apply");
            clicks.undo = enabled_button(ui, self.doc.history.can_undo(), "Undo");
            clicks.save = enabled_button(ui, self.doc.core.is_some(), "Save PNG…");
        });
    }
}

/// `ui.add_enabled(cond, Button::new(label)).clicked()` shortcut so the
/// menu and action-bar sites read as one line each.
#[inline]
fn enabled_button(ui: &mut egui::Ui, enabled: bool, label: &str) -> bool {
    ui.add_enabled(enabled, egui::Button::new(label)).clicked()
}
