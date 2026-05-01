//! Central image view: draws the texture, renders selection markers via
//! [`egui::Painter`] on top, handles hover + click coordinate math, and
//! rebuilds the texture from the base RGBA + active overlay.

use egui::{Color32, RichText};

use crate::composite::{composite_into, composite_rows_into};

use super::super::PngBendApp;
use super::super::overlay_cache::OverlayMode;
use super::super::select::SelectSource;

impl PngBendApp {
    pub(in crate::app) fn ui_central_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            if self.async_ops.is_loading() {
                ui.centered_and_justified(|ui| {
                    ui.spinner();
                    ui.label(&self.status);
                });
                return;
            }
            if self.view.texture.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.label(RichText::new("Drop a PNG here or use File > Open").size(18.0));
                });
                return;
            }

            let (iw, ih, cw, ch_img, c_bpp, c_row_stride) = {
                let g = self
                    .doc
                    .core
                    .as_ref()
                    .expect("checked: texture implies core")
                    .geom;
                (
                    g.w as f32,
                    g.h as f32,
                    g.w,
                    g.h,
                    g.bpp as usize,
                    g.row_stride as usize,
                )
            };
            let tex = self.view.texture.as_ref().expect("checked above");

            let available = ui.available_size();
            let scale = (available.x / iw).min(available.y / ih);
            let disp = egui::vec2(iw * scale, ih * scale);
            let offset = egui::vec2((available.x - disp.x) * 0.5, (available.y - disp.y) * 0.5);
            let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());
            let img_rect = egui::Rect::from_min_size(rect.min + offset, disp);

            ui.painter().image(
                tex.id(),
                img_rect,
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                Color32::WHITE,
            );

            // Selection markers — drawn on top of the texture so selection-only
            // changes never dirty the texture (no re-upload).
            self.draw_selection_markers(ui, img_rect, iw, ih);

            // Hover info
            if let Some(pos) = response.hover_pos()
                && img_rect.contains(pos)
            {
                let px = pixel_at(pos, img_rect, iw, cw - 1);
                let py = pixel_at_y(pos, img_rect, ih, ch_img - 1);
                let base = py as usize * c_row_stride + 1 + px as usize * c_bpp;
                let rgb: Vec<String> = self
                    .doc
                    .core
                    .as_ref()
                    .map(|c| {
                        (0..c_bpp.min(3))
                            .filter_map(|i| c.output.get(base + i).map(|v| v.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                self.hover_info = format!("  ({px}, {py})  ({})", rgb.join(", "));
            }

            // Click → select pixel
            if response.clicked()
                && let Some(pos) = response.interact_pointer_pos()
                && img_rect.contains(pos)
            {
                let px = pixel_at(pos, img_rect, iw, cw - 1);
                let py = pixel_at_y(pos, img_rect, ih, ch_img - 1);
                self.select_pixel(px, py, SelectSource::ImageClick);
                self.view.texture_dirty = true;
            }

            // Window title with dirty marker. Only send a viewport
            // command when the title actually changes — egui has no
            // "did this change?" check internally, so resending every
            // frame would spam the windowing layer with no-ops.
            let title = if let Some(ref p) = self.doc.path {
                format!(
                    "{}{}  —  Compressed PNG Editor",
                    p.file_name().unwrap_or_default().to_string_lossy(),
                    if self.doc.dirty { " •" } else { "" }
                )
            } else {
                "Compressed PNG Editor".to_string()
            };
            if title != self.last_title {
                ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
                self.last_title = title;
            }
        });
    }

    /// Build the texture from the current base + active overlay. Uploads
    /// directly from `base_rgba` when no overlay is active; otherwise
    /// composites into the reusable `composite_scratch` and uploads that.
    pub(in crate::app) fn rebuild_texture(&mut self, ctx: &egui::Context) {
        if !self.view.texture_dirty || self.view.base_rgba.is_empty() {
            return;
        }
        let Some(c) = self.doc.core.as_ref() else {
            return;
        };
        let dims = [c.geom.w as usize, c.geom.h as usize];
        if self.view.base_rgba.len() != dims[0] * dims[1] * 4 {
            return;
        }

        let upload_buf: &[u8] = if self.compose_into_scratch() {
            &self.view.composite_scratch
        } else {
            &self.view.base_rgba
        };
        let color_image = egui::ColorImage::from_rgba_unmultiplied(dims, upload_buf);
        self.view.texture =
            Some(ctx.load_texture("image", color_image, egui::TextureOptions::LINEAR));
        self.view.texture_dirty = false;
    }

    /// If the current overlay mode has bytes to composite, blend them into
    /// `composite_scratch` and return `true`. Otherwise leave the scratch
    /// alone and return `false` — the caller uploads `base_rgba` directly.
    ///
    /// When `partial_composite_rows` is set (by the incremental edit
    /// path), refresh just those rows inside `composite_scratch` — the
    /// other rows still hold last frame's valid composite.
    fn compose_into_scratch(&mut self) -> bool {
        // Split-borrow: ViewState fields are disjoint from `doc.core`, and
        // within ViewState we want `cascade_rgba` / `overlay_cache` shared
        // with a `&mut composite_scratch` write target.
        let view = &mut self.view;
        let overlay: Option<&Vec<u8>> = match view.overlay_mode {
            OverlayMode::None => None,
            OverlayMode::Cascade => view.cascade_rgba.as_ref(),
            other => view.overlay_cache.get(other),
        };
        let Some(ov) = overlay else {
            view.partial_composite_rows.take();
            return false;
        };
        let can_partial = view.composite_scratch.len() == view.base_rgba.len();
        if can_partial
            && let Some(rows) = view.partial_composite_rows.take()
            && let Some(c) = self.doc.core.as_ref()
        {
            composite_rows_into(
                &view.base_rgba,
                ov,
                &mut view.composite_scratch,
                c.geom.w,
                rows,
            );
            return true;
        }
        view.partial_composite_rows.take();
        composite_into(&view.base_rgba, ov, &mut view.composite_scratch);
        true
    }

    fn draw_selection_markers(
        &self,
        ui: &egui::Ui,
        img_rect: egui::Rect,
        image_w: f32,
        image_h: f32,
    ) {
        let Some((sx, sy)) = self.sel.sel_pixel else {
            return;
        };
        let painter = ui.painter_at(img_rect);
        let px_to_screen = |px: u32, py: u32| -> egui::Pos2 {
            egui::pos2(
                img_rect.left() + (px as f32 + 0.5) / image_w * img_rect.width(),
                img_rect.top() + (py as f32 + 0.5) / image_h * img_rect.height(),
            )
        };
        let center = px_to_screen(sx, sy);

        // Back-reference source: red line + circle at the src pixel.
        if let Some((bx, by)) = self.sel.backref_src {
            let src = px_to_screen(bx, by);
            let red = Color32::from_rgba_unmultiplied(255, 80, 80, 220);
            painter.line_segment([center, src], egui::Stroke::new(1.5, red));
            painter.circle_stroke(
                src,
                4.0,
                egui::Stroke::new(1.5, Color32::from_rgb(255, 80, 80)),
            );
        }

        // Yellow crosshair + 3×3 box.
        let arm = 12.0;
        let stroke = egui::Stroke::new(1.0, Color32::YELLOW);
        painter.line_segment(
            [
                egui::pos2(center.x - arm, center.y),
                egui::pos2(center.x + arm, center.y),
            ],
            stroke,
        );
        painter.line_segment(
            [
                egui::pos2(center.x, center.y - arm),
                egui::pos2(center.x, center.y + arm),
            ],
            stroke,
        );
        painter.rect_stroke(
            egui::Rect::from_center_size(center, egui::vec2(6.0, 6.0)),
            0.0,
            stroke,
            egui::StrokeKind::Middle,
        );
    }
}

fn pixel_at(pos: egui::Pos2, img_rect: egui::Rect, image_w: f32, max_x: u32) -> u32 {
    let raw = ((pos.x - img_rect.left()) / img_rect.width() * image_w) as u32;
    raw.min(max_x)
}

fn pixel_at_y(pos: egui::Pos2, img_rect: egui::Rect, image_h: f32, max_y: u32) -> u32 {
    let raw = ((pos.y - img_rect.top()) / img_rect.height() * image_h) as u32;
    raw.min(max_y)
}
