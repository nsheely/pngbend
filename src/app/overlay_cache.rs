//! Single-slot cache for the event-driven overlay buffers (Literals /
//! Distance / Blocks), keyed by which one is currently shown.

use crate::deflate::Event;
use crate::overlays::{
    make_block_overlay_bytes, make_distance_overlay_bytes, make_literal_overlay_bytes,
};
use crate::png::PngInfo;

use super::PngBendApp;

#[derive(PartialEq, Eq, Clone, Copy, Default)]
pub(super) enum OverlayMode {
    #[default]
    None,
    Literals,
    Distance,
    Blocks,
    Cascade,
}

impl OverlayMode {
    /// All modes the user can pick in the UI. The overlay selector and the
    /// `match` / `ensure` sites iterate this rather than re-listing the variants.
    pub const ALL: [OverlayMode; 5] = [
        OverlayMode::None,
        OverlayMode::Literals,
        OverlayMode::Distance,
        OverlayMode::Blocks,
        OverlayMode::Cascade,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Literals => "Literals",
            Self::Distance => "Distance",
            Self::Blocks => "Blocks",
            Self::Cascade => "Cascade",
        }
    }

    /// The cacheable event-driven overlay for this mode, or `None` for the
    /// modes without a cached buffer (`None` renders nothing; `Cascade` is
    /// rebuilt per click).
    pub fn event_overlay(self) -> Option<EventOverlay> {
        match self {
            Self::Literals => Some(EventOverlay::Literals),
            Self::Distance => Some(EventOverlay::Distance),
            Self::Blocks => Some(EventOverlay::Blocks),
            Self::None | Self::Cascade => None,
        }
    }
}

/// The overlays rendered from the event stream and cached: the cacheable
/// subset of [`OverlayMode`] (which also has `None` and the per-click
/// `Cascade`). Keying [`OverlayCache`] on this makes the "only these three"
/// rule a type rather than a runtime guard.
#[derive(PartialEq, Eq, Clone, Copy)]
pub(super) enum EventOverlay {
    Literals,
    Distance,
    Blocks,
}

/// Single-slot cache holding the most recently rendered overlay buffer.
///
/// Each event-driven overlay is `w * h * 4` bytes: 16 MB on a 4 MP RGB
/// photo, 256 MB on a 64 MP one. The user sees one overlay at a time
/// and re-rendering takes ~50 ms, so a single slot trades "instant mode
/// switch" for two-thirds of the overlay working set back.
///
/// On a switch the previous entry drops first, then the new one renders.
/// Dropping the `Vec` releases the pages so the working set shrinks
/// instead of sitting cold.
#[derive(Default)]
pub(super) struct OverlayCache {
    entry: Option<(EventOverlay, Vec<u8>)>,
}

impl OverlayCache {
    pub fn clear(&mut self) {
        self.entry = None;
    }

    /// Drop the cached overlay only when it's the distance one. After
    /// a `DistRedirect` edit, the redirected ref's distance changes,
    /// which recolours its pixels in the distance heatmap, but leaves
    /// literal and block overlays valid (literals didn't move, every
    /// event kept its `block`). Specialised over `clear` so the common
    /// case (current cache is Literals or Blocks) skips the eviction.
    pub fn invalidate_distance(&mut self) {
        if matches!(self.entry, Some((EventOverlay::Distance, _))) {
            self.entry = None;
        }
    }

    pub fn get(&self, overlay: EventOverlay) -> Option<&Vec<u8>> {
        match &self.entry {
            Some((cached, bytes)) if *cached == overlay => Some(bytes),
            _ => None,
        }
    }

    /// Render `overlay` if it isn't already the cached entry, evicting
    /// whatever was cached before, then return the cached bytes.
    pub fn ensure(
        &mut self,
        overlay: EventOverlay,
        events: &[Event],
        info: &PngInfo,
        block_starts: &[u32],
        max_distance: u32,
    ) -> Option<&Vec<u8>> {
        let already_cached = matches!(&self.entry, Some((cached, _)) if *cached == overlay);
        if !already_cached {
            // Drop the previous slot before rendering the new one so peak
            // memory during the switch stays at one buffer, not two.
            self.entry = None;
            let bytes = match overlay {
                EventOverlay::Literals => make_literal_overlay_bytes(events, info),
                EventOverlay::Distance => make_distance_overlay_bytes(events, info, max_distance),
                EventOverlay::Blocks => {
                    make_block_overlay_bytes(events, info, block_starts, block_starts.len())
                }
            };
            self.entry = Some((overlay, bytes));
        }
        self.entry.as_ref().map(|(_, bytes)| bytes)
    }
}

impl PngBendApp {
    /// Ensure the current overlay mode's buffer is cached before the texture
    /// rebuild reads it. No-op for `None` / `Cascade` (no event-driven buffer)
    /// and for interlaced images (overlays are progressive-only).
    pub(super) fn ensure_overlay_cached(&mut self) {
        let Some(c) = self.doc.core.as_ref() else {
            return;
        };
        let Some(overlay) = self.view.overlay_mode.event_overlay() else {
            return;
        };
        if !c.overlays_supported() {
            return;
        }
        let info = c.info;
        self.view
            .overlay_cache
            .ensure(overlay, &c.events, &info, &c.block_starts, c.max_distance);
    }
}
