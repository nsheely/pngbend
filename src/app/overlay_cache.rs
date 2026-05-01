use crate::coords::ImgGeom;
use crate::deflate::Event;
use crate::overlays::{
    make_block_overlay_bytes, make_distance_overlay_bytes, make_literal_overlay_bytes,
};

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
    /// All modes the user can pick in the UI. Use as the single source of
    /// truth for the overlay selector and any `match` / `ensure` sites.
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
}

/// Single-slot cache holding the most recently rendered overlay buffer.
///
/// Each event-driven overlay is `w * h * 4` bytes — 16 MB on a 4 MP RGB
/// photo, 256 MB on a 64 MP one. The user sees one overlay at a time
/// and re-rendering takes ~50 ms, so a single slot trades "instant mode
/// switch" for two-thirds of the overlay working set back.
///
/// On a switch the previous entry drops first, then the new one renders.
/// Dropping the `Vec` releases the pages so the working set shrinks
/// instead of sitting cold.
#[derive(Default)]
pub(super) struct OverlayCache {
    entry: Option<(OverlayMode, Vec<u8>)>,
}

impl OverlayCache {
    pub fn clear(&mut self) {
        self.entry = None;
    }

    /// Drop the cached overlay only when it's the distance one. After
    /// a `DistRedirect` edit, the redirected ref's distance changes —
    /// which recolours its pixels in the distance heatmap, but leaves
    /// literal and block overlays valid (literals didn't move, every
    /// event kept its `block`). Specialised over `clear` so the common
    /// case (current cache is Literals or Blocks) skips the eviction.
    pub fn invalidate_distance(&mut self) {
        if matches!(self.entry, Some((OverlayMode::Distance, _))) {
            self.entry = None;
        }
    }

    pub fn get(&self, mode: OverlayMode) -> Option<&Vec<u8>> {
        match &self.entry {
            Some((cached, bytes)) if *cached == mode => Some(bytes),
            _ => None,
        }
    }

    /// Render `mode`'s overlay if it isn't already the cached entry, then
    /// return the cached bytes. Evicts whatever was cached before. Returns
    /// `None` for modes that don't have a cacheable buffer (None, Cascade).
    pub fn ensure(
        &mut self,
        mode: OverlayMode,
        events: &[Event],
        geom: &ImgGeom,
        num_blocks: usize,
        max_distance: u32,
    ) -> Option<&Vec<u8>> {
        if !matches!(
            mode,
            OverlayMode::Literals | OverlayMode::Distance | OverlayMode::Blocks
        ) {
            return None;
        }
        let already_cached = matches!(&self.entry, Some((cached, _)) if *cached == mode);
        if !already_cached {
            // Drop the previous slot before rendering the new one so peak
            // memory during the switch stays at one buffer, not two.
            self.entry = None;
            let bytes = match mode {
                OverlayMode::Literals => make_literal_overlay_bytes(events, geom),
                OverlayMode::Distance => make_distance_overlay_bytes(events, geom, max_distance),
                OverlayMode::Blocks => make_block_overlay_bytes(events, geom, num_blocks),
                _ => unreachable!("guarded above"),
            };
            self.entry = Some((mode, bytes));
        }
        self.entry.as_ref().map(|(_, bytes)| bytes)
    }
}
