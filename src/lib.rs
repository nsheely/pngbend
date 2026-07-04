//! The `pngbend` editor. The PNG/DEFLATE codec and the pass-aware `Raster`
//! projection live in the `glasspng` crate and are re-exported here under the
//! same paths (`crate::deflate::…`, `pngbend::png::…`) so they resolve
//! unchanged across the app, its tests, and benches. On top of the codec sit
//! the derived [`index`] structures, the [`overlays`] and [`composite`]
//! rendering, and the [`app`] GUI.
pub use glasspng::{Raster, bitstream, coords, deflate, png};

pub mod app;
pub mod composite;
pub mod index;
pub mod overlays;
