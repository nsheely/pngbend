//! Apply / undo / redo for edits that have already been built (see
//! [`super::select`] for how a click becomes an [`EditAction`]).
//!
//! Two edit kinds, both applied in place without re-decoding: a **literal
//! swap** (same-length Huffman code, LZ77 topology unchanged) and a **distance
//! redirect** (one back-reference's source moves). Each keeps most of the index
//! valid and re-renders only the affected rows; the per-kind mechanics and the
//! invariants they depend on live on [`EditKind`]'s variants and the
//! `apply_*_incremental` methods.
//!
//! Split three ways: the edit types here, the [`PngBendApp`](super::PngBendApp)
//! apply/undo/redo methods in [`apply`], and the row-scoped re-render +
//! bit-patch leaf helpers in [`render`].

mod apply;
mod render;

/// One bit-precise rewrite in the deflate buffer: `value` is written
/// into the `code_len` bits starting at `bit_start`. Pairs with
/// [`render::apply_patches_capturing_prior`], which returns a Vec of these
/// describing the inverse rewrite for undo.
#[derive(Debug, Clone, Copy)]
pub(super) struct Patch {
    pub bit_start: u32,
    pub value: u32,
    pub code_len: u8,
}

/// The output-buffer side of a literal swap: byte at `out_pos` takes
/// `value`. Captured alongside the deflate-stream [`Patch`] so the
/// incremental apply path can update `output` (and its LZ77 descendants)
/// without re-decoding, and so undo can restore the prior bytes.
#[derive(Debug, Clone, Copy)]
pub(super) struct ByteWrite {
    pub out_pos: u32,
    pub value: u8,
}

/// The patches + kind needed to apply an edit forward and to derive its
/// inverse for undo/redo.
#[derive(Clone)]
pub(super) struct EditAction {
    /// Low-level bit writes that realise the edit in the deflate stream.
    /// Always the source of truth for what changes on disk.
    pub patches: Vec<Patch>,
    pub label: String,
    pub kind: EditKind,
}

/// Structural classification of an edit. Drives whether `apply_edit` can
/// take the fast in-place path or has to fall back to a full reload.
#[derive(Clone)]
pub(super) enum EditKind {
    /// Same-length Huffman-code literal swap. One entry per patched
    /// channel, naming the output byte and the new symbol value. Every
    /// patch maps to an `Event::Lit`; no LZ77 topology changes.
    LiteralSwap { byte_updates: Vec<ByteWrite> },
    /// Distance-symbol redirect. The LZ77 source moves but the rest of
    /// the topology (event count, every other event's `out_pos` /
    /// `copy_len` / channel role) is unchanged: same-length Huffman codes
    /// guarantee that. Apply updates `events[i]`, recopies
    /// `output[out_pos..out_pos+copy_len]` from the new src, and
    /// propagates via a rebuilt `reverse_graph` (one ref's outgoing edges
    /// moved), the only structural index that moves; no `decode_deflate`.
    DistRedirect {
        /// Output-byte offset of the redirected ref. Doubles as the
        /// "nothing before this can have changed" floor for
        /// row-scoped re-render.
        out_pos: u32,
        copy_len: u16,
        /// Target `src_out_pos` for this edit. For undo of a redirect,
        /// this is the *previous* src.
        src_after: u32,
        /// Target `dist_sym` to write into `events[i]`.
        dist_sym_after: u8,
    },
}
