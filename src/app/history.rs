//! Undo / redo history.
//!
//! Encapsulates the two stacks of inverse-[`EditAction`]s so the
//! "recording a new edit clears redo" invariant lives inside
//! [`UndoHistory::record`] and can't be forgotten by callers.

use super::edit::EditAction;

#[derive(Default)]
pub(super) struct UndoHistory {
    undo: Vec<EditAction>,
    redo: Vec<EditAction>,
}

impl UndoHistory {
    /// A new edit was applied: push the inverse onto undo, clear redo.
    pub fn record(&mut self, inverse: EditAction) {
        self.undo.push(inverse);
        self.redo.clear();
    }

    pub fn pop_undo(&mut self) -> Option<EditAction> {
        self.undo.pop()
    }

    pub fn pop_redo(&mut self) -> Option<EditAction> {
        self.redo.pop()
    }

    pub fn push_undo(&mut self, e: EditAction) {
        self.undo.push(e);
    }

    pub fn push_redo(&mut self, e: EditAction) {
        self.redo.push(e);
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }
}
