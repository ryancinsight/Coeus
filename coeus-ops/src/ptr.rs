// ── Parallel-safe raw pointer wrappers re-exports ──
// Expose the core-defined SendPtr and SendPtrMut using the local aliases Ptr and MutPtr.

pub(crate) use coeus_core::SendPtr as Ptr;
pub(crate) use coeus_core::SendPtrMut as MutPtr;
