// ── Coeus Core: Fundamental abstractions ──
// Provides dtype system, layout descriptors, storage primitives,
// and backend execution abstractions.

pub mod dtype;
pub mod layout;
pub mod storage;
pub mod backend;
pub mod ptr;

// Re-export the most commonly used items
pub use dtype::{Scalar, Float, Int, Complex};
pub use layout::{Shape, Strides, Layout, ConstLayout, ConstShape};
pub use storage::{Storage, StorageMut, CpuStorage, CpuAddressableStorage, CpuAddressableStorageMut};
pub use backend::{Backend, ComputeBackend, MoiraiBackend, SequentialBackend};
pub use ptr::{SendPtr, SendPtrMut};
