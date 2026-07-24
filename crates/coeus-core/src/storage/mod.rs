// ── Storage module ──
// Memory abstraction layer. Built on Mnemosyne allocator.

mod cow;
mod cpu;
/// Storage trait definitions.
pub mod traits;

pub use cow::CowStorage;
pub use cpu::CpuStorage;
pub use traits::{private, CpuAddressableStorage, CpuAddressableStorageMut, Storage, StorageMut};
