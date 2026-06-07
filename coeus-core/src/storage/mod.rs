// ── Storage module ──
// Memory abstraction layer. Built on Mnemosyne allocator.

pub mod traits;
mod cow;
mod cpu;

pub use traits::{Storage, StorageMut, CpuAddressableStorage, CpuAddressableStorageMut, private};
pub use cow::CowStorage;
pub use cpu::CpuStorage;
