//! NaN-aware reductions.
//!
//! `torch.nansum`/`torch.nanmean` equivalents that treat NaN as missing,
//! returning scalar `Var`s usable inside a tracked loss graph.

/// Sum of all finite elements, treating NaN as zero (`torch.nansum`).
///
/// Returns a scalar `Var` (shape `[1]`).
///
/// # Errors
///
/// Returns the backend error when an intermediate allocation or dispatch fails.
pub fn nansum<T: coeus_core::Float + coeus_cpu::RealScalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &coeus_autograd::Var<T, B>,
) -> Result<coeus_autograd::Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    coeus_autograd::nansum(x)
}

/// Mean of all finite elements, treating NaN as missing (`torch.nanmean`).
///
/// Returns a scalar `Var` (shape `[1]`).
///
/// # Errors
///
/// Returns the backend error when an intermediate allocation or dispatch fails.
pub fn nanmean<
    T: coeus_core::Float + coeus_cpu::RealScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    x: &coeus_autograd::Var<T, B>,
) -> Result<coeus_autograd::Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    coeus_autograd::nanmean(x)
}
