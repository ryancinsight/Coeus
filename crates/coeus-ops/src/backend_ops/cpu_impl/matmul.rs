//! Matrix-multiply CPU kernel delegations.

use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};

use super::error::map_leto_error;
use super::CpuBackend;

#[inline]
pub(super) fn matmul<T, B>(
    _backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    b: &B::DeviceBuffer<T>,
    b_layout: &Layout,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) -> Result<(), B::Error>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    coeus_leto::matmul_into(
        a_layout,
        a.as_slice(),
        b_layout,
        b.as_slice(),
        c_layout,
        c.as_mut_slice(),
    )
    .map_err(|error| map_leto_error("matmul", error))
}

#[inline]
pub(super) fn batched_matmul<T, B>(
    _backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    b: &B::DeviceBuffer<T>,
    b_layout: &Layout,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) -> Result<(), B::Error>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    coeus_leto::batched_matmul_into(
        a_layout,
        a.as_slice(),
        b_layout,
        b.as_slice(),
        c_layout,
        c.as_mut_slice(),
    )
    .map_err(|error| map_leto_error("batched matmul", error))
}

#[inline]
pub(super) fn matmul_accumulate<T, B>(
    _backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    b: &B::DeviceBuffer<T>,
    b_layout: &Layout,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) -> Result<(), B::Error>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    coeus_leto::matmul_accumulate_into(
        a_layout,
        a.as_slice(),
        b_layout,
        b.as_slice(),
        c_layout,
        c.as_mut_slice(),
    )
    .map_err(|error| map_leto_error("matmul accumulate", error))
}

#[inline]
pub(super) fn batched_matmul_accumulate<T, B>(
    _backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    b: &B::DeviceBuffer<T>,
    b_layout: &Layout,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) -> Result<(), B::Error>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    coeus_leto::batched_matmul_accumulate_into(
        a_layout,
        a.as_slice(),
        b_layout,
        b.as_slice(),
        c_layout,
        c.as_mut_slice(),
    )
    .map_err(|error| map_leto_error("batched matmul accumulate", error))
}
