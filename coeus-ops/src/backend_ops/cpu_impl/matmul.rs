//! Matrix-multiply CPU kernel delegations.
#![allow(clippy::too_many_arguments)]

use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};

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
) where
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
    .expect("coeus-leto matmul failed");
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
) where
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
    .expect("coeus-leto batched matmul failed");
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
) where
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
    .expect("coeus-leto matmul_accumulate failed");
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
) where
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
    .expect("coeus-leto batched_matmul_accumulate failed");
}
