//! Reduction CPU kernel delegations: reduce, argmax/argmin, topk, cumulative
//! sum/product scans.

use coeus_core::{BackendError, CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};

use super::error::map_leto_error;
use super::CpuBackend;
use crate::backend_ops::ReductionOp;

#[inline]
pub(super) fn reduce<T, B>(
    _backend: &B,
    op: ReductionOp,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) -> Result<(), BackendError>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    coeus_leto::reduce_into(op, a_layout, a.as_slice(), axis, c_layout, c.as_mut_slice())
        .map_err(|error| map_leto_error("reduction", error))
}

#[inline]
pub(super) fn argmax<T, B>(
    backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut B::DeviceBuffer<i64>,
    c_layout: &Layout,
) where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    coeus_leto::argmax_into(
        a_layout,
        a.as_slice(),
        axis,
        c_layout,
        backend.as_mut_slice_i64(c),
    )
    .expect("coeus-leto argmax failed");
}

#[inline]
pub(super) fn argmin<T, B>(
    backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut B::DeviceBuffer<i64>,
    c_layout: &Layout,
) where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    coeus_leto::argmin_into(
        a_layout,
        a.as_slice(),
        axis,
        c_layout,
        backend.as_mut_slice_i64(c),
    )
    .expect("coeus-leto argmin failed");
}

#[inline]
pub(super) fn topk<T, B>(
    backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    k: usize,
    axis: usize,
    largest: bool,
    values: &mut B::DeviceBuffer<T>,
    _values_layout: &Layout,
    indices: &mut B::DeviceBuffer<i64>,
    _indices_layout: &Layout,
) where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    crate::reduction::topk::topk_impl(
        a.as_slice(),
        a_layout.shape(),
        k,
        axis,
        largest,
        values.as_mut_slice(),
        backend.as_mut_slice_i64(indices),
    );
}

#[inline]
pub(super) fn cumsum<T, B>(
    _backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    coeus_leto::cumsum_into(a_layout, a.as_slice(), axis, c_layout, c.as_mut_slice())
        .expect("coeus-leto cumsum failed");
}

#[inline]
pub(super) fn suffix_sum<T, B>(
    _backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    coeus_leto::suffix_sum_into(a_layout, a.as_slice(), axis, c_layout, c.as_mut_slice())
        .expect("coeus-leto suffix_sum failed");
}

#[inline]
pub(super) fn cumprod<T, B>(
    _backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    coeus_leto::cumprod_into(a_layout, a.as_slice(), axis, c_layout, c.as_mut_slice())
        .expect("coeus-leto cumprod failed");
}

#[inline]
pub(super) fn suffix_prod<T, B>(
    _backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    coeus_leto::suffix_prod_into(a_layout, a.as_slice(), axis, c_layout, c.as_mut_slice())
        .expect("coeus-leto suffix_prod failed");
}
