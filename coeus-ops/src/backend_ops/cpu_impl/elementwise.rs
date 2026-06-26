//! Elementwise CPU kernel delegations: unary and binary ops.
#![allow(clippy::too_many_arguments)]

use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};

use super::CpuBackend;
use crate::backend_ops::{BinaryOp, UnaryOp};

#[inline]
pub(super) fn elementwise_binary<T, B>(
    _backend: &B,
    op: BinaryOp,
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
    coeus_leto::elementwise_binary_into(
        op,
        a_layout,
        a.as_slice(),
        b_layout,
        b.as_slice(),
        c_layout,
        c.as_mut_slice(),
    )
    .expect("coeus-leto elementwise_binary failed");
}

#[inline]
pub(super) fn elementwise_unary<T, B>(
    _backend: &B,
    op: UnaryOp,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    coeus_leto::elementwise_unary_into(op, a_layout, a.as_slice(), c_layout, c.as_mut_slice())
        .expect("coeus-leto elementwise_unary failed");
}
