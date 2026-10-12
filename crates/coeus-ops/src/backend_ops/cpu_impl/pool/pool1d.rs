//! One-dimensional pooling CPU kernel delegations.
#![allow(clippy::too_many_arguments)]

use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};

use super::super::error::map_leto_error;
use super::super::CpuBackend;

#[inline]
pub(crate) fn max_pool1d<T, B>(
    _backend: &B,
    input: &B::DeviceBuffer<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut B::DeviceBuffer<T>,
    output_layout: &Layout,
) -> Result<(), B::Error>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    coeus_leto::pooling_forward_into(
        leto_ops::PoolingMode::Maximum,
        input_layout,
        input.as_slice(),
        kernel_size,
        stride,
        padding,
        dilation,
        output_layout,
        output.as_mut_slice(),
    )
    .map_err(|error| map_leto_error("max pool 1d", error))
}

#[inline]
pub(crate) fn max_pool1d_backward<T, B>(
    _backend: &B,
    grad_out: &B::DeviceBuffer<T>,
    grad_out_layout: &Layout,
    input: &B::DeviceBuffer<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input: &mut B::DeviceBuffer<T>,
    grad_input_layout: &Layout,
) -> Result<(), B::Error>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    coeus_leto::pooling_backward_into(
        leto_ops::PoolingMode::Maximum,
        grad_out_layout,
        grad_out.as_slice(),
        Some((input_layout, input.as_slice())),
        kernel_size,
        stride,
        padding,
        dilation,
        grad_input_layout,
        grad_input.as_mut_slice(),
    )
    .map_err(|error| map_leto_error("max pool 1d backward", error))
}

#[inline]
pub(crate) fn avg_pool1d<T, B>(
    _backend: &B,
    input: &B::DeviceBuffer<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut B::DeviceBuffer<T>,
    output_layout: &Layout,
) -> Result<(), B::Error>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    coeus_leto::pooling_forward_into(
        leto_ops::PoolingMode::Average,
        input_layout,
        input.as_slice(),
        kernel_size,
        stride,
        padding,
        dilation,
        output_layout,
        output.as_mut_slice(),
    )
    .map_err(|error| map_leto_error("avg pool 1d", error))
}

#[inline]
pub(crate) fn avg_pool1d_backward<T, B>(
    _backend: &B,
    grad_out: &B::DeviceBuffer<T>,
    grad_out_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input: &mut B::DeviceBuffer<T>,
    grad_input_layout: &Layout,
) -> Result<(), B::Error>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    coeus_leto::pooling_backward_into(
        leto_ops::PoolingMode::Average,
        grad_out_layout,
        grad_out.as_slice(),
        None,
        kernel_size,
        stride,
        padding,
        dilation,
        grad_input_layout,
        grad_input.as_mut_slice(),
    )
    .map_err(|error| map_leto_error("avg pool 1d backward", error))
}
