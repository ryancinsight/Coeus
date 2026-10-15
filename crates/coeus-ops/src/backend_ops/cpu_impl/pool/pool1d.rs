//! One-dimensional pooling CPU kernel delegations.
#![allow(clippy::too_many_arguments)]

use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};

use super::super::error::map_leto_error;
use super::super::CpuBackend;
use super::check_rank;

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
    check_rank("max pool 1d", input_layout, 3)?;
    let input_slice = input.as_slice();
    let output_slice = output.as_mut_slice()?;
    coeus_leto::pooling_forward_into(
        leto_ops::PoolingMode::Maximum,
        input_layout,
        input_slice,
        kernel_size,
        stride,
        padding,
        dilation,
        output_layout,
        output_slice,
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
    check_rank("max pool 1d backward", grad_out_layout, 3)?;
    check_rank("max pool 1d backward", input_layout, 3)?;
    let grad_out_slice = grad_out.as_slice();
    let grad_input_slice = grad_input.as_mut_slice()?;
    coeus_leto::pooling_backward_into(
        leto_ops::PoolingMode::Maximum,
        grad_out_layout,
        grad_out_slice,
        Some((input_layout, input.as_slice())),
        kernel_size,
        stride,
        padding,
        dilation,
        grad_input_layout,
        grad_input_slice,
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
    check_rank("avg pool 1d", input_layout, 3)?;
    let input_slice = input.as_slice();
    let output_slice = output.as_mut_slice()?;
    coeus_leto::pooling_forward_into(
        leto_ops::PoolingMode::Average,
        input_layout,
        input_slice,
        kernel_size,
        stride,
        padding,
        dilation,
        output_layout,
        output_slice,
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
    check_rank("avg pool 1d backward", grad_out_layout, 3)?;
    let grad_out_slice = grad_out.as_slice();
    let grad_input_slice = grad_input.as_mut_slice()?;
    coeus_leto::pooling_backward_into(
        leto_ops::PoolingMode::Average,
        grad_out_layout,
        grad_out_slice,
        None,
        kernel_size,
        stride,
        padding,
        dilation,
        grad_input_layout,
        grad_input_slice,
    )
    .map_err(|error| map_leto_error("avg pool 1d backward", error))
}
