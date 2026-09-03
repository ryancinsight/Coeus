use crate::backend::{get_wgpu_context, WgpuBackend, WgpuBackendError, WgpuScalar};
use crate::storage::WgpuStorage;
use coeus_core::{BackendError, Layout};
use coeus_hephaestus::{UnfoldFoldBackend, UnfoldFoldProvider, WindowConfiguration};
use hephaestus_core::{ComputeDevice, HephaestusError};
use hephaestus_wgpu::{WgpuDevice, WgpuSlidingWindowOps, WgpuWindowScalar};
use leto::WindowParameters;


impl<T> UnfoldFoldProvider<T> for WgpuBackend
where
    T: WgpuScalar + leto_ops::Scalar + WgpuWindowScalar,
{
    type Operations = WgpuSlidingWindowOps;
}

impl<T> UnfoldFoldBackend<T> for WgpuBackend
where
    T: WgpuScalar + leto_ops::Scalar + WgpuWindowScalar,
{
    type Device = WgpuDevice;
    type Operations = WgpuSlidingWindowOps;

    fn unfold_fold_device() -> &'static Self::Device {
        &get_wgpu_context().hephaestus_device
    }

    fn unfold_fold_buffer(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T> {
        storage.buffer.as_ref()
    }

    fn unfold_fold_configuration_error(operation: &'static str, reason: String) -> Self::Error {
        WgpuBackendError::Validation(BackendError::Storage { operation, reason })
    }

    fn unfold_fold_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        WgpuBackendError::dispatch(operation, source)
    }
}

impl<T> coeus_ops::UnfoldFoldOps<T> for WgpuBackend
where
    T: WgpuScalar + leto_ops::Scalar + WgpuWindowScalar,
{
    fn unfold1d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        unfold::<T, 3, 1>(
            "unfold1d",
            (input, input_layout),
            WindowConfiguration {
                kernel: [kernel_size],
                stride: [stride],
                padding: [padding],
                dilation: [dilation],
            },
            (output, output_layout),
        )
    }

    fn fold1d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output_size: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        fold::<T, 3, 1>(
            "fold1d",
            (input, input_layout),
            [output_size],
            WindowConfiguration {
                kernel: [kernel_size],
                stride: [stride],
                padding: [padding],
                dilation: [dilation],
            },
            (output, output_layout),
        )
    }

    fn unfold2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        unfold::<T, 4, 2>(
            "unfold2d",
            (input, input_layout),
            WindowConfiguration {
                kernel: [kernel_h, kernel_w],
                stride: [stride_h, stride_w],
                padding: [padding_h, padding_w],
                dilation: [dilation_h, dilation_w],
            },
            (output, output_layout),
        )
    }

    fn fold2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output_h: usize,
        output_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        fold::<T, 4, 2>(
            "fold2d",
            (input, input_layout),
            [output_h, output_w],
            WindowConfiguration {
                kernel: [kernel_h, kernel_w],
                stride: [stride_h, stride_w],
                padding: [padding_h, padding_w],
                dilation: [dilation_h, dilation_w],
            },
            (output, output_layout),
        )
    }
}

fn unfold<T, const R: usize, const S: usize>(
    operation: &'static str,
    input: (&WgpuStorage<T>, &Layout),
    window: WindowConfiguration<S>,
    output: (&WgpuStorage<T>, &Layout),
) -> Result<(), WgpuBackendError>
where
    T: WgpuScalar + leto_ops::Scalar + WgpuWindowScalar,
    WgpuBackend: UnfoldFoldBackend<T>,
{
    let parameters = WindowParameters::new(
        window.kernel,
        window.stride,
        window.padding,
        window.dilation,
    )
    .map_err(|error| {
        WgpuBackendError::Validation(BackendError::Storage {
            operation,
            reason: error.to_string(),
        })
    })?;
    coeus_hephaestus::unfold_fold_unfold::<WgpuBackend, T, R, S>(
        operation, input, parameters, output,
    )
}

fn fold<T, const R: usize, const S: usize>(
    operation: &'static str,
    input: (&WgpuStorage<T>, &Layout),
    output_spatial_shape: [usize; S],
    window: WindowConfiguration<S>,
    output: (&WgpuStorage<T>, &Layout),
) -> Result<(), WgpuBackendError>
where
    T: WgpuScalar + leto_ops::Scalar + WgpuWindowScalar,
    WgpuBackend: UnfoldFoldBackend<T>,
{
    let parameters = WindowParameters::new(
        window.kernel,
        window.stride,
        window.padding,
        window.dilation,
    )
    .map_err(|error| {
        WgpuBackendError::Validation(BackendError::Storage {
            operation,
            reason: error.to_string(),
        })
    })?;
    coeus_hephaestus::unfold_fold_fold::<WgpuBackend, T, R, S>(
        operation,
        input,
        output_spatial_shape,
        parameters,
        output,
    )
}
