use crate::backend::{get_wgpu_context, WgpuBackend, WgpuBackendError, WgpuScalar};
use crate::storage::WgpuStorage;
use coeus_core::{BackendError, Layout};
use coeus_hephaestus::{PoolingBackend, PoolingProvider, WindowConfiguration};
use hephaestus_core::{ComputeDevice, HephaestusError, PoolingMode};
use hephaestus_wgpu::{WgpuDevice, WgpuPoolingOps, WgpuWindowScalar};
use leto::WindowParameters;

impl<T> PoolingProvider<T> for WgpuBackend
where
    T: WgpuScalar + leto_ops::Scalar + WgpuWindowScalar,
{
    type Operations = WgpuPoolingOps;
}

impl<T> PoolingBackend<T> for WgpuBackend
where
    T: WgpuScalar + leto_ops::Scalar + WgpuWindowScalar,
{
    type Device = WgpuDevice;
    type Operations = WgpuPoolingOps;

    fn pooling_device() -> &'static Self::Device {
        &get_wgpu_context().hephaestus_device
    }

    fn pooling_buffer(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T> {
        storage.buffer.as_ref()
    }

    fn pooling_configuration_error(operation: &'static str, reason: String) -> Self::Error {
        WgpuBackendError::Validation(BackendError::Storage { operation, reason })
    }

    fn pooling_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        WgpuBackendError::dispatch(operation, source)
    }
}

impl<T> coeus_ops::PoolOps<T> for WgpuBackend
where
    T: WgpuScalar + leto_ops::Scalar + WgpuWindowScalar,
{
    fn max_pool2d(
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
        forward::<T, 4, 2>(
            "max_pool2d",
            (input, input_layout),
            WindowConfiguration {
                kernel: [kernel_size; 2],
                stride: [stride; 2],
                padding: [padding; 2],
                dilation: [dilation; 2],
            },
            PoolingMode::Maximum,
            (output, output_layout),
        )
    }

    fn max_pool2d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error> {
        backward::<T, 4, 2>(
            "max_pool2d_backward",
            (grad_out, grad_out_layout),
            Some((input, input_layout)),
            WindowConfiguration {
                kernel: [kernel_size; 2],
                stride: [stride; 2],
                padding: [padding; 2],
                dilation: [dilation; 2],
            },
            PoolingMode::Maximum,
            (grad_input, grad_input_layout),
        )
    }

    fn avg_pool2d(
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
        forward::<T, 4, 2>(
            "avg_pool2d",
            (input, input_layout),
            WindowConfiguration {
                kernel: [kernel_size; 2],
                stride: [stride; 2],
                padding: [padding; 2],
                dilation: [dilation; 2],
            },
            PoolingMode::Average,
            (output, output_layout),
        )
    }

    fn avg_pool2d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error> {
        backward::<T, 4, 2>(
            "avg_pool2d_backward",
            (grad_out, grad_out_layout),
            None,
            WindowConfiguration {
                kernel: [kernel_size; 2],
                stride: [stride; 2],
                padding: [padding; 2],
                dilation: [dilation; 2],
            },
            PoolingMode::Average,
            (grad_input, grad_input_layout),
        )
    }

    fn max_pool3d(
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
        forward::<T, 5, 3>(
            "max_pool3d",
            (input, input_layout),
            WindowConfiguration {
                kernel: [kernel_size; 3],
                stride: [stride; 3],
                padding: [padding; 3],
                dilation: [dilation; 3],
            },
            PoolingMode::Maximum,
            (output, output_layout),
        )
    }

    fn max_pool3d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error> {
        backward::<T, 5, 3>(
            "max_pool3d_backward",
            (grad_out, grad_out_layout),
            Some((input, input_layout)),
            WindowConfiguration {
                kernel: [kernel_size; 3],
                stride: [stride; 3],
                padding: [padding; 3],
                dilation: [dilation; 3],
            },
            PoolingMode::Maximum,
            (grad_input, grad_input_layout),
        )
    }

    fn avg_pool3d(
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
        forward::<T, 5, 3>(
            "avg_pool3d",
            (input, input_layout),
            WindowConfiguration {
                kernel: [kernel_size; 3],
                stride: [stride; 3],
                padding: [padding; 3],
                dilation: [dilation; 3],
            },
            PoolingMode::Average,
            (output, output_layout),
        )
    }

    fn avg_pool3d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error> {
        backward::<T, 5, 3>(
            "avg_pool3d_backward",
            (grad_out, grad_out_layout),
            None,
            WindowConfiguration {
                kernel: [kernel_size; 3],
                stride: [stride; 3],
                padding: [padding; 3],
                dilation: [dilation; 3],
            },
            PoolingMode::Average,
            (grad_input, grad_input_layout),
        )
    }

    fn max_pool1d(
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
        forward::<T, 3, 1>(
            "max_pool1d",
            (input, input_layout),
            WindowConfiguration {
                kernel: [kernel_size],
                stride: [stride],
                padding: [padding],
                dilation: [dilation],
            },
            PoolingMode::Maximum,
            (output, output_layout),
        )
    }

    fn max_pool1d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error> {
        backward::<T, 3, 1>(
            "max_pool1d_backward",
            (grad_out, grad_out_layout),
            Some((input, input_layout)),
            WindowConfiguration {
                kernel: [kernel_size],
                stride: [stride],
                padding: [padding],
                dilation: [dilation],
            },
            PoolingMode::Maximum,
            (grad_input, grad_input_layout),
        )
    }

    fn avg_pool1d(
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
        forward::<T, 3, 1>(
            "avg_pool1d",
            (input, input_layout),
            WindowConfiguration {
                kernel: [kernel_size],
                stride: [stride],
                padding: [padding],
                dilation: [dilation],
            },
            PoolingMode::Average,
            (output, output_layout),
        )
    }

    fn avg_pool1d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error> {
        backward::<T, 3, 1>(
            "avg_pool1d_backward",
            (grad_out, grad_out_layout),
            None,
            WindowConfiguration {
                kernel: [kernel_size],
                stride: [stride],
                padding: [padding],
                dilation: [dilation],
            },
            PoolingMode::Average,
            (grad_input, grad_input_layout),
        )
    }
}

fn forward<T, const R: usize, const S: usize>(
    operation: &'static str,
    input: (&WgpuStorage<T>, &Layout),
    window: WindowConfiguration<S>,
    mode: PoolingMode,
    output: (&WgpuStorage<T>, &Layout),
) -> Result<(), WgpuBackendError>
where
    T: WgpuScalar + leto_ops::Scalar + WgpuWindowScalar,
    WgpuBackend: PoolingBackend<T>,
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
    coeus_hephaestus::pooling_forward::<WgpuBackend, T, R, S>(
        operation, input, parameters, mode, output,
    )
}

fn backward<T, const R: usize, const S: usize>(
    operation: &'static str,
    grad_output: (&WgpuStorage<T>, &Layout),
    input: Option<(&WgpuStorage<T>, &Layout)>,
    window: WindowConfiguration<S>,
    mode: PoolingMode,
    grad_input: (&WgpuStorage<T>, &Layout),
) -> Result<(), WgpuBackendError>
where
    T: WgpuScalar + leto_ops::Scalar + WgpuWindowScalar,
    WgpuBackend: PoolingBackend<T>,
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
    coeus_hephaestus::pooling_backward::<WgpuBackend, T, R, S>(
        operation,
        grad_output,
        input,
        parameters,
        mode,
        grad_input,
    )
}
