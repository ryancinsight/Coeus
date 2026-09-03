use crate::backend::{get_cuda_device, CudaBackend, CudaScalar};
use crate::storage::CudaStorage;
use crate::CudaBackendError;
use coeus_core::{BackendError, Layout};
use coeus_hephaestus::{UnfoldFoldBackend, UnfoldFoldProvider, WindowConfiguration};
use hephaestus_core::{ComputeDevice, CudaC, DialectScalar, HephaestusError};
use hephaestus_cuda::{CudaDevice, CudaSlidingWindowOps};
use leto::WindowParameters;

impl<T> UnfoldFoldProvider<T> for CudaBackend
where
    T: CudaScalar + DialectScalar<CudaC>,
{
    type Operations = CudaSlidingWindowOps;
}

impl<T> UnfoldFoldBackend<T> for CudaBackend
where
    T: CudaScalar + DialectScalar<CudaC>,
{
    type Device = CudaDevice;
    type Operations = CudaSlidingWindowOps;

    fn unfold_fold_device() -> &'static Self::Device {
        get_cuda_device()
    }

    fn unfold_fold_buffer(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T> {
        storage.buffer.as_ref()
    }

    fn unfold_fold_configuration_error(operation: &'static str, reason: String) -> Self::Error {
        CudaBackendError::Validation {
            source: BackendError::Storage { operation, reason },
        }
    }

    fn unfold_fold_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        CudaBackendError::dispatch(operation, source)
    }
}

impl<T> coeus_ops::UnfoldFoldOps<T> for CudaBackend
where
    T: CudaScalar + DialectScalar<CudaC>,
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
    input: (&CudaStorage<T>, &Layout),
    window: WindowConfiguration<S>,
    output: (&CudaStorage<T>, &Layout),
) -> Result<(), CudaBackendError>
where
    T: CudaScalar + DialectScalar<CudaC>,
    CudaBackend: UnfoldFoldBackend<T>,
{
    let parameters = WindowParameters::new(
        window.kernel,
        window.stride,
        window.padding,
        window.dilation,
    )
    .map_err(|error| CudaBackendError::Validation {
        source: BackendError::Storage {
            operation,
            reason: error.to_string(),
        },
    })?;
    coeus_hephaestus::unfold_fold_unfold::<CudaBackend, T, R, S>(
        operation, input, parameters, output,
    )
}

fn fold<T, const R: usize, const S: usize>(
    operation: &'static str,
    input: (&CudaStorage<T>, &Layout),
    output_spatial_shape: [usize; S],
    window: WindowConfiguration<S>,
    output: (&CudaStorage<T>, &Layout),
) -> Result<(), CudaBackendError>
where
    T: CudaScalar + DialectScalar<CudaC>,
    CudaBackend: UnfoldFoldBackend<T>,
{
    let parameters = WindowParameters::new(
        window.kernel,
        window.stride,
        window.padding,
        window.dilation,
    )
    .map_err(|error| CudaBackendError::Validation {
        source: BackendError::Storage {
            operation,
            reason: error.to_string(),
        },
    })?;
    coeus_hephaestus::unfold_fold_fold::<CudaBackend, T, R, S>(
        operation,
        input,
        output_spatial_shape,
        parameters,
        output,
    )
}
