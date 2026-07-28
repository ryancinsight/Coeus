use super::checked_numel;
use crate::backend::ops::cast::{cast_storage, cast_storage_mut};
use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::error::CudaBackendError;
use crate::kernels;
use crate::storage::CudaStorage;
use coeus_core::Layout;

impl CudaBackend {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_conv1d<T: CudaScalar>(
        &self,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        weight: &CudaStorage<T>,
        weight_layout: &Layout,
        bias: Option<&CudaStorage<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        let launched = if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            match checked_numel(output_layout) {
                Some(out_numel) => {
                    let input_f32 = cast_storage::<T, f32>(input);
                    let weight_f32 = cast_storage::<T, f32>(weight);
                    let bias_f32 = bias.map(|b| cast_storage::<T, f32>(b));
                    let mut output_f32 = cast_storage_mut::<T, f32>(output);
                    kernels::launch_conv1d(
                        &input_f32,
                        &weight_f32,
                        bias_f32.as_ref(),
                        &mut output_f32,
                        input_layout,
                        weight_layout,
                        output_layout,
                        stride,
                        padding,
                        dilation,
                        out_numel,
                    )
                }
                None => {
                    return Err(CudaBackendError::dispatch_unavailable(
                        "conv1d",
                        "output element-count arithmetic overflowed",
                    ));
                }
            }
        } else {
            false
        };
        if launched {
            return Ok(());
        }
        Err(CudaBackendError::dispatch_unavailable(
            "conv1d",
            "native CUDA dispatch requires an initialized context and supported f32 layouts",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_conv2d<T: CudaScalar>(
        &self,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        weight: &CudaStorage<T>,
        weight_layout: &Layout,
        bias: Option<&CudaStorage<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        let launched = if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            match checked_numel(output_layout) {
                Some(out_numel) => {
                    let input_f32 = cast_storage::<T, f32>(input);
                    let weight_f32 = cast_storage::<T, f32>(weight);
                    let bias_f32 = bias.map(|b| cast_storage::<T, f32>(b));
                    let mut output_f32 = cast_storage_mut::<T, f32>(output);
                    kernels::launch_conv2d(
                        &input_f32,
                        &weight_f32,
                        bias_f32.as_ref(),
                        &mut output_f32,
                        input_layout,
                        weight_layout,
                        output_layout,
                        stride,
                        padding,
                        dilation,
                        out_numel,
                    )
                }
                None => {
                    return Err(CudaBackendError::dispatch_unavailable(
                        "conv2d",
                        "output element-count arithmetic overflowed",
                    ));
                }
            }
        } else {
            false
        };
        if launched {
            return Ok(());
        }
        Err(CudaBackendError::dispatch_unavailable(
            "conv2d",
            "native CUDA dispatch requires an initialized context and supported f32 layouts",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_conv3d<T: CudaScalar>(
        &self,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        weight: &CudaStorage<T>,
        weight_layout: &Layout,
        bias: Option<&CudaStorage<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        let launched = if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            match checked_numel(output_layout) {
                Some(out_numel) => {
                    let input_f32 = cast_storage::<T, f32>(input);
                    let weight_f32 = cast_storage::<T, f32>(weight);
                    let bias_f32 = bias.map(|b| cast_storage::<T, f32>(b));
                    let mut output_f32 = cast_storage_mut::<T, f32>(output);
                    kernels::launch_conv3d(
                        &input_f32,
                        &weight_f32,
                        bias_f32.as_ref(),
                        &mut output_f32,
                        input_layout,
                        weight_layout,
                        output_layout,
                        stride,
                        padding,
                        dilation,
                        out_numel,
                    )
                }
                None => {
                    return Err(CudaBackendError::dispatch_unavailable(
                        "conv3d",
                        "output element-count arithmetic overflowed",
                    ));
                }
            }
        } else {
            false
        };
        if launched {
            return Ok(());
        }
        Err(CudaBackendError::dispatch_unavailable(
            "conv3d",
            "native CUDA dispatch requires an initialized context and supported f32 layouts",
        ))
    }
}
