use super::supports_native_conv_transpose_layouts;
use crate::backend::ops::cast::{cast_storage, cast_storage_mut};
use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::error::CudaBackendError;
use crate::kernels;
use crate::storage::CudaStorage;
use coeus_core::{Layout, Storage};

impl CudaBackend {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_conv_transpose1d<T: CudaScalar + coeus_core::Float>(
        &self,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        weight: &CudaStorage<T>,
        weight_layout: &Layout,
        bias: Option<&CudaStorage<T>>,
        stride: usize,
        padding: usize,
        _output_padding: usize,
        dilation: usize,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        if supports_native_conv_transpose_layouts::<1>(input_layout, weight_layout, output_layout)
            && get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let input_f32 = cast_storage::<T, f32>(input);
            let weight_f32 = cast_storage::<T, f32>(weight);
            let bias_f32 = bias.map(|b| cast_storage::<T, f32>(b));
            let mut output_f32 = cast_storage_mut::<T, f32>(output);
            // [n, c_in, l] / [c_in, c_out, k] / [n, c_out, l_out]
            let n = input_layout.shape()[0];
            let c_in = input_layout.shape()[1];
            let l = input_layout.shape()[2];
            let c_out = weight_layout.shape()[1];
            let k = weight_layout.shape()[2];
            let l_out = output_layout.shape()[2];
            if kernels::launch_conv_transpose1d(
                &input_f32,
                &weight_f32,
                bias_f32.as_ref(),
                &mut output_f32,
                n,
                c_in,
                l,
                c_out,
                k,
                l_out,
                stride,
                padding,
                dilation,
            ) {
                return Ok(());
            }
        }
        Err(CudaBackendError::dispatch_unavailable(
            "conv_transpose1d",
            "native CUDA dispatch requires supported contiguous f32 layouts and an initialized CUDA context",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_conv_transpose2d<T: CudaScalar + coeus_core::Float>(
        &self,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        weight: &CudaStorage<T>,
        weight_layout: &Layout,
        bias: Option<&CudaStorage<T>>,
        stride: usize,
        padding: usize,
        _output_padding: usize,
        dilation: usize,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        if supports_native_conv_transpose_layouts::<2>(input_layout, weight_layout, output_layout)
            && get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let input_f32 = cast_storage::<T, f32>(input);
            let weight_f32 = cast_storage::<T, f32>(weight);
            let bias_f32 = bias.map(|b| cast_storage::<T, f32>(b));
            let mut output_f32 = cast_storage_mut::<T, f32>(output);
            // [n, c_in, h, w] / [c_in, c_out, kh, kw] / [n, c_out, h_out, w_out]
            let n = input_layout.shape()[0];
            let c_in = input_layout.shape()[1];
            let h = input_layout.shape()[2];
            let w = input_layout.shape()[3];
            let c_out = weight_layout.shape()[1];
            let kh = weight_layout.shape()[2];
            let kw = weight_layout.shape()[3];
            let h_out = output_layout.shape()[2];
            let w_out = output_layout.shape()[3];
            if kernels::launch_conv_transpose2d(
                &input_f32,
                &weight_f32,
                bias_f32.as_ref(),
                &mut output_f32,
                n,
                c_in,
                h,
                w,
                c_out,
                kh,
                kw,
                h_out,
                w_out,
                stride,
                padding,
                dilation,
            ) {
                return Ok(());
            }
        }
        Err(CudaBackendError::dispatch_unavailable(
            "conv_transpose2d",
            "native CUDA dispatch requires supported contiguous f32 layouts and an initialized CUDA context",
        ))
    }
}
