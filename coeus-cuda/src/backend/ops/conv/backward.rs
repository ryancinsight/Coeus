use crate::backend::ops::cast::{cast_storage, cast_storage_mut};
use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::kernels;
use crate::storage::CudaStorage;
use coeus_core::Layout;

impl CudaBackend {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_conv1d_backward<T: CudaScalar>(
        &self,
        grad_out: &CudaStorage<T>,
        grad_out_layout: &Layout,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        weight: &CudaStorage<T>,
        weight_layout: &Layout,
        mut grad_input: Option<&mut CudaStorage<T>>,
        grad_input_layout: &Layout,
        mut grad_weight: Option<&mut CudaStorage<T>>,
        grad_weight_layout: &Layout,
        mut grad_bias: Option<&mut CudaStorage<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) {
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let grad_out_f32 = cast_storage::<T, f32>(grad_out);
            let input_f32 = cast_storage::<T, f32>(input);
            let weight_f32 = cast_storage::<T, f32>(weight);
            let mut grad_input_f32 = grad_input.as_mut().map(|gi| cast_storage_mut::<T, f32>(gi));
            let mut grad_weight_f32 = grad_weight
                .as_mut()
                .map(|gw| cast_storage_mut::<T, f32>(gw));
            let mut grad_bias_f32 = grad_bias.as_mut().map(|gb| cast_storage_mut::<T, f32>(gb));

            if kernels::launch_conv1d_backward(
                &grad_out_f32,
                grad_out_layout,
                &input_f32,
                input_layout,
                &weight_f32,
                weight_layout,
                grad_input_f32.as_mut(),
                grad_input_layout,
                grad_weight_f32.as_mut(),
                grad_weight_layout,
                grad_bias_f32.as_mut(),
                stride,
                padding,
                dilation,
            ) {
                if let Some(gi) = grad_input.as_mut() {
                    **gi = cast_storage(&grad_input_f32.unwrap());
                }
                if let Some(gw) = grad_weight.as_mut() {
                    **gw = cast_storage(&grad_weight_f32.unwrap());
                }
                if let Some(gb) = grad_bias.as_mut() {
                    **gb = cast_storage(&grad_bias_f32.unwrap());
                }
                return;
            }
        }
        self.fallback_conv1d_backward(
            grad_out,
            grad_out_layout,
            input,
            input_layout,
            weight,
            weight_layout,
            grad_input,
            grad_input_layout,
            grad_weight,
            grad_weight_layout,
            grad_bias,
            stride,
            padding,
            dilation,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_conv2d_backward<T: CudaScalar>(
        &self,
        grad_out: &CudaStorage<T>,
        grad_out_layout: &Layout,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        weight: &CudaStorage<T>,
        weight_layout: &Layout,
        mut grad_input: Option<&mut CudaStorage<T>>,
        grad_input_layout: &Layout,
        mut grad_weight: Option<&mut CudaStorage<T>>,
        grad_weight_layout: &Layout,
        mut grad_bias: Option<&mut CudaStorage<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) {
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let grad_out_f32 = cast_storage::<T, f32>(grad_out);
            let input_f32 = cast_storage::<T, f32>(input);
            let weight_f32 = cast_storage::<T, f32>(weight);
            let mut grad_input_f32 = grad_input.as_mut().map(|gi| cast_storage_mut::<T, f32>(gi));
            let mut grad_weight_f32 = grad_weight
                .as_mut()
                .map(|gw| cast_storage_mut::<T, f32>(gw));
            let mut grad_bias_f32 = grad_bias.as_mut().map(|gb| cast_storage_mut::<T, f32>(gb));

            if kernels::launch_conv2d_backward(
                &grad_out_f32,
                grad_out_layout,
                &input_f32,
                input_layout,
                &weight_f32,
                weight_layout,
                grad_input_f32.as_mut(),
                grad_input_layout,
                grad_weight_f32.as_mut(),
                grad_weight_layout,
                grad_bias_f32.as_mut(),
                stride,
                padding,
                dilation,
            ) {
                if let Some(gi) = grad_input.as_mut() {
                    **gi = cast_storage(&grad_input_f32.unwrap());
                }
                if let Some(gw) = grad_weight.as_mut() {
                    **gw = cast_storage(&grad_weight_f32.unwrap());
                }
                if let Some(gb) = grad_bias.as_mut() {
                    **gb = cast_storage(&grad_bias_f32.unwrap());
                }
                return;
            }
        }
        self.fallback_conv2d_backward(
            grad_out,
            grad_out_layout,
            input,
            input_layout,
            weight,
            weight_layout,
            grad_input,
            grad_input_layout,
            grad_weight,
            grad_weight_layout,
            grad_bias,
            stride,
            padding,
            dilation,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_conv3d_backward<T: CudaScalar>(
        &self,
        grad_out: &CudaStorage<T>,
        grad_out_layout: &Layout,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        weight: &CudaStorage<T>,
        weight_layout: &Layout,
        mut grad_input: Option<&mut CudaStorage<T>>,
        grad_input_layout: &Layout,
        mut grad_weight: Option<&mut CudaStorage<T>>,
        grad_weight_layout: &Layout,
        mut grad_bias: Option<&mut CudaStorage<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) {
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let grad_out_f32 = cast_storage::<T, f32>(grad_out);
            let input_f32 = cast_storage::<T, f32>(input);
            let weight_f32 = cast_storage::<T, f32>(weight);
            let mut grad_input_f32 = grad_input.as_mut().map(|gi| cast_storage_mut::<T, f32>(gi));
            let mut grad_weight_f32 = grad_weight
                .as_mut()
                .map(|gw| cast_storage_mut::<T, f32>(gw));
            let mut grad_bias_f32 = grad_bias.as_mut().map(|gb| cast_storage_mut::<T, f32>(gb));

            if kernels::launch_conv3d_backward(
                &grad_out_f32,
                grad_out_layout,
                &input_f32,
                input_layout,
                &weight_f32,
                weight_layout,
                grad_input_f32.as_mut(),
                grad_input_layout,
                grad_weight_f32.as_mut(),
                grad_weight_layout,
                grad_bias_f32.as_mut(),
                stride,
                padding,
                dilation,
            ) {
                if let Some(gi) = grad_input.as_mut() {
                    **gi = cast_storage(&grad_input_f32.unwrap());
                }
                if let Some(gw) = grad_weight.as_mut() {
                    **gw = cast_storage(&grad_weight_f32.unwrap());
                }
                if let Some(gb) = grad_bias.as_mut() {
                    **gb = cast_storage(&grad_bias_f32.unwrap());
                }
                return;
            }
        }
        self.fallback_conv3d_backward(
            grad_out,
            grad_out_layout,
            input,
            input_layout,
            weight,
            weight_layout,
            grad_input,
            grad_input_layout,
            grad_weight,
            grad_weight_layout,
            grad_bias,
            stride,
            padding,
            dilation,
        );
    }
}
