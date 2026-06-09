use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::kernels;
use crate::storage::CudaStorage;
use coeus_core::Layout;

fn cast_storage<T, U>(storage: &CudaStorage<T>) -> CudaStorage<U> {
    CudaStorage {
        buffer: storage.buffer.clone(),
        len: storage.len,
        _marker: std::marker::PhantomData,
    }
}

fn cast_storage_mut<T, U>(storage: &mut CudaStorage<T>) -> CudaStorage<U> {
    CudaStorage {
        buffer: storage.buffer.clone(),
        len: storage.len,
        _marker: std::marker::PhantomData,
    }
}

impl CudaBackend {
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
    ) {
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let input_f32 = cast_storage::<T, f32>(input);
            let weight_f32 = cast_storage::<T, f32>(weight);
            let bias_f32 = bias.map(|b| cast_storage::<T, f32>(b));
            let mut output_f32 = cast_storage_mut::<T, f32>(output);
            let out_numel = output_layout.shape().iter().product::<usize>();
            if kernels::launch_conv1d(
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
            ) {
                return;
            }
        }
        self.fallback_conv1d(
            input,
            input_layout,
            weight,
            weight_layout,
            bias,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        );
    }

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
        panic!("CUDA conv1d_backward failed to launch or context is not initialized");
    }

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
    ) {
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let input_f32 = cast_storage::<T, f32>(input);
            let weight_f32 = cast_storage::<T, f32>(weight);
            let bias_f32 = bias.map(|b| cast_storage::<T, f32>(b));
            let mut output_f32 = cast_storage_mut::<T, f32>(output);
            let out_numel = output_layout.shape().iter().product::<usize>();
            if kernels::launch_conv2d(
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
            ) {
                return;
            }
        }
        self.fallback_conv2d(
            input,
            input_layout,
            weight,
            weight_layout,
            bias,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        );
    }

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
        panic!("CUDA conv2d_backward failed to launch or context is not initialized");
    }
}
