use super::cast::{cast_storage, cast_storage_mut};
use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::kernels;
use crate::storage::CudaStorage;
use coeus_core::{ComputeBackend, Layout, Storage};

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
    ) {
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let input_f32 = cast_storage::<T, f32>(input);
            let weight_f32 = cast_storage::<T, f32>(weight);
            let bias_f32 = bias.map(|b| cast_storage::<T, f32>(b));
            let mut output_f32 = cast_storage_mut::<T, f32>(output);
            let out_numel = output_layout.shape().iter().product::<usize>();
            if kernels::launch_conv3d(
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
        self.fallback_conv3d(
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
        output_padding: usize,
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
                return;
            }
        }
        let mut host_input = vec![T::zero(); input.len()];
        self.copy_to_host(input, &mut host_input);
        let mut host_weight = vec![T::zero(); weight.len()];
        self.copy_to_host(weight, &mut host_weight);
        let host_bias = bias.map(|b| {
            let mut hb = vec![T::zero(); b.len()];
            self.copy_to_host(b, &mut hb);
            hb
        });

        let seq = coeus_core::SequentialBackend::new();
        let seq_in = coeus_core::CpuStorage::from_slice(&host_input);
        let seq_w = coeus_core::CpuStorage::from_slice(&host_weight);
        let seq_bias = host_bias.map(|hb| coeus_core::CpuStorage::from_slice(&hb));
        let mut seq_out = coeus_core::CpuStorage::from_slice(&vec![T::zero(); output.len()]);

        coeus_ops::ConvOps::conv_transpose1d(
            &seq,
            &seq_in,
            input_layout,
            &seq_w,
            weight_layout,
            seq_bias.as_ref(),
            stride,
            padding,
            output_padding,
            dilation,
            &mut seq_out,
            output_layout,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_out.as_slice(), output);
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
        output_padding: usize,
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
                return;
            }
        }
        let mut host_input = vec![T::zero(); input.len()];
        self.copy_to_host(input, &mut host_input);
        let mut host_weight = vec![T::zero(); weight.len()];
        self.copy_to_host(weight, &mut host_weight);
        let host_bias = bias.map(|b| {
            let mut hb = vec![T::zero(); b.len()];
            self.copy_to_host(b, &mut hb);
            hb
        });

        let seq = coeus_core::SequentialBackend::new();
        let seq_in = coeus_core::CpuStorage::from_slice(&host_input);
        let seq_w = coeus_core::CpuStorage::from_slice(&host_weight);
        let seq_bias = host_bias.map(|hb| coeus_core::CpuStorage::from_slice(&hb));
        let mut seq_out = coeus_core::CpuStorage::from_slice(&vec![T::zero(); output.len()]);

        coeus_ops::ConvOps::conv_transpose2d(
            &seq,
            &seq_in,
            input_layout,
            &seq_w,
            weight_layout,
            seq_bias.as_ref(),
            stride,
            padding,
            output_padding,
            dilation,
            &mut seq_out,
            output_layout,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_out.as_slice(), output);
    }
}
