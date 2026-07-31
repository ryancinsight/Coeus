use crate::backend::CudaScalar;
use crate::storage::CudaStorage;
use crate::CudaBackendError;
use coeus_core::Layout;
use leto::Layout as LetoLayout;
use std::sync::Arc;

pub(super) fn parameterized_activation_operation(
    operation: coeus_ops::UnaryOp,
) -> Option<&'static str> {
    match operation {
        coeus_ops::UnaryOp::Hardtanh(_) => Some("Hardtanh"),
        coeus_ops::UnaryOp::HardtanhGrad(_) => Some("HardtanhGrad"),
        coeus_ops::UnaryOp::Threshold(_) => Some("Threshold"),
        coeus_ops::UnaryOp::ThresholdGrad(_) => Some("ThresholdGrad"),
        _ => None,
    }
}

macro_rules! coeus_to_leto_layout {
    ($layout:expr, $n:expr) => {{
        let rank = $layout.ndim();
        let pad = $n - rank.min($n);
        let shape: [usize; $n] = {
            let source = $layout.shape();
            let mut target = [1usize; $n];
            for index in 0..rank.min($n) {
                target[pad + index] = source[index];
            }
            target
        };
        let strides: [isize; $n] = {
            let source = $layout.strides();
            let mut target = [0isize; $n];
            for index in 0..rank.min($n) {
                target[pad + index] = source[index] as isize;
            }
            target
        };
        LetoLayout::new(shape, strides, $layout.offset())
    }};
}

pub(crate) trait ParameterizedActivationScalar: CudaScalar {
    fn dispatch_parameterized<const N: usize>(
        operation: coeus_ops::UnaryOp,
        input: &CudaStorage<Self>,
        input_layout: &LetoLayout<N>,
        output: &CudaStorage<Self>,
        output_layout: &LetoLayout<N>,
    ) -> Result<bool, CudaBackendError>;
}

impl ParameterizedActivationScalar for f32 {
    fn dispatch_parameterized<const N: usize>(
        operation: coeus_ops::UnaryOp,
        input: &CudaStorage<Self>,
        input_layout: &LetoLayout<N>,
        output: &CudaStorage<Self>,
        output_layout: &LetoLayout<N>,
    ) -> Result<bool, CudaBackendError> {
        let Some(parameters) = operation.parameter_pair() else {
            return Ok(false);
        };
        let input = hephaestus_cuda::StridedOperand {
            buffer: input.buffer.as_ref(),
            layout: input_layout,
        };
        let output = hephaestus_cuda::StridedOperand {
            buffer: output.buffer.as_ref(),
            layout: output_layout,
        };
        let device = crate::backend::get_cuda_device();
        let result = match operation {
            coeus_ops::UnaryOp::Hardtanh(_) => {
                hephaestus_cuda::parameterized_unary_strided_into::<hephaestus_core::HardtanhOp, N>(
                    device,
                    input,
                    parameters,
                    output,
                    hephaestus_core::BlockWidth::DEFAULT,
                )
            }
            coeus_ops::UnaryOp::HardtanhGrad(_) => {
                hephaestus_cuda::parameterized_unary_strided_into::<
                    hephaestus_core::HardtanhGradOp,
                    N,
                >(
                    device,
                    input,
                    parameters,
                    output,
                    hephaestus_core::BlockWidth::DEFAULT,
                )
            }
            coeus_ops::UnaryOp::Threshold(_) => {
                hephaestus_cuda::parameterized_unary_strided_into::<hephaestus_core::ThresholdOp, N>(
                    device,
                    input,
                    parameters,
                    output,
                    hephaestus_core::BlockWidth::DEFAULT,
                )
            }
            coeus_ops::UnaryOp::ThresholdGrad(_) => {
                hephaestus_cuda::parameterized_unary_strided_into::<
                    hephaestus_core::ThresholdGradOp,
                    N,
                >(
                    device,
                    input,
                    parameters,
                    output,
                    hephaestus_core::BlockWidth::DEFAULT,
                )
            }
            _ => return Ok(false),
        };
        result
            .map(|()| true)
            .map_err(|source| CudaBackendError::dispatch("parameterized activation", source))
    }
}

macro_rules! impl_unsupported_parameterized_activation {
    ($($scalar:ty),+ $(,)?) => {
        $(
            impl ParameterizedActivationScalar for $scalar {
                fn dispatch_parameterized<const N: usize>(
                    _operation: coeus_ops::UnaryOp,
                    _input: &CudaStorage<Self>,
                    _input_layout: &LetoLayout<N>,
                    _output: &CudaStorage<Self>,
                    _output_layout: &LetoLayout<N>,
                ) -> Result<bool, CudaBackendError> {
                    Ok(false)
                }
            }
        )+
    };
}

impl_unsupported_parameterized_activation!(f64, eunomia::F16, eunomia::Bf16, i32);

pub(super) fn try_hephaestus_parameterized_unary<T: ParameterizedActivationScalar>(
    operation: coeus_ops::UnaryOp,
    input: &CudaStorage<T>,
    input_layout: &Layout,
    output: &CudaStorage<T>,
    output_layout: &Layout,
) -> Result<bool, CudaBackendError> {
    if Arc::ptr_eq(&input.buffer, &output.buffer)
        || !super::can_route_dynamic_strided(&[input_layout], output_layout)
    {
        return Ok(false);
    }
    macro_rules! dispatch_rank {
        ($rank:expr) => {{
            let input_layout = coeus_to_leto_layout!(input_layout, $rank);
            let output_layout = coeus_to_leto_layout!(output_layout, $rank);
            T::dispatch_parameterized(operation, input, &input_layout, output, &output_layout)
        }};
    }
    match input_layout.ndim().max(output_layout.ndim()) {
        1 => dispatch_rank!(1),
        2 => dispatch_rank!(2),
        3 => dispatch_rank!(3),
        4 => dispatch_rank!(4),
        _ => Ok(false),
    }
}
