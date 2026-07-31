use crate::backend::{WgpuBackendError, WgpuScalar, WgpuStorage};
use hephaestus_core::BlockWidth;
use hephaestus_wgpu::StridedOperand;
use leto::Layout;

pub(super) trait ParameterizedActivationScalar: WgpuScalar {
    fn dispatch_parameterized<const N: usize>(
        operation: coeus_ops::UnaryOp,
        input: &WgpuStorage<Self>,
        input_layout: &Layout<N>,
        output: &WgpuStorage<Self>,
        output_layout: &Layout<N>,
    ) -> Result<bool, WgpuBackendError>;
}

impl ParameterizedActivationScalar for f32 {
    fn dispatch_parameterized<const N: usize>(
        operation: coeus_ops::UnaryOp,
        input: &WgpuStorage<Self>,
        input_layout: &Layout<N>,
        output: &WgpuStorage<Self>,
        output_layout: &Layout<N>,
    ) -> Result<bool, WgpuBackendError> {
        let Some(parameters) = operation.parameter_pair() else {
            return Ok(false);
        };
        let input = StridedOperand {
            buffer: input.buffer.as_ref(),
            layout: input_layout,
        };
        let output = StridedOperand {
            buffer: output.buffer.as_ref(),
            layout: output_layout,
        };
        let device = &crate::backend::get_wgpu_context().hephaestus_device;
        let result =
            match operation {
                coeus_ops::UnaryOp::Hardtanh(_) => {
                    hephaestus_wgpu::parameterized_unary_strided_into::<
                        hephaestus_core::HardtanhOp,
                        N,
                    >(device, input, parameters, output, BlockWidth::DEFAULT)
                }
                coeus_ops::UnaryOp::HardtanhGrad(_) => {
                    hephaestus_wgpu::parameterized_unary_strided_into::<
                        hephaestus_core::HardtanhGradOp,
                        N,
                    >(device, input, parameters, output, BlockWidth::DEFAULT)
                }
                coeus_ops::UnaryOp::Threshold(_) => {
                    hephaestus_wgpu::parameterized_unary_strided_into::<
                        hephaestus_core::ThresholdOp,
                        N,
                    >(device, input, parameters, output, BlockWidth::DEFAULT)
                }
                coeus_ops::UnaryOp::ThresholdGrad(_) => {
                    hephaestus_wgpu::parameterized_unary_strided_into::<
                        hephaestus_core::ThresholdGradOp,
                        N,
                    >(device, input, parameters, output, BlockWidth::DEFAULT)
                }
                _ => return Ok(false),
            };
        result
            .map(|()| true)
            .map_err(|source| WgpuBackendError::dispatch("parameterized activation", source))
    }
}

macro_rules! impl_unsupported_parameterized_activation {
    ($($scalar:ty),+ $(,)?) => {
        $(
            impl ParameterizedActivationScalar for $scalar {
                fn dispatch_parameterized<const N: usize>(
                    _operation: coeus_ops::UnaryOp,
                    _input: &WgpuStorage<Self>,
                    _input_layout: &Layout<N>,
                    _output: &WgpuStorage<Self>,
                    _output_layout: &Layout<N>,
                ) -> Result<bool, WgpuBackendError> {
                    Ok(false)
                }
            }
        )+
    };
}

impl_unsupported_parameterized_activation!(i32, u32);
