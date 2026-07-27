use super::provider::RocmProvider;
use coeus_hephaestus::{ElementwiseProvider, RankedOperand};
use coeus_ops::{BinaryOp, UnaryOp};
use hephaestus_core::{ComputeDevice, HephaestusError};

fn unsupported_unary_operation(operation: UnaryOp) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!(
            "unary elementwise operation {operation:?} is not implemented by ROCm provider"
        ),
    }
}

macro_rules! activation_unary_dispatch {
    (activations, $operation:expr, $device:expr, $input:expr, $output:expr) => {
        match $operation {
            UnaryOp::Relu => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::ReluOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::ReluGrad => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::ReluGradOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Sigmoid => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::SigmoidOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::SigmoidGrad => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::SigmoidGradOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Tanh => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::TanhOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::TanhGrad => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::TanhGradOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::GeluTanh => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::GeluTanhOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::GeluTanhGrad => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::GeluTanhGradOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Silu => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::SiluOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::SiluGrad => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::SiluGradOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Softplus => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::SoftplusOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::SoftplusGrad => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::SoftplusGradOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Tan => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::TanOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Asin => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::AsinOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Acos => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::AcosOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Atan => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::AtanOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Sinh => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::SinhOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Cosh => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::CoshOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Log2 => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::Log2Op,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Log10 => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::Log10Op,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Exp2 => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::Exp2Op,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Atanh => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::AtanhOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Asinh => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::AsinhOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Acosh => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::AcoshOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Expm1 => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::Expm1Op,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Log1p => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::Log1pOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Sign => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::SignOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Floor => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::FloorOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Ceil => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::CeilOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Round => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::RoundOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Trunc => hephaestus_rocm::unary_elementwise_strided_into::<
            hephaestus_rocm::TruncOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            _ => Err(unsupported_unary_operation($operation)),
        }
    };
    (arithmetic_only, $operation:expr, $device:expr, $input:expr, $output:expr) => {
        Err(unsupported_unary_operation($operation))
    };
}

macro_rules! impl_elementwise_provider {
    ($scalar:ty, $activation_mode:ident) => {
        impl ElementwiseProvider<$scalar> for RocmProvider {
            fn binary<const N: usize>(
                device: &Self::Device,
                operation: BinaryOp,
                lhs: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<$scalar>, N>,
                rhs: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<$scalar>, N>,
                output: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<$scalar>, N>,
            ) -> hephaestus_core::Result<()> {
                let lhs = hephaestus_rocm::StridedOperand {
                    buffer: lhs.buffer,
                    layout: lhs.layout,
                };
                let rhs = hephaestus_rocm::StridedOperand {
                    buffer: rhs.buffer,
                    layout: rhs.layout,
                };
                let output = hephaestus_rocm::StridedOperand {
                    buffer: output.buffer,
                    layout: output.layout,
                };
                match operation {
                    BinaryOp::Add => hephaestus_rocm::binary_elementwise_strided_into::<
                        hephaestus_rocm::AddOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Sub => hephaestus_rocm::binary_elementwise_strided_into::<
                        hephaestus_rocm::SubOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Mul => hephaestus_rocm::binary_elementwise_strided_into::<
                        hephaestus_rocm::MulOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Div => hephaestus_rocm::binary_elementwise_strided_into::<
                        hephaestus_rocm::DivOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Eq => hephaestus_rocm::binary_elementwise_strided_typed_into::<
                        hephaestus_rocm::EqOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Ne => hephaestus_rocm::binary_elementwise_strided_typed_into::<
                        hephaestus_rocm::NeOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Lt => hephaestus_rocm::binary_elementwise_strided_typed_into::<
                        hephaestus_rocm::LtOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Gt => hephaestus_rocm::binary_elementwise_strided_typed_into::<
                        hephaestus_rocm::GtOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Le => hephaestus_rocm::binary_elementwise_strided_typed_into::<
                        hephaestus_rocm::LeOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Ge => hephaestus_rocm::binary_elementwise_strided_typed_into::<
                        hephaestus_rocm::GeOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                }
            }

            fn unary<const N: usize>(
                device: &Self::Device,
                operation: UnaryOp,
                input: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<$scalar>, N>,
                output: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<$scalar>, N>,
            ) -> hephaestus_core::Result<()> {
                let input = hephaestus_rocm::StridedOperand {
                    buffer: input.buffer,
                    layout: input.layout,
                };
                let output = hephaestus_rocm::StridedOperand {
                    buffer: output.buffer,
                    layout: output.layout,
                };
                match operation {
                    UnaryOp::Sin => hephaestus_rocm::unary_elementwise_strided_into::<
                        hephaestus_rocm::SinOp,
                        $scalar,
                        N,
                    >(
                        device, input, output, hephaestus_core::BlockWidth::DEFAULT
                    ),
                    UnaryOp::Cos => hephaestus_rocm::unary_elementwise_strided_into::<
                        hephaestus_rocm::CosOp,
                        $scalar,
                        N,
                    >(
                        device, input, output, hephaestus_core::BlockWidth::DEFAULT
                    ),
                    UnaryOp::Exp => hephaestus_rocm::unary_elementwise_strided_into::<
                        hephaestus_rocm::ExpOp,
                        $scalar,
                        N,
                    >(
                        device, input, output, hephaestus_core::BlockWidth::DEFAULT
                    ),
                    UnaryOp::Log => hephaestus_rocm::unary_elementwise_strided_into::<
                        hephaestus_rocm::LnOp,
                        $scalar,
                        N,
                    >(
                        device, input, output, hephaestus_core::BlockWidth::DEFAULT
                    ),
                    UnaryOp::Neg => hephaestus_rocm::unary_elementwise_strided_into::<
                        hephaestus_rocm::NegOp,
                        $scalar,
                        N,
                    >(
                        device, input, output, hephaestus_core::BlockWidth::DEFAULT
                    ),
                    UnaryOp::Abs => hephaestus_rocm::unary_elementwise_strided_into::<
                        hephaestus_rocm::AbsOp,
                        $scalar,
                        N,
                    >(
                        device, input, output, hephaestus_core::BlockWidth::DEFAULT
                    ),
                    UnaryOp::Sqrt => hephaestus_rocm::unary_elementwise_strided_into::<
                        hephaestus_rocm::SqrtOp,
                        $scalar,
                        N,
                    >(
                        device, input, output, hephaestus_core::BlockWidth::DEFAULT
                    ),
                    UnaryOp::Recip => hephaestus_rocm::unary_elementwise_strided_into::<
                        hephaestus_rocm::RecipOp,
                        $scalar,
                        N,
                    >(
                        device, input, output, hephaestus_core::BlockWidth::DEFAULT
                    ),
                    _ => activation_unary_dispatch!(
                        $activation_mode,
                        operation,
                        device,
                        input,
                        output
                    ),
                }
            }
        }
    };
}

impl_elementwise_provider!(f32, activations);
impl_elementwise_provider!(u32, arithmetic_only);
impl_elementwise_provider!(i32, arithmetic_only);

#[cfg(test)]
mod tests {
    use super::unsupported_unary_operation;
    use coeus_ops::UnaryOp;
    use hephaestus_core::HephaestusError;

    #[test]
    fn unsupported_operations_are_reported_as_typed_provider_errors() {
        assert!(matches!(
            unsupported_unary_operation(UnaryOp::Gelu),
            HephaestusError::DispatchFailed { .. }
        ));
    }
}
