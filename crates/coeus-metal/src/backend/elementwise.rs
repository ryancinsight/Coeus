use super::provider::MetalProvider;
use coeus_hephaestus::{ElementwiseProvider, RankedOperand};
use coeus_ops::{BinaryOp, UnaryOp};
use hephaestus_core::{ComputeDevice, HephaestusError};

fn unsupported_unary_operation(operation: UnaryOp) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!(
            "unary elementwise operation {operation:?} is not implemented by Metal provider"
        ),
    }
}

macro_rules! activation_unary_dispatch {
    (activations, $operation:expr, $device:expr, $input:expr, $output:expr) => {
        match $operation {
            UnaryOp::Relu => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::ReluOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::ReluGrad => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::ReluGradOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Sigmoid => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::SigmoidOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::SigmoidGrad => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::SigmoidGradOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Tanh => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::TanhOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::TanhGrad => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::TanhGradOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Gelu => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::GeluOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::GeluGrad => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::GeluGradOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::GeluTanh => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::GeluTanhOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::GeluTanhGrad => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::GeluTanhGradOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Silu => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::SiluOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::SiluGrad => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::SiluGradOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Softplus => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::SoftplusOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::SoftplusGrad => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::SoftplusGradOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Tan => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::TanOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Asin => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::AsinOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Acos => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::AcosOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Atan => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::AtanOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Sinh => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::SinhOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Cosh => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::CoshOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Log2 => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::Log2Op,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Log10 => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::Log10Op,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Exp2 => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::Exp2Op,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Atanh => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::AtanhOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Asinh => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::AsinhOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Acosh => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::AcoshOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Expm1 => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::Expm1Op,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Log1p => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::Log1pOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Sign => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::SignOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Floor => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::FloorOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Ceil => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::CeilOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Round => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::RoundOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Trunc => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::TruncOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Erf => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::ErfOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Erfc => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::ErfcOp,
            f32,
            N,
        >($device, $input, $output, hephaestus_core::BlockWidth::DEFAULT),
            UnaryOp::Lgamma => hephaestus_metal::unary_elementwise_strided_into::<
            hephaestus_metal::LgammaOp,
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
        impl ElementwiseProvider<$scalar> for MetalProvider {
            fn binary<const N: usize>(
                device: &Self::Device,
                operation: BinaryOp,
                lhs: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<$scalar>, N>,
                rhs: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<$scalar>, N>,
                output: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<$scalar>, N>,
            ) -> hephaestus_core::Result<()> {
                let lhs = hephaestus_metal::StridedOperand {
                    buffer: lhs.buffer,
                    layout: lhs.layout,
                };
                let rhs = hephaestus_metal::StridedOperand {
                    buffer: rhs.buffer,
                    layout: rhs.layout,
                };
                let output = hephaestus_metal::StridedOperand {
                    buffer: output.buffer,
                    layout: output.layout,
                };
                match operation {
                    BinaryOp::Add => hephaestus_metal::binary_elementwise_strided_into::<
                        hephaestus_metal::AddOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Sub => hephaestus_metal::binary_elementwise_strided_into::<
                        hephaestus_metal::SubOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Mul => hephaestus_metal::binary_elementwise_strided_into::<
                        hephaestus_metal::MulOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Div => hephaestus_metal::binary_elementwise_strided_into::<
                        hephaestus_metal::DivOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Eq => hephaestus_metal::binary_elementwise_strided_typed_into::<
                        hephaestus_metal::EqOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Ne => hephaestus_metal::binary_elementwise_strided_typed_into::<
                        hephaestus_metal::NeOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Lt => hephaestus_metal::binary_elementwise_strided_typed_into::<
                        hephaestus_metal::LtOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Gt => hephaestus_metal::binary_elementwise_strided_typed_into::<
                        hephaestus_metal::GtOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Le => hephaestus_metal::binary_elementwise_strided_typed_into::<
                        hephaestus_metal::LeOp,
                        $scalar,
                        N,
                    >(
                        device,
                        lhs,
                        rhs,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    BinaryOp::Ge => hephaestus_metal::binary_elementwise_strided_typed_into::<
                        hephaestus_metal::GeOp,
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
                let input = hephaestus_metal::StridedOperand {
                    buffer: input.buffer,
                    layout: input.layout,
                };
                let output = hephaestus_metal::StridedOperand {
                    buffer: output.buffer,
                    layout: output.layout,
                };
                match operation {
                    UnaryOp::Sin => hephaestus_metal::unary_elementwise_strided_into::<
                        hephaestus_metal::SinOp,
                        $scalar,
                        N,
                    >(
                        device, input, output, hephaestus_core::BlockWidth::DEFAULT
                    ),
                    UnaryOp::Cos => hephaestus_metal::unary_elementwise_strided_into::<
                        hephaestus_metal::CosOp,
                        $scalar,
                        N,
                    >(
                        device, input, output, hephaestus_core::BlockWidth::DEFAULT
                    ),
                    UnaryOp::Exp => hephaestus_metal::unary_elementwise_strided_into::<
                        hephaestus_metal::ExpOp,
                        $scalar,
                        N,
                    >(
                        device, input, output, hephaestus_core::BlockWidth::DEFAULT
                    ),
                    UnaryOp::Log => hephaestus_metal::unary_elementwise_strided_into::<
                        hephaestus_metal::LnOp,
                        $scalar,
                        N,
                    >(
                        device, input, output, hephaestus_core::BlockWidth::DEFAULT
                    ),
                    UnaryOp::Neg => hephaestus_metal::unary_elementwise_strided_into::<
                        hephaestus_metal::NegOp,
                        $scalar,
                        N,
                    >(
                        device, input, output, hephaestus_core::BlockWidth::DEFAULT
                    ),
                    UnaryOp::Abs => hephaestus_metal::unary_elementwise_strided_into::<
                        hephaestus_metal::AbsOp,
                        $scalar,
                        N,
                    >(
                        device, input, output, hephaestus_core::BlockWidth::DEFAULT
                    ),
                    UnaryOp::Sqrt => hephaestus_metal::unary_elementwise_strided_into::<
                        hephaestus_metal::SqrtOp,
                        $scalar,
                        N,
                    >(
                        device, input, output, hephaestus_core::BlockWidth::DEFAULT
                    ),
                    UnaryOp::Recip => hephaestus_metal::unary_elementwise_strided_into::<
                        hephaestus_metal::RecipOp,
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
