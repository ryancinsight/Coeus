use coeus_core::{ComputeBackend, Layout, Scalar};
use coeus_hephaestus::{
    ElementwiseProvider, HephaestusBackend, HephaestusBackendError, HephaestusProvider,
    HephaestusStorage, RankedOperand, ReductionProvider, ScanOperation,
};
use coeus_ops::{BinaryOp, ElementwiseOps, ReductionOp, ReductionOps, UnaryOp};
use hephaestus_core::{ComputeDevice, HephaestusError, ScanDirection};
use hephaestus_rocm::{RocmDevice, StridedOperand};
use std::sync::OnceLock;

/// Provider marker for the native ROCm device.
#[derive(Debug, Clone, Copy, Default)]
pub struct RocmProvider;

// SAFETY: ROCm buffers retain their owning context and HIP launches bind that
// context before accessing the allocation; the handle is thread-transferable.
unsafe impl HephaestusProvider for RocmProvider {
    type Device = RocmDevice;
    const NAME: &'static str = "rocm";

    fn device() -> &'static Self::Device {
        static DEVICE: OnceLock<RocmDevice> = OnceLock::new();
        DEVICE.get_or_init(|| RocmDevice::try_default().expect("ROCm device acquisition failed"))
    }
}

fn unsupported_unary_operation(operation: UnaryOp) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!(
            "unary elementwise operation {operation:?} is not implemented by ROCm provider"
        ),
    }
}

macro_rules! impl_reduction_provider {
    ($scalar:ty) => {
        impl ReductionProvider<$scalar> for RocmProvider {
            fn reduce(
                device: &Self::Device,
                op: ReductionOp,
                input: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<$scalar>, 2>,
                axis: usize,
                output: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<$scalar>, 2>,
            ) -> hephaestus_core::Result<()> {
                let input = StridedOperand {
                    buffer: input.buffer,
                    layout: input.layout,
                };
                let output = StridedOperand {
                    buffer: output.buffer,
                    layout: output.layout,
                };
                match op {
                    ReductionOp::Sum => hephaestus_rocm::sum_axis_into::<$scalar>(
                        device,
                        input,
                        axis,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    ReductionOp::Prod => hephaestus_rocm::prod_axis_into::<$scalar>(
                        device,
                        input,
                        axis,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    ReductionOp::Mean => hephaestus_rocm::mean_axis_into::<$scalar>(
                        device,
                        input,
                        axis,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    ReductionOp::Max => hephaestus_rocm::max_axis_into::<$scalar>(
                        device,
                        input,
                        axis,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    ReductionOp::Min => hephaestus_rocm::min_axis_into::<$scalar>(
                        device,
                        input,
                        axis,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                }
            }

            fn scan(
                device: &Self::Device,
                input: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<$scalar>, 2>,
                axis: usize,
                operation: ScanOperation,
                direction: ScanDirection,
                output: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<$scalar>, 2>,
            ) -> hephaestus_core::Result<()> {
                let input = StridedOperand {
                    buffer: input.buffer,
                    layout: input.layout,
                };
                let output = StridedOperand {
                    buffer: output.buffer,
                    layout: output.layout,
                };
                match operation {
                    ScanOperation::Sum => {
                        hephaestus_rocm::scan_axis_into::<hephaestus_rocm::CumSumOp, $scalar>(
                            device,
                            input,
                            axis,
                            direction,
                            output,
                            hephaestus_core::BlockWidth::DEFAULT,
                        )
                    }
                    ScanOperation::Product => {
                        hephaestus_rocm::scan_axis_into::<hephaestus_rocm::CumProdOp, $scalar>(
                            device,
                            input,
                            axis,
                            direction,
                            output,
                            hephaestus_core::BlockWidth::DEFAULT,
                        )
                    }
                }
            }
        }
    };
}

impl_reduction_provider!(f32);
impl_reduction_provider!(u32);
impl_reduction_provider!(i32);

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

/// Coeus ROCm backend with native Hephaestus storage and rank-2 reductions.
#[derive(Debug, Clone, Copy, Default)]
pub struct RocmBackend(HephaestusBackend<RocmProvider>);

impl coeus_core::backend::private::Sealed for RocmBackend {}

impl RocmBackend {
    /// Construct the ROCm backend selector.
    #[must_use]
    pub const fn new() -> Self {
        Self(HephaestusBackend::new())
    }
}

impl ComputeBackend for RocmBackend {
    type Error = HephaestusBackendError;
    type DeviceBuffer<T: Scalar> = HephaestusStorage<RocmProvider, T>;
    type KernelDescriptor = ();
    type DispatchFuture<T: Scalar> = std::future::Ready<T>;

    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn num_threads(&self) -> usize {
        self.0.num_threads()
    }

    fn allocate<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        self.0.allocate(len)
    }

    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) {
        self.0.fill(dst, val)
    }

    fn copy_to_device<T: Scalar>(&self, src: &[T], dst: &mut Self::DeviceBuffer<T>) {
        self.0.copy_to_device(src, dst)
    }

    fn copy_to_host<T: Scalar>(&self, src: &Self::DeviceBuffer<T>, dst: &mut [T]) {
        self.0.copy_to_host(src, dst)
    }
}

impl<T> ReductionOps<T> for RocmBackend
where
    T: Scalar + leto_ops::Scalar,
    RocmProvider: ReductionProvider<T>,
{
    fn reduce(
        &self,
        op: ReductionOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.0.reduce(op, a, a_layout, axis, c, c_layout)
    }

    fn cumsum(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.0.cumsum(a, a_layout, axis, c, c_layout)
    }

    fn suffix_sum(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.0.suffix_sum(a, a_layout, axis, c, c_layout)
    }

    fn cumprod(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.0.cumprod(a, a_layout, axis, c, c_layout)
    }

    fn suffix_prod(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.0.suffix_prod(a, a_layout, axis, c, c_layout)
    }
}

impl<T> ElementwiseOps<T> for RocmBackend
where
    T: Scalar + leto_ops::Scalar,
    RocmProvider: ElementwiseProvider<T>,
{
    fn elementwise_binary(
        &self,
        operation: BinaryOp,
        lhs: &Self::DeviceBuffer<T>,
        lhs_layout: &Layout,
        rhs: &Self::DeviceBuffer<T>,
        rhs_layout: &Layout,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.0.elementwise_binary(
            operation,
            lhs,
            lhs_layout,
            rhs,
            rhs_layout,
            output,
            output_layout,
        )
    }

    fn elementwise_unary(
        &self,
        operation: UnaryOp,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.0
            .elementwise_unary(operation, input, input_layout, output, output_layout)
    }
}

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
