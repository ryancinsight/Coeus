use super::provider::MetalProvider;
use coeus_hephaestus::{RankedOperand, ReductionProvider, ScanOperation};
use coeus_ops::ReductionOp;
use hephaestus_core::{ComputeDevice, ScanDirection};
use hephaestus_metal::StridedOperand;

macro_rules! impl_reduction_provider {
    ($scalar:ty) => {
        impl ReductionProvider<$scalar> for MetalProvider {
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
                    ReductionOp::Sum => hephaestus_metal::sum_axis_into::<$scalar>(
                        device,
                        input,
                        axis,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    ReductionOp::Prod => hephaestus_metal::prod_axis_into::<$scalar>(
                        device,
                        input,
                        axis,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    ReductionOp::Mean => hephaestus_metal::mean_axis_into::<$scalar>(
                        device,
                        input,
                        axis,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    ReductionOp::Max => hephaestus_metal::max_axis_into::<$scalar>(
                        device,
                        input,
                        axis,
                        output,
                        hephaestus_core::BlockWidth::DEFAULT,
                    ),
                    ReductionOp::Min => hephaestus_metal::min_axis_into::<$scalar>(
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
                        hephaestus_metal::scan_axis_into::<hephaestus_metal::CumSumOp, $scalar>(
                            device,
                            input,
                            axis,
                            direction,
                            output,
                            hephaestus_core::BlockWidth::DEFAULT,
                        )
                    }
                    ScanOperation::Product => {
                        hephaestus_metal::scan_axis_into::<hephaestus_metal::CumProdOp, $scalar>(
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
