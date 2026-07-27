use super::provider::RocmProvider;
use coeus_hephaestus::{RankedOperand, ReductionProvider, ScanOperation};
use coeus_ops::ReductionOp;
use hephaestus_core::{ComputeDevice, ScanDirection};
use hephaestus_rocm::StridedOperand;

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
