use coeus_core::{ComputeBackend, Layout, Scalar};
use coeus_hephaestus::{
    HephaestusBackend, HephaestusBackendError, HephaestusProvider, HephaestusStorage,
    RankTwoOperand, ReductionProvider, ScanOperation,
};
use coeus_ops::{ReductionOp, ReductionOps};
use hephaestus_core::{ComputeDevice, ScanDirection};
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

macro_rules! impl_reduction_provider {
    ($scalar:ty) => {
        impl ReductionProvider<$scalar> for RocmProvider {
            fn reduce(
                device: &Self::Device,
                op: ReductionOp,
                input: RankTwoOperand<'_, <Self::Device as ComputeDevice>::Buffer<$scalar>>,
                axis: usize,
                output: RankTwoOperand<'_, <Self::Device as ComputeDevice>::Buffer<$scalar>>,
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
                input: RankTwoOperand<'_, <Self::Device as ComputeDevice>::Buffer<$scalar>>,
                axis: usize,
                operation: ScanOperation,
                direction: ScanDirection,
                output: RankTwoOperand<'_, <Self::Device as ComputeDevice>::Buffer<$scalar>>,
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
