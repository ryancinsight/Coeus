use coeus_core::{ComputeBackend, Layout, Scalar};
use coeus_hephaestus::{
    HephaestusBackend, HephaestusBackendError, HephaestusProvider, HephaestusStorage,
    RankTwoOperand, ReductionProvider, ScanOperation,
};
use coeus_ops::{ReductionOp, ReductionOps};
use hephaestus_core::{ComputeDevice, ScanDirection};
use hephaestus_metal::{MetalDevice, StridedOperand};
use std::sync::OnceLock;

/// Provider marker for the native Metal device.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalProvider;

// SAFETY: Metal buffers retain the WGPU/Metal device context and WGPU queue
// submission supplies the synchronization boundary required by the handle.
unsafe impl HephaestusProvider for MetalProvider {
    type Device = MetalDevice;
    const NAME: &'static str = "metal";

    fn device() -> &'static Self::Device {
        static DEVICE: OnceLock<MetalDevice> = OnceLock::new();
        DEVICE.get_or_init(|| MetalDevice::try_default().expect("Metal device acquisition failed"))
    }
}

macro_rules! impl_reduction_provider {
    ($scalar:ty) => {
        impl ReductionProvider<$scalar> for MetalProvider {
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

/// Coeus Metal backend with native Hephaestus storage and rank-2 reductions.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalBackend(HephaestusBackend<MetalProvider>);

impl coeus_core::backend::private::Sealed for MetalBackend {}

impl MetalBackend {
    /// Construct the Metal backend selector.
    #[must_use]
    pub const fn new() -> Self {
        Self(HephaestusBackend::new())
    }
}

impl ComputeBackend for MetalBackend {
    type Error = HephaestusBackendError;
    type DeviceBuffer<T: Scalar> = HephaestusStorage<MetalProvider, T>;
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

impl<T> ReductionOps<T> for MetalBackend
where
    T: Scalar + leto_ops::Scalar,
    MetalProvider: ReductionProvider<T>,
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
