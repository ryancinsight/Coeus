use coeus_core::Scalar;
use coeus_hephaestus::{
    get_or_try_init, ActivationUnaryOperations, ArithmeticUnaryOperations, AttentionProvider,
    ConvolutionProvider, CrossEntropyProvider, ElementwiseProvider, HephaestusProvider,
    MatmulProvider, ParameterizedElementwiseProvider, PoolingProvider, RandomInitProvider,
    ReductionProvider, RotateHalfProvider, ScalarPowerProvider, StatefulUpdateProvider,
    UnfoldFoldProvider,
};
use hephaestus_core::{PoolingOps, SlidingWindowOps};
use hephaestus_metal::{
    MetalAttentionOps, MetalAxisReductionOps, MetalConvolutionOps, MetalCrossEntropyOps,
    MetalDenseProductOps, MetalDevice, MetalElementwiseOps, MetalParameterizedUnaryOps,
    MetalPoolingOps, MetalRandomOps, MetalScanOps, MetalSlidingWindowOps,
};
use std::sync::OnceLock;

static METAL_DEVICE: OnceLock<MetalDevice> = OnceLock::new();

/// Provider marker for the native Metal device.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalProvider;

// SAFETY: Metal buffers retain the WGPU/Metal device context and WGPU queue
// submission supplies the synchronization boundary required by the handle.
unsafe impl HephaestusProvider for MetalProvider {
    type Device = MetalDevice;
    const NAME: &'static str = "metal";

    fn device() -> &'static Self::Device {
        METAL_DEVICE
            .get_or_init(|| MetalDevice::try_default().expect("Metal device acquisition failed"))
    }

    fn try_device() -> hephaestus_core::Result<&'static Self::Device> {
        get_or_try_init(
            &METAL_DEVICE,
            "Metal device initialization did not publish the acquired device",
            MetalDevice::try_default,
        )
    }
}

impl ConvolutionProvider<f32> for MetalProvider {
    type Operations = MetalConvolutionOps;
}

impl MatmulProvider<f32> for MetalProvider {
    type Operations = MetalDenseProductOps;
}

impl AttentionProvider<f32> for MetalProvider {
    type Operations = MetalAttentionOps;
}

impl ElementwiseProvider<f32> for MetalProvider {
    type Operations = MetalElementwiseOps;
    type UnaryOperations = ActivationUnaryOperations;
}

impl ElementwiseProvider<u32> for MetalProvider {
    type Operations = MetalElementwiseOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

impl ElementwiseProvider<i32> for MetalProvider {
    type Operations = MetalElementwiseOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

impl ScalarPowerProvider<f32> for MetalProvider {
    type Operations = MetalElementwiseOps;
}

impl ReductionProvider<f32> for MetalProvider {
    type AxisOperations = MetalAxisReductionOps;
    type ScanOperations = MetalScanOps;
}

impl ReductionProvider<u32> for MetalProvider {
    type AxisOperations = MetalAxisReductionOps;
    type ScanOperations = MetalScanOps;
}

impl ReductionProvider<i32> for MetalProvider {
    type AxisOperations = MetalAxisReductionOps;
    type ScanOperations = MetalScanOps;
}

impl CrossEntropyProvider for MetalProvider {
    type Operations = MetalCrossEntropyOps;
}

impl RandomInitProvider<f32> for MetalProvider {
    type Operations = MetalRandomOps;
}

impl RotateHalfProvider<f32> for MetalProvider {
    type Operations = MetalElementwiseOps;
}

impl ParameterizedElementwiseProvider for MetalProvider {
    type Operations = MetalParameterizedUnaryOps;
}

impl StatefulUpdateProvider for MetalProvider {
    type Operations = hephaestus_metal::MetalStatefulUpdateOps;
}

impl<T> PoolingProvider<T> for MetalProvider
where
    T: Scalar,
    MetalPoolingOps: PoolingOps<MetalDevice, T>,
{
    type Operations = MetalPoolingOps;
}

impl<T> UnfoldFoldProvider<T> for MetalProvider
where
    T: Scalar,
    MetalSlidingWindowOps: SlidingWindowOps<MetalDevice, T>,
{
    type Operations = MetalSlidingWindowOps;
}
