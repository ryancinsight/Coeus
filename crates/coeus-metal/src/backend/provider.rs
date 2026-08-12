use coeus_hephaestus::{
    ActivationUnaryOperations, ArithmeticUnaryOperations, AttentionProvider, ConvolutionProvider,
    CrossEntropyProvider, ElementwiseProvider, HephaestusProvider,
    ParameterizedElementwiseProvider, RandomInitProvider, ReductionProvider, RotateHalfProvider,
    ScalarPowerProvider, StatefulUpdateProvider,
};
use hephaestus_metal::{
    MetalAttentionOps, MetalAxisReductionOps, MetalConvolutionOps, MetalCrossEntropyOps,
    MetalDevice, MetalElementwiseOps, MetalParameterizedUnaryOps, MetalRandomOps, MetalScanOps,
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
        if let Some(device) = METAL_DEVICE.get() {
            return Ok(device);
        }
        let candidate = MetalDevice::try_default()?;
        let _ = METAL_DEVICE.set(candidate);
        METAL_DEVICE
            .get()
            .ok_or_else(|| hephaestus_core::HephaestusError::DeviceUnavailable {
                message: "Metal device initialization did not publish the acquired device"
                    .to_owned(),
            })
    }
}

impl ConvolutionProvider<f32> for MetalProvider {
    type Operations = MetalConvolutionOps;
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
