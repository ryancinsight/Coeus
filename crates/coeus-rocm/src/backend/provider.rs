use coeus_hephaestus::{
    ActivationUnaryOperations, ArithmeticUnaryOperations, CrossEntropyProvider,
    ElementwiseProvider, HephaestusProvider, MatmulProvider, ParameterizedElementwiseProvider,
    RandomInitProvider, ReductionProvider, RotateHalfProvider, ScalarPowerProvider,
    StatefulUpdateProvider,
};
#[cfg(all(feature = "rocm", target_os = "linux"))]
use coeus_hephaestus::{AttentionProvider, ConvolutionProvider};
use hephaestus_rocm::RocmDevice;
#[cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_rocm::{RocmAttentionOps, RocmConvolutionOps};
use hephaestus_rocm::{RocmAxisReductionOps, RocmDenseProductOps, RocmElementwiseOps, RocmScanOps};
use std::sync::OnceLock;

static ROCM_DEVICE: OnceLock<RocmDevice> = OnceLock::new();

/// Provider marker for the native ROCm device.
#[derive(Debug, Clone, Copy, Default)]
pub struct RocmProvider;

// SAFETY: ROCm buffers retain their owning context and HIP launches bind that
// context before accessing the allocation; the handle is thread-transferable.
unsafe impl HephaestusProvider for RocmProvider {
    type Device = RocmDevice;
    const NAME: &'static str = "rocm";

    fn device() -> &'static Self::Device {
        ROCM_DEVICE
            .get_or_init(|| RocmDevice::try_default().expect("ROCm device acquisition failed"))
    }

    fn try_device() -> hephaestus_core::Result<&'static Self::Device> {
        if let Some(device) = ROCM_DEVICE.get() {
            return Ok(device);
        }
        let candidate = RocmDevice::try_default()?;
        let _ = ROCM_DEVICE.set(candidate);
        ROCM_DEVICE
            .get()
            .ok_or_else(|| hephaestus_core::HephaestusError::DeviceUnavailable {
                message: "ROCm device initialization did not publish the acquired device"
                    .to_owned(),
            })
    }
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
impl ConvolutionProvider<f32> for RocmProvider {
    type Operations = RocmConvolutionOps;
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
impl AttentionProvider<f32> for RocmProvider {
    type Operations = RocmAttentionOps;
}

impl ElementwiseProvider<f32> for RocmProvider {
    type Operations = RocmElementwiseOps;
    type UnaryOperations = ActivationUnaryOperations;
}

impl ElementwiseProvider<u32> for RocmProvider {
    type Operations = RocmElementwiseOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

impl ElementwiseProvider<i32> for RocmProvider {
    type Operations = RocmElementwiseOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

impl ScalarPowerProvider<f32> for RocmProvider {
    type Operations = RocmElementwiseOps;
}

impl ReductionProvider<f32> for RocmProvider {
    type AxisOperations = RocmAxisReductionOps;
    type ScanOperations = RocmScanOps;
}

impl MatmulProvider<f32> for RocmProvider {
    type Operations = RocmDenseProductOps;
}

impl ReductionProvider<u32> for RocmProvider {
    type AxisOperations = RocmAxisReductionOps;
    type ScanOperations = RocmScanOps;
}

impl ReductionProvider<i32> for RocmProvider {
    type AxisOperations = RocmAxisReductionOps;
    type ScanOperations = RocmScanOps;
}

impl CrossEntropyProvider for RocmProvider {
    type Operations = hephaestus_rocm::RocmCrossEntropyOps;
}

impl RandomInitProvider<f32> for RocmProvider {
    type Operations = hephaestus_rocm::RocmRandomOps;
}

impl RotateHalfProvider<f32> for RocmProvider {
    type Operations = RocmElementwiseOps;
}

impl ParameterizedElementwiseProvider for RocmProvider {
    type Operations = hephaestus_rocm::RocmParameterizedUnaryOps;
}

impl StatefulUpdateProvider for RocmProvider {
    type Operations = hephaestus_rocm::RocmStatefulUpdateOps;
}
