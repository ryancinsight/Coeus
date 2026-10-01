use coeus_core::Scalar;
use coeus_hephaestus::{
    ActivationUnaryOperations, ArithmeticUnaryOperations, CrossEntropyProvider,
    ElementwiseProvider, HephaestusProvider, MatmulProvider, ParameterizedElementwiseProvider,
    PoolingProvider, RandomInitProvider, ReductionProvider, RotateHalfProvider,
    ScalarPowerProvider, StatefulUpdateProvider, UnfoldFoldProvider,
};
#[cfg(all(feature = "rocm", target_os = "linux"))]
use coeus_hephaestus::{AttentionProvider, ConvolutionProvider};
use hephaestus_core::{PoolingOps, SlidingWindowOps};
use hephaestus_rocm::RocmDevice;
#[cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_rocm::{RocmAttentionOps, RocmConvolutionOps};
use hephaestus_rocm::{
    RocmAxisReductionOps, RocmDenseProductOps, RocmElementwiseOps, RocmPoolingOps, RocmScanOps,
    RocmSlidingWindowOps,
};
use std::sync::OnceLock;

static ROCM_DEVICE: OnceLock<RocmDevice> = OnceLock::new();

/// Provider marker for the native ROCm device.
#[derive(Debug, Clone, Copy, Default)]
pub struct RocmProvider;

// SAFETY: ROCm buffers retain their owning context and HIP launches bind that
// context before accessing the allocation; the handle is thread-transferable.
unsafe impl HephaestusProvider for RocmProvider {
    type Device = RocmDevice;
    type Error = coeus_hephaestus::HephaestusBackendError;
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
// SAFETY: `RocmAttentionOps` initializes both forward outputs and only
// accumulates backward gradients into initialized destinations.
unsafe impl AttentionProvider<f32> for RocmProvider {
    type Operations = RocmAttentionOps;
}

// SAFETY: Provider kernels overwrite every logical output on success without reading prior contents.
unsafe impl ElementwiseProvider<f32> for RocmProvider {
    type Operations = RocmElementwiseOps;
    type UnaryOperations = ActivationUnaryOperations;
}

// SAFETY: Provider kernels overwrite every logical output on success without reading prior contents.
unsafe impl ElementwiseProvider<u32> for RocmProvider {
    type Operations = RocmElementwiseOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

// SAFETY: Provider kernels overwrite every logical output on success without reading prior contents.
unsafe impl ElementwiseProvider<i32> for RocmProvider {
    type Operations = RocmElementwiseOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

// SAFETY: Provider kernels overwrite every logical output on success without reading prior contents.
unsafe impl ScalarPowerProvider<f32> for RocmProvider {
    type Operations = RocmElementwiseOps;
}

// SAFETY: Provider kernels overwrite every logical output on success without reading prior contents.
unsafe impl ReductionProvider<f32> for RocmProvider {
    type AxisOperations = RocmAxisReductionOps;
    type ScanOperations = RocmScanOps;
}

// SAFETY: `RocmDenseProductOps` fully initializes each product output before
// returning success.
unsafe impl MatmulProvider<f32> for RocmProvider {
    type Operations = RocmDenseProductOps;
}

// SAFETY: Provider kernels overwrite every logical output on success without reading prior contents.
unsafe impl ReductionProvider<u32> for RocmProvider {
    type AxisOperations = RocmAxisReductionOps;
    type ScanOperations = RocmScanOps;
}

// SAFETY: Provider kernels overwrite every logical output on success without reading prior contents.
unsafe impl ReductionProvider<i32> for RocmProvider {
    type AxisOperations = RocmAxisReductionOps;
    type ScanOperations = RocmScanOps;
}

// SAFETY: `RocmCrossEntropyOps` initializes both forward outputs and only
// accumulates backward gradients into initialized destinations.
unsafe impl CrossEntropyProvider for RocmProvider {
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

// SAFETY: `RocmPoolingOps` initializes forward outputs and only accumulates
// backward gradients into initialized destinations.
unsafe impl<T> PoolingProvider<T> for RocmProvider
where
    T: Scalar + leto_ops::Scalar,
    RocmPoolingOps: PoolingOps<RocmDevice, T>,
{
    type Operations = RocmPoolingOps;
}

// SAFETY: `RocmSlidingWindowOps` initializes unfold outputs and clears fold
// outputs before accumulating, as required by the Hephaestus contract.
unsafe impl<T> UnfoldFoldProvider<T> for RocmProvider
where
    T: Scalar + leto_ops::Scalar,
    RocmSlidingWindowOps: SlidingWindowOps<RocmDevice, T>,
{
    type Operations = RocmSlidingWindowOps;
}
