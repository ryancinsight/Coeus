use coeus_core::Scalar;
use coeus_hephaestus::{
    get_or_try_init, ActivationUnaryOperations, ArithmeticUnaryOperations, CrossEntropyProvider,
    CtcProvider, ElementwiseProvider, FixedFdProvider, HephaestusProvider, MatmulProvider,
    ParameterizedElementwiseProvider, PoolingProvider, RandomInitProvider, ReductionProvider,
    RotateHalfProvider, ScalarPowerProvider, StaggeredProvider, StatefulUpdateProvider,
    UnfoldFoldProvider,
};
#[cfg(all(feature = "rocm", target_os = "linux"))]
use coeus_hephaestus::{AttentionProvider, ConvolutionProvider};
use hephaestus_core::{CtcOps, FixedFd3DOps, PoolingOps, SlidingWindowOps};
use hephaestus_rocm::RocmDevice;
use hephaestus_rocm::{
    CtcKernel, FixedFd3DKernel, RocmAxisReductionOps, RocmCtcOps, RocmDenseProductOps,
    RocmElementwiseOps, RocmFixedFd3DOps, RocmPoolingOps, RocmScanOps, RocmSlidingWindowOps,
    RocmStaggered3DOps,
};
#[cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_rocm::{RocmAttentionOps, RocmConvolutionOps};
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
        get_or_try_init(
            &ROCM_DEVICE,
            "ROCm device initialization did not publish the acquired device",
            RocmDevice::try_default,
        )
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

impl StaggeredProvider for RocmProvider {
    type Operations = RocmStaggered3DOps;
}

/// One compiled fixed-scheme sweep kernel for the process-wide ROCm device.
static FIXED_FD_KERNEL: OnceLock<FixedFd3DKernel> = OnceLock::new();

impl FixedFdProvider for RocmProvider {
    type Operations = RocmFixedFd3DOps;

    fn fixed_fd_kernel(
    ) -> hephaestus_core::Result<&'static <Self::Operations as FixedFd3DOps<Self::Device>>::FixedFd3D>
    {
        get_or_try_init(
            &FIXED_FD_KERNEL,
            "fixed-fd kernel initialization did not publish the compiled kernel",
            || RocmFixedFd3DOps.prepare_fixed_fd_3d(Self::device()),
        )
    }
}

/// One compiled CTC kernel set for the process-wide ROCm device.
static CTC_KERNEL: OnceLock<CtcKernel> = OnceLock::new();

impl CtcProvider for RocmProvider {
    type Operations = RocmCtcOps;

    fn ctc_kernel(
    ) -> hephaestus_core::Result<&'static <Self::Operations as CtcOps<Self::Device>>::Ctc> {
        get_or_try_init(
            &CTC_KERNEL,
            "ctc kernel initialization did not publish the compiled kernel",
            || RocmCtcOps.prepare_ctc(Self::device()),
        )
    }
}

impl<T> PoolingProvider<T> for RocmProvider
where
    T: Scalar,
    RocmPoolingOps: PoolingOps<RocmDevice, T>,
{
    type Operations = RocmPoolingOps;
}

impl<T> UnfoldFoldProvider<T> for RocmProvider
where
    T: Scalar,
    RocmSlidingWindowOps: SlidingWindowOps<RocmDevice, T>,
{
    type Operations = RocmSlidingWindowOps;
}
