//! WGPU provider wiring for the generic Hephaestus backend.
//!
//! [`WgpuProvider`] names the WGPU device and its operation bundles; every
//! Coeus operator resolves through the generic
//! [`HephaestusBackend<WgpuProvider>`](coeus_hephaestus::HephaestusBackend)
//! implementations in `coeus-hephaestus`. Scalar coverage per operator mirrors
//! the historical `WgpuBackend` bounds exactly: the WGSL dialect predicates
//! and per-operation bundle bounds are the same ones the removed per-vendor
//! shims carried.

use std::sync::OnceLock;

use coeus_core::Scalar;
use coeus_hephaestus::{
    ActivationUnaryOperations, ArithmeticUnaryOperations, AttentionProvider, ConvolutionProvider,
    CrossEntropyProvider, CrossProductProvider, ElementwiseProvider, FusionProvider,
    HephaestusProvider, MatmulProvider, ParameterizedElementwiseProvider, PoolingProvider,
    RandomInitProvider, ReductionProvider, RotateHalfProvider, ScalarPowerProvider,
    StaggeredProvider, StatefulUpdateProvider, UnfoldFoldProvider,
};
use hephaestus_core::{ConvolutionOps, CrossProductOps, DenseProductOps, DialectScalar};
use hephaestus_wgpu::{
    WgpuAttentionOps, WgpuAxisReductionOps, WgpuConvolutionOps, WgpuCrossEntropyOps,
    WgpuCrossProductOps, WgpuDenseProductOps, WgpuDevice, WgpuElementwiseOps, WgpuFusionOps,
    WgpuFusionScalar, WgpuParameterizedUnaryOps, WgpuPoolingOps, WgpuRandomOps, WgpuScanOps,
    WgpuSlidingWindowOps, WgpuStaggered3DOps, WgpuStatefulUpdateOps, WgpuWindowScalar, Wgsl,
};

/// Process-global WGPU execution context shared by every provider dispatch.
pub(crate) struct WgpuContext {
    pub(crate) hephaestus_device: WgpuDevice,
}

static WGPU_CONTEXT: OnceLock<WgpuContext> = OnceLock::new();

/// Retrieve a reference to the global lazily-initialized wgpu context.
pub(crate) fn get_wgpu_context() -> &'static WgpuContext {
    try_get_wgpu_context().expect("Failed to initialize hephaestus-wgpu device")
}

/// Try to retrieve the process-global WGPU context.
///
/// # Errors
///
/// Returns the typed Hephaestus acquisition failure when WGPU is unavailable.
pub(crate) fn try_get_wgpu_context() -> hephaestus_core::Result<&'static WgpuContext> {
    if let Some(context) = WGPU_CONTEXT.get() {
        return Ok(context);
    }
    // No backend is forced here. Hephaestus selects the compiled backend set
    // unless the process explicitly sets `WGPU_BACKEND`; a request for an
    // unavailable backend therefore fails through the provider's typed
    // acquisition error. Backend selection belongs to the provider and to
    // whoever runs the process, not to this library.
    let candidate = WgpuContext {
        // Fused expressions bind the tensor inputs, output, and layout table.
        // The provider's downlevel baseline exposes only four storage slots,
        // while Coeus' public fusion contract permits three tensor inputs.
        hephaestus_device:
            hephaestus_wgpu::WgpuDevice::try_with_device_preference_and_optional_device_features_and_limits(
            "coeus-wgpu-device",
            hephaestus_core::DevicePreference::HighPerformance,
            &[],
            hephaestus_wgpu::WgpuDevice::default_device_limits(),
        )?,
    };
    let _ = WGPU_CONTEXT.set(candidate);
    WGPU_CONTEXT
        .get()
        .ok_or_else(|| hephaestus_core::HephaestusError::DeviceUnavailable {
            message: "WGPU context initialization did not publish the acquired device".to_owned(),
        })
}

/// Hephaestus WGPU provider marker for [`coeus_hephaestus::HephaestusBackend`].
///
/// ```
/// use coeus_hephaestus::HephaestusBackend;
/// use coeus_wgpu::WgpuProvider;
///
/// let backend = HephaestusBackend::<WgpuProvider>::new();
/// assert_eq!(coeus_core::ComputeBackend::name(&backend), "wgpu");
/// ```
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WgpuProvider;

// SAFETY: `HephaestusStorage` retains the provider buffer behind `Arc`, and the
// process-global device owns queue synchronization for every submitted kernel.
unsafe impl HephaestusProvider for WgpuProvider {
    type Device = WgpuDevice;

    const NAME: &'static str = "wgpu";

    fn device() -> &'static Self::Device {
        &get_wgpu_context().hephaestus_device
    }

    fn try_device() -> hephaestus_core::Result<&'static Self::Device> {
        try_get_wgpu_context().map(|context| &context.hephaestus_device)
    }
}

impl ParameterizedElementwiseProvider for WgpuProvider {
    type Operations = WgpuParameterizedUnaryOps;
}

impl ElementwiseProvider<f32> for WgpuProvider {
    type Operations = WgpuElementwiseOps;
    type UnaryOperations = ActivationUnaryOperations;
}

impl ElementwiseProvider<i32> for WgpuProvider {
    type Operations = WgpuElementwiseOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

impl ElementwiseProvider<u32> for WgpuProvider {
    type Operations = WgpuElementwiseOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

impl ScalarPowerProvider<f32> for WgpuProvider {
    type Operations = WgpuElementwiseOps;
}

impl ReductionProvider<f32> for WgpuProvider {
    type AxisOperations = WgpuAxisReductionOps;
    type ScanOperations = WgpuScanOps;
}

impl ReductionProvider<i32> for WgpuProvider {
    type AxisOperations = WgpuAxisReductionOps;
    type ScanOperations = WgpuScanOps;
}

impl ReductionProvider<u32> for WgpuProvider {
    type AxisOperations = WgpuAxisReductionOps;
    type ScanOperations = WgpuScanOps;
}

impl<T> MatmulProvider<T> for WgpuProvider
where
    T: Scalar + leto_ops::Scalar + DialectScalar<Wgsl>,
    WgpuDenseProductOps: DenseProductOps<WgpuDevice, T>,
{
    type Operations = WgpuDenseProductOps;
}

impl AttentionProvider<f32> for WgpuProvider {
    type Operations = WgpuAttentionOps;
}

impl<T> ConvolutionProvider<T> for WgpuProvider
where
    T: Scalar + leto_ops::Scalar + DialectScalar<Wgsl>,
    WgpuConvolutionOps: ConvolutionOps<WgpuDevice, T>,
{
    type Operations = WgpuConvolutionOps;
}

impl<T> PoolingProvider<T> for WgpuProvider
where
    T: Scalar + leto_ops::Scalar + WgpuWindowScalar,
{
    type Operations = WgpuPoolingOps;
}

impl RandomInitProvider<f32> for WgpuProvider {
    type Operations = WgpuRandomOps;
}

impl RotateHalfProvider<f32> for WgpuProvider {
    type Operations = WgpuElementwiseOps;
}

impl StatefulUpdateProvider for WgpuProvider {
    type Operations = WgpuStatefulUpdateOps;
}

impl CrossEntropyProvider for WgpuProvider {
    type Operations = WgpuCrossEntropyOps;
}

impl<T> UnfoldFoldProvider<T> for WgpuProvider
where
    T: Scalar + leto_ops::Scalar + WgpuWindowScalar,
{
    type Operations = WgpuSlidingWindowOps;
}

impl StaggeredProvider for WgpuProvider {
    type Operations = WgpuStaggered3DOps;
}

impl<T> CrossProductProvider<T> for WgpuProvider
where
    T: Scalar + DialectScalar<Wgsl>,
    WgpuCrossProductOps: CrossProductOps<WgpuDevice, T>,
{
    type Operations = WgpuCrossProductOps;
}

impl<T> FusionProvider<T> for WgpuProvider
where
    T: Scalar + leto_ops::Scalar + WgpuFusionScalar,
{
    type Dialect = Wgsl;
    type Operations = WgpuFusionOps;
}
