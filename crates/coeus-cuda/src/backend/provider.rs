//! CUDA provider wiring for the generic Hephaestus backend.
//!
//! [`CudaProvider`] names the CUDA device and its operation bundles; every
//! Coeus operator resolves through the generic
//! [`HephaestusBackend<CudaProvider>`](coeus_hephaestus::HephaestusBackend)
//! implementations in `coeus-hephaestus`. Scalar coverage per operator mirrors
//! the historical `CudaBackend` bounds exactly: the CUDA-C dialect predicates
//! and per-operation bundle bounds are the same ones the removed per-vendor
//! shims carried.
//!
//! The provider binds unconditionally over `hephaestus_cuda::CudaDevice`:
//! without the `cuda` feature that type is hephaestus-cuda's stub, so the
//! crate still compiles and device acquisition fails honestly with
//! [`HephaestusError::AdapterUnavailable`](hephaestus_core::HephaestusError::AdapterUnavailable)
//! instead of a fabricated device.

use std::sync::OnceLock;

use coeus_core::{Float, Scalar};
use coeus_hephaestus::{
    ActivationUnaryOperations, ArithmeticUnaryOperations, AttentionProvider, ConvolutionProvider,
    CrossEntropyProvider, CrossProductProvider, ElementwiseProvider, FusionProvider,
    HephaestusProvider, HostCrossProductOps, MatmulProvider, ParameterizedElementwiseProvider,
    PoolingProvider, RandomInitProvider, ReductionProvider, RotateHalfProvider,
    ScalarPowerProvider, StatefulUpdateProvider, UnfoldFoldProvider,
};
use hephaestus_core::{
    AttentionOps, AttentionScalar, ConvolutionOps, CudaC, DenseProductOps, DialectScalar,
};
use hephaestus_cuda::{
    CudaAttentionOps, CudaAxisReductionOps, CudaConvolutionOps, CudaCrossEntropyOps,
    CudaDenseProductOps, CudaDevice, CudaElementwiseOps, CudaFusionOps, CudaFusionScalar,
    CudaParameterizedUnaryOps, CudaPoolingOps, CudaRandomOps, CudaScanOps, CudaSlidingWindowOps,
    CudaStatefulUpdateOps, CumProdOp, CumSumOp, IdentityToken, MaxOp, MinOp, OpIdentity, ProdOp,
    SumOp,
};

/// Scalar types supported by the CUDA backend and Hephaestus fusion.
pub trait CudaScalar: Scalar + leto_ops::Scalar + CudaFusionScalar {}

impl CudaScalar for f32 {}
impl CudaScalar for f64 {}
impl CudaScalar for eunomia::F16 {}
impl CudaScalar for eunomia::Bf16 {}
impl CudaScalar for i32 {}

static CUDA_DEVICE: OnceLock<CudaDevice> = OnceLock::new();

/// Retrieve a reference to the global lazily-initialized hephaestus CUDA device.
pub(crate) fn get_cuda_device() -> &'static CudaDevice {
    try_get_cuda_device().expect("Failed to initialize hephaestus-cuda device")
}

/// Try to retrieve the process-global CUDA device.
///
/// # Errors
///
/// Returns the typed Hephaestus acquisition failure when CUDA is unavailable.
pub(crate) fn try_get_cuda_device() -> hephaestus_core::Result<&'static CudaDevice> {
    if let Some(device) = CUDA_DEVICE.get() {
        return Ok(device);
    }
    let candidate = CudaDevice::try_default()?;
    let _ = CUDA_DEVICE.set(candidate);
    CUDA_DEVICE
        .get()
        .ok_or_else(|| hephaestus_core::HephaestusError::DeviceUnavailable {
            message: "CUDA device initialization did not publish the acquired device".to_owned(),
        })
}

/// Hephaestus CUDA provider marker for [`coeus_hephaestus::HephaestusBackend`].
///
/// ```
/// use coeus_hephaestus::HephaestusBackend;
/// use coeus_cuda::CudaProvider;
///
/// let backend = HephaestusBackend::<CudaProvider>::new();
/// assert_eq!(coeus_core::ComputeBackend::name(&backend), "cuda");
/// ```
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaProvider;

// SAFETY: `HephaestusStorage` retains provider buffers behind `Arc`, and the
// process-global CUDA device owns stream synchronization for every dispatch.
unsafe impl HephaestusProvider for CudaProvider {
    type Device = CudaDevice;

    const NAME: &'static str = "cuda";

    fn device() -> &'static Self::Device {
        get_cuda_device()
    }

    fn try_device() -> hephaestus_core::Result<&'static Self::Device> {
        try_get_cuda_device()
    }
}

impl<T> AttentionProvider<T> for CudaProvider
where
    T: CudaScalar + Float + AttentionScalar + coeus_ops::AttentionScalar,
    CudaAttentionOps: AttentionOps<CudaDevice, T>,
{
    type Operations = CudaAttentionOps;
}

impl<T> ConvolutionProvider<T> for CudaProvider
where
    T: CudaScalar,
    CudaConvolutionOps: ConvolutionOps<CudaDevice, T>,
{
    type Operations = CudaConvolutionOps;
}

impl CrossEntropyProvider for CudaProvider {
    type Operations = CudaCrossEntropyOps;
}

/// Host-fold cross product: hephaestus has no CUDA `CrossProductOps` seam
/// (ADR 0077).
impl<T: Scalar> CrossProductProvider<T> for CudaProvider {
    type Operations = HostCrossProductOps;
}

impl ParameterizedElementwiseProvider for CudaProvider {
    type Operations = CudaParameterizedUnaryOps;
}

impl ElementwiseProvider<f32> for CudaProvider {
    type Operations = CudaElementwiseOps;
    type UnaryOperations = ActivationUnaryOperations;
}

impl ElementwiseProvider<f64> for CudaProvider {
    type Operations = CudaElementwiseOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

impl ElementwiseProvider<i32> for CudaProvider {
    type Operations = CudaElementwiseOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

impl ScalarPowerProvider<f32> for CudaProvider {
    type Operations = CudaElementwiseOps;
}

impl ScalarPowerProvider<f64> for CudaProvider {
    type Operations = CudaElementwiseOps;
}

impl<T> MatmulProvider<T> for CudaProvider
where
    T: CudaScalar,
    CudaDenseProductOps: DenseProductOps<CudaDevice, T>,
{
    type Operations = CudaDenseProductOps;
}

impl<T> PoolingProvider<T> for CudaProvider
where
    T: CudaScalar + DialectScalar<CudaC>,
{
    type Operations = CudaPoolingOps;
}

impl RandomInitProvider<f32> for CudaProvider {
    type Operations = CudaRandomOps;
}

impl<T> ReductionProvider<T> for CudaProvider
where
    T: CudaScalar
        + DialectScalar<CudaC>
        + bytemuck::Pod
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, CudaC>
        + OpIdentity<ProdOp>
        + IdentityToken<ProdOp, CudaC>
        + OpIdentity<MinOp>
        + IdentityToken<MinOp, CudaC>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, CudaC>
        + OpIdentity<CumSumOp>
        + IdentityToken<CumSumOp, CudaC>
        + OpIdentity<CumProdOp>
        + IdentityToken<CumProdOp, CudaC>,
{
    type AxisOperations = CudaAxisReductionOps;
    type ScanOperations = CudaScanOps;
}

impl RotateHalfProvider<f32> for CudaProvider {
    type Operations = CudaElementwiseOps;
}

impl StatefulUpdateProvider for CudaProvider {
    type Operations = CudaStatefulUpdateOps;
}

impl<T> UnfoldFoldProvider<T> for CudaProvider
where
    T: CudaScalar + DialectScalar<CudaC>,
{
    type Operations = CudaSlidingWindowOps;
}

impl<T> FusionProvider<T> for CudaProvider
where
    T: CudaScalar,
{
    type Dialect = CudaC;
    type Operations = CudaFusionOps;
}
