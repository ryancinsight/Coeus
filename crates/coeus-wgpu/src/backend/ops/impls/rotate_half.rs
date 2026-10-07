use crate::backend::{WgpuBackend, WgpuBackendError};
use coeus_core::{Layout, Scalar};
use coeus_hephaestus::{rotate_half, RotateHalfProvider};
use hephaestus_core::{ElementwiseOps, IdentityOp, NegOp, UnaryExpr};
use hephaestus_wgpu::{DialectScalar, WgpuDevice, WgpuElementwiseOps, Wgsl};

// NOTE: `WgpuScalar` is deliberately not required: it gates the fused
// dispatch (f32/i32/u32 only), while rotate-half runs on the plain
// elementwise seam, whose `DialectScalar<Wgsl>` admits f64 as well.
impl<T> RotateHalfProvider<T> for WgpuBackend
where
    T: Scalar + DialectScalar<Wgsl>,
    WgpuElementwiseOps: ElementwiseOps<WgpuDevice, T>,
{
    type Operations = WgpuElementwiseOps;
}

impl<T> coeus_ops::RotateHalfOps<T> for WgpuBackend
where
    T: Scalar + DialectScalar<Wgsl>,
    WgpuElementwiseOps: ElementwiseOps<WgpuDevice, T>,
    IdentityOp: UnaryExpr<<WgpuElementwiseOps as ElementwiseOps<WgpuDevice, T>>::Dialect>,
    NegOp: UnaryExpr<<WgpuElementwiseOps as ElementwiseOps<WgpuDevice, T>>::Dialect>,
{
    fn rotate_half_storage(
        &self,
        input: &Self::DeviceBuffer<T>,
        layout: &Layout,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        rotate_half::<Self, _>(input.buffer(), layout)
            .map(coeus_hephaestus::HephaestusStorage::from_buffer)
            .map_err(|source| WgpuBackendError::dispatch("rotate_half", source))
    }
}
