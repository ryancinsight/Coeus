use crate::backend::{WgpuBackend, WgpuBackendError};
use coeus_core::{ComputeBackend, Layout, NumericElement, Scalar};
use coeus_hephaestus::{cross_product, CrossProductProvider, HephaestusStorage};
use hephaestus_core::CrossProductOps;
use hephaestus_wgpu::{DialectScalar, WgpuCrossProductOps, WgpuDevice, Wgsl};

impl<T> CrossProductProvider<T> for WgpuBackend
where
    T: Scalar + DialectScalar<Wgsl>,
    WgpuCrossProductOps: CrossProductOps<WgpuDevice, T>,
{
    type Operations = WgpuCrossProductOps;
}

/// Bridges `coeus_ops::CrossOps` to hephaestus's on-device `CrossProductOps`
/// (ADR 0077) for the layout it can express — `dim` the last axis, both
/// operands contiguous, so the tensor is already `numel / 3` consecutive
/// `(x, y, z)` triples. Any other layout falls back to the shared host-fold
/// every seam-less backend uses; that fallback is explicit and typed, never
/// a silent runtime probe.
impl<T> coeus_ops::CrossOps<T> for WgpuBackend
where
    T: Scalar + DialectScalar<Wgsl>,
    WgpuCrossProductOps: CrossProductOps<WgpuDevice, T>,
{
    fn cross_storage(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        dim: usize,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        let numel = a_layout.numel();
        if dim == a_layout.ndim().saturating_sub(1) && a_layout.is_contiguous() {
            return cross_product::<Self, T>(a.buffer(), b.buffer(), numel / 3)
                .map(HephaestusStorage::from_buffer)
                .map_err(|source| WgpuBackendError::dispatch("cross", source));
        }
        let mut a_host = vec![<T as NumericElement>::ZERO; numel];
        let mut b_host = vec![<T as NumericElement>::ZERO; numel];
        self.copy_to_host(a, &mut a_host);
        self.copy_to_host(b, &mut b_host);
        let mut out_host = vec![<T as NumericElement>::ZERO; numel];
        coeus_ops::cross_fold(&a_host, &b_host, a_layout, dim, &mut out_host);
        let mut output = self.allocate(numel);
        self.copy_to_device(&out_host, &mut output);
        Ok(output)
    }
}
