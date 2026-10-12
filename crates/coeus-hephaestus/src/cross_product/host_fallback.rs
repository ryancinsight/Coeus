use super::{cross_product, CrossProductProvider};
use crate::{HephaestusBackend, HephaestusBackendError, HephaestusStorage};
use bytemuck::Pod;
use coeus_core::{ComputeBackend, Layout, Scalar};
use hephaestus_core::{ComputeDevice, CrossProductOps, DeviceBuffer};

/// Host-fold cross product for the layouts — or providers — without an
/// on-device `CrossProductOps` seam.
///
/// The fold itself is the same [`coeus_ops::cross_fold`] every seam-less
/// backend uses, reached via `copy_to_host`/`copy_to_device` since
/// Hephaestus-backed storage is not host-addressable.
pub fn cross_host_fallback<B, T>(
    backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    b: &B::DeviceBuffer<T>,
    dim: usize,
) -> Result<B::DeviceBuffer<T>, B::Error>
where
    B: ComputeBackend,
    T: Scalar,
{
    let numel = a_layout.numel();
    let mut a_host = vec![T::ZERO; numel];
    let mut b_host = vec![T::ZERO; numel];
    backend.copy_to_host(a, &mut a_host);
    backend.copy_to_host(b, &mut b_host);
    let mut out_host = vec![T::ZERO; numel];
    coeus_ops::cross_fold(&a_host, &b_host, a_layout, dim, &mut out_host);
    let mut output = backend.allocate(numel);
    backend.copy_to_device(&out_host, &mut output);
    Ok(output)
}

/// Host-fold [`CrossProductOps`] bundle for providers without an on-device
/// cross-product kernel.
///
/// Providers name this bundle in their [`CrossProductProvider`] impl so the
/// shared [`coeus_ops::CrossOps`] blanket below resolves for them; the values
/// equal the on-device path bit-for-bit because both fold through
/// [`coeus_ops::cross_fold`].
#[derive(Debug, Default)]
pub struct HostCrossProductOps;

impl<D: ComputeDevice, T: Pod + Scalar> CrossProductOps<D, T> for HostCrossProductOps {
    fn cross_into(
        &self,
        device: &D,
        a: &D::Buffer<T>,
        b: &D::Buffer<T>,
        out: &D::Buffer<T>,
    ) -> hephaestus_core::Result<()> {
        let triples = hephaestus_core::validate_cross_product_lengths(a.len(), b.len(), out.len())?;
        let mut a_host = vec![T::ZERO; triples * 3];
        let mut b_host = vec![T::ZERO; triples * 3];
        device.download(a, &mut a_host)?;
        device.download(b, &mut b_host)?;
        let mut out_host = vec![T::ZERO; triples * 3];
        let layout = Layout::new([triples, 3].into());
        coeus_ops::cross_fold(&a_host, &b_host, &layout, 1, &mut out_host);
        let staged = device.upload(&out_host)?;
        device.copy_buffer(&staged, out)
    }
}

/// Bridges `coeus_ops::CrossOps` to each provider's cross-product bundle
/// (ADR 0077) for the layout an on-device kernel can express — `dim` the
/// last axis, both operands contiguous, so the tensor is already
/// `numel / 3` consecutive `(x, y, z)` triples. Any other layout falls back
/// to the shared host-fold; that fallback is explicit and typed, never a
/// silent runtime probe.
impl<P, T> coeus_ops::CrossOps<T> for HephaestusBackend<P>
where
    P: CrossProductProvider<T>,
    T: Scalar,
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
            return cross_product::<P, T>(a.buffer(), b.buffer(), numel / 3)
                .map(HephaestusStorage::from_buffer)
                .map_err(|source| HephaestusBackendError::device("cross", source));
        }
        cross_host_fallback(self, a, a_layout, b, dim)
    }
}
