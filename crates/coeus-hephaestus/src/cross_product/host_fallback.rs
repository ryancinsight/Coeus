use crate::{HephaestusBackend, HephaestusProvider};
use coeus_core::{ComputeBackend, Layout, Scalar};

/// Generic host-fold `CrossOps` for every Hephaestus provider without a
/// `CrossProductProvider<T>` impl — currently `RocmProvider` and
/// `MetalProvider` (ADR 0077: hephaestus has no `CrossProductOps` for either
/// vendor yet). Uses the same shared host-fold every other seam-less backend
/// uses, via `copy_to_host`/`copy_to_device` since Hephaestus-backed storage
/// is not host-addressable.
///
/// A provider that later gains `CrossProductProvider<T>` overrides this
/// generic impl by implementing `coeus_ops::CrossOps<T>` directly on its own
/// concrete backend type instead of going through this blanket — exactly the
/// path `WgpuBackend` already takes, since `WgpuBackend` is not a
/// `HephaestusBackend<P>` instantiation.
impl<P, T> coeus_ops::CrossOps<T> for HephaestusBackend<P>
where
    P: HephaestusProvider,
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
        let mut a_host = vec![T::zero(); numel];
        let mut b_host = vec![T::zero(); numel];
        self.copy_to_host(a, &mut a_host)?;
        self.copy_to_host(b, &mut b_host)?;
        let mut out_host = vec![T::zero(); numel];
        coeus_ops::cross_fold(&a_host, &b_host, a_layout, dim, &mut out_host);
        // SAFETY: copy_to_device uploads every element of the completed output.
        let mut output = unsafe { self.allocate(numel)? };
        self.copy_to_device(&out_host, &mut output)?;
        Ok(output)
    }
}
