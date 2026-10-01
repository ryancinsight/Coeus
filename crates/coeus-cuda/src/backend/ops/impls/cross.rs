use crate::backend::CudaBackend;
use coeus_core::{ComputeBackend, Layout, Scalar};

/// hephaestus has no `CrossProductOps` implementation for CUDA yet (only
/// host/CPU and WGPU, ADR 0077) — this backend uses the same shared
/// host-fold every other seam-less backend uses, via `copy_to_host`/
/// `copy_to_device` since CUDA storage is not host-addressable.
impl<T: Scalar> coeus_ops::CrossOps<T> for CudaBackend {
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
