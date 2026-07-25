#![allow(
    clippy::too_many_arguments,
    reason = "fallback methods mirror the BackendOps math boundary signatures"
)]

use crate::backend::{CudaBackend, CudaScalar};
use crate::storage::CudaStorage;
use coeus_core::{ComputeBackend, Layout, Storage};

impl CudaBackend {
    pub(crate) fn fallback_binary<T: CudaScalar>(
        &self,
        op: coeus_ops::BinaryOp,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        b: &CudaStorage<T>,
        b_layout: &Layout,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) -> Result<(), crate::CudaBackendError> {
        let mut host_a = vec![T::zero(); a.len()];
        self.copy_to_host(a, &mut host_a);
        let mut host_b = vec![T::zero(); b.len()];
        self.copy_to_host(b, &mut host_b);
        let mut host_c = vec![T::zero(); c.len()];
        self.copy_to_host(c, &mut host_c);

        let seq = coeus_core::SequentialBackend::new();
        let seq_a = coeus_core::CpuStorage::from_slice(&host_a);
        let seq_b = coeus_core::CpuStorage::from_slice(&host_b);
        let mut seq_c = coeus_core::CpuStorage::from_slice(&host_c);

        coeus_ops::ElementwiseOps::elementwise_binary(
            &seq, op, &seq_a, a_layout, &seq_b, b_layout, &mut seq_c, c_layout,
        )?;

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_c.as_slice(), c);
        Ok(())
    }

    pub(crate) fn fallback_unary<T: CudaScalar>(
        &self,
        op: coeus_ops::UnaryOp,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) -> Result<(), crate::CudaBackendError> {
        let mut host_a = vec![T::zero(); a.len()];
        self.copy_to_host(a, &mut host_a);
        let mut host_c = vec![T::zero(); c.len()];
        self.copy_to_host(c, &mut host_c);

        let seq = coeus_core::SequentialBackend::new();
        let seq_a = coeus_core::CpuStorage::from_slice(&host_a);
        let mut seq_c = coeus_core::CpuStorage::from_slice(&host_c);

        coeus_ops::ElementwiseOps::elementwise_unary(
            &seq, op, &seq_a, a_layout, &mut seq_c, c_layout,
        )?;

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_c.as_slice(), c);
        Ok(())
    }

    pub(crate) fn fallback_matmul<T: CudaScalar>(
        &self,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        b: &CudaStorage<T>,
        b_layout: &Layout,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) -> Result<(), crate::CudaBackendError> {
        let mut host_a = vec![T::zero(); a.len()];
        self.copy_to_host(a, &mut host_a);
        let mut host_b = vec![T::zero(); b.len()];
        self.copy_to_host(b, &mut host_b);
        let mut host_c = vec![T::zero(); c.len()];
        self.copy_to_host(c, &mut host_c);

        let seq = coeus_core::SequentialBackend::new();
        let seq_a = coeus_core::CpuStorage::from_slice(&host_a);
        let seq_b = coeus_core::CpuStorage::from_slice(&host_b);
        let mut seq_c = coeus_core::CpuStorage::from_slice(&host_c);

        coeus_ops::MatmulOps::matmul(
            &seq, &seq_a, a_layout, &seq_b, b_layout, &mut seq_c, c_layout,
        )?;

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_c.as_slice(), c);
        Ok(())
    }

    pub(crate) fn fallback_reduce<T: CudaScalar>(
        &self,
        op: coeus_ops::ReductionOp,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) -> Result<(), crate::CudaBackendError> {
        let mut host_a = vec![T::zero(); a.len()];
        self.copy_to_host(a, &mut host_a);
        let mut host_c = vec![T::zero(); c.len()];
        self.copy_to_host(c, &mut host_c);

        let seq = coeus_core::SequentialBackend::new();
        let seq_a = coeus_core::CpuStorage::from_slice(&host_a);
        let mut seq_c = coeus_core::CpuStorage::from_slice(&host_c);

        coeus_ops::ReductionOps::reduce(&seq, op, &seq_a, a_layout, axis, &mut seq_c, c_layout)
            .map_err(|source| crate::CudaBackendError::cpu_capability("reduction", source))?;

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_c.as_slice(), c);
        Ok(())
    }
}
