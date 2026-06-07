use coeus_core::{Layout, ComputeBackend, Storage};
use crate::backend::{CudaBackend, CudaScalar};
use crate::storage::CudaStorage;

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
    ) {
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

        coeus_ops::BackendOps::elementwise_binary(
            &seq,
            op,
            &seq_a,
            a_layout,
            &seq_b,
            b_layout,
            &mut seq_c,
            c_layout,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_c.as_slice(), c);
    }

    pub(crate) fn fallback_unary<T: CudaScalar>(
        &self,
        op: coeus_ops::UnaryOp,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) {
        let mut host_a = vec![T::zero(); a.len()];
        self.copy_to_host(a, &mut host_a);
        let mut host_c = vec![T::zero(); c.len()];
        self.copy_to_host(c, &mut host_c);

        let seq = coeus_core::SequentialBackend::new();
        let seq_a = coeus_core::CpuStorage::from_slice(&host_a);
        let mut seq_c = coeus_core::CpuStorage::from_slice(&host_c);

        coeus_ops::BackendOps::elementwise_unary(
            &seq,
            op,
            &seq_a,
            a_layout,
            &mut seq_c,
            c_layout,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_c.as_slice(), c);
    }

    pub(crate) fn fallback_matmul<T: CudaScalar>(
        &self,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        b: &CudaStorage<T>,
        b_layout: &Layout,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) {
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

        coeus_ops::BackendOps::matmul(
            &seq,
            &seq_a,
            a_layout,
            &seq_b,
            b_layout,
            &mut seq_c,
            c_layout,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_c.as_slice(), c);
    }

    pub(crate) fn fallback_reduce<T: CudaScalar>(
        &self,
        op: coeus_ops::ReductionOp,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) {
        let mut host_a = vec![T::zero(); a.len()];
        self.copy_to_host(a, &mut host_a);
        let mut host_c = vec![T::zero(); c.len()];
        self.copy_to_host(c, &mut host_c);

        let seq = coeus_core::SequentialBackend::new();
        let seq_a = coeus_core::CpuStorage::from_slice(&host_a);
        let mut seq_c = coeus_core::CpuStorage::from_slice(&host_c);

        coeus_ops::BackendOps::reduce(
            &seq,
            op,
            &seq_a,
            a_layout,
            axis,
            &mut seq_c,
            c_layout,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_c.as_slice(), c);
    }
}
