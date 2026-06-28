#![allow(
    clippy::too_many_arguments,
    reason = "fallback methods mirror the BackendOps pooling boundary signatures"
)]

use crate::backend::{CudaBackend, CudaScalar};
use crate::storage::CudaStorage;
use coeus_core::{ComputeBackend, Layout, Storage};

impl CudaBackend {
    pub(crate) fn fallback_max_pool2d<T: CudaScalar>(
        &self,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
    ) {
        let mut host_in = vec![T::zero(); input.len()];
        self.copy_to_host(input, &mut host_in);
        let mut host_out = vec![T::zero(); output.len()];
        self.copy_to_host(output, &mut host_out);

        let seq = coeus_core::SequentialBackend::new();
        let seq_in = coeus_core::CpuStorage::from_slice(&host_in);
        let mut seq_out = coeus_core::CpuStorage::from_slice(&host_out);

        coeus_ops::PoolOps::max_pool2d(
            &seq,
            &seq_in,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &mut seq_out,
            output_layout,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_out.as_slice(), output);
    }

    pub(crate) fn fallback_max_pool2d_backward<T: CudaScalar>(
        &self,
        grad_out: &CudaStorage<T>,
        grad_out_layout: &Layout,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut CudaStorage<T>,
        grad_input_layout: &Layout,
    ) {
        let mut host_go = vec![T::zero(); grad_out.len()];
        self.copy_to_host(grad_out, &mut host_go);
        let mut host_in = vec![T::zero(); input.len()];
        self.copy_to_host(input, &mut host_in);
        let mut host_gi = vec![T::zero(); grad_input.len()];
        self.copy_to_host(grad_input, &mut host_gi);

        let seq = coeus_core::SequentialBackend::new();
        let seq_go = coeus_core::CpuStorage::from_slice(&host_go);
        let seq_in = coeus_core::CpuStorage::from_slice(&host_in);
        let mut seq_gi = coeus_core::CpuStorage::from_slice(&host_gi);

        coeus_ops::PoolOps::max_pool2d_backward(
            &seq,
            &seq_go,
            grad_out_layout,
            &seq_in,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &mut seq_gi,
            grad_input_layout,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_gi.as_slice(), grad_input);
    }

    pub(crate) fn fallback_avg_pool2d<T: CudaScalar>(
        &self,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
    ) {
        let mut host_in = vec![T::zero(); input.len()];
        self.copy_to_host(input, &mut host_in);
        let mut host_out = vec![T::zero(); output.len()];
        self.copy_to_host(output, &mut host_out);

        let seq = coeus_core::SequentialBackend::new();
        let seq_in = coeus_core::CpuStorage::from_slice(&host_in);
        let mut seq_out = coeus_core::CpuStorage::from_slice(&host_out);

        coeus_ops::PoolOps::avg_pool2d(
            &seq,
            &seq_in,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &mut seq_out,
            output_layout,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_out.as_slice(), output);
    }

    pub(crate) fn fallback_avg_pool2d_backward<T: CudaScalar>(
        &self,
        grad_out: &CudaStorage<T>,
        grad_out_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut CudaStorage<T>,
        grad_input_layout: &Layout,
    ) {
        let mut host_go = vec![T::zero(); grad_out.len()];
        self.copy_to_host(grad_out, &mut host_go);
        let mut host_gi = vec![T::zero(); grad_input.len()];
        self.copy_to_host(grad_input, &mut host_gi);

        let seq = coeus_core::SequentialBackend::new();
        let seq_go = coeus_core::CpuStorage::from_slice(&host_go);
        let mut seq_gi = coeus_core::CpuStorage::from_slice(&host_gi);

        coeus_ops::PoolOps::avg_pool2d_backward(
            &seq,
            &seq_go,
            grad_out_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &mut seq_gi,
            grad_input_layout,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_gi.as_slice(), grad_input);
    }

    pub(crate) fn fallback_max_pool3d<T: CudaScalar>(
        &self,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
    ) {
        let mut host_in = vec![T::zero(); input.len()];
        self.copy_to_host(input, &mut host_in);
        let mut host_out = vec![T::zero(); output.len()];
        self.copy_to_host(output, &mut host_out);

        let seq = coeus_core::SequentialBackend::new();
        let seq_in = coeus_core::CpuStorage::from_slice(&host_in);
        let mut seq_out = coeus_core::CpuStorage::from_slice(&host_out);

        coeus_ops::PoolOps::max_pool3d(
            &seq,
            &seq_in,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &mut seq_out,
            output_layout,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_out.as_slice(), output);
    }

    pub(crate) fn fallback_max_pool3d_backward<T: CudaScalar>(
        &self,
        grad_out: &CudaStorage<T>,
        grad_out_layout: &Layout,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut CudaStorage<T>,
        grad_input_layout: &Layout,
    ) {
        let mut host_go = vec![T::zero(); grad_out.len()];
        self.copy_to_host(grad_out, &mut host_go);
        let mut host_in = vec![T::zero(); input.len()];
        self.copy_to_host(input, &mut host_in);
        let mut host_gi = vec![T::zero(); grad_input.len()];
        self.copy_to_host(grad_input, &mut host_gi);

        let seq = coeus_core::SequentialBackend::new();
        let seq_go = coeus_core::CpuStorage::from_slice(&host_go);
        let seq_in = coeus_core::CpuStorage::from_slice(&host_in);
        let mut seq_gi = coeus_core::CpuStorage::from_slice(&host_gi);

        coeus_ops::PoolOps::max_pool3d_backward(
            &seq,
            &seq_go,
            grad_out_layout,
            &seq_in,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &mut seq_gi,
            grad_input_layout,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_gi.as_slice(), grad_input);
    }

    pub(crate) fn fallback_avg_pool3d<T: CudaScalar>(
        &self,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
    ) {
        let mut host_in = vec![T::zero(); input.len()];
        self.copy_to_host(input, &mut host_in);
        let mut host_out = vec![T::zero(); output.len()];
        self.copy_to_host(output, &mut host_out);

        let seq = coeus_core::SequentialBackend::new();
        let seq_in = coeus_core::CpuStorage::from_slice(&host_in);
        let mut seq_out = coeus_core::CpuStorage::from_slice(&host_out);

        coeus_ops::PoolOps::avg_pool3d(
            &seq,
            &seq_in,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &mut seq_out,
            output_layout,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_out.as_slice(), output);
    }

    pub(crate) fn fallback_avg_pool3d_backward<T: CudaScalar>(
        &self,
        grad_out: &CudaStorage<T>,
        grad_out_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut CudaStorage<T>,
        grad_input_layout: &Layout,
    ) {
        let mut host_go = vec![T::zero(); grad_out.len()];
        self.copy_to_host(grad_out, &mut host_go);
        let mut host_gi = vec![T::zero(); grad_input.len()];
        self.copy_to_host(grad_input, &mut host_gi);

        let seq = coeus_core::SequentialBackend::new();
        let seq_go = coeus_core::CpuStorage::from_slice(&host_go);
        let mut seq_gi = coeus_core::CpuStorage::from_slice(&host_gi);

        coeus_ops::PoolOps::avg_pool3d_backward(
            &seq,
            &seq_go,
            grad_out_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &mut seq_gi,
            grad_input_layout,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_gi.as_slice(), grad_input);
    }
}
