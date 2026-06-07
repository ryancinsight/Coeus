use coeus_core::{Layout, ComputeBackend, Storage};
use crate::backend::{CudaBackend, CudaScalar};
use crate::storage::CudaStorage;

impl CudaBackend {
    pub(crate) fn fallback_conv1d<T: CudaScalar>(
        &self,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        weight: &CudaStorage<T>,
        weight_layout: &Layout,
        bias: Option<&CudaStorage<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
    ) {
        let mut host_input = vec![T::zero(); input.len()];
        self.copy_to_host(input, &mut host_input);
        let mut host_weight = vec![T::zero(); weight.len()];
        self.copy_to_host(weight, &mut host_weight);

        let host_bias = bias.map(|b| {
            let mut hb = vec![T::zero(); b.len()];
            self.copy_to_host(b, &mut hb);
            hb
        });

        let seq = coeus_core::SequentialBackend::new();
        let seq_in = coeus_core::CpuStorage::from_slice(&host_input);
        let seq_w = coeus_core::CpuStorage::from_slice(&host_weight);
        let seq_bias = host_bias.map(|hb| coeus_core::CpuStorage::from_slice(&hb));
        let mut seq_out = coeus_core::CpuStorage::from_slice(&vec![T::zero(); output.len()]);

        coeus_ops::BackendOps::conv1d(
            &seq,
            &seq_in,
            input_layout,
            &seq_w,
            weight_layout,
            seq_bias.as_ref(),
            stride,
            padding,
            dilation,
            &mut seq_out,
            output_layout,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_out.as_slice(), output);
    }

    pub(crate) fn fallback_conv2d<T: CudaScalar>(
        &self,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        weight: &CudaStorage<T>,
        weight_layout: &Layout,
        bias: Option<&CudaStorage<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
    ) {
        let mut host_input = vec![T::zero(); input.len()];
        self.copy_to_host(input, &mut host_input);
        let mut host_weight = vec![T::zero(); weight.len()];
        self.copy_to_host(weight, &mut host_weight);

        let host_bias = bias.map(|b| {
            let mut hb = vec![T::zero(); b.len()];
            self.copy_to_host(b, &mut hb);
            hb
        });

        let seq = coeus_core::SequentialBackend::new();
        let seq_in = coeus_core::CpuStorage::from_slice(&host_input);
        let seq_w = coeus_core::CpuStorage::from_slice(&host_weight);
        let seq_bias = host_bias.map(|hb| coeus_core::CpuStorage::from_slice(&hb));
        let mut seq_out = coeus_core::CpuStorage::from_slice(&vec![T::zero(); output.len()]);

        coeus_ops::BackendOps::conv2d(
            &seq,
            &seq_in,
            input_layout,
            &seq_w,
            weight_layout,
            seq_bias.as_ref(),
            stride,
            padding,
            dilation,
            &mut seq_out,
            output_layout,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_out.as_slice(), output);
    }
}
