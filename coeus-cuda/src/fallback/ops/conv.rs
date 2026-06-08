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

    pub(crate) fn fallback_conv3d<T: CudaScalar>(
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

        coeus_ops::BackendOps::conv3d(
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

    pub(crate) fn fallback_conv3d_backward<T: CudaScalar>(
        &self,
        grad_out: &CudaStorage<T>,
        grad_out_layout: &Layout,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        weight: &CudaStorage<T>,
        weight_layout: &Layout,
        grad_input: Option<&mut CudaStorage<T>>,
        grad_input_layout: &Layout,
        grad_weight: Option<&mut CudaStorage<T>>,
        grad_weight_layout: &Layout,
        grad_bias: Option<&mut CudaStorage<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) {
        let mut host_grad_out = vec![T::zero(); grad_out.len()];
        self.copy_to_host(grad_out, &mut host_grad_out);
        let mut host_input = vec![T::zero(); input.len()];
        self.copy_to_host(input, &mut host_input);
        let mut host_weight = vec![T::zero(); weight.len()];
        self.copy_to_host(weight, &mut host_weight);

        let seq = coeus_core::SequentialBackend::new();
        let seq_go = coeus_core::CpuStorage::from_slice(&host_grad_out);
        let seq_in = coeus_core::CpuStorage::from_slice(&host_input);
        let seq_w = coeus_core::CpuStorage::from_slice(&host_weight);

        let mut seq_gi = grad_input.as_ref().map(|gi| coeus_core::CpuStorage::from_slice(&vec![T::zero(); gi.len()]));
        let mut seq_gw = grad_weight.as_ref().map(|gw| coeus_core::CpuStorage::from_slice(&vec![T::zero(); gw.len()]));
        let mut seq_gb = grad_bias.as_ref().map(|gb| coeus_core::CpuStorage::from_slice(&vec![T::zero(); gb.len()]));

        coeus_ops::BackendOps::conv3d_backward(
            &seq,
            &seq_go,
            grad_out_layout,
            &seq_in,
            input_layout,
            &seq_w,
            weight_layout,
            seq_gi.as_mut(),
            grad_input_layout,
            seq_gw.as_mut(),
            grad_weight_layout,
            seq_gb.as_mut(),
            stride,
            padding,
            dilation,
        );

        use coeus_core::CpuAddressableStorage;
        if let (Some(seq_gi_val), Some(gi)) = (seq_gi, grad_input) {
            self.copy_to_device(seq_gi_val.as_slice(), gi);
        }
        if let (Some(seq_gw_val), Some(gw)) = (seq_gw, grad_weight) {
            self.copy_to_device(seq_gw_val.as_slice(), gw);
        }
        if let (Some(seq_gb_val), Some(gb)) = (seq_gb, grad_bias) {
            self.copy_to_device(seq_gb_val.as_slice(), gb);
        }
    }
}
