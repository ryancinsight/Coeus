#![allow(
    clippy::multiple_bound_locations,
    clippy::too_many_arguments,
    reason = "fallback methods mirror the BackendOps device boundary signatures"
)]

use crate::backend::{CudaBackend, CudaScalar};
use crate::storage::CudaStorage;
use coeus_core::{ComputeBackend, Layout, Storage};
use coeus_ops::BackendOps;

impl CudaBackend {
    pub(crate) fn fallback_sdp_attention<T: CudaScalar>(
        &self,
        query: &CudaStorage<T>,
        query_layout: &Layout,
        key: &CudaStorage<T>,
        key_layout: &Layout,
        value: &CudaStorage<T>,
        value_layout: &Layout,
        key_padding_mask: Option<&CudaStorage<T>>,
        key_padding_mask_layout: Option<&Layout>,
        is_causal: bool,
        scale: T,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
        attn_weights: &mut CudaStorage<T>,
        attn_weights_layout: &Layout,
    ) where
        T: coeus_core::Float,
    {
        let mut host_q = vec![T::zero(); query.len()];
        self.copy_to_host(query, &mut host_q);
        let mut host_k = vec![T::zero(); key.len()];
        self.copy_to_host(key, &mut host_k);
        let mut host_v = vec![T::zero(); value.len()];
        self.copy_to_host(value, &mut host_v);

        let host_mask = key_padding_mask.map(|mask| {
            let mut hm = vec![T::zero(); mask.len()];
            self.copy_to_host(mask, &mut hm);
            hm
        });

        let seq = coeus_core::SequentialBackend::new();
        let seq_q = coeus_core::CpuStorage::from_slice(&host_q);
        let seq_k = coeus_core::CpuStorage::from_slice(&host_k);
        let seq_v = coeus_core::CpuStorage::from_slice(&host_v);
        let seq_mask = host_mask.map(|hm| coeus_core::CpuStorage::from_slice(&hm));

        let mut seq_out = coeus_core::CpuStorage::from_slice(&vec![T::zero(); output.len()]);
        let mut seq_aw = coeus_core::CpuStorage::from_slice(&vec![T::zero(); attn_weights.len()]);

        seq.sdp_attention(
            &seq_q,
            query_layout,
            &seq_k,
            key_layout,
            &seq_v,
            value_layout,
            seq_mask.as_ref(),
            key_padding_mask_layout,
            is_causal,
            scale,
            &mut seq_out,
            output_layout,
            &mut seq_aw,
            attn_weights_layout,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_out.as_slice(), output);
        self.copy_to_device(seq_aw.as_slice(), attn_weights);
    }

    pub(crate) fn fallback_sdp_attention_backward<T: CudaScalar>(
        &self,
        grad_out: &CudaStorage<T>,
        grad_out_layout: &Layout,
        query: &CudaStorage<T>,
        query_layout: &Layout,
        key: &CudaStorage<T>,
        key_layout: &Layout,
        value: &CudaStorage<T>,
        value_layout: &Layout,
        attn_weights: &CudaStorage<T>,
        attn_weights_layout: &Layout,
        scale: T,
        grad_q: Option<&mut CudaStorage<T>>,
        grad_k: Option<&mut CudaStorage<T>>,
        grad_v: Option<&mut CudaStorage<T>>,
    ) where
        T: coeus_core::Float,
    {
        let mut host_go = vec![T::zero(); grad_out.len()];
        self.copy_to_host(grad_out, &mut host_go);
        let mut host_q = vec![T::zero(); query.len()];
        self.copy_to_host(query, &mut host_q);
        let mut host_k = vec![T::zero(); key.len()];
        self.copy_to_host(key, &mut host_k);
        let mut host_v = vec![T::zero(); value.len()];
        self.copy_to_host(value, &mut host_v);
        let mut host_aw = vec![T::zero(); attn_weights.len()];
        self.copy_to_host(attn_weights, &mut host_aw);

        let seq = coeus_core::SequentialBackend::new();
        let seq_go = coeus_core::CpuStorage::from_slice(&host_go);
        let seq_q = coeus_core::CpuStorage::from_slice(&host_q);
        let seq_k = coeus_core::CpuStorage::from_slice(&host_k);
        let seq_v = coeus_core::CpuStorage::from_slice(&host_v);
        let seq_aw = coeus_core::CpuStorage::from_slice(&host_aw);

        let mut seq_gq = grad_q
            .as_ref()
            .map(|g| coeus_core::CpuStorage::from_slice(&vec![T::zero(); g.len()]));
        let mut seq_gk = grad_k
            .as_ref()
            .map(|g| coeus_core::CpuStorage::from_slice(&vec![T::zero(); g.len()]));
        let mut seq_gv = grad_v
            .as_ref()
            .map(|g| coeus_core::CpuStorage::from_slice(&vec![T::zero(); g.len()]));

        seq.sdp_attention_backward(
            &seq_go,
            grad_out_layout,
            &seq_q,
            query_layout,
            &seq_k,
            key_layout,
            &seq_v,
            value_layout,
            &seq_aw,
            attn_weights_layout,
            scale,
            seq_gq.as_mut(),
            seq_gk.as_mut(),
            seq_gv.as_mut(),
        );

        use coeus_core::CpuAddressableStorage;
        if let (Some(seq_gq_val), Some(gq)) = (seq_gq, grad_q) {
            self.copy_to_device(seq_gq_val.as_slice(), gq);
        }
        if let (Some(seq_gk_val), Some(gk)) = (seq_gk, grad_k) {
            self.copy_to_device(seq_gk_val.as_slice(), gk);
        }
        if let (Some(seq_gv_val), Some(gv)) = (seq_gv, grad_v) {
            self.copy_to_device(seq_gv_val.as_slice(), gv);
        }
    }
}
