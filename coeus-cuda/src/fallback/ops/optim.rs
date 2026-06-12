#![allow(
    clippy::multiple_bound_locations,
    clippy::too_many_arguments,
    reason = "fallback methods mirror the BackendOps optimizer boundary signatures"
)]

use crate::backend::{CudaBackend, CudaScalar};
use crate::storage::CudaStorage;
use coeus_core::{ComputeBackend, Layout, Storage};

impl CudaBackend {
    pub(crate) fn fallback_sgd_step<T: CudaScalar>(
        &self,
        param: &mut CudaStorage<T>,
        param_layout: &Layout,
        grad: &CudaStorage<T>,
        grad_layout: &Layout,
        velocity: &mut CudaStorage<T>,
        velocity_layout: &Layout,
        lr: T,
        momentum: T,
    ) where
        T: coeus_core::Float,
    {
        let mut host_param = vec![T::zero(); param.len()];
        self.copy_to_host(param, &mut host_param);
        let mut host_grad = vec![T::zero(); grad.len()];
        self.copy_to_host(grad, &mut host_grad);
        let mut host_velocity = vec![T::zero(); velocity.len()];
        self.copy_to_host(velocity, &mut host_velocity);

        let seq = coeus_core::SequentialBackend::new();
        let mut seq_param = coeus_core::CpuStorage::from_slice(&host_param);
        let seq_grad = coeus_core::CpuStorage::from_slice(&host_grad);
        let mut seq_velocity = coeus_core::CpuStorage::from_slice(&host_velocity);

        coeus_ops::BackendOps::sgd_step(
            &seq,
            &mut seq_param,
            param_layout,
            &seq_grad,
            grad_layout,
            &mut seq_velocity,
            velocity_layout,
            lr,
            momentum,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_param.as_slice(), param);
        self.copy_to_device(seq_velocity.as_slice(), velocity);
    }

    pub(crate) fn fallback_adam_step<T: CudaScalar>(
        &self,
        param: &mut CudaStorage<T>,
        param_layout: &Layout,
        grad: &CudaStorage<T>,
        grad_layout: &Layout,
        m: &mut CudaStorage<T>,
        m_layout: &Layout,
        v: &mut CudaStorage<T>,
        v_layout: &Layout,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        t: usize,
    ) where
        T: coeus_core::Float,
    {
        let mut host_param = vec![T::zero(); param.len()];
        self.copy_to_host(param, &mut host_param);
        let mut host_grad = vec![T::zero(); grad.len()];
        self.copy_to_host(grad, &mut host_grad);
        let mut host_m = vec![T::zero(); m.len()];
        self.copy_to_host(m, &mut host_m);
        let mut host_v = vec![T::zero(); v.len()];
        self.copy_to_host(v, &mut host_v);

        let seq = coeus_core::SequentialBackend::new();
        let mut seq_param = coeus_core::CpuStorage::from_slice(&host_param);
        let seq_grad = coeus_core::CpuStorage::from_slice(&host_grad);
        let mut seq_m = coeus_core::CpuStorage::from_slice(&host_m);
        let mut seq_v = coeus_core::CpuStorage::from_slice(&host_v);

        coeus_ops::BackendOps::adam_step(
            &seq,
            &mut seq_param,
            param_layout,
            &seq_grad,
            grad_layout,
            &mut seq_m,
            m_layout,
            &mut seq_v,
            v_layout,
            lr,
            beta1,
            beta2,
            eps,
            t,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_param.as_slice(), param);
        self.copy_to_device(seq_m.as_slice(), m);
        self.copy_to_device(seq_v.as_slice(), v);
    }

    pub(crate) fn fallback_rmsprop_step<T: CudaScalar>(
        &self,
        param: &mut CudaStorage<T>,
        param_layout: &Layout,
        grad: &CudaStorage<T>,
        grad_layout: &Layout,
        v: &mut CudaStorage<T>,
        v_layout: &Layout,
        lr: T,
        alpha: T,
        eps: T,
    ) where
        T: coeus_core::Float,
    {
        let mut host_param = vec![T::zero(); param.len()];
        self.copy_to_host(param, &mut host_param);
        let mut host_grad = vec![T::zero(); grad.len()];
        self.copy_to_host(grad, &mut host_grad);
        let mut host_v = vec![T::zero(); v.len()];
        self.copy_to_host(v, &mut host_v);

        let seq = coeus_core::SequentialBackend::new();
        let mut seq_param = coeus_core::CpuStorage::from_slice(&host_param);
        let seq_grad = coeus_core::CpuStorage::from_slice(&host_grad);
        let mut seq_v = coeus_core::CpuStorage::from_slice(&host_v);

        coeus_ops::BackendOps::rmsprop_step(
            &seq,
            &mut seq_param,
            param_layout,
            &seq_grad,
            grad_layout,
            &mut seq_v,
            v_layout,
            lr,
            alpha,
            eps,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_param.as_slice(), param);
        self.copy_to_device(seq_v.as_slice(), v);
    }

    pub(crate) fn fallback_adagrad_step<T: CudaScalar>(
        &self,
        param: &mut CudaStorage<T>,
        param_layout: &Layout,
        grad: &CudaStorage<T>,
        grad_layout: &Layout,
        history: &mut CudaStorage<T>,
        history_layout: &Layout,
        lr: T,
        eps: T,
    ) where
        T: coeus_core::Float,
    {
        let mut host_param = vec![T::zero(); param.len()];
        self.copy_to_host(param, &mut host_param);
        let mut host_grad = vec![T::zero(); grad.len()];
        self.copy_to_host(grad, &mut host_grad);
        let mut host_history = vec![T::zero(); history.len()];
        self.copy_to_host(history, &mut host_history);

        let seq = coeus_core::SequentialBackend::new();
        let mut seq_param = coeus_core::CpuStorage::from_slice(&host_param);
        let seq_grad = coeus_core::CpuStorage::from_slice(&host_grad);
        let mut seq_history = coeus_core::CpuStorage::from_slice(&host_history);

        coeus_ops::BackendOps::adagrad_step(
            &seq,
            &mut seq_param,
            param_layout,
            &seq_grad,
            grad_layout,
            &mut seq_history,
            history_layout,
            lr,
            eps,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_param.as_slice(), param);
        self.copy_to_device(seq_history.as_slice(), history);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fallback_adamw_step<T: CudaScalar>(
        &self,
        param: &mut CudaStorage<T>,
        param_layout: &Layout,
        grad: &CudaStorage<T>,
        grad_layout: &Layout,
        m: &mut CudaStorage<T>,
        m_layout: &Layout,
        v: &mut CudaStorage<T>,
        v_layout: &Layout,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        weight_decay: T,
        t: usize,
    ) where
        T: coeus_core::Float,
    {
        let mut host_param = vec![T::zero(); param.len()];
        self.copy_to_host(param, &mut host_param);
        let mut host_grad = vec![T::zero(); grad.len()];
        self.copy_to_host(grad, &mut host_grad);
        let mut host_m = vec![T::zero(); m.len()];
        self.copy_to_host(m, &mut host_m);
        let mut host_v = vec![T::zero(); v.len()];
        self.copy_to_host(v, &mut host_v);

        let seq = coeus_core::SequentialBackend::new();
        let mut seq_param = coeus_core::CpuStorage::from_slice(&host_param);
        let seq_grad = coeus_core::CpuStorage::from_slice(&host_grad);
        let mut seq_m = coeus_core::CpuStorage::from_slice(&host_m);
        let mut seq_v = coeus_core::CpuStorage::from_slice(&host_v);

        coeus_ops::BackendOps::adamw_step(
            &seq,
            &mut seq_param,
            param_layout,
            &seq_grad,
            grad_layout,
            &mut seq_m,
            m_layout,
            &mut seq_v,
            v_layout,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t,
        );

        use coeus_core::CpuAddressableStorage;
        self.copy_to_device(seq_param.as_slice(), param);
        self.copy_to_device(seq_m.as_slice(), m);
        self.copy_to_device(seq_v.as_slice(), v);
    }
}
