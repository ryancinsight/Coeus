use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::kernels;
use crate::storage::CudaStorage;
use coeus_core::Layout;

fn cast_storage<T, U>(storage: &CudaStorage<T>) -> CudaStorage<U> {
    CudaStorage {
        buffer: storage.buffer.clone(),
        len: storage.len,
        _marker: std::marker::PhantomData,
    }
}

fn cast_storage_mut<T, U>(storage: &mut CudaStorage<T>) -> CudaStorage<U> {
    CudaStorage {
        buffer: storage.buffer.clone(),
        len: storage.len,
        _marker: std::marker::PhantomData,
    }
}

impl CudaBackend {
    pub(crate) fn cuda_sgd_step<T: CudaScalar>(
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
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let mut param_f32 = cast_storage_mut::<T, f32>(param);
            let grad_f32 = cast_storage::<T, f32>(grad);
            let mut velocity_f32 = cast_storage_mut::<T, f32>(velocity);
            let lr_f32 = lr.to_f64() as f32;
            let momentum_f32 = momentum.to_f64() as f32;

            if kernels::launch_sgd_step(
                &mut param_f32,
                param_layout,
                &grad_f32,
                grad_layout,
                &mut velocity_f32,
                velocity_layout,
                lr_f32,
                momentum_f32,
            ) {
                return;
            }
        }
        self.fallback_sgd_step(
            param,
            param_layout,
            grad,
            grad_layout,
            velocity,
            velocity_layout,
            lr,
            momentum,
        );
    }

    pub(crate) fn cuda_adam_step<T: CudaScalar>(
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
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let mut param_f32 = cast_storage_mut::<T, f32>(param);
            let grad_f32 = cast_storage::<T, f32>(grad);
            let mut m_f32 = cast_storage_mut::<T, f32>(m);
            let mut v_f32 = cast_storage_mut::<T, f32>(v);
            let lr_f32 = lr.to_f64() as f32;
            let beta1_f32 = beta1.to_f64() as f32;
            let beta2_f32 = beta2.to_f64() as f32;
            let eps_f32 = eps.to_f64() as f32;

            if kernels::launch_adam_step(
                &mut param_f32,
                param_layout,
                &grad_f32,
                grad_layout,
                &mut m_f32,
                m_layout,
                &mut v_f32,
                v_layout,
                lr_f32,
                beta1_f32,
                beta2_f32,
                eps_f32,
                t,
            ) {
                return;
            }
        }
        self.fallback_adam_step(
            param,
            param_layout,
            grad,
            grad_layout,
            m,
            m_layout,
            v,
            v_layout,
            lr,
            beta1,
            beta2,
            eps,
            t,
        );
    }

    pub(crate) fn cuda_rmsprop_step<T: CudaScalar>(
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
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let mut param_f32 = cast_storage_mut::<T, f32>(param);
            let grad_f32 = cast_storage::<T, f32>(grad);
            let mut v_f32 = cast_storage_mut::<T, f32>(v);
            let lr_f32 = lr.to_f64() as f32;
            let alpha_f32 = alpha.to_f64() as f32;
            let eps_f32 = eps.to_f64() as f32;

            if kernels::launch_rmsprop_step(
                &mut param_f32,
                param_layout,
                &grad_f32,
                grad_layout,
                &mut v_f32,
                v_layout,
                lr_f32,
                alpha_f32,
                eps_f32,
            ) {
                return;
            }
        }
        self.fallback_rmsprop_step(
            param,
            param_layout,
            grad,
            grad_layout,
            v,
            v_layout,
            lr,
            alpha,
            eps,
        );
    }

    pub(crate) fn cuda_adagrad_step<T: CudaScalar>(
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
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let mut param_f32 = cast_storage_mut::<T, f32>(param);
            let grad_f32 = cast_storage::<T, f32>(grad);
            let mut history_f32 = cast_storage_mut::<T, f32>(history);
            let lr_f32 = lr.to_f64() as f32;
            let eps_f32 = eps.to_f64() as f32;

            if kernels::launch_adagrad_step(
                &mut param_f32,
                param_layout,
                &grad_f32,
                grad_layout,
                &mut history_f32,
                history_layout,
                lr_f32,
                eps_f32,
            ) {
                return;
            }
        }
        self.fallback_adagrad_step(
            param,
            param_layout,
            grad,
            grad_layout,
            history,
            history_layout,
            lr,
            eps,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_adamw_step<T: CudaScalar>(
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
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let mut param_f32 = cast_storage_mut::<T, f32>(param);
            let grad_f32 = cast_storage::<T, f32>(grad);
            let mut m_f32 = cast_storage_mut::<T, f32>(m);
            let mut v_f32 = cast_storage_mut::<T, f32>(v);
            let lr_f32 = lr.to_f64() as f32;
            let beta1_f32 = beta1.to_f64() as f32;
            let beta2_f32 = beta2.to_f64() as f32;
            let eps_f32 = eps.to_f64() as f32;
            let weight_decay_f32 = weight_decay.to_f64() as f32;

            if kernels::launch_adamw_step(
                &mut param_f32,
                param_layout,
                &grad_f32,
                grad_layout,
                &mut m_f32,
                m_layout,
                &mut v_f32,
                v_layout,
                lr_f32,
                beta1_f32,
                beta2_f32,
                eps_f32,
                weight_decay_f32,
                t,
            ) {
                return;
            }
        }
        self.fallback_adamw_step(
            param,
            param_layout,
            grad,
            grad_layout,
            m,
            m_layout,
            v,
            v_layout,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t,
        );
    }
}
