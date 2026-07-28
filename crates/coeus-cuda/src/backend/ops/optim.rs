use super::cast::{cast_storage, cast_storage_mut};
use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::error::CudaBackendError;
use crate::kernels;
use crate::storage::CudaStorage;
use coeus_core::Layout;

impl CudaBackend {
    #[allow(clippy::too_many_arguments, clippy::multiple_bound_locations)]
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
    ) -> Result<(), CudaBackendError>
    where
        T: coeus_core::Float,
    {
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let mut param_f32 = cast_storage_mut::<T, f32>(param);
            let grad_f32 = cast_storage::<T, f32>(grad);
            let mut velocity_f32 = cast_storage_mut::<T, f32>(velocity);
            let lr_f32 = coeus_core::Scalar::to_f64(lr) as f32;
            let momentum_f32 = coeus_core::Scalar::to_f64(momentum) as f32;

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
                return Ok(());
            }
        }
        Err(CudaBackendError::dispatch_unavailable(
            "sgd_step",
            "native CUDA kernel compilation or launch failed",
        ))
    }

    #[allow(clippy::too_many_arguments, clippy::multiple_bound_locations)]
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
    ) -> Result<(), CudaBackendError>
    where
        T: coeus_core::Float,
    {
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let mut param_f32 = cast_storage_mut::<T, f32>(param);
            let grad_f32 = cast_storage::<T, f32>(grad);
            let mut m_f32 = cast_storage_mut::<T, f32>(m);
            let mut v_f32 = cast_storage_mut::<T, f32>(v);
            let lr_f32 = coeus_core::Scalar::to_f64(lr) as f32;
            let beta1_f32 = coeus_core::Scalar::to_f64(beta1) as f32;
            let beta2_f32 = coeus_core::Scalar::to_f64(beta2) as f32;
            let eps_f32 = coeus_core::Scalar::to_f64(eps) as f32;

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
                return Ok(());
            }
        }
        Err(CudaBackendError::dispatch_unavailable(
            "adam_step",
            "native CUDA kernel compilation or launch failed",
        ))
    }

    #[allow(clippy::too_many_arguments, clippy::multiple_bound_locations)]
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
    ) -> Result<(), CudaBackendError>
    where
        T: coeus_core::Float,
    {
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let mut param_f32 = cast_storage_mut::<T, f32>(param);
            let grad_f32 = cast_storage::<T, f32>(grad);
            let mut v_f32 = cast_storage_mut::<T, f32>(v);
            let lr_f32 = coeus_core::Scalar::to_f64(lr) as f32;
            let alpha_f32 = coeus_core::Scalar::to_f64(alpha) as f32;
            let eps_f32 = coeus_core::Scalar::to_f64(eps) as f32;

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
                return Ok(());
            }
        }
        Err(CudaBackendError::dispatch_unavailable(
            "rmsprop_step",
            "native CUDA kernel compilation or launch failed",
        ))
    }

    #[allow(clippy::too_many_arguments, clippy::multiple_bound_locations)]
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
    ) -> Result<(), CudaBackendError>
    where
        T: coeus_core::Float,
    {
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let mut param_f32 = cast_storage_mut::<T, f32>(param);
            let grad_f32 = cast_storage::<T, f32>(grad);
            let mut history_f32 = cast_storage_mut::<T, f32>(history);
            let lr_f32 = coeus_core::Scalar::to_f64(lr) as f32;
            let eps_f32 = coeus_core::Scalar::to_f64(eps) as f32;

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
                return Ok(());
            }
        }
        Err(CudaBackendError::dispatch_unavailable(
            "adagrad_step",
            "native CUDA kernel compilation or launch failed",
        ))
    }

    #[allow(clippy::too_many_arguments, clippy::multiple_bound_locations)]
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
    ) -> Result<(), CudaBackendError>
    where
        T: coeus_core::Float,
    {
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let mut param_f32 = cast_storage_mut::<T, f32>(param);
            let grad_f32 = cast_storage::<T, f32>(grad);
            let mut m_f32 = cast_storage_mut::<T, f32>(m);
            let mut v_f32 = cast_storage_mut::<T, f32>(v);
            let lr_f32 = coeus_core::Scalar::to_f64(lr) as f32;
            let beta1_f32 = coeus_core::Scalar::to_f64(beta1) as f32;
            let beta2_f32 = coeus_core::Scalar::to_f64(beta2) as f32;
            let eps_f32 = coeus_core::Scalar::to_f64(eps) as f32;
            let weight_decay_f32 = coeus_core::Scalar::to_f64(weight_decay) as f32;

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
                return Ok(());
            }
        }
        Err(CudaBackendError::dispatch_unavailable(
            "adamw_step",
            "native CUDA kernel compilation or launch failed",
        ))
    }
}
