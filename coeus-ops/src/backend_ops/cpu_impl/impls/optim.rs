use super::super::optim;
use super::super::CpuBackend;
use crate::backend_ops::traits::OptimizerOps;
use coeus_core::{CpuAddressableStorageMut, Layout, Scalar};

#[allow(clippy::too_many_arguments)]
impl<T: Scalar + leto_ops::Scalar, B: CpuBackend> OptimizerOps<T> for B
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    #[inline]
    fn sgd_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        velocity: &mut Self::DeviceBuffer<T>,
        velocity_layout: &Layout,
        lr: T,
        momentum: T,
    ) {
        optim::sgd_step(
            self,
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

    #[inline]
    fn adam_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        m: &mut Self::DeviceBuffer<T>,
        m_layout: &Layout,
        v: &mut Self::DeviceBuffer<T>,
        v_layout: &Layout,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        t: usize,
    ) where
        T: coeus_core::Float,
    {
        optim::adam_step(
            self,
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

    #[inline]
    fn rmsprop_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        v: &mut Self::DeviceBuffer<T>,
        v_layout: &Layout,
        lr: T,
        alpha: T,
        eps: T,
    ) {
        optim::rmsprop_step(
            self,
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

    #[inline]
    fn adamw_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        m: &mut Self::DeviceBuffer<T>,
        m_layout: &Layout,
        v: &mut Self::DeviceBuffer<T>,
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
        optim::adamw_step(
            self,
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

    #[inline]
    fn adagrad_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        history: &mut Self::DeviceBuffer<T>,
        history_layout: &Layout,
        lr: T,
        eps: T,
    ) where
        T: coeus_core::Float,
    {
        optim::adagrad_step(
            self,
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
}
