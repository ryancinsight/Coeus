use super::*;

pub(super) fn dispatch_sgd_step<T: WgpuScalar + coeus_core::Float>(
    param: &mut crate::backend::WgpuStorage<T>,
    param_layout: &Layout,
    grad: &crate::backend::WgpuStorage<T>,
    grad_layout: &Layout,
    velocity: &mut crate::backend::WgpuStorage<T>,
    velocity_layout: &Layout,
    lr: T,
    momentum: T,
) {
    let len = param_layout.shape().iter().product::<usize>();
    kernels::dispatch_sgd_step::<T>(
        &param.buffer,
        param_layout,
        &grad.buffer,
        grad_layout,
        &velocity.buffer,
        velocity_layout,
        lr,
        momentum,
        len,
    );
}

pub(super) fn dispatch_adam_step<T: WgpuScalar + coeus_core::Float>(
    param: &mut crate::backend::WgpuStorage<T>,
    param_layout: &Layout,
    grad: &crate::backend::WgpuStorage<T>,
    grad_layout: &Layout,
    m: &mut crate::backend::WgpuStorage<T>,
    m_layout: &Layout,
    v: &mut crate::backend::WgpuStorage<T>,
    v_layout: &Layout,
    lr: T,
    beta1: T,
    beta2: T,
    eps: T,
    t: usize,
) {
    let len = param_layout.shape().iter().product::<usize>();
    kernels::dispatch_adam_step::<T>(
        &param.buffer,
        param_layout,
        &grad.buffer,
        grad_layout,
        &m.buffer,
        m_layout,
        &v.buffer,
        v_layout,
        lr,
        beta1,
        beta2,
        eps,
        t,
        len,
    );
}

pub(super) fn dispatch_rmsprop_step<T: WgpuScalar + coeus_core::Float>(
    param: &mut crate::backend::WgpuStorage<T>,
    param_layout: &Layout,
    grad: &crate::backend::WgpuStorage<T>,
    grad_layout: &Layout,
    v: &mut crate::backend::WgpuStorage<T>,
    v_layout: &Layout,
    lr: T,
    alpha: T,
    eps: T,
) {
    let len = param_layout.shape().iter().product::<usize>();
    kernels::dispatch_rmsprop_step::<T>(
        &param.buffer,
        param_layout,
        &grad.buffer,
        grad_layout,
        &v.buffer,
        v_layout,
        lr,
        alpha,
        eps,
        len,
    );
}

pub(super) fn dispatch_adamw_step<T: WgpuScalar + coeus_core::Float>(
    param: &mut crate::backend::WgpuStorage<T>,
    param_layout: &Layout,
    grad: &crate::backend::WgpuStorage<T>,
    grad_layout: &Layout,
    m: &mut crate::backend::WgpuStorage<T>,
    m_layout: &Layout,
    v: &mut crate::backend::WgpuStorage<T>,
    v_layout: &Layout,
    lr: T,
    beta1: T,
    beta2: T,
    eps: T,
    weight_decay: T,
    t: usize,
) {
    let len = param_layout.shape().iter().product::<usize>();
    kernels::dispatch_adamw_step::<T>(
        &param.buffer,
        param_layout,
        &grad.buffer,
        grad_layout,
        &m.buffer,
        m_layout,
        &v.buffer,
        v_layout,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        t,
        len,
    );
}

pub(super) fn dispatch_adagrad_step<T: WgpuScalar + coeus_core::Float>(
    param: &mut crate::backend::WgpuStorage<T>,
    param_layout: &Layout,
    grad: &crate::backend::WgpuStorage<T>,
    grad_layout: &Layout,
    history: &mut crate::backend::WgpuStorage<T>,
    history_layout: &Layout,
    lr: T,
    eps: T,
) {
    let len = param_layout.shape().iter().product::<usize>();
    kernels::dispatch_adagrad_step::<T>(
        &param.buffer,
        param_layout,
        &grad.buffer,
        grad_layout,
        &history.buffer,
        history_layout,
        lr,
        eps,
        len,
    );
}
