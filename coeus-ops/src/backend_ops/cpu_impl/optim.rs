use crate::ptr::{MutPtr, Ptr};
use coeus_core::FloatOps;
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};

pub fn sgd_step<T: Scalar, B: Backend>(
    backend: &B,
    param: &mut B::DeviceBuffer<T>,
    param_layout: &Layout,
    grad: &B::DeviceBuffer<T>,
    grad_layout: &Layout,
    velocity: &mut B::DeviceBuffer<T>,
    velocity_layout: &Layout,
    lr: T,
    momentum: T,
) where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let numel = param_layout.numel();
    assert_eq!(numel, grad_layout.numel());
    assert_eq!(numel, velocity_layout.numel());

    let p_slice = param.as_mut_slice();
    let g_slice = grad.as_slice();
    let v_slice = velocity.as_mut_slice();

    let p_ptr = MutPtr(p_slice.as_mut_ptr());
    let g_ptr = Ptr(g_slice.as_ptr());
    let v_ptr = MutPtr(v_slice.as_mut_ptr());

    let p_off = param_layout.offset();
    let g_off = grad_layout.offset();
    let v_off = velocity_layout.offset();

    let is_contiguous = param_layout.is_contiguous()
        && grad_layout.is_contiguous()
        && velocity_layout.is_contiguous();

    if is_contiguous {
        backend.parallel_for(0, numel, move |i| unsafe {
            let g = g_ptr.read(g_off + i);
            let v = v_ptr.read(v_off + i) * momentum + g;
            v_ptr.write(v_off + i, v);
            p_ptr.write(p_off + i, p_ptr.read(p_off + i) - lr * v);
        });
    } else {
        let ndim = param_layout.ndim();
        let p_shape = param_layout.shape_cloned();
        let p_strides = param_layout.strides_cloned();
        let g_shape = grad_layout.shape_cloned();
        let g_strides = grad_layout.strides_cloned();
        let v_shape = velocity_layout.shape_cloned();
        let v_strides = velocity_layout.strides_cloned();

        backend.parallel_for(0, numel, move |i| {
            let mut temp = i;
            let mut coords = smallvec::SmallVec::<[usize; 4]>::from_elem(0, ndim);
            for d in (0..ndim).rev() {
                coords[d] = temp % p_shape[d];
                temp /= p_shape[d];
            }

            let mut pi = p_off;
            let mut gi = g_off;
            let mut vi = v_off;

            for d in 0..ndim {
                if d < p_shape.len() && p_shape[d] > 1 {
                    pi += coords[d] * p_strides[d];
                }
                if d < g_shape.len() && g_shape[d] > 1 {
                    gi += coords[d] * g_strides[d];
                }
                if d < v_shape.len() && v_shape[d] > 1 {
                    vi += coords[d] * v_strides[d];
                }
            }

            unsafe {
                let g = g_ptr.read(gi);
                let v = v_ptr.read(vi) * momentum + g;
                v_ptr.write(vi, v);
                p_ptr.write(pi, p_ptr.read(pi) - lr * v);
            }
        });
    }
}

pub fn adam_step<T: Scalar + FloatOps, B: Backend>(
    backend: &B,
    param: &mut B::DeviceBuffer<T>,
    param_layout: &Layout,
    grad: &B::DeviceBuffer<T>,
    grad_layout: &Layout,
    m: &mut B::DeviceBuffer<T>,
    m_layout: &Layout,
    v: &mut B::DeviceBuffer<T>,
    v_layout: &Layout,
    lr: T,
    beta1: T,
    beta2: T,
    eps: T,
    t: usize,
) where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let numel = param_layout.numel();
    assert_eq!(numel, grad_layout.numel());
    assert_eq!(numel, m_layout.numel());
    assert_eq!(numel, v_layout.numel());

    let p_slice = param.as_mut_slice();
    let g_slice = grad.as_slice();
    let m_slice = m.as_mut_slice();
    let v_slice = v.as_mut_slice();

    let p_ptr = MutPtr(p_slice.as_mut_ptr());
    let g_ptr = Ptr(g_slice.as_ptr());
    let m_ptr = MutPtr(m_slice.as_mut_ptr());
    let v_ptr = MutPtr(v_slice.as_mut_ptr());

    let p_off = param_layout.offset();
    let g_off = grad_layout.offset();
    let m_off = m_layout.offset();
    let v_off = v_layout.offset();

    let bias_correction1 = T::one() - (beta1.log_op() * T::from_f64(t as f64)).exp_op();
    let bias_correction2 = T::one() - (beta2.log_op() * T::from_f64(t as f64)).exp_op();

    let is_contiguous = param_layout.is_contiguous()
        && grad_layout.is_contiguous()
        && m_layout.is_contiguous()
        && v_layout.is_contiguous();

    if is_contiguous {
        backend.parallel_for(0, numel, move |i| unsafe {
            let g = g_ptr.read(g_off + i);
            let m_val = m_ptr.read(m_off + i) * beta1 + (T::one() - beta1) * g;
            let v_val = v_ptr.read(v_off + i) * beta2 + (T::one() - beta2) * g * g;

            m_ptr.write(m_off + i, m_val);
            v_ptr.write(v_off + i, v_val);

            let m_hat = m_val / bias_correction1;
            let v_hat = v_val / bias_correction2;
            let denom = v_hat.sqrt_val() + eps;
            p_ptr.write(p_off + i, p_ptr.read(p_off + i) - lr * m_hat / denom);
        });
    } else {
        let ndim = param_layout.ndim();
        let p_shape = param_layout.shape_cloned();
        let p_strides = param_layout.strides_cloned();
        let g_shape = grad_layout.shape_cloned();
        let g_strides = grad_layout.strides_cloned();
        let m_shape = m_layout.shape_cloned();
        let m_strides = m_layout.strides_cloned();
        let v_shape = v_layout.shape_cloned();
        let v_strides = v_layout.strides_cloned();

        backend.parallel_for(0, numel, move |i| {
            let mut temp = i;
            let mut coords = smallvec::SmallVec::<[usize; 4]>::from_elem(0, ndim);
            for d in (0..ndim).rev() {
                coords[d] = temp % p_shape[d];
                temp /= p_shape[d];
            }

            let mut pi = p_off;
            let mut gi = g_off;
            let mut mi = m_off;
            let mut vi = v_off;

            for d in 0..ndim {
                if d < p_shape.len() && p_shape[d] > 1 {
                    pi += coords[d] * p_strides[d];
                }
                if d < g_shape.len() && g_shape[d] > 1 {
                    gi += coords[d] * g_strides[d];
                }
                if d < m_shape.len() && m_shape[d] > 1 {
                    mi += coords[d] * m_strides[d];
                }
                if d < v_shape.len() && v_shape[d] > 1 {
                    vi += coords[d] * v_strides[d];
                }
            }

            unsafe {
                let g = g_ptr.read(gi);
                let m_val = m_ptr.read(mi) * beta1 + (T::one() - beta1) * g;
                let v_val = v_ptr.read(vi) * beta2 + (T::one() - beta2) * g * g;

                m_ptr.write(mi, m_val);
                v_ptr.write(vi, v_val);

                let m_hat = m_val / bias_correction1;
                let v_hat = v_val / bias_correction2;
                let denom = v_hat.sqrt_val() + eps;
                p_ptr.write(pi, p_ptr.read(pi) - lr * m_hat / denom);
            }
        });
    }
}

pub fn rmsprop_step<T: Scalar, B: Backend>(
    backend: &B,
    param: &mut B::DeviceBuffer<T>,
    param_layout: &Layout,
    grad: &B::DeviceBuffer<T>,
    grad_layout: &Layout,
    v: &mut B::DeviceBuffer<T>,
    v_layout: &Layout,
    lr: T,
    alpha: T,
    eps: T,
) where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let numel = param_layout.numel();
    assert_eq!(numel, grad_layout.numel());
    assert_eq!(numel, v_layout.numel());

    let p_slice = param.as_mut_slice();
    let g_slice = grad.as_slice();
    let v_slice = v.as_mut_slice();

    let p_ptr = MutPtr(p_slice.as_mut_ptr());
    let g_ptr = Ptr(g_slice.as_ptr());
    let v_ptr = MutPtr(v_slice.as_mut_ptr());

    let p_off = param_layout.offset();
    let g_off = grad_layout.offset();
    let v_off = v_layout.offset();

    let is_contiguous =
        param_layout.is_contiguous() && grad_layout.is_contiguous() && v_layout.is_contiguous();

    if is_contiguous {
        backend.parallel_for(0, numel, move |i| unsafe {
            let g = g_ptr.read(g_off + i);
            let v_val = v_ptr.read(v_off + i) * alpha + (T::one() - alpha) * g * g;
            v_ptr.write(v_off + i, v_val);

            let denom = v_val.sqrt_val() + eps;
            p_ptr.write(p_off + i, p_ptr.read(p_off + i) - lr * g / denom);
        });
    } else {
        let ndim = param_layout.ndim();
        let p_shape = param_layout.shape_cloned();
        let p_strides = param_layout.strides_cloned();
        let g_shape = grad_layout.shape_cloned();
        let g_strides = grad_layout.strides_cloned();
        let v_shape = v_layout.shape_cloned();
        let v_strides = v_layout.strides_cloned();

        backend.parallel_for(0, numel, move |i| {
            let mut temp = i;
            let mut coords = smallvec::SmallVec::<[usize; 4]>::from_elem(0, ndim);
            for d in (0..ndim).rev() {
                coords[d] = temp % p_shape[d];
                temp /= p_shape[d];
            }

            let mut pi = p_off;
            let mut gi = g_off;
            let mut vi = v_off;

            for d in 0..ndim {
                if d < p_shape.len() && p_shape[d] > 1 {
                    pi += coords[d] * p_strides[d];
                }
                if d < g_shape.len() && g_shape[d] > 1 {
                    gi += coords[d] * g_strides[d];
                }
                if d < v_shape.len() && v_shape[d] > 1 {
                    vi += coords[d] * v_strides[d];
                }
            }

            unsafe {
                let g = g_ptr.read(gi);
                let v_val = v_ptr.read(vi) * alpha + (T::one() - alpha) * g * g;
                v_ptr.write(vi, v_val);

                let denom = v_val.sqrt_val() + eps;
                p_ptr.write(pi, p_ptr.read(pi) - lr * g / denom);
            }
        });
    }
}

/// Fused AdamW (decoupled weight decay) step.
///
/// Algorithm (Loshchilov & Hutter, 2019):
/// 1. m = β₁·m + (1−β₁)·g
/// 2. v = β₂·v + (1−β₂)·g²
/// 3. m̂ = m / (1 − β₁ᵗ),  v̂ = v / (1 − β₂ᵗ)
/// 4. p = p − lr·(m̂/(√v̂+ε) + λ·p)      [decoupled weight-decay λ]
#[allow(clippy::too_many_arguments)]
pub fn adamw_step<T: Scalar + FloatOps, B: Backend>(
    backend: &B,
    param: &mut B::DeviceBuffer<T>,
    param_layout: &Layout,
    grad: &B::DeviceBuffer<T>,
    grad_layout: &Layout,
    m: &mut B::DeviceBuffer<T>,
    m_layout: &Layout,
    v: &mut B::DeviceBuffer<T>,
    v_layout: &Layout,
    lr: T,
    beta1: T,
    beta2: T,
    eps: T,
    weight_decay: T,
    t: usize,
) where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let numel = param_layout.numel();
    assert_eq!(numel, grad_layout.numel());
    assert_eq!(numel, m_layout.numel());
    assert_eq!(numel, v_layout.numel());

    let p_slice = param.as_mut_slice();
    let g_slice = grad.as_slice();
    let m_slice = m.as_mut_slice();
    let v_slice = v.as_mut_slice();

    let p_ptr = MutPtr(p_slice.as_mut_ptr());
    let g_ptr = Ptr(g_slice.as_ptr());
    let m_ptr = MutPtr(m_slice.as_mut_ptr());
    let v_ptr = MutPtr(v_slice.as_mut_ptr());

    let p_off = param_layout.offset();
    let g_off = grad_layout.offset();
    let m_off = m_layout.offset();
    let v_off = v_layout.offset();

    let bias_correction1 = T::one() - (beta1.log_op() * T::from_f64(t as f64)).exp_op();
    let bias_correction2 = T::one() - (beta2.log_op() * T::from_f64(t as f64)).exp_op();
    let one_minus_b1 = T::one() - beta1;
    let one_minus_b2 = T::one() - beta2;

    let is_contiguous = param_layout.is_contiguous()
        && grad_layout.is_contiguous()
        && m_layout.is_contiguous()
        && v_layout.is_contiguous();

    if is_contiguous {
        backend.parallel_for(0, numel, move |i| unsafe {
            let g = g_ptr.read(g_off + i);
            let m_val = m_ptr.read(m_off + i) * beta1 + one_minus_b1 * g;
            let v_val = v_ptr.read(v_off + i) * beta2 + one_minus_b2 * g * g;
            m_ptr.write(m_off + i, m_val);
            v_ptr.write(v_off + i, v_val);

            let m_hat = m_val / bias_correction1;
            let v_hat = v_val / bias_correction2;
            let adam_update = lr * m_hat / (v_hat.sqrt_val() + eps);
            let wd_update = lr * weight_decay * p_ptr.read(p_off + i);
            p_ptr.write(p_off + i, p_ptr.read(p_off + i) - adam_update - wd_update);
        });
    } else {
        let ndim = param_layout.ndim();
        let p_shape = param_layout.shape_cloned();
        let p_strides = param_layout.strides_cloned();
        let g_shape = grad_layout.shape_cloned();
        let g_strides = grad_layout.strides_cloned();
        let m_shape = m_layout.shape_cloned();
        let m_strides = m_layout.strides_cloned();
        let v_shape = v_layout.shape_cloned();
        let v_strides = v_layout.strides_cloned();

        backend.parallel_for(0, numel, move |i| {
            let mut temp = i;
            let mut coords = smallvec::SmallVec::<[usize; 4]>::from_elem(0, ndim);
            for d in (0..ndim).rev() {
                coords[d] = temp % p_shape[d];
                temp /= p_shape[d];
            }

            let mut pi = p_off;
            let mut gi = g_off;
            let mut mi = m_off;
            let mut vi = v_off;

            for d in 0..ndim {
                if d < p_shape.len() && p_shape[d] > 1 {
                    pi += coords[d] * p_strides[d];
                }
                if d < g_shape.len() && g_shape[d] > 1 {
                    gi += coords[d] * g_strides[d];
                }
                if d < m_shape.len() && m_shape[d] > 1 {
                    mi += coords[d] * m_strides[d];
                }
                if d < v_shape.len() && v_shape[d] > 1 {
                    vi += coords[d] * v_strides[d];
                }
            }

            unsafe {
                let g = g_ptr.read(gi);
                let m_val = m_ptr.read(mi) * beta1 + one_minus_b1 * g;
                let v_val = v_ptr.read(vi) * beta2 + one_minus_b2 * g * g;
                m_ptr.write(mi, m_val);
                v_ptr.write(vi, v_val);

                let m_hat = m_val / bias_correction1;
                let v_hat = v_val / bias_correction2;
                let adam_update = lr * m_hat / (v_hat.sqrt_val() + eps);
                let wd_update = lr * weight_decay * p_ptr.read(pi);
                p_ptr.write(pi, p_ptr.read(pi) - adam_update - wd_update);
            }
        });
    }
}

pub fn adagrad_step<T: Scalar, B: Backend>(
    backend: &B,
    param: &mut B::DeviceBuffer<T>,
    param_layout: &Layout,
    grad: &B::DeviceBuffer<T>,
    grad_layout: &Layout,
    history: &mut B::DeviceBuffer<T>,
    history_layout: &Layout,
    lr: T,
    eps: T,
) where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let numel = param_layout.numel();
    assert_eq!(numel, grad_layout.numel());
    assert_eq!(numel, history_layout.numel());

    let p_slice = param.as_mut_slice();
    let g_slice = grad.as_slice();
    let h_slice = history.as_mut_slice();

    let p_ptr = MutPtr(p_slice.as_mut_ptr());
    let g_ptr = Ptr(g_slice.as_ptr());
    let h_ptr = MutPtr(h_slice.as_mut_ptr());

    let p_off = param_layout.offset();
    let g_off = grad_layout.offset();
    let h_off = history_layout.offset();

    let is_contiguous = param_layout.is_contiguous()
        && grad_layout.is_contiguous()
        && history_layout.is_contiguous();

    if is_contiguous {
        backend.parallel_for(0, numel, move |i| unsafe {
            let g = g_ptr.read(g_off + i);
            let h = h_ptr.read(h_off + i) + g * g;
            h_ptr.write(h_off + i, h);
            p_ptr.write(
                p_off + i,
                p_ptr.read(p_off + i) - lr * g / (h.sqrt_val() + eps),
            );
        });
    } else {
        let ndim = param_layout.ndim();
        let p_shape = param_layout.shape_cloned();
        let p_strides = param_layout.strides_cloned();
        let g_shape = grad_layout.shape_cloned();
        let g_strides = grad_layout.strides_cloned();
        let h_shape = history_layout.shape_cloned();
        let h_strides = history_layout.strides_cloned();

        backend.parallel_for(0, numel, move |i| {
            let mut temp = i;
            let mut coords = smallvec::SmallVec::<[usize; 4]>::from_elem(0, ndim);
            for d in (0..ndim).rev() {
                coords[d] = temp % p_shape[d];
                temp /= p_shape[d];
            }

            let mut pi = p_off;
            let mut gi = g_off;
            let mut hi = h_off;

            for d in 0..ndim {
                if d < p_shape.len() && p_shape[d] > 1 {
                    pi += coords[d] * p_strides[d];
                }
                if d < g_shape.len() && g_shape[d] > 1 {
                    gi += coords[d] * g_strides[d];
                }
                if d < h_shape.len() && h_shape[d] > 1 {
                    hi += coords[d] * h_strides[d];
                }
            }

            unsafe {
                let g = g_ptr.read(gi);
                let h = h_ptr.read(hi) + g * g;
                h_ptr.write(hi, h);
                p_ptr.write(pi, p_ptr.read(pi) - lr * g / (h.sqrt_val() + eps));
            }
        });
    }
}
