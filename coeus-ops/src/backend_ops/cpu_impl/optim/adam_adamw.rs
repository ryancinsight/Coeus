use crate::ptr::{MutPtr, Ptr};
use coeus_core::FloatOps;
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};

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
