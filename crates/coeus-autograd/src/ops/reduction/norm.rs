// ── Autograd nodes: norm reductions (norm, norm_p, norm_p_axis) ──

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float, Scalar};
use coeus_tensor::Tensor;
use std::ops::Neg;
use std::sync::Arc;

// ── NormNode (L2 short-circuit) ────────────────────────────────────────────

/// Bespoke autograd node for `norm` (L2 norm over all elements, scalar output).
///
/// Forward: `y = sqrt(sum(x_i²))`, output shape `[1]`.
/// Backward: `∂y/∂x_i = x_i / y`, computed as tensor `div` + `mul` so all work
/// stays on the backend (no host-side copy).
pub struct NormNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    /// Input tensor saved for backward.
    pub input_tensor: Tensor<T, B>,
    /// Forward output (scalar norm as `[1]` tensor).
    pub norm_tensor: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for NormNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "norm"
    }
    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        let Some(Some(ref g)) = input_grads.first() else {
            return;
        };

        let norm_broad = self.norm_tensor.broadcast(self.input_tensor.shape_cloned());
        let scale = coeus_ops::div(grad_out, &norm_broad, &backend);
        let grad_in = coeus_ops::mul(&scale, &self.input_tensor, &backend);

        let lock = g.write();
        coeus_ops::add_assign(lock, &grad_in, &backend);
    }
}

/// Tracked L2 norm over all elements, output shape `[1]`.
///
/// Forward uses the efficient `mul` + `sum` + `sqrt` backend path (no
/// host-side fold). Backward: `∂y/∂x_i = x_i / y`.
#[inline]
pub fn norm<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    let backend = B::default();
    let norm_val = coeus_ops::norm(&a.tensor, &backend).expect("norm");
    let out_tensor = Tensor::full_on([1], norm_val, &backend);

    let requires_grad = crate::grad_mode::should_track_var(a);
    let grad = requires_grad.then(|| {
        Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )))
    });

    let creator = requires_grad.then(|| {
        let output_grad = grad.as_ref().unwrap().clone();
        Arc::new(NormNode {
            output_grad,
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            norm_tensor: out_tensor.clone(),
        }) as Arc<dyn BackwardNode<T, B>>
    });
    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}

// ── NormPNode (general Lp norm, scalar output) ─────────────────────────────

/// Bespoke autograd node for `norm_p` (general Lp norm, scalar `[1]` output).
///
/// Forward: `y = (Σ|xᵢ|^p)^(1/p)`, output shape `[1]`.
/// Backward: `∂y/∂x_i = y^(1-p) * |xᵢ|^(p-1) * sign(xᵢ)`, computed as a
/// host-side fold since `T::powf` is not available as a tensor op.
pub struct NormPNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub input_tensor: Tensor<T, B>,
    pub p: T,
    /// Scalar norm value (forward output).
    pub norm_value: T,
}

impl<T: Float + Neg<Output = T>, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for NormPNode<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "norm_p"
    }
    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        let Some(Some(ref g)) = input_grads.first() else {
            return;
        };

        let n = self.input_tensor.numel();
        let grad_val = grad_out.to_contiguous_on(&backend).as_slice()[0];

        let input_contig =
            if self.input_tensor.is_contiguous() && self.input_tensor.layout().offset() == 0 {
                self.input_tensor.reshape([n])
            } else {
                self.input_tensor.to_contiguous_on(&backend).reshape([n])
            };
        let mut host = vec![T::zero(); n];
        backend.copy_to_host(input_contig.storage(), &mut host);

        let y = self.norm_value;
        let p = self.p;
        let mut grad_host = vec![T::zero(); n];

        if y != T::zero() {
            let scale = y.powf(T::one() - p) * grad_val;
            for i in 0..n {
                let abs_x = <T as Float>::abs(host[i]);
                if abs_x != T::zero() {
                    let sign = if host[i] > T::zero() {
                        T::one()
                    } else {
                        -T::one()
                    };
                    grad_host[i] = scale * abs_x.powf(p - T::one()) * sign;
                }
            }
        }

        let grad_t = Tensor::from_slice(self.input_tensor.shape().to_vec(), &grad_host);
        let lock = g.write();
        coeus_ops::add_assign(lock, &grad_t, &backend);
    }
}

/// Tracked general Lp norm over all elements, output shape `[1]`.
///
/// Matches `coeus_ops::norm_p` but returns a `[1]` tensor for autograd.
#[inline]
pub fn norm_p<T: Float + Neg<Output = T>, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    p: T,
) -> Var<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let norm_val = coeus_ops::norm_p(&a.tensor, p, &backend);
    let out_tensor = Tensor::full_on([1], norm_val, &backend);

    let requires_grad = crate::grad_mode::should_track_var(a);
    let grad = requires_grad.then(|| {
        Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )))
    });

    let creator = requires_grad.then(|| {
        let output_grad = grad.as_ref().unwrap().clone();
        Arc::new(NormPNode {
            output_grad,
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            p,
            norm_value: norm_val,
        }) as Arc<dyn BackwardNode<T, B>>
    });
    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}

// ── NormPAxisNode (per-axis Lp norm) ───────────────────────────────────────

/// Bespoke autograd node for `norm_p_axis`.
///
/// Forward: per-axis `(Σ|xⱼₖ|^p)^(1/p)` where `k` indexes the reduced axis.
/// Backward: `∂yⱼ/∂xⱼₖ = yⱼ^(1-p) * |xⱼₖ|^(p-1) * sign(xⱼₖ)`, computed as a
/// host-side fold.
pub struct NormPAxisNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub input_tensor: Tensor<T, B>,
    pub p: T,
    pub axis: usize,
    /// Forward output tensor (norm values, axis dim = 1).
    pub norm_tensor: Tensor<T, B>,
}

impl<T: Float + Neg<Output = T>, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for NormPAxisNode<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "norm_p_axis"
    }
    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        let Some(Some(ref g)) = input_grads.first() else {
            return;
        };

        let n = self.input_tensor.numel();
        let input_contig =
            if self.input_tensor.is_contiguous() && self.input_tensor.layout().offset() == 0 {
                self.input_tensor
                    .reshape(self.input_tensor.shape().to_vec())
            } else {
                self.input_tensor.to_contiguous_on(&backend)
            };
        let mut host = vec![T::zero(); n];
        backend.copy_to_host(input_contig.storage(), &mut host);

        let norm_n = self.norm_tensor.numel();
        let norm_contig =
            if self.norm_tensor.is_contiguous() && self.norm_tensor.layout().offset() == 0 {
                self.norm_tensor.reshape(self.norm_tensor.shape().to_vec())
            } else {
                self.norm_tensor.to_contiguous_on(&backend)
            };
        let mut norm_host = vec![T::zero(); norm_n];
        backend.copy_to_host(norm_contig.storage(), &mut norm_host);

        let mut grad_host_vec = vec![T::zero(); norm_n];
        backend.copy_to_host(
            grad_out
                .to_contiguous_on(&backend)
                .reshape(norm_contig.shape().to_vec())
                .storage(),
            &mut grad_host_vec,
        );

        let p = self.p;
        let axis = self.axis;
        let shape = self.input_tensor.shape();
        let axis_dim = shape[axis];
        let pre_count: usize = shape[..axis].iter().product();
        let post_count: usize = shape[axis + 1..].iter().product();

        let mut grad_in_host = vec![T::zero(); n];

        for pre_idx in 0..pre_count {
            for post_idx in 0..post_count {
                let out_idx = pre_idx * post_count + post_idx;
                let y_j = norm_host[out_idx];
                let grad_j = grad_host_vec[out_idx];
                if y_j == T::zero() || grad_j == T::zero() {
                    continue;
                }
                let scale = y_j.powf(T::one() - p) * grad_j;
                let base = pre_idx * (axis_dim * post_count) + post_idx;
                for k in 0..axis_dim {
                    let linear = base + k * post_count;
                    let val = host[linear];
                    let abs_x = <T as Float>::abs(val);
                    if abs_x == T::zero() {
                        continue;
                    }
                    let sign = if val > T::zero() { T::one() } else { -T::one() };
                    grad_in_host[linear] = scale * abs_x.powf(p - T::one()) * sign;
                }
            }
        }

        let grad_t = Tensor::from_slice(self.input_tensor.shape().to_vec(), &grad_in_host);
        let lock = g.write();
        coeus_ops::add_assign(lock, &grad_t, &backend);
    }
}

/// Tracked per-axis Lp norm, output has `axis` reduced to size 1.
#[inline]
pub fn norm_p_axis<T: Float + Neg<Output = T>, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    p: T,
    axis: usize,
) -> Var<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let out_tensor = coeus_ops::norm_p_axis(&a.tensor, p, axis, &backend);

    let requires_grad = crate::grad_mode::should_track_var(a);
    let grad = requires_grad.then(|| {
        Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )))
    });

    let creator = requires_grad.then(|| {
        let output_grad = grad.as_ref().unwrap().clone();
        Arc::new(NormPAxisNode {
            output_grad,
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            p,
            axis,
            norm_tensor: out_tensor.clone(),
        }) as Arc<dyn BackwardNode<T, B>>
    });
    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
