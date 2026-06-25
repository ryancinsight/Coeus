// ── Autograd nodes: axis reductions (max_axis, min_axis, log_sum_exp) ──
//
// `cumsum` is implemented in shape.rs (tracked, backward = suffix_sum).
//
// # max_axis / min_axis
//   Forward: standard max/min reduction, output axis dim = 1.
//   Backward: indicator gradient — 1 at max/min positions, 0 elsewhere.
//   Ties: gradient is split equally (sum-normalised mask), matching PyTorch's
//   `max` backward convention.
//   Mask construction:
//     diff = abs(x - broadcast(max, input_shape))
//     mask_raw = ReluGrad(eps - diff)   →  1 where diff < eps (exact max match)
//     tie_count = sum_axis(mask_raw, axis)
//     mask = mask_raw / broadcast(tie_count, input_shape)
//     grad_in = broadcast(grad_out, input_shape) * mask
//   eps = T::from_f64(1e-30) — exploits the fact that floating-point max is exact.
//
// # log_sum_exp
//   Composed entirely from existing tracked ops — no new backward node needed.
//     lse(x, axis) = log(sum(exp(x − max_axis(x)), axis)) + max_axis(x)
//   max-subtraction stabilises the exp range to (−∞, 0] preventing overflow.
//   Gradient flows through max_axis → sub → exp → sum_axis → log → add.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::ops::activation::{exp, log};
use crate::ops::arithmetic::{add, sub, sum_axis};
use crate::var::Var;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float, FloatOps, Scalar};
use coeus_tensor::Tensor;
use std::{ops::Neg, sync::Arc};

// ── MaxAxisNode ────────────────────────────────────────────────────────────

/// Bespoke autograd node for `max_axis`.
pub struct MaxAxisNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    /// Copy of the input tensor stored for backward mask construction.
    pub input_tensor: Tensor<T, B>,
    /// Forward output (max values along axis), shape has axis dim = 1.
    pub max_tensor: Tensor<T, B>,
    pub axis: usize,
}

impl<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for MaxAxisNode<T, B>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "max_axis"
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

        // 1. Broadcast max_tensor (axis dim = 1) back to full input shape.
        let max_broad = self.max_tensor.broadcast(self.input_tensor.shape_cloned());

        // 2. Build indicator mask: 1 at max positions, 0 elsewhere.
        //    diff = abs(x - max_broad)  →  0 exactly at max positions.
        //    ReluGrad(eps - diff): 1 where diff < eps (i.e. at exact max), 0 otherwise.
        let eps = Tensor::full_on(self.input_tensor.shape(), T::from_f64(1e-30), &backend);
        let diff = coeus_ops::sub(&self.input_tensor, &max_broad, &backend);
        let abs_diff = coeus_ops::abs(&diff, &backend);
        let shifted = coeus_ops::sub(&eps, &abs_diff, &backend);
        let mask_raw =
            coeus_ops::elementwise_unary(&shifted, &backend, coeus_ops::UnaryOp::ReluGrad);

        // 3. Tie-normalise: divide mask by the count of max-valued elements along axis.
        let tie_count = coeus_ops::sum_axis(&mask_raw, self.axis, &backend);
        let tie_broad = tie_count.broadcast(self.input_tensor.shape_cloned());
        let mask = coeus_ops::div(&mask_raw, &tie_broad, &backend);

        // 4. Broadcast grad_out to input shape and apply mask.
        let grad_broad = grad_out.broadcast(self.input_tensor.shape_cloned());
        let grad_in = coeus_ops::mul(&grad_broad, &mask, &backend);

        let lock = g.write();
        coeus_ops::add_assign(lock, &grad_in, &backend);
    }
}

/// Tracked `max_axis`: maximum over `axis`, output has that axis = 1.
///
/// Backward: indicator gradient distributed equally across tied maxima.
#[inline]
pub fn max_axis<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    axis: usize,
) -> Var<T, B> {
    let backend = B::default();
    let out_tensor = coeus_ops::max_axis(&a.tensor, axis, &backend);

    let requires_grad = a.grad.is_some();
    let grad = requires_grad.then(|| {
        Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )))
    });

    let creator = requires_grad.then(|| {
        let output_grad = grad.as_ref().unwrap().clone();
        Arc::new(MaxAxisNode {
            output_grad,
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            max_tensor: out_tensor.clone(),
            axis,
        }) as Arc<dyn BackwardNode<T, B>>
    });
    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}

// ── MinAxisNode ────────────────────────────────────────────────────────────

/// Bespoke autograd node for `min_axis`.
pub struct MinAxisNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub input_tensor: Tensor<T, B>,
    pub min_tensor: Tensor<T, B>,
    pub axis: usize,
}

impl<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for MinAxisNode<T, B>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "min_axis"
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

        // Identical logic to MaxAxisNode but using min_tensor.
        let min_broad = self.min_tensor.broadcast(self.input_tensor.shape_cloned());
        let eps = Tensor::full_on(self.input_tensor.shape(), T::from_f64(1e-30), &backend);
        let diff = coeus_ops::sub(&self.input_tensor, &min_broad, &backend);
        let abs_diff = coeus_ops::abs(&diff, &backend);
        let shifted = coeus_ops::sub(&eps, &abs_diff, &backend);
        let mask_raw =
            coeus_ops::elementwise_unary(&shifted, &backend, coeus_ops::UnaryOp::ReluGrad);
        let tie_count = coeus_ops::sum_axis(&mask_raw, self.axis, &backend);
        let tie_broad = tie_count.broadcast(self.input_tensor.shape_cloned());
        let mask = coeus_ops::div(&mask_raw, &tie_broad, &backend);
        let grad_broad = grad_out.broadcast(self.input_tensor.shape_cloned());
        let grad_in = coeus_ops::mul(&grad_broad, &mask, &backend);

        let lock = g.write();
        coeus_ops::add_assign(lock, &grad_in, &backend);
    }
}

/// Tracked `min_axis`: minimum over `axis`, output has that axis = 1.
///
/// Backward: indicator gradient distributed equally across tied minima.
#[inline]
pub fn min_axis<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    axis: usize,
) -> Var<T, B> {
    let backend = B::default();
    let out_tensor = coeus_ops::min_axis(&a.tensor, axis, &backend);

    let requires_grad = a.grad.is_some();
    let grad = requires_grad.then(|| {
        Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )))
    });

    let creator = requires_grad.then(|| {
        let output_grad = grad.as_ref().unwrap().clone();
        Arc::new(MinAxisNode {
            output_grad,
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            min_tensor: out_tensor.clone(),
            axis,
        }) as Arc<dyn BackwardNode<T, B>>
    });
    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}

// ── log_sum_exp ────────────────────────────────────────────────────────────
//
// Numerically stable log-sum-exp along `axis`.
//
// Composed from existing tracked ops so that grad flows through the DAG
// automatically without a bespoke node:
//
//   x_max  = max_axis(x, axis)             → tracked, broadcasts back as size-1 dim
//   x_sh   = x - x_max                     → stable shifted input, exp in (-inf, 0]
//   sum_e  = sum_axis(exp(x_sh), axis)      → sum of stabilised exponentials
//   lse    = log(sum_e) + x_max             → add back the max offset
//
// The backward through log and exp reproduces the softmax probabilities, so
// d(lse)/dx_i = softmax(x)_i — correct by composition.

/// Tracked numerically stable log-sum-exp along `axis`.
///
/// Output shape equals input shape with `axis` dimension reduced to 1.
/// No new backward node — gradients flow through the composed tracked ops.
///
/// Precision: all computation in native `T` precision; max-subtraction
/// constrains exp input to (−∞, 0], eliminating overflow at any precision.
#[inline]
pub fn log_sum_exp<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    axis: usize,
) -> Var<T, B> {
    let x_max = max_axis(x, axis); // shape: [..., 1, ...]
    let x_shifted = sub(x, &x_max); // broadcasts axis=1 dim over x
    let exp_sh = exp(&x_shifted);
    let sum_exp = sum_axis(&exp_sh, axis);
    let log_sum = log(&sum_exp);
    add(&log_sum, &x_max) // adds back max offset
}

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

        // grad_in = grad_out * x / y   (broadcast grad_out and norm_tensor to input shape)
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
    let norm_val = coeus_ops::norm(&a.tensor, &backend);
    let out_tensor = Tensor::full_on([1], norm_val, &backend);

    let requires_grad = a.grad.is_some();
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
                let abs_x = host[i].abs();
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

    let requires_grad = a.grad.is_some();
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

        // Materialise input and norm tensors to contiguous host arrays.
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

        // Also read grad_out into host.
        let mut grad_host = vec![T::zero(); norm_n];
        backend.copy_to_host(
            grad_out
                .to_contiguous_on(&backend)
                .reshape(norm_contig.shape().to_vec())
                .storage(),
            &mut grad_host,
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
                let grad_j = grad_host[out_idx];
                if y_j == T::zero() || grad_j == T::zero() {
                    continue;
                }
                let scale = y_j.powf(T::one() - p) * grad_j;
                let base = pre_idx * (axis_dim * post_count) + post_idx;
                for k in 0..axis_dim {
                    let linear = base + k * post_count;
                    let val = host[linear];
                    let abs_x = val.abs();
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

    let requires_grad = a.grad.is_some();
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
