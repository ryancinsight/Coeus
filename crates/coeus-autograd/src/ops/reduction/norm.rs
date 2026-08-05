// ── Autograd nodes: norm reductions (norm, norm_p, norm_p_axis) ──

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
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

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        let Some(Some(ref g)) = input_grads.first() else {
            return Ok(());
        };

        let norm_broad = self.norm_tensor.broadcast(self.input_tensor.shape_cloned());
        let scale = coeus_ops::div(grad_out, &norm_broad, &backend);
        let grad_in = coeus_ops::mul(&scale, &self.input_tensor, &backend);

        let lock = g.write();
        coeus_ops::add_assign(lock, &grad_in, &backend)?;
        Ok(())
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
/// Backward: `∂y/∂x_i = y^(1-p) * |xᵢ|^(p-1) * sign(xᵢ)`, composed from
/// provider-resident scalar-power and elementwise operations.
pub struct NormPNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub input_tensor: Tensor<T, B>,
    pub p: T,
    /// Provider-resident scalar norm output.
    pub norm_tensor: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T> + Default>
    BackwardNode<T, B> for NormPNode<T, B>
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

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        let Some(Some(ref g)) = input_grads.first() else {
            return Ok(());
        };

        let shape = self.input_tensor.shape_cloned();
        let ones = Tensor::full_on(shape.clone(), T::one(), &backend);
        let zeros = Tensor::zeros_on(shape.clone(), &backend);
        let abs_x = coeus_ops::abs(&self.input_tensor, &backend);
        let safe_abs = coeus_ops::where_cond(&abs_x, &abs_x, &ones, &backend)
            .expect("norm_p backward: safe absolute value");
        let abs_power = coeus_ops::pow_scalar(&safe_abs, self.p - T::one(), &backend);

        let norm_broad = self.norm_tensor.broadcast(shape.clone());
        let safe_norm = coeus_ops::where_cond(&norm_broad, &norm_broad, &ones, &backend)
            .expect("norm_p backward: safe norm");
        let norm_factor = coeus_ops::pow_scalar(&safe_norm, T::one() - self.p, &backend);
        let signed = coeus_ops::mul(
            &coeus_ops::sign(&self.input_tensor, &backend),
            &abs_power,
            &backend,
        );
        let scaled = coeus_ops::mul(&signed, &norm_factor, &backend);
        let grad_broad = grad_out.broadcast(shape);
        let local = coeus_ops::mul(&scaled, &grad_broad, &backend);
        let grad_t = coeus_ops::where_cond(&self.input_tensor, &local, &zeros, &backend)
            .expect("norm_p backward: zero-input mask");
        let lock = g.write();
        coeus_ops::add_assign(lock, &grad_t, &backend)?;
        Ok(())
    }
}

/// Tracked general Lp norm over all elements, output shape `[1]`.
///
/// Matches `coeus_ops::norm_p` but returns a `[1]` tensor for autograd.
#[inline]
pub fn norm_p<T: Float, B: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T> + Default>(
    a: &Var<T, B>,
    p: T,
) -> Var<T, B> {
    let backend = B::default();
    let out_tensor = coeus_ops::norm_p_tensor(&a.tensor, p, &backend);

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
            norm_tensor: out_tensor.clone(),
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
/// Backward: `∂yⱼ/∂xⱼₖ = yⱼ^(1-p) * |xⱼₖ|^(p-1) * sign(xⱼₖ)`, composed from
/// provider-resident scalar-power and elementwise operations.
pub struct NormPAxisNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub input_tensor: Tensor<T, B>,
    pub p: T,
    /// Forward output tensor (norm values, axis dim = 1).
    pub norm_tensor: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T> + Default>
    BackwardNode<T, B> for NormPAxisNode<T, B>
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

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        let Some(Some(ref g)) = input_grads.first() else {
            return Ok(());
        };

        let shape = self.input_tensor.shape_cloned();
        let ones = Tensor::full_on(shape.clone(), T::one(), &backend);
        let zeros = Tensor::zeros_on(shape.clone(), &backend);
        let abs_x = coeus_ops::abs(&self.input_tensor, &backend);
        let safe_abs = coeus_ops::where_cond(&abs_x, &abs_x, &ones, &backend)
            .expect("norm_p_axis backward: safe absolute value");
        let abs_power = coeus_ops::pow_scalar(&safe_abs, self.p - T::one(), &backend);

        let norm_broad = self.norm_tensor.broadcast(shape.clone());
        let safe_norm = coeus_ops::where_cond(&norm_broad, &norm_broad, &ones, &backend)
            .expect("norm_p_axis backward: safe norm");
        let norm_factor = coeus_ops::pow_scalar(&safe_norm, T::one() - self.p, &backend);
        let signed = coeus_ops::mul(
            &coeus_ops::sign(&self.input_tensor, &backend),
            &abs_power,
            &backend,
        );
        let scaled = coeus_ops::mul(&signed, &norm_factor, &backend);
        let grad_broad = grad_out.broadcast(shape.clone());
        let local = coeus_ops::mul(&scaled, &grad_broad, &backend);
        let grad_t = coeus_ops::where_cond(&self.input_tensor, &local, &zeros, &backend)
            .expect("norm_p_axis backward: zero-input mask");
        let lock = g.write();
        coeus_ops::add_assign(lock, &grad_t, &backend)?;
        Ok(())
    }
}

/// Tracked per-axis Lp norm, output has `axis` reduced to size 1.
#[inline]
pub fn norm_p_axis<
    T: Float,
    B: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T> + Default,
>(
    a: &Var<T, B>,
    p: T,
    axis: usize,
) -> Var<T, B> {
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
            norm_tensor: out_tensor.clone(),
        }) as Arc<dyn BackwardNode<T, B>>
    });
    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
