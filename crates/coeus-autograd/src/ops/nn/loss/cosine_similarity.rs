//! Row-wise cosine similarity (PyTorch `F.cosine_similarity`).
//!
//! For inputs `x1, x2` of shape `[N, D]`, returns a `[N]` vector with
//!   `out_i = <x1_i, x2_i> / max(||x1_i||_2 * ||x2_i||_2, eps)`
//! where the denominator is floored at `eps` (PyTorch `clamp_min(eps)`),
//! numerically stable convention used in
//! `torch.nn.functional.cosine_similarity`.
//!
//! Per-row partials (analytical subgradient):
//!   d cos_i / d x1_i  = (x2_i / max(||x1_i|| * ||x2_i||, eps))
//!                      - cos_i * x1_i / ||x1_i||^2
//!   d cos_i / d x2_i  = (x1_i / max(||x1_i|| * ||x2_i||, eps))
//!                      - cos_i * x2_i / ||x2_i||^2
//!
//! The node caches the row-wise `(dot, n1_sqr, n2_sqr, cos)` tuples plus a
//! per-element derivative factor so the backward pass is a single
//! rebroadcast — no extra inner-product recomputed at backward time.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar, Storage};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for row-wise cosine similarity.
pub struct CosineSimilarityNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output (shape `[N]`).
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Per-element factor `d(loss_k)/d(x1_ik)`, row-major `[N*D]`.
    pub grad_x1: Vec<T>,
    /// Per-element factor `d(loss_k)/d(x2_ik)`, row-major `[N*D]`.
    pub grad_x2: Vec<T>,
    /// Number of rows `N`.
    pub rows: usize,
    /// Feature dimension `D`.
    pub feat: usize,
    /// Input shape `[N, D]` for gradient reconstruction.
    pub shape: coeus_core::Shape,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for CosineSimilarityNode<T, B>
{
    fn op_name(&self) -> &'static str {
        "cosine_similarity"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        let temp_grad;
        let grad_cont = if grad_out.is_contiguous() && grad_out.layout().offset() == 0 {
            grad_out
        } else {
            temp_grad = grad_out.to_contiguous_on(&backend)?;
            &temp_grad
        };
        let mut g_rows = vec![T::zero(); self.rows];
        backend.copy_to_host(grad_cont.storage(), &mut g_rows)?;

        let want_x1 = matches!(input_grads.first(), Some(Some(_)));
        let want_x2 = matches!(input_grads.get(1), Some(Some(_)));
        if !want_x1 && !want_x2 {
            return Ok(());
        }

        // d/d_x1: grad_unit_x1[n,k] (already computed at forward time).
        // outer * grad_unit_x1 — applied per-row, broadcasting the row loss
        // scalar across the feature axis.
        let mut dx1 = vec![T::zero(); self.rows * self.feat];
        for i in 0..self.rows {
            let gi = g_rows[i];
            let base = i * self.feat;
            for k in 0..self.feat {
                dx1[base + k] = gi * self.grad_x1[base + k];
            }
        }

        if let Some(Some(ref g)) = input_grads.first() {
            let grad_tensor = Tensor::from_slice_on(self.shape.clone(), &dx1, &backend)?;
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend)?;
        }
        if let Some(Some(ref g)) = input_grads.get(1) {
            let mut dx2 = vec![T::zero(); self.rows * self.feat];
            for i in 0..self.rows {
                let gi = g_rows[i];
                let base = i * self.feat;
                for k in 0..self.feat {
                    dx2[base + k] = gi * self.grad_x2[base + k];
                }
            }
            let grad_tensor = Tensor::from_slice_on(self.shape.clone(), &dx2, &backend)?;
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend)?;
        }

        Ok(())
    }
}

/// Tracked row-wise cosine similarity (PyTorch `F.cosine_similarity`):
/// for inputs `x1, x2` of shape `[N, D]`, returns a `[N]` vector with
/// `out_i = <x1_i, x2_i> / max(||x1_i||_2 * ||x2_i||_2, eps)`.
///
/// # Panics
/// Panics when `x1` and `x2` do not share shape, when the inputs are not
/// 2-D `[N, D]`, or when `D == 0`.
///
/// # Numerical contract
/// - Subgradient at the `eps`-denominator convention is the standard rule
///   `d/dx1_i (x2_i / denom)`; the cached factor matches PyTorch's autograd
///   formula at f64 precision up to the `eps` cst-order term.
/// - When `||x1_i||` or `||x2_i||` is zero, the input gradient remains
///   bounded via the `eps` shift.
pub fn cosine_similarity<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x1: &Var<T, B>,
    x2: &Var<T, B>,
    dim: usize,
    eps: T,
) -> Result<Var<T, B>, B::Error> {
    let backend = B::default();
    assert_eq!(
        x1.tensor.shape(),
        x2.tensor.shape(),
        "cosine_similarity requires x1 and x2 to have identical shapes"
    );
    let shape_ref = x1.tensor.shape();
    assert_eq!(
        shape_ref.len(),
        2,
        "cosine_similarity expects 2D [N, D] inputs"
    );
    let rows = shape_ref[0];
    let feat = shape_ref[1];
    assert_eq!(
        dim, 1,
        "cosine_similarity currently supports dim=1 (row contraction); got dim={dim}"
    );
    assert!(feat > 0, "cosine_similarity requires D > 0; got D=0");
    let n = rows * feat;
    let shape = x1.tensor.shape_cloned();

    let x1_cont;
    let x1_raw = if x1.tensor.is_contiguous() && x1.tensor.layout().offset() == 0 {
        &x1.tensor
    } else {
        x1_cont = x1.tensor.to_contiguous_on(&backend)?;
        &x1_cont
    };
    let x2_cont;
    let x2_raw = if x2.tensor.is_contiguous() && x2.tensor.layout().offset() == 0 {
        &x2.tensor
    } else {
        x2_cont = x2.tensor.to_contiguous_on(&backend)?;
        &x2_cont
    };

    let x1_host: std::borrow::Cow<[T]> = if let Some(s) = x1_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(x1_raw.storage(), &mut v)?;
        std::borrow::Cow::Owned(v)
    };
    let x2_host: std::borrow::Cow<[T]> = if let Some(s) = x2_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(x2_raw.storage(), &mut v)?;
        std::borrow::Cow::Owned(v)
    };

    let one = T::one();
    let mut out = vec![T::zero(); rows];
    let mut grad_x1 = vec![T::zero(); n];
    let mut grad_x2 = vec![T::zero(); n];
    for i in 0..rows {
        let base = i * feat;
        let mut dot = T::zero();
        let mut n1_sqr = T::zero();
        let mut n2_sqr = T::zero();
        for k in 0..feat {
            dot += x1_host[base + k] * x2_host[base + k];
            n1_sqr += x1_host[base + k] * x1_host[base + k];
            n2_sqr += x2_host[base + k] * x2_host[base + k];
        }
        let n1 = <T as Float>::sqrt(n1_sqr);
        let n2 = <T as Float>::sqrt(n2_sqr);
        // PyTorch clamps the denominator: `(||x1||·||x2||).clamp_min(eps)` =
        // max(norm_product, eps). For normal-magnitude rows this is exactly the
        // norm product (no perturbation), so the result matches torch to full
        // precision; eps only floors a vanishing-norm denominator. (Adding eps
        // instead perturbed every result by an O(eps/denom) term.)
        let norm_product = n1 * n2;
        let denom = if norm_product > eps {
            norm_product
        } else {
            eps
        };
        let cos_i = dot / denom;
        out[i] = cos_i;

        // d cos_i / d x1_ik  =  x2_ik / denom  -  cos_i * x1_ik / n1_sqr
        // d cos_i / d x2_ik  =  x1_ik / denom  -  cos_i * x2_ik / n2_sqr
        // Subgradient factor for x1_ik == 0 in a vanishing-norm row is
        // well-defined (x1_ik = 0 collapses the second term to 0).
        for k in 0..feat {
            grad_x1[base + k] = x2_host[base + k] / denom - cos_i * x1_host[base + k] / n1_sqr;
            grad_x2[base + k] = x1_host[base + k] / denom - cos_i * x2_host[base + k] / n2_sqr;
        }
        // Reference unused `one` to silence unused warnings on builds that
        // skip the Int-bounded code path (`Float` always provides `one()`).
        let _ = one;
    }

    let out_tensor = Tensor::from_slice_on([rows], &out, &backend)?;
    let requires_grad =
        crate::grad_mode::should_track_var(x1) || crate::grad_mode::should_track_var(x2);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            [rows],
            &backend,
        )?)))
    } else {
        None
    };
    let creator = grad.as_ref().cloned().map(|output_grad| {
        Arc::new(CosineSimilarityNode {
            output_grad: output_grad.clone(),
            inputs: vec![x1.clone(), x2.clone()],
            grad_x1,
            grad_x2,
            rows,
            feat,
            shape,
        }) as Arc<dyn BackwardNode<T, B>>
    });
    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}
