//! Row-wise cosine similarity (PyTorch `F.cosine_similarity`).
//!
//! For inputs `x1, x2` of shape `[N, D]`, returns a `[N]` vector with
//! `out_i = <x1_i, x2_i> / max(||x1_i||_2 * ||x2_i||_2, eps)`.
//!
//! Forward and backward remain on the selected backend. The backward mask
//! applies the norm derivative only where the unclamped norm product is at
//! least `eps`; below the clamp, the denominator is constant and the gradient
//! is `x_other / eps`.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for row-wise cosine similarity.
pub struct CosineSimilarityNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output (shape `[N]`).
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// First input snapshot.
    pub x1: Tensor<T, B>,
    /// Second input snapshot.
    pub x2: Tensor<T, B>,
    /// Clamped row denominator, shape `[N]`.
    pub denominator: Tensor<T, B>,
    /// Squared norm of the first input, shape `[N]`.
    pub x1_norm_squared: Tensor<T, B>,
    /// Squared norm of the second input, shape `[N]`.
    pub x2_norm_squared: Tensor<T, B>,
    /// Forward cosine value, shape `[N]`.
    pub cosine: Tensor<T, B>,
    /// One where the norm product is at least `eps`, zero below it.
    pub norm_derivative_mask: Tensor<T, B>,
    /// One below the denominator clamp, zero at and above it.
    pub clamped_mask: Tensor<T, B>,
    /// Number of rows.
    pub rows: usize,
    /// Feature dimension.
    pub features: usize,
}

#[inline]
fn broadcast_rows<T: Scalar, B: coeus_ops::BackendOps<T>>(
    rows: &Tensor<T, B>,
    row_count: usize,
    feature_count: usize,
) -> Tensor<T, B> {
    rows.reshape([row_count, 1])
        .broadcast([row_count, feature_count])
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
        let want_x1 = matches!(input_grads.first(), Some(Some(_)));
        let want_x2 = matches!(input_grads.get(1), Some(Some(_)));
        if !want_x1 && !want_x2 {
            return Ok(());
        }

        let backend = B::default();
        let grad = broadcast_rows(grad_out, self.rows, self.features);
        let denominator = broadcast_rows(&self.denominator, self.rows, self.features);
        let cosine = broadcast_rows(&self.cosine, self.rows, self.features);
        let active = broadcast_rows(&self.norm_derivative_mask, self.rows, self.features);
        let inactive = broadcast_rows(&self.clamped_mask, self.rows, self.features);

        if let Some(Some(buffer)) = input_grads.first() {
            let norm_squared = broadcast_rows(&self.x1_norm_squared, self.rows, self.features);
            let safe_norm_squared = coeus_ops::add(&norm_squared, &inactive, &backend);
            let direct = coeus_ops::div(&self.x2, &denominator, &backend);
            let radial = coeus_ops::div(
                &coeus_ops::mul(&cosine, &self.x1, &backend),
                &safe_norm_squared,
                &backend,
            );
            let derivative = coeus_ops::sub(
                &direct,
                &coeus_ops::mul(&active, &radial, &backend),
                &backend,
            );
            let gradient = coeus_ops::mul(&grad, &derivative, &backend);
            coeus_ops::add_assign(buffer.write(), &gradient, &backend)?;
        }

        if let Some(Some(buffer)) = input_grads.get(1) {
            let norm_squared = broadcast_rows(&self.x2_norm_squared, self.rows, self.features);
            let safe_norm_squared = coeus_ops::add(&norm_squared, &inactive, &backend);
            let direct = coeus_ops::div(&self.x1, &denominator, &backend);
            let radial = coeus_ops::div(
                &coeus_ops::mul(&cosine, &self.x2, &backend),
                &safe_norm_squared,
                &backend,
            );
            let derivative = coeus_ops::sub(
                &direct,
                &coeus_ops::mul(&active, &radial, &backend),
                &backend,
            );
            let gradient = coeus_ops::mul(&grad, &derivative, &backend);
            coeus_ops::add_assign(buffer.write(), &gradient, &backend)?;
        }

        Ok(())
    }
}

/// Tracked row-wise cosine similarity.
///
/// For inputs `x1, x2` of shape `[N, D]`, returns `[N]` with
/// `out_i = <x1_i, x2_i> / max(||x1_i||_2 * ||x2_i||_2, eps)`.
///
/// At the clamp boundary, the norm-product derivative is retained. Below the
/// boundary, the denominator is constant and contributes no derivative.
///
/// # Panics
///
/// Panics when the inputs do not share a two-dimensional non-empty shape,
/// `dim` is not one, or `eps` is not finite and strictly positive.
#[must_use]
pub fn cosine_similarity<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x1: &Var<T, B>,
    x2: &Var<T, B>,
    dim: usize,
    eps: T,
) -> Var<T, B> {
    assert_eq!(
        x1.tensor.shape(),
        x2.tensor.shape(),
        "cosine_similarity requires x1 and x2 to have identical shapes"
    );
    assert_eq!(
        x1.tensor.ndim(),
        2,
        "cosine_similarity expects 2D [N, D] inputs"
    );
    assert_eq!(
        dim, 1,
        "cosine_similarity currently supports dim=1; got dim={dim}"
    );
    assert!(
        eps > T::zero() && !<T as Float>::is_nan(eps) && !<T as Float>::is_infinite(eps),
        "cosine_similarity requires finite eps > 0"
    );

    let rows = x1.tensor.shape()[0];
    let features = x1.tensor.shape()[1];
    assert!(
        rows > 0 && features > 0,
        "cosine_similarity requires non-empty [N, D] inputs"
    );

    let backend = B::default();
    let dot = coeus_ops::sum_axis(
        &coeus_ops::mul(&x1.tensor, &x2.tensor, &backend),
        dim,
        &backend,
    )
    .expect("invariant: cosine similarity validates the reduction axis")
    .reshape([rows]);
    let x1_norm_squared = coeus_ops::sum_axis(
        &coeus_ops::mul(&x1.tensor, &x1.tensor, &backend),
        dim,
        &backend,
    )
    .expect("invariant: cosine similarity validates the reduction axis")
    .reshape([rows]);
    let x2_norm_squared = coeus_ops::sum_axis(
        &coeus_ops::mul(&x2.tensor, &x2.tensor, &backend),
        dim,
        &backend,
    )
    .expect("invariant: cosine similarity validates the reduction axis")
    .reshape([rows]);
    let norm_product = coeus_ops::mul(
        &coeus_ops::sqrt(&x1_norm_squared, &backend),
        &coeus_ops::sqrt(&x2_norm_squared, &backend),
        &backend,
    );
    // Transfer the runtime scalar once, then broadcast it as a zero-copy view.
    // Materializing `[rows]` through `fill` would stage a host vector whose
    // transfer scales with the batch size on accelerator backends.
    let epsilon = Tensor::from_slice_on([1], &[eps], &backend).broadcast([rows]);
    let ones = coeus_ops::div(&epsilon, &epsilon, &backend);
    // `1 - ReluGrad(eps - norm_product)` is one for norm_product >= eps and
    // zero below it. This keeps the inclusive clamp convention while routing
    // through the backend-portable unary provider seam; accelerator providers
    // do not all expose comparison opcodes.
    let inactive = coeus_ops::elementwise_unary(
        &coeus_ops::sub(&epsilon, &norm_product, &backend),
        &backend,
        coeus_ops::UnaryOp::ReluGrad,
    )
    .expect("invariant: cosine similarity uses a supported scalar unary operation");
    let norm_derivative_mask = coeus_ops::sub(&ones, &inactive, &backend);
    let denominator = coeus_ops::add(
        &coeus_ops::mul(&norm_derivative_mask, &norm_product, &backend),
        &coeus_ops::mul(&inactive, &epsilon, &backend),
        &backend,
    );
    let cosine = coeus_ops::div(&dot, &denominator, &backend);

    let requires_grad =
        crate::grad_mode::should_track_var(x1) || crate::grad_mode::should_track_var(x2);
    let grad = requires_grad.then(|| {
        Arc::new(GradBuffer::new(Tensor::zeros_on(
            cosine.shape_cloned(),
            &backend,
        )))
    });
    let creator = grad.as_ref().map(|output_grad| {
        Arc::new(CosineSimilarityNode {
            output_grad: Arc::clone(output_grad),
            inputs: vec![x1.clone(), x2.clone()],
            x1: x1.tensor.clone(),
            x2: x2.tensor.clone(),
            denominator,
            x1_norm_squared,
            x2_norm_squared,
            cosine: cosine.clone(),
            norm_derivative_mask,
            clamped_mask: inactive,
            rows,
            features,
        }) as Arc<dyn BackwardNode<T, B>>
    });

    Var {
        tensor: cosine,
        grad,
        creator,
    }
}
