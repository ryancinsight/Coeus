// ── Embedding lookup operations ──

use coeus_core::{ComputeBackend, Scalar};
use coeus_tensor::Tensor;

/// Apply embedding lookup: maps integer indices to dense vectors from a weight matrix.
///
/// # Shape logic
/// - `weight`: `[num_embeddings, embedding_dim]`
/// - `indices`: `[d0, d1, ..., dk]`
/// - Output: `[d0, d1, ..., dk, embedding_dim]`
///
/// # Memory
/// Output is `zeros_on` (accumulates via row-copy; sparse indices may not cover
/// all output rows if the index tensor is non-contiguous). Embedding backward
/// (gradient w.r.t. `weight`) is also `zeros_on` because it scatter-adds.
pub fn embedding<T: Scalar, I: Scalar, B: ComputeBackend + Default>(
    weight: &Tensor<T, B>,
    indices: &Tensor<I, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    let weight_shape = weight.shape();
    assert_eq!(
        weight_shape.len(),
        2,
        "Weight tensor must be 2D [num_embeddings, embedding_dim]"
    );
    let num_embeddings = weight_shape[0];
    let embedding_dim = weight_shape[1];
    let weight_values = weight.to_vec_on(backend)?;
    let index_values = indices.to_vec_on(backend)?;
    let mut output = Vec::with_capacity(index_values.len() * embedding_dim);

    for value in index_values {
        let index = <I as Scalar>::to_f64(value) as isize;
        assert!(
            index >= 0 && index < num_embeddings as isize,
            "Embedding index {index} out of bounds [0, {num_embeddings})"
        );
        let start = index as usize * embedding_dim;
        output.extend_from_slice(&weight_values[start..start + embedding_dim]);
    }

    let mut output_shape = indices.shape_cloned();
    output_shape.push(embedding_dim);
    Tensor::from_slice_on(output_shape, &output, backend)
}

/// Compute backward pass of embedding lookup, accumulating gradients into weights.
pub fn embedding_backward<T: Scalar, I: Scalar, B: ComputeBackend + Default>(
    grad_out: &Tensor<T, B>,
    indices: &Tensor<I, B>,
    num_embeddings: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    embedding_backward_impl(grad_out, indices, num_embeddings, None, backend)
}

/// Compute embedding lookup gradients while suppressing an optional padding row.
pub fn embedding_backward_with_padding_idx<T: Scalar, I: Scalar, B: ComputeBackend + Default>(
    grad_out: &Tensor<T, B>,
    indices: &Tensor<I, B>,
    num_embeddings: usize,
    padding_idx: Option<usize>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    assert!(
        padding_idx.is_none_or(|idx| idx < num_embeddings),
        "embedding_backward: padding_idx {:?} out of bounds [0, {})",
        padding_idx,
        num_embeddings
    );
    embedding_backward_impl(
        grad_out,
        indices,
        num_embeddings,
        padding_idx.map(|idx| idx as isize),
        backend,
    )
}

fn embedding_backward_impl<T: Scalar, I: Scalar, B: ComputeBackend + Default>(
    grad_out: &Tensor<T, B>,
    indices: &Tensor<I, B>,
    num_embeddings: usize,
    skip_index: Option<isize>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    let grad_shape = grad_out.shape();
    let ndim_grad = grad_shape.len();
    assert!(ndim_grad >= 2, "grad_out must have at least 2 dimensions");
    let embedding_dim = grad_shape[ndim_grad - 1];
    assert_eq!(
        &grad_shape[..ndim_grad - 1],
        indices.shape(),
        "grad_out shape prefix must match indices shape"
    );

    let gradients = grad_out.to_vec_on(backend)?;
    let index_values = indices.to_vec_on(backend)?;
    let mut weight_gradient = vec![T::zero(); num_embeddings * embedding_dim];
    for (row, value) in index_values.into_iter().enumerate() {
        let index = <I as Scalar>::to_f64(value) as isize;
        if index >= 0 && index < num_embeddings as isize && Some(index) != skip_index {
            let source = row * embedding_dim;
            let destination = index as usize * embedding_dim;
            for column in 0..embedding_dim {
                weight_gradient[destination + column] += gradients[source + column];
            }
        }
    }

    Tensor::from_slice_on([num_embeddings, embedding_dim], &weight_gradient, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    #[test]
    fn embedding_backward_padding_idx_skips_padding_row() {
        let backend = SequentialBackend::new();
        let grad_out = Tensor::<f32, SequentialBackend>::from_slice(
            vec![3, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .expect("invariant: test backend operation succeeds");
        let indices = Tensor::<i32, SequentialBackend>::from_slice(vec![3], &[0, 1, 0])
            .expect("invariant: test backend operation succeeds");

        let grad = embedding_backward_with_padding_idx(&grad_out, &indices, 2, Some(0), &backend)
            .expect("invariant: test operation succeeds");

        assert_eq!(grad.shape(), &[2, 2]);
        assert_eq!(grad.as_slice(), &[0.0, 0.0, 3.0, 4.0]);
    }
}
