// ── Embedding lookup operations ──

use coeus_core::{ComputeBackend, Scalar, Storage, StorageMut};
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
) -> Tensor<T, B> {
    let weight_shape = weight.shape();
    assert_eq!(
        weight_shape.len(),
        2,
        "Weight tensor must be 2D [num_embeddings, embedding_dim]"
    );
    let num_embeddings = weight_shape[0];
    let embedding_dim = weight_shape[1];

    let indices_shape = indices.shape();
    let mut out_shape = indices_shape.to_vec();
    out_shape.push(embedding_dim);
    let out_shape = coeus_core::Shape::from(out_shape);

    let mut out_tensor = Tensor::zeros_on(out_shape, backend);

    let w_layout = weight.layout().clone();
    let idx_layout = indices.layout().clone();
    let out_layout = out_tensor.layout().clone();

    let w_strides = w_layout.strides();
    let w_offset = w_layout.offset();
    let out_strides = out_layout.strides();
    let out_offset = out_layout.offset();

    if let (Some(w_slice), Some(idx_slice), Some(out_slice)) = (
        weight.storage().try_as_slice(),
        indices.storage().try_as_slice(),
        out_tensor.storage_mut().try_as_mut_slice(),
    ) {
        let ndim_idx = indices.ndim();
        let indices_shape_ref = indices.shape();
        let num_indices = indices.numel();
        let mut idx_coords = smallvec::SmallVec::<[usize; 4]>::from_elem(0, ndim_idx);

        for i in 0..num_indices {
            let physical_idx = idx_layout.physical_index(&idx_coords);
            let token_val = idx_slice[physical_idx];
            let token_idx = <I as Scalar>::to_f64(token_val) as isize;

            assert!(
                token_idx >= 0 && token_idx < num_embeddings as isize,
                "Embedding index {} out of bounds [0, {})",
                token_idx,
                num_embeddings
            );

            let w_row_start = w_offset + (token_idx as usize) * w_strides[0];
            let out_row_stride = if ndim_idx > 0 {
                out_strides[ndim_idx - 1]
            } else {
                0
            };
            let out_row_start = out_offset + i * out_row_stride;

            for j in 0..embedding_dim {
                let w_idx = w_row_start + j * w_strides[1];
                let out_idx = out_row_start + j * out_strides[ndim_idx];
                out_slice[out_idx] = w_slice[w_idx];
            }

            for d in (0..ndim_idx).rev() {
                idx_coords[d] += 1;
                if idx_coords[d] < indices_shape_ref[d] {
                    break;
                }
                idx_coords[d] = 0;
            }
        }
    } else {
        let host_backend = coeus_core::MoiraiBackend::new();
        let w_host = weight.to_backend(&host_backend);
        let idx_host = indices.to_backend(&host_backend);
        let out_host = embedding(&w_host, &idx_host, &host_backend);
        out_tensor = out_host.to_backend(backend);
    }

    out_tensor
}

/// Compute backward pass of embedding lookup, accumulating gradients into weights.
pub fn embedding_backward<T: Scalar, I: Scalar, B: ComputeBackend + Default>(
    grad_out: &Tensor<T, B>,
    indices: &Tensor<I, B>,
    num_embeddings: usize,
    backend: &B,
) -> Tensor<T, B> {
    embedding_backward_impl(grad_out, indices, num_embeddings, None, backend)
}

/// Compute embedding lookup gradients while suppressing an optional padding row.
pub fn embedding_backward_with_padding_idx<T: Scalar, I: Scalar, B: ComputeBackend + Default>(
    grad_out: &Tensor<T, B>,
    indices: &Tensor<I, B>,
    num_embeddings: usize,
    padding_idx: Option<usize>,
    backend: &B,
) -> Tensor<T, B> {
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
) -> Tensor<T, B> {
    let grad_shape = grad_out.shape();
    let ndim_grad = grad_shape.len();
    assert!(ndim_grad >= 2, "grad_out must have at least 2 dimensions");
    let embedding_dim = grad_shape[ndim_grad - 1];

    let indices_shape = indices.shape();
    assert_eq!(
        &grad_shape[..ndim_grad - 1],
        indices_shape,
        "grad_out shape prefix must match indices shape"
    );

    let mut grad_weight = Tensor::zeros_on([num_embeddings, embedding_dim], backend);

    let go_layout = grad_out.layout().clone();
    let idx_layout = indices.layout().clone();
    let gw_layout = grad_weight.layout().clone();

    let go_strides = go_layout.strides();
    let go_offset = go_layout.offset();
    let gw_strides = gw_layout.strides();
    let gw_offset = gw_layout.offset();

    if let (Some(go_slice), Some(idx_slice), Some(gw_slice)) = (
        grad_out.storage().try_as_slice(),
        indices.storage().try_as_slice(),
        grad_weight.storage_mut().try_as_mut_slice(),
    ) {
        let ndim_idx = indices.ndim();
        let num_indices = indices.numel();
        let mut idx_coords = smallvec::SmallVec::<[usize; 4]>::from_elem(0, ndim_idx);

        for i in 0..num_indices {
            let physical_idx = idx_layout.physical_index(&idx_coords);
            let token_val = idx_slice[physical_idx];
            let token_idx = <I as Scalar>::to_f64(token_val) as isize;

            if token_idx >= 0
                && token_idx < num_embeddings as isize
                && Some(token_idx) != skip_index
            {
                let go_row_stride = if ndim_idx > 0 {
                    go_strides[ndim_idx - 1]
                } else {
                    0
                };
                let go_row_start = go_offset + i * go_row_stride;
                let gw_row_start = gw_offset + (token_idx as usize) * gw_strides[0];

                for j in 0..embedding_dim {
                    let go_idx = go_row_start + j * go_strides[ndim_idx];
                    let gw_idx = gw_row_start + j * gw_strides[1];
                    gw_slice[gw_idx] += go_slice[go_idx];
                }
            }

            for d in (0..ndim_idx).rev() {
                idx_coords[d] += 1;
                if idx_coords[d] < indices_shape[d] {
                    break;
                }
                idx_coords[d] = 0;
            }
        }
    } else {
        let host_backend = coeus_core::MoiraiBackend::new();
        let go_host = grad_out.to_backend(&host_backend);
        let idx_host = indices.to_backend(&host_backend);
        let gw_host = embedding_backward_impl(
            &go_host,
            &idx_host,
            num_embeddings,
            skip_index,
            &host_backend,
        );
        grad_weight = gw_host.to_backend(backend);
    }

    grad_weight
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
        );
        let indices = Tensor::<i32, SequentialBackend>::from_slice(vec![3], &[0, 1, 0]);

        let grad = embedding_backward_with_padding_idx(&grad_out, &indices, 2, Some(0), &backend);

        assert_eq!(grad.shape(), &[2, 2]);
        assert_eq!(grad.as_slice(), &[0.0, 0.0, 3.0, 4.0]);
    }
}
