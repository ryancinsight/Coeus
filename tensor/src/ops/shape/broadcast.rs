
/// Helper to broadcast tensor data to a target shape
pub fn broadcast_tensor_data<T: Clone>(
    data: &[T],
    current_shape: &[usize],
    target_shape: &[usize],
) -> crate::Result<Vec<T>> {
    if current_shape == target_shape {
        return Ok(data.to_vec());
    }

    let target_numel: usize = target_shape.iter().product();
    let mut result = Vec::with_capacity(target_numel);

    for i in 0..target_numel {
        let mut current_idx = 0;
        let mut remaining = i;
        let mut stride = target_numel;

        for (d, &target_dim) in target_shape.iter().enumerate() {
            stride /= target_dim;
            let target_coord = remaining / stride;
            remaining %= stride;

            // Mapping target coord to source coord
            // If target_dim > current_dim and current_dim == 1, we use coord 0 (broadcast)
            // We need to handle padding of current_shape with 1s at the front
            let current_dim_idx =
                d as i32 - (target_shape.len() as i32 - current_shape.len() as i32);
            if current_dim_idx >= 0 {
                let current_dim = current_shape[current_dim_idx as usize];
                if current_dim != target_dim && current_dim != 1 {
                    return Err(crate::TensorError::ShapeError {
                        expected: current_dim,
                        actual: target_dim,
                        message: format!(
                            "Cannot broadcast dimension {} from {} to {}",
                            d, current_dim, target_dim
                        ),
                    });
                }
                if current_dim > 1 {
                    // Calculate current stride
                    let current_stride: usize = current_shape
                        .iter()
                        .skip(current_dim_idx as usize + 1)
                        .product();
                    current_idx += target_coord * current_stride;
                }
            }
        }
        result.push(data[current_idx].clone());
    }

    Ok(result)
}

/// Helper to calculate broadcasted shapes
pub fn broadcast_shapes(shape_a: &[usize], shape_b: &[usize]) -> crate::Result<Vec<usize>> {
    let len_a = shape_a.len();
    let len_b = shape_b.len();
    let out_len = len_a.max(len_b);
    let mut out_shape = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let dim_a = if i < out_len - len_a {
            1
        } else {
            shape_a[i - (out_len - len_a)]
        };
        let dim_b = if i < out_len - len_b {
            1
        } else {
            shape_b[i - (out_len - len_b)]
        };

        if dim_a == dim_b {
            out_shape.push(dim_a);
        } else if dim_a == 1 {
            out_shape.push(dim_b);
        } else if dim_b == 1 {
            out_shape.push(dim_a);
        } else {
            return Err(crate::TensorError::ShapeError {
                expected: dim_a,
                actual: dim_b,
                message: format!("Cannot broadcast shapes {shape_a:?} and {shape_b:?}"),
            });
        }
    }
    Ok(out_shape)
}
