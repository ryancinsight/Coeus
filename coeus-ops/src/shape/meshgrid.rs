// ── meshgrid — create coordinate grids from 1-D tensors ──
//
// `meshgrid(tensors, indexing)` creates N tensors of shape
// `[tensors[0].len(), tensors[1].len(), ..., tensors[N-1].len()]`
// where tensor `i` varies along dimension `i`.
//
// Matches `torch.meshgrid(tensors, indexing="ij")` (default "ij").
//
// - `"ij"` indexing (default, matrix / NumPy convention):
//     output[0] varies along dim 0, output[1] along dim 1, etc.
// - `"xy"` indexing (Cartesian/Matplotlib convention):
//     output[0] (first arg) varies along dim 1, output[1] along dim 0.

use crate::backend_ops::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Create coordinate grids from a slice of 1-D tensors.
///
/// Returns a `Vec<Tensor<T, B>>` of length `tensors.len()`, each with shape
/// equal to the product of all input lengths.
///
/// # Panics
/// - Panics if any input tensor is not 1-D.
#[inline]
pub fn meshgrid<T: Scalar, B: BackendOps<T> + Default>(
    tensors: &[&Tensor<T, B>],
    indexing: &str,
    _backend: &B,
) -> Vec<Tensor<T, B>>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert!(
        matches!(indexing, "ij" | "xy"),
        "meshgrid: indexing must be \"ij\" or \"xy\", got {indexing:?}"
    );
    for (i, t) in tensors.iter().enumerate() {
        assert_eq!(
            t.ndim(),
            1,
            "meshgrid: tensor {i} must be 1-D, got {}-D",
            t.ndim()
        );
    }

    let n = tensors.len();
    let sizes: Vec<usize> = tensors.iter().map(|t| t.shape()[0]).collect();

    // Under "xy" indexing, the first two outputs are swapped relative to "ij".
    let ij_to_grid = |grid_idx: usize| -> usize {
        if indexing == "xy" && n >= 2 {
            match grid_idx {
                0 => 1, // first grid varies along dim 1 (xy)
                1 => 0, // second grid varies along dim 0 (xy)
                i => i,
            }
        } else {
            grid_idx
        }
    };

    let total: usize = sizes.iter().product();

    // For each output tensor `g`, element at multi-dim index `idx` (in the
    // grid shape) takes the value from `tensors[ij_to_grid(g)]` at
    // coordinate `idx[ij_to_grid(g)]`.
    (0..n)
        .map(|g| {
            let src_dim = ij_to_grid(g);
            let src_cont = tensors[g].to_contiguous();
            let src_s = src_cont.as_slice();

            // Compute row-major strides for the output shape.
            let out_shape = if indexing == "xy" && n >= 2 {
                // under "xy", output shape has first two dims swapped
                let mut s = sizes.clone();
                s.swap(0, 1);
                s
            } else {
                sizes.clone()
            };

            let mut strides = vec![1usize; n];
            for d in (0..n - 1).rev() {
                strides[d] = strides[d + 1] * out_shape[d + 1];
            }

            let data: Vec<T> = (0..total)
                .map(|flat| {
                    // Decode flat → multi-dim index.
                    let coord_in_src_dim = (flat / strides[src_dim]) % out_shape[src_dim];
                    src_s[coord_in_src_dim]
                })
                .collect();

            Tensor::from_slice(out_shape, &data)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn meshgrid_ij_2d_creates_correct_grids() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![3], &[0.0f32, 1.0, 2.0]);
        let y = Tensor::from_slice(vec![2], &[10.0f32, 20.0]);
        let grids = meshgrid(&[&x, &y], "ij", &b);
        assert_eq!(grids.len(), 2);
        // x-grid [3,2]: each row is [0,0], [1,1], [2,2]
        assert_eq!(grids[0].shape(), &[3, 2]);
        assert_eq!(
            grids[0].as_slice(),
            &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
        );
        // y-grid [3,2]: each column is [10,20] repeated 3 times
        assert_eq!(grids[1].shape(), &[3, 2]);
        assert_eq!(
            grids[1].as_slice(),
            &[10.0, 20.0, 10.0, 20.0, 10.0, 20.0]
        );
    }

    #[test]
    fn meshgrid_ij_1d_is_identity() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![4], &[1.0f32, 2.0, 3.0, 4.0]);
        let grids = meshgrid(&[&x], "ij", &b);
        assert_eq!(grids.len(), 1);
        assert_eq!(grids[0].as_slice(), x.as_slice());
    }

    #[test]
    fn meshgrid_ij_matches_numpy_convention() {
        // For (x=[0,1,2], y=[0,1]):
        // ij indexing: grid_x varies along axis 0, grid_y along axis 1.
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![3], &[0.0f32, 1.0, 2.0]);
        let y = Tensor::from_slice(vec![2], &[0.0f32, 1.0]);
        let grids = meshgrid(&[&x, &y], "ij", &b);
        // grid_x[i,j] = x[i]
        for row in 0..3 {
            for col in 0..2 {
                assert_eq!(grids[0].as_slice()[row * 2 + col], row as f32);
            }
        }
        // grid_y[i,j] = y[j]
        for row in 0..3 {
            for col in 0..2 {
                assert_eq!(grids[1].as_slice()[row * 2 + col], col as f32);
            }
        }
    }
}
