//! Differential verification of public CPU arg reductions.
//!
//! `coeus_ops::argmax` and `coeus_ops::argmin` route through the dynamic-rank
//! `coeus-leto` arg-reduction shim. The reference indices below are derived
//! from independent row-major scans over exactly represented values.

use coeus_core::{
    ComputeBackend, CpuAddressableStorageMut, MoiraiBackend, Scalar, SequentialBackend,
};
use coeus_tensor::{Tensor, Transpose};

fn tensor_from_slice<T, B>(shape: &[usize], data: &[T], backend: &B) -> Tensor<T, B>
where
    T: Scalar,
    B: ComputeBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    Tensor::from_slice_on(shape.to_vec(), data, backend)
}

fn assert_indices(got: &[i64], expected: &[i64], context: &str) {
    assert_eq!(got, expected, "{context}");
}

fn check_backend<T, B>(backend: &B)
where
    T: Scalar + leto_ops::Scalar,
    B: coeus_ops::BackendOps<T> + coeus_ops::BackendOps<i64> + Default,
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorageMut<i64>,
{
    let data: Vec<T> = [1.0, 4.0, -2.0, 5.0, 3.0, 6.0]
        .into_iter()
        .map(T::from_f64)
        .collect();
    let tensor = tensor_from_slice::<T, B>(&[2, 3], &data, backend);

    let axis1_max = coeus_ops::argmax(&tensor, 1);
    assert_eq!(axis1_max.shape(), &[2, 1]);
    assert_indices(axis1_max.as_slice(), &[1, 2], "axis-1 argmax");

    let axis1_min = coeus_ops::argmin(&tensor, 1);
    assert_eq!(axis1_min.shape(), &[2, 1]);
    assert_indices(axis1_min.as_slice(), &[2, 1], "axis-1 argmin");

    let axis0_max = coeus_ops::argmax(&tensor, 0);
    assert_eq!(axis0_max.shape(), &[1, 3]);
    assert_indices(axis0_max.as_slice(), &[1, 0, 1], "axis-0 argmax");

    let axis0_min = coeus_ops::argmin(&tensor, 0);
    assert_eq!(axis0_min.shape(), &[1, 3]);
    assert_indices(axis0_min.as_slice(), &[0, 1, 0], "axis-0 argmin");

    let transposed = tensor.transpose();
    let transposed_axis1_max = coeus_ops::argmax(&transposed, 1);
    assert_eq!(transposed_axis1_max.shape(), &[3, 1]);
    assert_indices(
        transposed_axis1_max.as_slice(),
        &[1, 0, 1],
        "transposed axis-1 argmax",
    );

    let transposed_axis1_min = coeus_ops::argmin(&transposed, 1);
    assert_eq!(transposed_axis1_min.shape(), &[3, 1]);
    assert_indices(
        transposed_axis1_min.as_slice(),
        &[0, 1, 0],
        "transposed axis-1 argmin",
    );
}

#[test]
fn sequential_arg_reductions_match_reference() {
    let backend = SequentialBackend;
    check_backend::<f32, _>(&backend);
    check_backend::<f64, _>(&backend);
}

#[test]
fn moirai_arg_reductions_match_reference() {
    let backend = MoiraiBackend;
    check_backend::<f32, _>(&backend);
    check_backend::<f64, _>(&backend);
}

/// topk along a non-terminal dim writes back through the output strides —
/// regression for the line-contiguous layout bug (values were transposed for
/// dim = 0). Reference: torch.topk(x, 2, dim=0, largest=False, sorted=True).
#[test]
fn topk_dim0_matches_torch_reference() {
    use coeus_core::MoiraiBackend;
    let backend = MoiraiBackend::new();
    let x = coeus_tensor::Tensor::<f64, MoiraiBackend>::from_slice_on(
        vec![3, 4],
        &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0],
        &backend,
    );
    let (vals, idxs) = coeus_ops::topk(&x, 2, 0, false);
    assert_eq!(vals.shape(), &[2, 4]);
    // Per-column two smallest, sorted ascending, laid out along dim 0:
    // row0 = [3, 1, 2, 1], row1 = [5, 3, 4, 6].
    let expected = [3.0, 1.0, 2.0, 1.0, 5.0, 3.0, 4.0, 6.0];
    let got = vals.to_contiguous_on(&backend);
    for (i, (g, w)) in got.as_slice().iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, w, "vals[{i}]");
    }
    // Indices are positions along dim 0 (per column, ascending-value order):
    // col0 -> [0, 1] (3 then the row-1 five), col1 -> [0, 2] (1 then 3),
    // col2 -> [1, 0] (2 then 4), col3 -> [0, 1] (1 then 6).
    let expected_idx: [i64; 8] = [0, 0, 1, 0, 1, 2, 0, 1];
    let gi = idxs.to_contiguous_on(&backend);
    for (i, (g, w)) in gi.as_slice().iter().zip(expected_idx.iter()).enumerate() {
        assert_eq!(g, w, "idxs[{i}]");
    }
}
