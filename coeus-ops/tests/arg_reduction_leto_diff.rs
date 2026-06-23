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
