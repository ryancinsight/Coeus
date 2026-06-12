//! Differential verification of public CPU reduction APIs.
//!
//! The CPU `BackendOps::reduce` path delegates sum/min/max reductions through
//! `coeus-leto::reduce_into`; `mean_axis` composes that reduction with a public
//! binary division. Inputs use exactly representable values, so bitwise equality
//! is the correct oracle for both scalar widths.

use coeus_core::{
    ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, Scalar,
    SequentialBackend,
};
use coeus_ops::BackendOps;
use coeus_tensor::{Tensor, Transpose};

fn tensor_from_slice<T, B>(shape: &[usize], data: &[T], backend: &B) -> Tensor<T, B>
where
    T: Scalar,
    B: ComputeBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    Tensor::from_slice_on(shape.to_vec(), data, backend)
}

fn assert_same_bits<T: Scalar, const N: usize>(got: &[T], expected: [T; N], context: &str) {
    assert_eq!(got.len(), expected.len(), "{context} length mismatch");
    for (index, (&actual, &reference)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(
            actual.to_f64().to_bits(),
            reference.to_f64().to_bits(),
            "{context} mismatch at index {index}",
        );
    }
}

fn check_reductions<T, B>(backend: &B)
where
    T: Scalar + leto_ops::Scalar,
    B: BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let data: Vec<T> = (1..=6).map(|value| T::from_f64(value as f64)).collect();
    let tensor = tensor_from_slice::<T, B>(&[2, 3], &data, backend);

    let total = coeus_ops::sum(&tensor, backend);
    assert_eq!(
        total.to_f64().to_bits(),
        T::from_f64(21.0).to_f64().to_bits()
    );

    let mean = coeus_ops::mean(&tensor, backend);
    assert_eq!(mean.to_f64().to_bits(), T::from_f64(3.5).to_f64().to_bits());

    let sum_axis = coeus_ops::sum_axis(&tensor, 0, backend);
    assert_eq!(sum_axis.shape(), &[1, 3]);
    assert_same_bits(
        sum_axis.as_slice(),
        [5.0, 7.0, 9.0].map(T::from_f64),
        "axis-0 sum",
    );

    let mean_axis = coeus_ops::mean_axis(&tensor, 1, backend);
    assert_eq!(mean_axis.shape(), &[2, 1]);
    assert_same_bits(
        mean_axis.as_slice(),
        [2.0, 5.0].map(T::from_f64),
        "axis-1 mean",
    );

    let max_axis = coeus_ops::max_axis(&tensor, 1, backend);
    assert_eq!(max_axis.shape(), &[2, 1]);
    assert_same_bits(
        max_axis.as_slice(),
        [3.0, 6.0].map(T::from_f64),
        "axis-1 max",
    );

    let min_axis = coeus_ops::min_axis(&tensor, 0, backend);
    assert_eq!(min_axis.shape(), &[1, 3]);
    assert_same_bits(
        min_axis.as_slice(),
        [1.0, 2.0, 3.0].map(T::from_f64),
        "axis-0 min",
    );

    let transposed = tensor.transpose();
    let transposed_sum = coeus_ops::sum_axis(&transposed, 1, backend);
    assert_eq!(transposed_sum.shape(), &[3, 1]);
    assert_same_bits(
        transposed_sum.as_slice(),
        [5.0, 7.0, 9.0].map(T::from_f64),
        "transposed axis-1 sum",
    );

    let transposed_mean = coeus_ops::mean_axis(&transposed, 1, backend);
    assert_eq!(transposed_mean.shape(), &[3, 1]);
    assert_same_bits(
        transposed_mean.as_slice(),
        [2.5, 3.5, 4.5].map(T::from_f64),
        "transposed axis-1 mean",
    );
}

#[test]
fn sequential_public_reductions_match_reference() {
    let backend = SequentialBackend;
    check_reductions::<f32, _>(&backend);
    check_reductions::<f64, _>(&backend);
}

#[test]
fn moirai_public_reductions_match_reference() {
    let backend = MoiraiBackend;
    check_reductions::<f32, _>(&backend);
    check_reductions::<f64, _>(&backend);
}
