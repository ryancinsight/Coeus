//! Differential verification of public CPU cumulative-sum scans.
//!
//! `coeus_ops::cumsum` and `coeus_ops::suffix_sum` route through the
//! dynamic-rank `coeus-leto` scan shim. The references below are independent
//! row-major prefix/suffix scans over exactly representable values.

use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

fn tensor_from_slice<T, B>(shape: &[usize], data: &[T], backend: &B) -> Tensor<T, B>
where
    T: Scalar,
    B: ComputeBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    Tensor::from_slice_on(shape.to_vec(), data, backend)
}

fn check_backend<T, B>(backend: &B)
where
    T: Scalar + leto_ops::Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let data: Vec<T> = (1..=6).map(|value| T::from_f64(value as f64)).collect();
    let tensor = tensor_from_slice::<T, B>(&[2, 3], &data, backend);

    let axis1_prefix = coeus_ops::cumsum(&tensor, 1);
    assert_same_bits(
        axis1_prefix.as_slice(),
        [1.0, 3.0, 6.0, 4.0, 9.0, 15.0].map(T::from_f64),
    );

    let axis0_prefix = coeus_ops::cumsum(&tensor, 0);
    assert_same_bits(
        axis0_prefix.as_slice(),
        [1.0, 2.0, 3.0, 5.0, 7.0, 9.0].map(T::from_f64),
    );

    let axis1_suffix = coeus_ops::suffix_sum(&tensor, 1);
    assert_same_bits(
        axis1_suffix.as_slice(),
        [6.0, 5.0, 3.0, 15.0, 11.0, 6.0].map(T::from_f64),
    );

    let axis0_suffix = coeus_ops::suffix_sum(&tensor, 0);
    assert_same_bits(
        axis0_suffix.as_slice(),
        [5.0, 7.0, 9.0, 4.0, 5.0, 6.0].map(T::from_f64),
    );
}

fn assert_same_bits<T: Scalar, const N: usize>(got: &[T], expected: [T; N]) {
    assert_eq!(got.len(), expected.len());
    for (index, (&actual, &reference)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(
            actual.to_f64().to_bits(),
            reference.to_f64().to_bits(),
            "scan mismatch at index {index}",
        );
    }
}

#[test]
fn sequential_scans_match_reference() {
    let backend = coeus_core::SequentialBackend;
    check_backend::<f32, _>(&backend);
    check_backend::<f64, _>(&backend);
}

#[test]
fn moirai_scans_match_reference() {
    let backend = coeus_core::MoiraiBackend;
    check_backend::<f32, _>(&backend);
    check_backend::<f64, _>(&backend);
}
