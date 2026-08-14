//! Differential verification for public tensor linspace constructors.
//!
//! `Tensor::linspace_on` routes coordinate traversal through
//! `coeus-leto::from_shape_fn_values` while preserving its `Scalar::from_f64`
//! value contract.
#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]

use coeus_core::{MoiraiBackend, Scalar, SequentialBackend};
use coeus_tensor::Tensor;

fn check_backend<B>()
where
    B: coeus_core::ComputeBackend + Default,
    B::DeviceBuffer<i32>:
        coeus_core::CpuAddressableStorage<i32> + coeus_core::CpuAddressableStorageMut<i32>,
{
    let backend = B::default();

    let tensor = Tensor::<i32, B>::linspace_on(2, 8, 4, &backend);
    let expected = coeus_leto::from_shape_fn_values(&[4usize], |index| {
        i32::from_f64(2.0 + 2.0 * index[0] as f64)
    })
    .unwrap();

    assert!(tensor.is_contiguous());
    assert_eq!(tensor.shape(), &[4]);
    assert_eq!(tensor.as_slice(), expected.as_slice());
    assert_eq!(tensor.as_slice(), &[2, 4, 6, 8]);

    let singleton = Tensor::<i32, B>::linspace_on(7, 99, 1, &backend);
    assert_eq!(singleton.shape(), &[1]);
    assert_eq!(singleton.as_slice(), &[7]);

    let empty = Tensor::<i32, B>::linspace_on(7, 99, 0, &backend);
    let expected_empty =
        coeus_leto::from_shape_fn_values(&[0usize], |index| i32::from_usize(index[0])).unwrap();
    assert!(empty.is_contiguous());
    assert_eq!(empty.shape(), &[0]);
    assert_eq!(empty.as_slice(), expected_empty.as_slice());
    assert_eq!(empty.as_slice(), &[] as &[i32]);
}

#[test]
fn sequential_linspace_matches_leto_dispatch() {
    check_backend::<SequentialBackend>();
}

#[test]
fn moirai_linspace_matches_leto_dispatch() {
    check_backend::<MoiraiBackend>();
}
