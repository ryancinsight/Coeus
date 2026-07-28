//! Differential verification for public tensor range constructors.
//!
//! `Tensor::arange_on` routes index-derived value generation through
//! `coeus-leto::from_shape_fn_values` and `Scalar::from_usize`.

use coeus_core::{MoiraiBackend, Scalar, SequentialBackend};
use coeus_tensor::Tensor;

fn check_backend<B>()
where
    B: coeus_core::ComputeBackend + Default,
    B::DeviceBuffer<i32>:
        coeus_core::CpuAddressableStorage<i32> + coeus_core::CpuAddressableStorageMut<i32>,
{
    let backend = B::default();

    let tensor = Tensor::<i32, B>::arange_on(8, &backend).expect("construct tensor");
    let expected =
        coeus_leto::from_shape_fn_values(&[8usize], |index| i32::from_usize(index[0])).unwrap();

    assert!(tensor.is_contiguous());
    assert_eq!(tensor.shape(), &[8]);
    assert_eq!(tensor.as_slice(), expected.as_slice());
    assert_eq!(tensor.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7]);

    let empty = Tensor::<i32, B>::arange_on(0, &backend).expect("construct tensor");
    let expected_empty =
        coeus_leto::from_shape_fn_values(&[0usize], |index| i32::from_usize(index[0])).unwrap();

    assert!(empty.is_contiguous());
    assert_eq!(empty.shape(), &[0]);
    assert_eq!(empty.as_slice(), expected_empty.as_slice());
    assert_eq!(empty.as_slice(), &[] as &[i32]);
}

#[test]
fn sequential_arange_matches_leto_dispatch() {
    check_backend::<SequentialBackend>();
}

#[test]
fn moirai_arange_matches_leto_dispatch() {
    check_backend::<MoiraiBackend>();
}
