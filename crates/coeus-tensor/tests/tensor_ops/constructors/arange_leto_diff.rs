//! Differential verification for public tensor range constructors.
//!
//! `Tensor::arange_on` converts each index through `TryFromCount`; the
//! `coeus-leto::from_shape_fn_values` traversal of the same conversion is the
//! differential oracle, and an index outside the element range is a typed error.

use coeus_core::{MoiraiBackend, SequentialBackend, TryFromCount};
use coeus_tensor::Tensor;

fn check_backend<B>()
where
    B: coeus_core::ComputeBackend + Default,
    B::DeviceBuffer<i32>:
        coeus_core::CpuAddressableStorage<i32> + coeus_core::CpuAddressableStorageMut<i32>,
{
    let backend = B::default();

    let tensor = Tensor::<i32, B>::arange_on(8, &backend).expect("0..8 fits i32");
    let expected = coeus_leto::from_shape_fn_values(&[8usize], |index| {
        i32::try_from_count(index[0]).expect("0..8 fits i32")
    })
    .unwrap();

    assert!(tensor.is_contiguous());
    assert_eq!(tensor.shape(), &[8]);
    assert_eq!(tensor.as_slice(), expected.as_slice());
    assert_eq!(tensor.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7]);

    let empty = Tensor::<i32, B>::arange_on(0, &backend).expect("an empty range fits i32");
    let expected_empty = coeus_leto::from_shape_fn_values(&[0usize], |index| {
        i32::try_from_count(index[0]).expect("an empty range has no index")
    })
    .unwrap();

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

#[test]
fn arange_reports_the_first_index_outside_the_element_range() {
    let backend = SequentialBackend::new();

    let widest =
        Tensor::<i8, SequentialBackend>::arange_on(128, &backend).expect("0..=127 fits i8");
    assert_eq!(widest.shape(), &[128]);
    assert_eq!(widest.as_slice()[127], 127);

    let Err(err) = Tensor::<i8, SequentialBackend>::arange_on(129, &backend) else {
        panic!("index 128 does not fit i8");
    };
    assert_eq!(err.count(), 128);
    assert_eq!(err.target(), "i8");
}
