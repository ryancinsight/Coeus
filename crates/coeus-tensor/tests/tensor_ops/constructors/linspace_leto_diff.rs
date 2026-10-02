//! Differential verification for public tensor linspace constructors.
//!
//! `Tensor::linspace_on` converts each coordinate through `TryFromCount`; the
//! `coeus-leto::from_shape_fn_values` traversal with the `Scalar::from_f64`
//! value contract is the differential oracle, and a coordinate outside the
//! element range is a typed error.

use coeus_core::{MoiraiBackend, Scalar, SequentialBackend, TryFromCount};
use coeus_tensor::Tensor;

fn check_backend<B>()
where
    B: coeus_core::ComputeBackend + Default,
    B::DeviceBuffer<i32>:
        coeus_core::CpuAddressableStorage<i32> + coeus_core::CpuAddressableStorageMut<i32>,
{
    let backend = B::default();

    let tensor = Tensor::<i32, B>::linspace_on(2, 8, 4, &backend).expect("0..4 fits i32");
    let expected = coeus_leto::from_shape_fn_values(&[4usize], |index| {
        i32::from_f64(2.0 + 2.0 * index[0] as f64)
    })
    .unwrap();

    assert!(tensor.is_contiguous());
    assert_eq!(tensor.shape(), &[4]);
    assert_eq!(tensor.as_slice(), expected.as_slice());
    assert_eq!(tensor.as_slice(), &[2, 4, 6, 8]);

    let singleton = Tensor::<i32, B>::linspace_on(7, 99, 1, &backend).expect("0 fits i32");
    assert_eq!(singleton.shape(), &[1]);
    assert_eq!(singleton.as_slice(), &[7]);

    let empty = Tensor::<i32, B>::linspace_on(7, 99, 0, &backend).expect("an empty range fits i32");
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
fn sequential_linspace_matches_leto_dispatch() {
    check_backend::<SequentialBackend>();
}

#[test]
fn moirai_linspace_matches_leto_dispatch() {
    check_backend::<MoiraiBackend>();
}

#[test]
fn linspace_reports_the_divisor_outside_the_element_range() {
    let backend = SequentialBackend::new();

    let Err(err) = Tensor::<i8, SequentialBackend>::linspace_on(0, 100, 129, &backend) else {
        panic!("divisor 128 does not fit i8");
    };
    assert_eq!(err.count(), 128);
    assert_eq!(err.target(), "i8");
}
