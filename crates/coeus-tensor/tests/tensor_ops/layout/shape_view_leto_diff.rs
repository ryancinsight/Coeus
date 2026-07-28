//! Differential verification for public tensor shape views.
//!
//! `Tensor::{reshape, permute}` route metadata validation through `coeus-leto`
//! while preserving zero-copy storage sharing.

use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_tensor::{Tensor, Transpose};

fn check_backend<B>()
where
    B: coeus_core::ComputeBackend + Default,
    B::DeviceBuffer<i32>:
        coeus_core::CpuAddressableStorage<i32> + coeus_core::CpuAddressableStorageMut<i32>,
{
    let backend = B::default();
    let data = (0..24).collect::<Vec<i32>>();
    let tensor = Tensor::<i32, B>::from_slice_on([2usize, 3, 4], &data, &backend).expect("construct tensor");

    let permuted = tensor.permute(&[2, 0, 1]);
    let expected_permuted = coeus_leto::permute_layout(tensor.layout(), &[2, 0, 1]).unwrap();
    assert_eq!(permuted.shape(), expected_permuted.shape());
    assert_eq!(permuted.strides(), expected_permuted.strides());
    assert_eq!(permuted.get(&[3, 1, 2]), tensor.get(&[1, 2, 3]));

    let transposed = tensor.t_nd();
    let expected_transposed = coeus_leto::permute_layout(tensor.layout(), &[0, 2, 1]).unwrap();
    assert_eq!(transposed.shape(), expected_transposed.shape());
    assert_eq!(transposed.strides(), expected_transposed.strides());
    assert_eq!(transposed.get(&[1, 3, 2]), tensor.get(&[1, 2, 3]));

    let matrix = Tensor::<i32, B>::from_slice_on([2usize, 3], &data[..6], &backend).expect("construct tensor");
    let matrix_t = matrix.transpose();
    assert_eq!(matrix_t.shape(), &[3, 2]);
    assert_eq!(matrix_t.get(&[2, 1]), matrix.get(&[1, 2]));

    let sliced = Tensor::<i32, B>::from_slice_on([8usize], &data[..8], &backend).expect("construct tensor").slice(&[(2, 6)]);
    let reshaped = sliced.reshape([2usize, 2]);
    let expected_reshaped = coeus_leto::reshape_layout(sliced.layout(), &[2, 2]).unwrap();
    assert_eq!(reshaped.shape(), expected_reshaped.shape());
    assert_eq!(reshaped.strides(), expected_reshaped.strides());
    assert_eq!(reshaped.layout().offset(), 2);
    assert_eq!(reshaped.as_slice(), &[2, 3, 4, 5]);
}

#[test]
fn sequential_shape_views_match_leto_dispatch() {
    check_backend::<SequentialBackend>();
}

#[test]
fn moirai_shape_views_match_leto_dispatch() {
    check_backend::<MoiraiBackend>();
}
