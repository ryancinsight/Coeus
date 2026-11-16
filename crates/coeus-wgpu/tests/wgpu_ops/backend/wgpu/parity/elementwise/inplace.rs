//! Parity of the in-place assignment and partial-update paths.

use coeus_core::{ComputeBackend, Layout};
use coeus_ops::{BinaryOp, ElementwiseOps};
use coeus_tensor::Tensor;

use super::super::{assert_parity, seq, to_cpu, to_gpu, wgpu};

#[test]
fn test_wgpu_assign_compacts_shared_rank_five_view() {
    let s = seq();
    let w = wgpu();
    let values: Vec<f32> = (0i16..64).map(f32::from).collect();
    let base = Tensor::from_slice([2, 2, 2, 2, 4], &values)
        .expect("invariant: test backend operation succeeds");
    let rhs = Tensor::from_slice([1, 1, 1, 1, 2], &[10.0, 20.0])
        .expect("invariant: test backend operation succeeds");
    let ranges = [(0, 2), (0, 2), (0, 2), (0, 2), (1, 3)];

    let mut expected = base.slice(&ranges);
    coeus_ops::add_assign(&mut expected, &rhs, &s).expect("valid CPU rank-five assignment");
    let expected = expected
        .to_vec_on(&s)
        .expect("invariant: test backend operation succeeds");

    let mut actual = to_gpu(&base).slice(&ranges);
    let shared = actual.clone();
    let shared_before = to_cpu(&shared);
    let rhs_gpu = to_gpu(&rhs);
    coeus_ops::add_assign(&mut actual, &rhs_gpu, &w).expect("valid WGPU rank-five assignment");

    assert!(actual.is_contiguous(), "replacement output must be compact");
    assert_eq!(actual.layout().offset(), 0);
    let actual = to_cpu(&actual);
    assert_parity("shared_rank_five_assign", &expected, actual.as_slice());
    let shared_after = to_cpu(&shared);
    assert_parity(
        "shared_rank_five_source",
        shared_before.as_slice(),
        shared_after.as_slice(),
    );
}

#[test]
fn test_wgpu_unary_assign_detaches_shared_view() {
    let w = wgpu();
    let base = Tensor::from_slice([2, 3], &[-3.0_f32, -1.0, 0.0, 2.0, 4.0, -5.0])
        .expect("invariant: test backend operation succeeds");
    let mut actual = to_gpu(&base).slice(&[(0, 2), (1, 3)]);
    let shared = actual.clone();

    coeus_ops::neg_assign(&mut actual, &w).expect("WGPU neg assignment");

    assert!(actual.is_contiguous(), "replacement output must be compact");
    assert_eq!(actual.layout().offset(), 0);
    assert_parity(
        "shared_unary_assign",
        &[1.0, 0.0, -4.0, 5.0],
        to_cpu(&actual).as_slice(),
    );
    let shared = to_cpu(&shared);
    assert_parity(
        "shared_unary_source",
        &[-1.0, 0.0, 4.0, -5.0],
        shared.as_slice(),
    );
}

#[test]
fn test_wgpu_partial_update_preserves_parent_and_shared_source() {
    let backend = wgpu();
    let parent_layout = Layout::new([2, 3].into());
    let destination_layout = parent_layout.slice(&[(0, 2), (1, 3)]);
    let rhs_layout = Layout::new([2, 2].into());
    // SAFETY: the following `copy_to_device` initializes every element before any read.
    let mut destination =
        unsafe { backend.allocate::<f32>(6) }.expect("invariant: test backend operation succeeds");
    // SAFETY: the following `copy_to_device` initializes every element before any read.
    let mut rhs =
        unsafe { backend.allocate::<f32>(4) }.expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &mut destination)
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&[10.0, 20.0, 30.0, 40.0], &mut rhs)
        .expect("invariant: test backend operation succeeds");
    let shared = destination.clone();

    backend
        .elementwise_binary_update(
            BinaryOp::Add,
            &mut destination,
            &destination_layout,
            &rhs,
            &rhs_layout,
        )
        .expect("WGPU partial update");

    let mut actual = [0.0; 6];
    backend
        .copy_to_host(&destination, &mut actual)
        .expect("invariant: test backend operation succeeds");
    assert_parity(
        "partial_update",
        &[1.0, 12.0, 23.0, 4.0, 35.0, 46.0],
        &actual,
    );
    let mut shared_values = [0.0; 6];
    backend
        .copy_to_host(&shared, &mut shared_values)
        .expect("invariant: test backend operation succeeds");
    assert_parity(
        "partial_update_shared",
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &shared_values,
    );
}
