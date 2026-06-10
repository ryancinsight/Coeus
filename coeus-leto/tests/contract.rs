//! Cross-repo contract tests: pin coeus's assumptions about the leto kernels it
//! delegates to. A failure here is a leto regression from coeus's perspective.

use coeus_core::{CpuAddressableStorage, CpuStorage, Layout, Shape, Strides};
use coeus_leto::{elementwise_add_into, matmul_into, to_leto_view};

fn layout(shape: &[usize]) -> Layout {
    Layout::new(Shape::from(shape.to_vec()))
}

#[test]
fn add_matches_reference_rank2() {
    let a = vec![1.0f64, 2.0, 3.0, 4.0];
    let b = vec![10.0f64, 20.0, 30.0, 40.0];
    let mut out = vec![0.0f64; 4];
    let la = layout(&[2, 2]);

    elementwise_add_into(&la, &a, &la, &b, &la, &mut out).unwrap();
    assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn add_broadcasts_rowvec_into_matrix() {
    // [2,1] + [1,2] -> [2,2], exercising the broadcast-aware leto kernel from
    // coeus's dynamic-rank entry point.
    let a = vec![1.0f64, 2.0]; // shape [2,1]
    let b = vec![10.0f64, 20.0]; // shape [1,2]
    let mut out = vec![0.0f64; 4];

    elementwise_add_into(
        &layout(&[2, 1]),
        &a,
        &layout(&[1, 2]),
        &b,
        &layout(&[2, 2]),
        &mut out,
    )
    .unwrap();
    // rows: [1+10, 1+20], [2+10, 2+20]
    assert_eq!(out, vec![11.0, 21.0, 12.0, 22.0]);
}

#[test]
fn matmul_matches_reference() {
    // [[1,2,3],[4,5,6]] x [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
    let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut out = vec![0.0f64; 4];

    matmul_into(
        &layout(&[2, 3]),
        &a,
        &layout(&[3, 2]),
        &b,
        &layout(&[2, 2]),
        &mut out,
    )
    .unwrap();
    assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn matmul_handles_transposed_input_view() {
    // a stored as [3,2] but used transposed as [2,3] via explicit strides.
    let a_storage = vec![1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0]; // logical [3,2]
                                                           // transposed layout: shape [2,3], strides swapped.
    let a_t = Layout::from_shape_strides(
        Shape::from(vec![2usize, 3]),
        Strides::from_slice(&[1usize, 2]),
        0,
    );
    let b = vec![7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut out = vec![0.0f64; 4];

    matmul_into(
        &a_t,
        &a_storage,
        &layout(&[3, 2]),
        &b,
        &layout(&[2, 2]),
        &mut out,
    )
    .unwrap();
    // transposed a is [[1,2,3],[4,5,6]] -> same product as the contiguous case.
    assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn view_over_cpu_storage_reads_logical_values() {
    // Prove the adapter binds directly to coeus CpuStorage slices.
    let storage = CpuStorage::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let la = layout(&[2, 3]);
    let view = to_leto_view::<f64, 2>(&la, storage.as_slice()).unwrap();
    assert_eq!(view.shape(), [2, 3]);
    assert_eq!(*view.get([1, 2]).unwrap(), 6.0);
}

#[test]
fn rank_beyond_dispatch_bound_is_rejected() {
    let a = vec![0.0f64; 32];
    let la = layout(&[2, 2, 2, 2, 2]); // rank 5 > MAX_DISPATCH_RANK
    let mut out = vec![0.0f64; 32];
    assert!(elementwise_add_into(&la, &a, &la, &a, &la, &mut out).is_err());
}
