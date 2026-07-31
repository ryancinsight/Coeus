use super::scaled_dot_product_attention;
use crate::AttentionOps;
use coeus_core::{CpuAddressableStorage, CpuStorage, Layout, SequentialBackend, Shape, Strides};
use coeus_tensor::Tensor;

#[test]
fn backward_uses_destination_layout() {
    let backend = SequentialBackend::new();
    let query_layout = Layout::new([1, 1, 1].into());
    let key_layout = Layout::new([1, 2, 1].into());
    let value_layout = Layout::new([1, 2, 1].into());
    let weights_layout = Layout::new([1, 1, 2].into());
    let output_gradient_layout = Layout::new([1, 1, 1].into());
    let value_gradient_layout = Layout::from_shape_strides(
        Shape::from(vec![1, 2, 1]),
        Strides::from_slice(&[6, 2, 1]),
        1,
    );
    let output_gradient = CpuStorage::from_slice(&[2.0]);
    let query = CpuStorage::from_slice(&[1.0]);
    let key = CpuStorage::from_slice(&[1.0, 1.0]);
    let value = CpuStorage::from_slice(&[2.0, 4.0]);
    let weights = CpuStorage::from_slice(&[0.25, 0.75]);
    let mut value_gradient = CpuStorage::from_slice(&[-9.0, 10.0, -9.0, 20.0]);

    backend
        .sdp_attention_backward(
            &output_gradient,
            &output_gradient_layout,
            &query,
            &query_layout,
            &key,
            &key_layout,
            &value,
            &value_layout,
            &weights,
            &weights_layout,
            1.0,
            None,
            None,
            Some((&mut value_gradient, &value_gradient_layout)),
        )
        .expect("invariant: compatible strided gradient destination is valid");

    assert_eq!(value_gradient.as_slice(), &[-9.0, 10.5, -9.0, 21.5]);
}

#[test]
fn rejects_unpaired_mask_metadata_without_writes() {
    let backend = SequentialBackend::new();
    let tensor_layout = Layout::new([1, 1, 1].into());
    let mask_layout = Layout::new([1].into());
    let query = CpuStorage::from_slice(&[1.0]);
    let key = CpuStorage::from_slice(&[1.0]);
    let value = CpuStorage::from_slice(&[3.0]);
    let mut output = CpuStorage::from_slice(&[7.0]);
    let mut weights = CpuStorage::from_slice(&[8.0]);

    let error = backend
        .sdp_attention(
            &query,
            &tensor_layout,
            &key,
            &tensor_layout,
            &value,
            &tensor_layout,
            None,
            Some(&mask_layout),
            false,
            1.0,
            &mut output,
            &tensor_layout,
            &mut weights,
            &tensor_layout,
        )
        .expect_err("unpaired mask metadata must fail");

    assert!(matches!(error, coeus_core::BackendError::Storage { .. }));
    assert_eq!(output.as_slice(), &[7.0]);
    assert_eq!(weights.as_slice(), &[8.0]);
}

#[test]
fn rejects_invalid_rank_before_allocation() {
    let backend = SequentialBackend::new();
    let query = Tensor::from_slice_on([2, 2], &[1.0; 4], &backend);
    let key = Tensor::from_slice_on([1, 2, 2], &[1.0; 4], &backend);
    let value = Tensor::from_slice_on([1, 2, 2], &[1.0; 4], &backend);

    let result = scaled_dot_product_attention(&query, &key, &value, None, false, 1.0, &backend);
    let Err(error) = result else {
        panic!("rank-two query must fail without indexing panic");
    };

    assert!(matches!(
        error,
        coeus_core::BackendError::UnsupportedRank { rank: 2, .. }
    ));
}

#[test]
fn rejects_rank_three_keep_mask() {
    let backend = SequentialBackend::new();
    let query = Tensor::from_slice_on([1, 1, 1], &[1.0], &backend);
    let key = Tensor::from_slice_on([1, 2, 1], &[1.0, 1.0], &backend);
    let value = Tensor::from_slice_on([1, 2, 1], &[2.0, 4.0], &backend);
    let mask = Tensor::from_slice_on([1, 1, 2], &[1.0, 0.0], &backend);

    let result =
        scaled_dot_product_attention(&query, &key, &value, Some(&mask), false, 1.0, &backend);
    let Err(error) = result else {
        panic!("rank-three keep mask is outside the public contract");
    };

    assert!(matches!(
        error,
        coeus_core::BackendError::IncompatibleBroadcast { .. }
    ));
}
