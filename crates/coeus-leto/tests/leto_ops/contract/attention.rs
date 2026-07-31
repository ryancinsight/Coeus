use coeus_core::{Layout, Shape, Strides};
use coeus_leto::{
    scaled_dot_product_attention_backward_accumulate, scaled_dot_product_attention_into,
    AttentionBackward, AttentionForward, AttentionGradientTargets, ReadOperand, WriteOperand,
};

fn strided_layout(shape: &[usize], strides: &[usize], offset: usize) -> Layout {
    Layout::from_shape_strides(
        Shape::from(shape.to_vec()),
        Strides::from_slice(strides),
        offset,
    )
}

fn read_rank_three(layout: &Layout, storage: &[f64]) -> Vec<f64> {
    let shape = layout.shape();
    let strides = layout.strides();
    let mut values = Vec::new();
    for batch in 0..shape[0] {
        for row in 0..shape[1] {
            for column in 0..shape[2] {
                values.push(
                    storage[layout.offset()
                        + batch * strides[0]
                        + row * strides[1]
                        + column * strides[2]],
                );
            }
        }
    }
    values
}

#[test]
fn attention_forward_preserves_strides_and_grouped_mask_borrows() {
    let query_layout = strided_layout(&[4, 1, 1], &[3, 2, 1], 1);
    let key_layout = strided_layout(&[4, 2, 1], &[6, 2, 1], 1);
    let value_layout = strided_layout(&[4, 2, 1], &[6, 2, 1], 1);
    let output_layout = strided_layout(&[4, 1, 1], &[4, 2, 1], 1);
    let weights_layout = strided_layout(&[4, 1, 2], &[6, 3, 1], 1);
    let mask_layout = strided_layout(&[2, 2], &[4, 2], 1);

    let mut query = vec![-91.0; 12];
    let mut key = vec![-92.0; 24];
    let mut value = vec![-93.0; 24];
    for batch in 0..4 {
        query[1 + batch * 3] = 1.0;
        key[1 + batch * 6] = 1.0;
        key[3 + batch * 6] = 1.0;
        value[1 + batch * 6] = 10.0 + batch as f64;
        value[3 + batch * 6] = 20.0 + batch as f64;
    }
    let mask = [-94.0, 1.0, -94.0, 0.0, -94.0, 0.0, -94.0, 1.0];
    let mut output = vec![-95.0; 16];
    let mut weights = vec![-96.0; 24];

    scaled_dot_product_attention_into(AttentionForward {
        query: ReadOperand {
            layout: &query_layout,
            data: &query,
        },
        key: ReadOperand {
            layout: &key_layout,
            data: &key,
        },
        value: ReadOperand {
            layout: &value_layout,
            data: &value,
        },
        keep_mask: Some(ReadOperand {
            layout: &mask_layout,
            data: &mask,
        }),
        is_causal: false,
        scale: 1.0,
        output: WriteOperand {
            layout: &output_layout,
            data: &mut output,
        },
        weights: WriteOperand {
            layout: &weights_layout,
            data: &mut weights,
        },
    })
    .unwrap();

    assert_eq!(
        read_rank_three(&output_layout, &output),
        [10.0, 11.0, 22.0, 23.0]
    );
    assert_eq!(
        read_rank_three(&weights_layout, &weights),
        [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    );
    assert_eq!(mask, [-94.0, 1.0, -94.0, 0.0, -94.0, 0.0, -94.0, 1.0]);
}

#[test]
fn attention_backward_accumulates_selected_strided_gradients() {
    let query_layout = strided_layout(&[1, 1, 1], &[4, 2, 1], 1);
    let key_layout = strided_layout(&[1, 2, 1], &[6, 2, 1], 1);
    let value_layout = strided_layout(&[1, 2, 1], &[6, 2, 1], 1);
    let weights_layout = strided_layout(&[1, 1, 2], &[6, 3, 1], 1);
    let output_gradient_layout = strided_layout(&[1, 1, 1], &[4, 2, 1], 1);
    let value_gradient_layout = strided_layout(&[1, 2, 1], &[6, 2, 1], 1);

    let query = [-9.0, 1.0];
    let key = [-9.0, 1.0, -9.0, 1.0];
    let value = [-9.0, 2.0, -9.0, 4.0];
    let weights = [-9.0, 0.25, 0.75];
    let output_gradient = [-9.0, 2.0];
    let mut value_gradient = [-9.0, 10.0, -9.0, 20.0];

    scaled_dot_product_attention_backward_accumulate(AttentionBackward {
        output_gradient: ReadOperand {
            layout: &output_gradient_layout,
            data: &output_gradient,
        },
        query: ReadOperand {
            layout: &query_layout,
            data: &query,
        },
        key: ReadOperand {
            layout: &key_layout,
            data: &key,
        },
        value: ReadOperand {
            layout: &value_layout,
            data: &value,
        },
        weights: ReadOperand {
            layout: &weights_layout,
            data: &weights,
        },
        scale: 1.0,
        gradients: AttentionGradientTargets {
            query: None,
            key: None,
            value: Some(WriteOperand {
                layout: &value_gradient_layout,
                data: &mut value_gradient,
            }),
        },
    })
    .unwrap();

    assert_eq!(
        read_rank_three(&value_gradient_layout, &value_gradient),
        [10.5, 21.5]
    );
}

#[test]
fn attention_grouped_mask_rejects_nondivisible_batch_without_writes() {
    let query_layout = Layout::new([3, 1, 1].into());
    let key_layout = Layout::new([3, 2, 1].into());
    let value_layout = Layout::new([3, 2, 1].into());
    let output_layout = Layout::new([3, 1, 1].into());
    let weights_layout = Layout::new([3, 1, 2].into());
    let mask_layout = Layout::new([2, 2].into());
    let query = [1.0; 3];
    let key = [1.0; 6];
    let value = [1.0; 6];
    let mask = [1.0; 4];
    let mut output = [7.0; 3];
    let mut weights = [8.0; 6];

    let error = scaled_dot_product_attention_into(AttentionForward {
        query: ReadOperand {
            layout: &query_layout,
            data: &query,
        },
        key: ReadOperand {
            layout: &key_layout,
            data: &key,
        },
        value: ReadOperand {
            layout: &value_layout,
            data: &value,
        },
        keep_mask: Some(ReadOperand {
            layout: &mask_layout,
            data: &mask,
        }),
        is_causal: false,
        scale: 1.0,
        output: WriteOperand {
            layout: &output_layout,
            data: &mut output,
        },
        weights: WriteOperand {
            layout: &weights_layout,
            data: &mut weights,
        },
    })
    .unwrap_err();

    assert!(matches!(error, leto_ops::AttentionError::MaskShape { .. }));
    assert_eq!(output, [7.0; 3]);
    assert_eq!(weights, [8.0; 6]);
}

#[test]
fn attention_grouped_mask_validates_later_groups_before_writes() {
    let query_layout = Layout::new([4, 1, 1].into());
    let key_layout = Layout::new([4, 2, 1].into());
    let value_layout = Layout::new([4, 2, 1].into());
    let output_layout = Layout::new([4, 1, 1].into());
    let weights_layout = Layout::new([4, 1, 2].into());
    let mask_layout = Layout::new([2, 2].into());
    let query = [1.0; 4];
    let key = [1.0; 8];
    let value = [1.0; 8];
    let mask = [1.0, 0.0, 0.0, f64::NAN];
    let mut output = [7.0; 4];
    let mut weights = [8.0; 8];

    let error = scaled_dot_product_attention_into(AttentionForward {
        query: ReadOperand {
            layout: &query_layout,
            data: &query,
        },
        key: ReadOperand {
            layout: &key_layout,
            data: &key,
        },
        value: ReadOperand {
            layout: &value_layout,
            data: &value,
        },
        keep_mask: Some(ReadOperand {
            layout: &mask_layout,
            data: &mask,
        }),
        is_causal: false,
        scale: 1.0,
        output: WriteOperand {
            layout: &output_layout,
            data: &mut output,
        },
        weights: WriteOperand {
            layout: &weights_layout,
            data: &mut weights,
        },
    })
    .unwrap_err();

    assert_eq!(
        error,
        leto_ops::AttentionError::NonFinite {
            operand: leto_ops::AttentionOperand::Mask,
        }
    );
    assert_eq!(output, [7.0; 4]);
    assert_eq!(weights, [8.0; 8]);
}
