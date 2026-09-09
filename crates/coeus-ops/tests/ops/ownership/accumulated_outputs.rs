use super::optimizer::{assert_values, upload};
use coeus_core::{Layout, Scalar};
use coeus_ops::{
    AttentionOps, AttentionScalar, ConvOps, ConvolutionBackward, ConvolutionForward, PoolOps,
    UnfoldFoldOps,
};

pub(crate) fn convolution_preserves_output_clones<T: Scalar, B: ConvOps<T>>(backend: &B, one: T) {
    let _span = tracing::info_span!("convolution_gradient_ownership").entered();
    let two = one + one;
    let three = two + one;
    let four = two + two;
    let input_layout = Layout::new([1, 1, 3].into());
    let weight_layout = Layout::new([1, 1, 2].into());
    let output_layout = Layout::new([1, 1, 2].into());
    let input = upload(backend, &[one, two, four]);
    let weight = upload(backend, &[two, three]);
    let original_output = upload(backend, &[three, one]);
    let mut output = original_output.clone();
    // [1,2] dot [2,3] = 8; [2,4] dot [2,3] = 16.
    backend
        .convolution_forward::<3, 1>(
            ConvolutionForward {
                input: &input,
                input_layout: &input_layout,
                weight: &weight,
                weight_layout: &weight_layout,
                bias: None,
                output: &mut output,
                output_layout: &output_layout,
            },
            [1],
            [0],
            [1],
        )
        .expect("valid convolution forward output");
    assert_values(backend, &input, &[one, two, four]);
    assert_values(backend, &weight, &[two, three]);
    assert_values(backend, &original_output, &[three, one]);
    assert_values(backend, &output, &[four + four, four * four]);

    let upstream = upload(backend, &[one, two]);
    let original_input = upload(backend, &[three, one, two]);
    let original_weight = upload(backend, &[four, one]);
    let original_bias = upload(backend, &[two]);
    let mut grad_input = original_input.clone();
    let mut grad_weight = original_weight.clone();
    let mut grad_bias = original_bias.clone();
    // The later weight-gradient descriptor exceeds its allocation.
    let invalid_weight_layout = Layout::new([1, 1, 3].into());
    backend
        .convolution_backward::<3, 1>(
            ConvolutionBackward {
                grad_output: &upstream,
                grad_output_layout: &output_layout,
                input: &input,
                input_layout: &input_layout,
                weight: &weight,
                weight_layout: &weight_layout,
                grad_input: Some(&mut grad_input),
                grad_input_layout: &input_layout,
                grad_weight: Some(&mut grad_weight),
                grad_weight_layout: &invalid_weight_layout,
                grad_bias: Some(&mut grad_bias),
            },
            [1],
            [0],
            [1],
        )
        .expect_err("the weight-gradient layout exceeds its two-element allocation");
    assert_values(backend, &original_input, &[three, one, two]);
    assert_values(backend, &grad_input, &[three, one, two]);
    assert_values(backend, &original_weight, &[four, one]);
    assert_values(backend, &grad_weight, &[four, one]);
    assert_values(backend, &original_bias, &[two]);
    assert_values(backend, &grad_bias, &[two]);

    grad_input = original_input.clone();
    grad_weight = original_weight.clone();
    grad_bias = original_bias.clone();

    // dInput=[2,7,6], dWeight=[5,10], dBias=3; two calls must add twice.
    for repetition in [one, two] {
        backend
            .convolution_backward::<3, 1>(
                ConvolutionBackward {
                    grad_output: &upstream,
                    grad_output_layout: &output_layout,
                    input: &input,
                    input_layout: &input_layout,
                    weight: &weight,
                    weight_layout: &weight_layout,
                    grad_input: Some(&mut grad_input),
                    grad_input_layout: &input_layout,
                    grad_weight: Some(&mut grad_weight),
                    grad_weight_layout: &weight_layout,
                    grad_bias: Some(&mut grad_bias),
                },
                [1],
                [0],
                [1],
            )
            .expect("valid convolution backward accumulation");
        assert_values(backend, &original_input, &[three, one, two]);
        assert_values(backend, &original_weight, &[four, one]);
        assert_values(backend, &original_bias, &[two]);
        assert_values(
            backend,
            &grad_input,
            &[
                three + repetition * two,
                one + repetition * (four + three),
                two + repetition * (three + three),
            ],
        );
        assert_values(
            backend,
            &grad_weight,
            &[
                four + repetition * (four + one),
                one + repetition * (four + four + two),
            ],
        );
        assert_values(backend, &grad_bias, &[two + repetition * three]);
    }
}

pub(crate) fn attention_preserves_output_clones<T: AttentionScalar, B: AttentionOps<T>>(
    backend: &B,
    one: T,
) {
    let _span = tracing::info_span!("attention_output_ownership").entered();
    let zero = T::zero();
    let two = one + one;
    let three = two + one;
    let four = two + two;
    let six = three + three;
    let half = one / two;
    let query_layout = Layout::new([1, 1, 2].into());
    let key_layout = Layout::new([1, 2, 2].into());
    let value_layout = Layout::new([1, 2, 1].into());
    let output_layout = Layout::new([1, 1, 1].into());
    let weights_layout = Layout::new([1, 1, 2].into());
    let query = upload(backend, &[one, zero]);
    let key = upload(backend, &[zero, one, zero, three]);
    let value = upload(backend, &[two, six]);
    let original_output = upload(backend, &[three]);
    let original_weights = upload(backend, &[two, three]);
    let mut output = original_output.clone();
    let mut weights = original_weights.clone();
    // The second forward output is invalid while the first is writable.
    let invalid_weights_layout = Layout::new([1, 1, 3].into());
    backend
        .sdp_attention(
            &query,
            &query_layout,
            &key,
            &key_layout,
            &value,
            &value_layout,
            None,
            None,
            false,
            one,
            &mut output,
            &output_layout,
            &mut weights,
            &invalid_weights_layout,
        )
        .expect_err("the attention weights layout exceeds its two-element allocation");
    assert_values(backend, &original_output, &[three]);
    assert_values(backend, &output, &[three]);
    assert_values(backend, &original_weights, &[two, three]);
    assert_values(backend, &weights, &[two, three]);

    output = original_output.clone();
    weights = original_weights.clone();

    // Orthogonal query/key vectors give exact zero logits and weights [1/2,1/2].
    backend
        .sdp_attention(
            &query,
            &query_layout,
            &key,
            &key_layout,
            &value,
            &value_layout,
            None,
            None,
            false,
            one,
            &mut output,
            &output_layout,
            &mut weights,
            &weights_layout,
        )
        .expect("valid attention forward");
    assert_values(backend, &original_output, &[three]);
    assert_values(backend, &original_weights, &[two, three]);
    assert_values(backend, &output, &[four]);
    assert_values(backend, &weights, &[half, half]);
    let upstream = upload(backend, &[two]);
    let original_q = upload(backend, &[three, two]);
    let original_k = upload(backend, &[four, three, two, one]);
    // Equal-sized gradient roles may share their initial allocation too.
    let original_v = original_q.clone();
    let mut grad_q = original_q.clone();
    let mut grad_k = original_k.clone();
    let mut grad_v = original_v.clone();
    // The last gradient is invalid; earlier gradients must not accumulate.
    let invalid_value_layout = Layout::new([1, 3, 1].into());
    backend
        .sdp_attention_backward(
            &upstream,
            &output_layout,
            &query,
            &query_layout,
            &key,
            &key_layout,
            &value,
            &value_layout,
            &weights,
            &weights_layout,
            one,
            Some((&mut grad_q, &query_layout)),
            Some((&mut grad_k, &key_layout)),
            Some((&mut grad_v, &invalid_value_layout)),
        )
        .expect_err("the value-gradient layout exceeds its two-element allocation");
    assert_values(backend, &original_q, &[three, two]);
    assert_values(backend, &grad_q, &[three, two]);
    assert_values(backend, &original_k, &[four, three, two, one]);
    assert_values(backend, &grad_k, &[four, three, two, one]);
    assert_values(backend, &original_v, &[three, two]);
    assert_values(backend, &grad_v, &[three, two]);

    grad_q = original_q.clone();
    grad_k = original_k.clone();
    grad_v = original_v.clone();

    // dLogits=[-2,2], hence dQ=[0,4], dK=[-2,0,2,0], dV=[1,1].
    for repetition in [one, two] {
        backend
            .sdp_attention_backward(
                &upstream,
                &output_layout,
                &query,
                &query_layout,
                &key,
                &key_layout,
                &value,
                &value_layout,
                &weights,
                &weights_layout,
                one,
                Some((&mut grad_q, &query_layout)),
                Some((&mut grad_k, &key_layout)),
                Some((&mut grad_v, &value_layout)),
            )
            .expect("valid attention backward accumulation");
        assert_values(backend, &original_q, &[three, two]);
        assert_values(backend, &original_k, &[four, three, two, one]);
        assert_values(backend, &original_v, &[three, two]);
        assert_values(backend, &grad_q, &[three, two + repetition * four]);
        assert_values(
            backend,
            &grad_k,
            &[four - repetition * two, three, two + repetition * two, one],
        );
        assert_values(backend, &grad_v, &[three + repetition, two + repetition]);
    }
}

pub(crate) fn pooling_preserves_output_clones<T: Scalar, B: PoolOps<T>>(backend: &B, one: T) {
    let two = one + one;
    let three = two + one;
    let four = two + two;
    let input_layout = Layout::new([1, 1, 3].into());
    let output_layout = Layout::new([1, 1, 2].into());
    let input = upload(backend, &[one, four, two]);
    let original_output = upload(backend, &[three, one]);
    let mut output = original_output.clone();
    // Both [1,4] and [4,2] have maximum 4.
    backend
        .max_pool1d(
            &input,
            &input_layout,
            2,
            1,
            0,
            1,
            &mut output,
            &output_layout,
        )
        .expect("valid overlapping max-pool forward output");
    assert_values(backend, &input, &[one, four, two]);
    assert_values(backend, &original_output, &[three, one]);
    assert_values(backend, &output, &[four, four]);

    let upstream = upload(backend, &[two, three]);
    let original = upload(backend, &[three, two, one]);
    let mut gradient = original.clone();
    // Both overlapping windows select the middle element: dInput=[0,5,0].
    for repetition in [one, two] {
        backend
            .max_pool1d_backward(
                &upstream,
                &output_layout,
                &input,
                &input_layout,
                2,
                1,
                0,
                1,
                &mut gradient,
                &input_layout,
            )
            .expect("valid overlapping max-pool backward");
        assert_values(backend, &original, &[three, two, one]);
        assert_values(
            backend,
            &gradient,
            &[three, two + repetition * (two + three), one],
        );
    }
}

pub(crate) fn windows_preserve_output_clones<T: Scalar, B: UnfoldFoldOps<T>>(backend: &B, one: T) {
    let two = one + one;
    let three = two + one;
    let four = two + two;
    let input_layout = Layout::new([1, 1, 3].into());
    let columns_layout = Layout::new([1, 2, 2].into());
    let input = upload(backend, &[one, two, four]);
    let original_columns = upload(backend, &[four, three, two, one]);
    let original_fold = upload(backend, &[three, one, two]);
    let mut columns = original_columns.clone();
    let mut folded = original_fold.clone();
    backend
        .unfold1d(
            &input,
            &input_layout,
            2,
            1,
            0,
            1,
            &mut columns,
            &columns_layout,
        )
        .expect("valid sliding-window extraction");
    assert_values(backend, &original_columns, &[four, three, two, one]);
    assert_values(backend, &columns, &[one, two, two, four]);
    // Fold starts a new sum: the middle cell appears in both windows.
    backend
        .fold1d(
            &columns,
            &columns_layout,
            3,
            2,
            1,
            0,
            1,
            &mut folded,
            &input_layout,
        )
        .expect("valid overlapping-window fold");
    assert_values(backend, &original_fold, &[three, one, two]);
    assert_values(backend, &folded, &[one, four, four]);
}
