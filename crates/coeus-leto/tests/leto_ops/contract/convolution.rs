use coeus_core::Layout;
use coeus_leto::{
    convolution_backward_accumulate, convolution_forward_into,
    convolution_transposed_backward_accumulate, convolution_transposed_forward_into,
    ConvolutionBackward, ConvolutionForward, ConvolutionGradients, ReadOperand, WriteOperand,
};
use leto_ops::{ConvolutionParameters, TransposedConvolutionParameters};

#[test]
fn regular_convolution_routes_forward_and_backward_through_leto() {
    let input_layout = Layout::new([1, 1, 3].into());
    let weight_layout = Layout::new([1, 1, 2].into());
    let output_layout = Layout::new([1, 1, 2].into());
    let input = [1.0_f32, 2.0, 3.0];
    let weight = [2.0_f32, 1.0];
    let bias = [1.0_f32];
    let mut output = [0.0_f32; 2];
    let parameters =
        ConvolutionParameters::new([1], [0], [1]).expect("valid convolution parameters");

    convolution_forward_into::<_, 3, 1>(
        ConvolutionForward {
            input: ReadOperand {
                layout: &input_layout,
                data: &input,
            },
            weight: ReadOperand {
                layout: &weight_layout,
                data: &weight,
            },
            bias: Some(&bias),
            output: WriteOperand {
                layout: &output_layout,
                data: &mut output,
            },
        },
        parameters,
    )
    .expect("regular convolution forward");
    assert_eq!(output, [5.0, 8.0]);

    let grad_output = [1.0_f32; 2];
    let mut grad_input = [0.0_f32; 3];
    let mut grad_weight = [0.0_f32; 2];
    let mut grad_bias = [0.0_f32];
    convolution_backward_accumulate::<_, 3, 1>(
        ConvolutionBackward {
            input: ReadOperand {
                layout: &input_layout,
                data: &input,
            },
            weight: ReadOperand {
                layout: &weight_layout,
                data: &weight,
            },
            grad_output: ReadOperand {
                layout: &output_layout,
                data: &grad_output,
            },
            gradients: ConvolutionGradients {
                input: Some(WriteOperand {
                    layout: &input_layout,
                    data: &mut grad_input,
                }),
                weight: Some(WriteOperand {
                    layout: &weight_layout,
                    data: &mut grad_weight,
                }),
                bias: Some(&mut grad_bias),
            },
        },
        parameters,
    )
    .expect("regular convolution backward");

    assert_eq!(grad_input, [2.0, 3.0, 1.0]);
    assert_eq!(grad_weight, [3.0, 5.0]);
    assert_eq!(grad_bias, [2.0]);
}

#[test]
fn transposed_convolution_routes_forward_and_backward_through_leto() {
    let input_layout = Layout::new([1, 1, 2].into());
    let weight_layout = Layout::new([1, 1, 2].into());
    let output_layout = Layout::new([1, 1, 3].into());
    let input = [1.0_f32, 2.0];
    let weight = [2.0_f32, 1.0];
    let mut output = [0.0_f32; 3];
    let parameters = TransposedConvolutionParameters::new([1], [0], [0], [1])
        .expect("valid transposed-convolution parameters");

    convolution_transposed_forward_into::<_, 3, 1>(
        ConvolutionForward {
            input: ReadOperand {
                layout: &input_layout,
                data: &input,
            },
            weight: ReadOperand {
                layout: &weight_layout,
                data: &weight,
            },
            bias: None,
            output: WriteOperand {
                layout: &output_layout,
                data: &mut output,
            },
        },
        parameters,
    )
    .expect("transposed convolution forward");
    assert_eq!(output, [2.0, 5.0, 2.0]);

    let grad_output = [1.0_f32; 3];
    let mut grad_input = [0.0_f32; 2];
    let mut grad_weight = [0.0_f32; 2];
    let mut grad_bias = [0.0_f32];
    convolution_transposed_backward_accumulate::<_, 3, 1>(
        ConvolutionBackward {
            input: ReadOperand {
                layout: &input_layout,
                data: &input,
            },
            weight: ReadOperand {
                layout: &weight_layout,
                data: &weight,
            },
            grad_output: ReadOperand {
                layout: &output_layout,
                data: &grad_output,
            },
            gradients: ConvolutionGradients {
                input: Some(WriteOperand {
                    layout: &input_layout,
                    data: &mut grad_input,
                }),
                weight: Some(WriteOperand {
                    layout: &weight_layout,
                    data: &mut grad_weight,
                }),
                bias: Some(&mut grad_bias),
            },
        },
        parameters,
    )
    .expect("transposed convolution backward");

    assert_eq!(grad_input, [3.0, 3.0]);
    assert_eq!(grad_weight, [3.0, 3.0]);
    assert_eq!(grad_bias, [3.0]);
}
