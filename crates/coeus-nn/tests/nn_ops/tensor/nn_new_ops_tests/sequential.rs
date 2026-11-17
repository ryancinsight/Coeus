use super::*;

fn assert_param_gradient_finite_differences<F>(
    label: &str,
    params: &mut [Var<f64, B>],
    param_index: usize,
    element_index: usize,
    evaluate: &mut F,
) where
    F: FnMut(&[Var<f64, B>]) -> f64,
{
    let step = 1e-6_f64;
    let original = params[param_index].tensor.as_slice()[element_index];
    params[param_index]
        .tensor
        .as_mut_slice()
        .expect("invariant: parameter tensor is host addressable")[element_index] = original + step;
    let plus = evaluate(params);
    params[param_index]
        .tensor
        .as_mut_slice()
        .expect("invariant: parameter tensor is host addressable")[element_index] = original - step;
    let minus = evaluate(params);
    params[param_index]
        .tensor
        .as_mut_slice()
        .expect("invariant: parameter tensor is host addressable")[element_index] = original;
    evaluate(params);
    let numeric = (plus - minus) / (2.0 * step);
    let analytic = params[param_index]
        .grad()
        .expect("invariant: the parameter receives a gradient")
        .as_slice()[element_index];
    assert!(
        (analytic - numeric).abs() <= 1e-5,
        "{label} grad[{element_index}]: analytic {analytic} vs numeric {numeric}"
    );
}

/// Assert a normalization layer's affine gradients from the fixture geometry:
/// each channel receives one upstream unit per element of its group, and the
/// per-channel weight gradients sum to zero within each group.
#[test]
fn test_sequential_chaining() {
    let backend = B::default();
    let mut seq = Sequential::<f64, B>::new();
    seq.add(
        Linear::new(4, 3, true).expect("invariant: the fixture's layer dimensions are non-zero"),
    );
    seq.add(
        Linear::new(3, 2, true).expect("invariant: the fixture's layer dimensions are non-zero"),
    );

    let input = Var::new(
        Tensor::ones_on([2, 4], &backend).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = seq.forward(&input).expect("valid Sequential input");

    assert_eq!(output.tensor.shape(), &[2, 2]);

    let loss = coeus_autograd::sum(&output).expect("invariant: test operation succeeds");
    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");

    let input_grad = input
        .grad()
        .expect("invariant: sequential gives its input a gradient");
    let mut params = seq.parameters();
    assert_eq!(params.len(), 4); // 2 weights + 2 biases
    let input_baseline = [1.0_f64; 8];
    let evaluate_input = |data: &[f64]| -> f64 {
        let probe = Var::new(
            Tensor::from_slice_on([2, 4], data, &backend)
                .expect("invariant: test backend operation succeeds"),
            false,
        )
        .expect("invariant: test backend operation succeeds");
        seq.forward(&probe)
            .expect("invariant: probe forward succeeds")
            .tensor
            .as_slice()
            .iter()
            .sum::<f64>()
    };
    assert_input_gradient_finite_differences(
        &input_baseline,
        input_grad.as_slice(),
        &[0, 3, 7],
        evaluate_input,
    );
    let mut evaluate_param = |loaded: &[Var<f64, B>]| -> f64 {
        seq.load_parameters(loaded);
        seq.forward(&input)
            .expect("invariant: probe forward succeeds")
            .tensor
            .as_slice()
            .iter()
            .sum::<f64>()
    };
    for index in 0..params.len() {
        assert_param_gradient_finite_differences(
            &format!("sequential param {index}"),
            &mut params,
            index,
            0,
            &mut evaluate_param,
        );
    }
}

#[test]
fn test_static_sequential_chaining() {
    let backend = B::default();
    let mut model: StaticSeq<Linear<f64, B>, Linear<f64, B>> = Linear::<f64, B>::new(4, 3, true)
        .expect("invariant: the fixture's layer dimensions are non-zero")
        .append(
            Linear::<f64, B>::new(3, 2, true)
                .expect("invariant: the fixture's layer dimensions are non-zero"),
        );

    let input = Var::new(
        Tensor::ones_on([2, 4], &backend).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = model.forward(&input).expect("valid StaticSeq input");

    assert_eq!(output.tensor.shape(), &[2, 2]);

    let loss = coeus_autograd::sum(&output).expect("invariant: test operation succeeds");
    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");

    let input_grad = input
        .grad()
        .expect("invariant: static sequential gives its input a gradient");
    let mut params = model.parameters();
    assert_eq!(params.len(), 4); // 2 weights + 2 biases
    let input_baseline = [1.0_f64; 8];
    let evaluate_input = |data: &[f64]| -> f64 {
        let probe = Var::new(
            Tensor::from_slice_on([2, 4], data, &backend)
                .expect("invariant: test backend operation succeeds"),
            false,
        )
        .expect("invariant: test backend operation succeeds");
        model
            .forward(&probe)
            .expect("invariant: probe forward succeeds")
            .tensor
            .as_slice()
            .iter()
            .sum::<f64>()
    };
    assert_input_gradient_finite_differences(
        &input_baseline,
        input_grad.as_slice(),
        &[0, 3, 7],
        evaluate_input,
    );
    let mut evaluate_param = |loaded: &[Var<f64, B>]| -> f64 {
        model.load_parameters(loaded);
        model
            .forward(&input)
            .expect("invariant: probe forward succeeds")
            .tensor
            .as_slice()
            .iter()
            .sum::<f64>()
    };
    for index in 0..params.len() {
        assert_param_gradient_finite_differences(
            &format!("static sequential param {index}"),
            &mut params,
            index,
            0,
            &mut evaluate_param,
        );
    }
}
