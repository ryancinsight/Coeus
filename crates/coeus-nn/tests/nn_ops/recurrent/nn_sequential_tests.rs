use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::{Linear, Module, ReLU, Sequential};
use coeus_tensor::Tensor;

/// Check input-gradient entries against central differences of the scalar
/// forward reduction, varying the input data while the module stays fixed.
fn assert_input_gradient_finite_differences<F>(
    baseline: &[f64],
    grad: &[f64],
    indices: &[usize],
    evaluate: F,
) where
    F: Fn(&[f64]) -> f64,
{
    let step = 1e-6_f64;
    for &index in indices {
        let mut plus = baseline.to_vec();
        plus[index] += step;
        let mut minus = baseline.to_vec();
        minus[index] -= step;
        let numeric = (evaluate(&plus) - evaluate(&minus)) / (2.0 * step);
        let analytic = grad[index];
        assert!(
            (analytic - numeric).abs() <= 1e-5,
            "input grad[{index}]: analytic {analytic} vs numeric {numeric}"
        );
    }
}

/// Check a parameter-gradient entry against a central difference. `evaluate`
/// installs the perturbed parameter list into the module and returns the
/// scalar forward reduction; `Module::parameters` returns copies, so the
/// perturbed value must be loaded rather than mutated behind the module.
fn assert_param_gradient_finite_differences<F>(
    label: &str,
    params: &mut [Var<f64, MoiraiBackend>],
    param_index: usize,
    element_index: usize,
    evaluate: &mut F,
) where
    F: FnMut(&[Var<f64, MoiraiBackend>]) -> f64,
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

#[test]
fn test_sequential_container() {
    let mut seq = Sequential::<f64, MoiraiBackend>::new();
    assert!(seq.is_empty());
    assert_eq!(seq.len(), 0);

    seq.add(
        Linear::new(3, 4, true).expect("invariant: the fixture's layer dimensions are non-zero"),
    );
    seq.add(ReLU);
    seq.add(
        Linear::new(4, 2, false).expect("invariant: the fixture's layer dimensions are non-zero"),
    );

    assert!(!seq.is_empty());
    assert_eq!(seq.len(), 3);

    // parameters check
    let mut params = seq.parameters();
    assert_eq!(params.len(), 3); // weight1, bias1, weight2
    assert_eq!(params[0].tensor.shape(), &[4, 3]); // weight1
    assert_eq!(params[1].tensor.shape(), &[4]); // bias1
    assert_eq!(params[2].tensor.shape(), &[2, 4]); // weight2

    // Forward pass
    // Input: [batch=2, in_features=3]
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let output = seq.forward(&input).expect("valid Sequential input");
    assert_eq!(output.tensor.shape(), &[2, 2]);

    // Backward pass
    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    // Central differences on sum(output): vary the input data while the
    // module stays fixed, then perturb each parameter leaf in place.
    let input_baseline = [1.0_f64, 2.0, 3.0, -1.0, -2.0, -3.0];
    let evaluate_input = |data: &[f64]| -> f64 {
        let probe = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([2, 3], data)
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
        input
            .grad()
            .expect("invariant: input receives a gradient")
            .as_slice(),
        &[0, 2, 4],
        evaluate_input,
    );
    let mut evaluate_param = |loaded: &[Var<f64, MoiraiBackend>]| -> f64 {
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
