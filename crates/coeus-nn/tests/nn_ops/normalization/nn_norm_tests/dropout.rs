use coeus_autograd::Var;
use coeus_nn::{Dropout, Module};
use coeus_tensor::Tensor;

#[test]
fn test_dropout() {
    let mut do_layer = Dropout::new(0.5);
    let input: Var<f64> = Var::new(
        Tensor::ones(vec![100]).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    // Evaluation mode: no dropout, output should be identical
    do_layer.set_training(false);
    let out_eval = do_layer.forward(&input).expect("valid Dropout input");
    assert_eq!(out_eval.tensor.as_slice(), input.tensor.as_slice());

    // Training mode: elements should be dropped or scaled by 2.0
    do_layer.set_training(true);
    let out_train = do_layer.forward(&input).expect("valid Dropout input");
    let o_slice = out_train.tensor.as_slice();

    let zero_count = o_slice.iter().filter(|&&x| x == 0.0).count();
    let scale_count = o_slice.iter().filter(|&&x| x == 2.0).count();

    assert!(zero_count > 0);
    assert!(scale_count > 0);
    assert_eq!(zero_count + scale_count, 100);

    // Backward pass should propagate through non-zeroed masks
    out_train
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let input_grad = input
        .grad()
        .expect("invariant: dropout gives its input a gradient");
    assert_eq!(input_grad.shape(), &[100]);
    // Inverted dropout keeps an element with the scale 1 / (1 - p) and zeroes
    // the dropped ones, so for a unit input the input gradient equals the
    // forward output elementwise.
    let mut grad_zero_count = 0;
    let mut grad_scale_count = 0;
    for (index, (&grad, &output)) in input_grad.as_slice().iter().zip(o_slice.iter()).enumerate() {
        assert!(
            (grad - output).abs() <= 1e-12,
            "dropout grad[{index}] = {grad}, expected {output}"
        );
        if grad == 0.0 {
            grad_zero_count += 1;
        } else {
            grad_scale_count += 1;
        }
    }
    assert_eq!(grad_zero_count, zero_count);
    assert_eq!(grad_scale_count, scale_count);
}
