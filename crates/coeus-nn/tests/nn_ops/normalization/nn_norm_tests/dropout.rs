use coeus_autograd::Var;
use coeus_nn::{Dropout, Module};
use coeus_tensor::Tensor;

#[test]
fn test_dropout() {
    let mut do_layer = Dropout::new(0.5);
    let input: Var<f64> = Var::new(Tensor::ones(vec![100]), true);

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
    assert!(input.grad().is_some());
}
