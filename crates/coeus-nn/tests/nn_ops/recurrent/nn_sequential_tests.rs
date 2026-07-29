use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::{Linear, Module, ReLU, Sequential};
use coeus_tensor::Tensor;

#[test]
fn test_sequential_container() {
    let mut seq = Sequential::<f64, MoiraiBackend>::new();
    assert!(seq.is_empty());
    assert_eq!(seq.len(), 0);

    seq.add(Linear::new(3, 4, true));
    seq.add(ReLU);
    seq.add(Linear::new(4, 2, false));

    assert!(!seq.is_empty());
    assert_eq!(seq.len(), 3);

    // parameters check
    let params = seq.parameters();
    assert_eq!(params.len(), 3); // weight1, bias1, weight2
    assert_eq!(params[0].tensor.shape(), &[4, 3]); // weight1
    assert_eq!(params[1].tensor.shape(), &[4]); // bias1
    assert_eq!(params[2].tensor.shape(), &[2, 4]); // weight2

    // Forward pass
    // Input: [batch=2, in_features=3]
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]),
        true,
    );

    let output = seq.forward(&input).expect("valid Sequential input");
    assert_eq!(output.tensor.shape(), &[2, 2]);

    // Backward pass
    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert!(input.grad().is_some());
    assert!(params[0].grad().is_some()); // weight1
    assert!(params[1].grad().is_some()); // bias1
    assert!(params[2].grad().is_some()); // weight2
}
