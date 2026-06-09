use coeus_autograd::Var;
use coeus_nn::{init, Embedding, Module};
use coeus_tensor::{Tensor, Transpose};

#[test]
fn test_embedding_forward_backward_indices() {
    let mut layer = Embedding::<f64>::new(5, 3);
    // Initialize weight matrix to constant value of 2.0
    init::constant(&mut layer.weight, 2.0);

    // Indices to look up: shape [2, 2]
    let indices = Tensor::<i32, _>::from_slice(vec![2, 2], &[0, 2, 4, 1]);

    // Perform forward pass using forward_indices
    let output = layer.forward_indices(&indices);

    // Expected output shape: [2, 2, 3]
    assert_eq!(output.tensor.shape(), &[2, 2, 3]);

    // Since weights are all 2.0, output should be all 2.0
    for &val in output.tensor.as_slice() {
        assert_eq!(val, 2.0);
    }

    // Set custom weight values to verify lookup values
    // Weight shape: [5, 3]
    let w_data = vec![
        1.0, 1.1, 1.2, // row 0
        2.0, 2.1, 2.2, // row 1
        3.0, 3.1, 3.2, // row 2
        4.0, 4.1, 4.2, // row 3
        5.0, 5.1, 5.2, // row 4
    ];
    layer.weight.tensor = Tensor::from_slice(vec![5, 3], &w_data);

    let output = layer.forward_indices(&indices);
    // Indices are:
    // [0, 2]
    // [4, 1]
    // Expected output:
    // [row 0, row 2]
    // [row 4, row 1]
    let out_slice = output.tensor.as_slice();
    // row 0
    assert_eq!(&out_slice[0..3], &[1.0, 1.1, 1.2]);
    // row 2
    assert_eq!(&out_slice[3..6], &[3.0, 3.1, 3.2]);
    // row 4
    assert_eq!(&out_slice[6..9], &[5.0, 5.1, 5.2]);
    // row 1
    assert_eq!(&out_slice[9..12], &[2.0, 2.1, 2.2]);

    // Backward pass
    output.backward();

    // Verify gradients on weights
    let weight_grad = layer.weight.grad().unwrap();
    assert_eq!(weight_grad.shape(), &[5, 3]);

    // Gradients should be accumulated based on the lookup frequency of each token index:
    // index 0: 1 time -> grad row 0 should be [1.0, 1.0, 1.0]
    // index 1: 1 time -> grad row 1 should be [1.0, 1.0, 1.0]
    // index 2: 1 time -> grad row 2 should be [1.0, 1.0, 1.0]
    // index 3: 0 times -> grad row 3 should be [0.0, 0.0, 0.0]
    // index 4: 1 time -> grad row 4 should be [1.0, 1.0, 1.0]
    let g_slice = weight_grad.as_slice();
    assert_eq!(&g_slice[0..3], &[1.0, 1.0, 1.0]);
    assert_eq!(&g_slice[3..6], &[1.0, 1.0, 1.0]);
    assert_eq!(&g_slice[6..9], &[1.0, 1.0, 1.0]);
    assert_eq!(&g_slice[9..12], &[0.0, 0.0, 0.0]);
    assert_eq!(&g_slice[12..15], &[1.0, 1.0, 1.0]);
}

#[test]
fn test_embedding_module_forward() {
    let mut layer = Embedding::<f64>::new(3, 2);
    let w_data = vec![
        1.0, 2.0, // row 0
        3.0, 4.0, // row 1
        5.0, 6.0, // row 2
    ];
    layer.weight.tensor = Tensor::from_slice(vec![3, 2], &w_data);

    // Module::forward takes &Var<T, B>
    let input = Var::new(Tensor::from_slice(vec![2], &[2.0f64, 0.0]), false);
    let output = layer.forward(&input);

    assert_eq!(output.tensor.shape(), &[2, 2]);
    let out_slice = output.tensor.as_slice();
    // row 2
    assert_eq!(&out_slice[0..2], &[5.0, 6.0]);
    // row 0
    assert_eq!(&out_slice[2..4], &[1.0, 2.0]);
}

#[test]
fn test_embedding_non_contiguous() {
    let mut layer = Embedding::<f64>::new(3, 2);
    let w_raw = Tensor::from_slice(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let w_t = w_raw.transpose(); // shape [3, 2], non-contiguous
    assert!(!w_t.is_contiguous());
    layer.weight.tensor = w_t;

    // Index tensor is also non-contiguous:
    let idx_raw = Tensor::<i32, _>::from_slice(vec![2, 2], &[0, 1, 2, 0]);
    let idx_t = idx_raw.transpose(); // shape [2, 2], non-contiguous
    assert!(!idx_t.is_contiguous());

    let output = layer.forward_indices(&idx_t);
    // idx_t is:
    // [0, 2]
    // [1, 0]
    // Weight transposed:
    // row 0: [1.0, 4.0]
    // row 1: [2.0, 5.0]
    // row 2: [3.0, 6.0]
    // Output:
    // [row 0, row 2]
    // [row 1, row 0]
    assert_eq!(output.tensor.shape(), &[2, 2, 2]);
    let out_slice = output.tensor.as_slice();
    assert_eq!(&out_slice[0..2], &[1.0, 4.0]);
    assert_eq!(&out_slice[2..4], &[3.0, 6.0]);
    assert_eq!(&out_slice[4..6], &[2.0, 5.0]);
    assert_eq!(&out_slice[6..8], &[1.0, 4.0]);

    // Backward pass
    let grad_out = Tensor::from_slice(vec![2, 2, 2], &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
    output.backward_with_seed(grad_out);

    let weight_grad = layer.weight.grad().unwrap();
    assert_eq!(weight_grad.shape(), &[3, 2]);
    let g_slice = weight_grad.as_slice();
    assert_eq!(&g_slice[0..2], &[5.0, 5.0]);
    assert_eq!(&g_slice[2..4], &[3.0, 3.0]);
    assert_eq!(&g_slice[4..6], &[2.0, 2.0]);
}
