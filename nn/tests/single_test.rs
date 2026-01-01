use backend::CpuBackend;
use dtype::float::Float32;
use nn::{Module, PReLU};
use storage::DenseStorage;
use tensor::Tensor;

#[test]
fn test_prelu_gradient_flow() {
    // Modified to force recompile
    let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1, None);

    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
            vec![Float32::new(1.0), Float32::new(-1.0)],
            &[2],
            CpuBackend::new(),
        )
        .unwrap()
        .requires_grad_(true);

    let output = prelu.forward(&input).unwrap();

    // Output should require gradients
    assert!(output.requires_grad());

    // Parameters should require gradients
    let params = prelu.parameters();
    assert!(params.iter().all(|p| p.requires_grad()));
}
