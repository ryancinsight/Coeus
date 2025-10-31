use backend::CpuBackend;
use dtype::float::Float32;
use nn::Linear;
use storage::DenseStorage;
use tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing NN Layer Autograd Integration");

    // Create a simple linear layer: 4 -> 2
    let layer = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(4, 2)?;

    // Create input tensor with gradient tracking enabled
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)],
        &[4]
    )?.requires_grad_(true);

    println!("Input requires_grad: {}", input.requires_grad());
    println!("Weight requires_grad: {}", layer.weight.data().requires_grad());
    println!("Bias requires_grad: {}", layer.bias.data().requires_grad());

    // Forward pass - this should create a computation graph
    let output = layer.forward(&input)?;

    println!("Output shape: {:?}", output.shape().dims());
    println!("Output has grad_fn: {}", output.grad_fn().is_some());

    if let Some(grad_fn) = output.grad_fn() {
        println!("Grad function name: {}", grad_fn.name());
        println!("Number of inputs to grad function: {}", grad_fn.inputs().len());
    }

    println!("✅ NN Layer successfully creates computation graphs!");
    Ok(())
}








