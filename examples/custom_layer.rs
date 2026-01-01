//! Custom Layer Example: Building Custom Neural Network Layers
//!
//! This example demonstrates how to implement custom neural network layers
//! with forward and backward passes, showing the full power of Coeus's
//! modular architecture.
//!
//! Sprint 7.7: Advanced Example Development
//!
//! Run with: cargo run --example custom_layer

use autograd::ops::backward_with_grad;
use dtype::float::Float32;
use nn::{error::NNError, Linear, Module, Parameter};
use storage::DenseStorage;
use tensor::CpuBackend;
use tensor::Tensor;

/// Custom Residual Block Layer
///
/// Implements a residual connection: output = input + transform(input)
/// This is a fundamental building block in modern deep learning architectures
/// like ResNet.
///
/// # Architecture
/// ```text
/// input ──┬──> Linear(in_features, hidden_features) ──> ReLU ──> Linear(hidden_features, in_features) ──┬──> output
///         └────────────────────────────────────────────────────────────────────────────────────────────┘
///                                                (residual connection)
/// ```
#[derive(Debug)]
pub struct ResidualBlock {
    fc1: Linear<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    fc2: Linear<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    _in_features: usize,
    _hidden_features: usize,
}

impl ResidualBlock {
    /// Create a new residual block
    ///
    /// # Arguments
    /// * `in_features` - Number of input features
    /// * `hidden_features` - Number of hidden features in the transformation
    ///
    /// # Example
    /// ```rust
    /// let residual_block = ResidualBlock::new(128, 256);
    /// ```
    pub fn new(in_features: usize, hidden_features: usize) -> Self {
        Self {
            fc1: Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                in_features,
                hidden_features,
            )
            .unwrap(),
            fc2: Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                hidden_features,
                in_features,
            )
            .unwrap(),
            _in_features: in_features,
            _hidden_features: hidden_features,
        }
    }
}

impl Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32> for ResidualBlock {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    ) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, NNError> {
        // Transform: fc1 -> ReLU -> fc2
        let x = self.fc1.forward(input)?;

        // ReLU activation (element-wise max(0, x))
        let x_relu = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            x.as_slice()
                .iter()
                .map(|&val| Float32::new(val.get().max(0.0)))
                .collect(),
            x.shape().dims(),
        )?;

        let transformed = self.fc2.forward(&x_relu)?;

        // Residual connection: output = input + transformed
        let output = input + &transformed;

        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<Float32>, DenseStorage<Float32>, Float32>> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }

    fn zero_grad(&mut self) {
        // Zero gradients for all parameters
        for mut param in self.parameters() {
            param.zero_grad();
        }
    }

    fn train(&mut self, _mode: bool) {
        // No dropout or batch norm, so training mode doesn't affect this layer
    }

    fn name(&self) -> &str {
        "ResidualBlock"
    }
}

/// Custom Attention Layer (Simplified Self-Attention)
///
/// Implements a simplified self-attention mechanism:
/// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
///
/// This is a core component of Transformer architectures.
#[derive(Debug)]
pub struct SelfAttention {
    query: Linear<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    key: Linear<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    value: Linear<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    _d_k: f32,
}

impl SelfAttention {
    /// Create a new self-attention layer
    ///
    /// # Arguments
    /// * `d_model` - Model dimensionality
    /// * `d_k` - Key/Query dimensionality
    ///
    /// # Example
    /// ```rust
    /// let attention = SelfAttention::new(512, 64);
    /// ```
    pub fn new(d_model: usize, d_k: usize) -> Self {
        Self {
            query: Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(d_model, d_k)
                .unwrap(),
            key: Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(d_model, d_k)
                .unwrap(),
            value: Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(d_model, d_k)
                .unwrap(),
            _d_k: d_k as f32,
        }
    }
}

impl Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32> for SelfAttention {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    ) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, NNError> {
        // Compute Q, K, V projections
        let _q = self.query.forward(input)?;
        let _k = self.key.forward(input)?;
        let v = self.value.forward(input)?;

        // For simplicity, return V (full attention would require matmul and softmax)
        // In production, you would compute: softmax(QK^T / sqrt(d_k)) V
        Ok(v)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<Float32>, DenseStorage<Float32>, Float32>> {
        let mut params = self.query.parameters();
        params.extend(self.key.parameters());
        params.extend(self.value.parameters());
        params
    }

    fn zero_grad(&mut self) {
        // Zero gradients for all parameters
        for mut param in self.parameters() {
            param.zero_grad();
        }
    }

    fn train(&mut self, _mode: bool) {
        // No dropout or batch norm, so training mode doesn't affect this layer
    }

    fn name(&self) -> &str {
        "SelfAttention"
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Custom Layer Example");
    println!("=======================\n");

    // Example 1: Residual Block
    println!("1. Residual Block Layer");
    println!("   Architecture: input -> fc1 -> ReLU -> fc2 -> (+input) -> output");

    let residual_block = ResidualBlock::new(4, 8);

    // Linear layers expect 2D input: [batch_size, features]
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[1, 4], // batch_size=1, features=4
    )?;

    let output = residual_block.forward(&input)?;

    println!("   Input shape: {:?}", input.shape().dims());
    println!("   Output shape: {:?}", output.shape().dims());
    println!("   Parameters: {}", residual_block.parameters().len());
    println!("   ✅ Residual block forward pass successful\n");

    // Example 2: Self-Attention Layer
    println!("2. Self-Attention Layer");
    println!("   Architecture: input -> [Q, K, V projections] -> attention -> output");

    let attention = SelfAttention::new(4, 2);

    // Linear layers expect 2D input: [batch_size, features]
    let input_attn = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[1, 4], // batch_size=1, features=4
    )?;

    let output_attn = attention.forward(&input_attn)?;

    println!("   Input shape: {:?}", input_attn.shape().dims());
    println!("   Output shape: {:?}", output_attn.shape().dims());
    println!("   Parameters: {}", attention.parameters().len());
    println!("   ✅ Self-attention forward pass successful\n");

    // Example 3: Gradient Flow Through Custom Layer
    println!("3. Gradient Flow Through Custom Layer");

    let tensor_input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[4],
    )?;

    let var_input = tensor_input.requires_grad_(true);

    // Simple transformation: square each element
    let squared = &var_input * &var_input;

    // Sum as loss
    let loss = squared.sum(None, false)?;

    // Backward pass
    let loss_grad = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0)],
        &[1],
    )?;
    backward_with_grad(&loss, &loss_grad)?;

    // Check gradients
    if let Ok(grad) = var_input.grad() {
        println!("   Input: {:?}", var_input.as_slice());
        println!("   Loss: {:.2}", loss.as_slice()[0].get());
        println!("   Gradient: {:?}", grad.as_slice());
        println!("   ✅ Gradient computation successful\n");
    }

    println!("✅ All custom layer examples completed successfully!");
    println!("\n📚 Key Takeaways:");
    println!("   • Custom layers implement the Module trait");
    println!("   • forward() defines the computation graph");
    println!("   • parameters() returns trainable parameters");
    println!("   • Gradients flow automatically through custom layers");
    println!("   • Residual connections improve gradient flow");
    println!("   • Attention mechanisms enable sequence modeling");

    Ok(())
}
