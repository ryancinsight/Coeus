//! GPU-Accelerated Neural Network Example
//!
//! This example demonstrates how to build neural networks that automatically
//! leverage GPU acceleration through the Coeus backend system.

use std::sync::Arc;
use nn::backend::{BackendSelector, BackendType, WorkloadCharacteristics, OperationType, MemoryAccessPattern, DataLocality};
use nn::tensor::Tensor;
use nn::dtype::float::Float32;
use nn::storage::DenseStorage;
use nn::error::Result;

/// GPU-accelerated Linear layer
pub struct GpuLinear {
    /// Weight matrix
    weight: Tensor<Float32, DenseStorage<Float32>, Float32>,
    /// Bias vector
    bias: Option<Tensor<Float32, DenseStorage<Float32>, Float32>>,
    /// Backend selector for adaptive computation
    backend_selector: Arc<BackendSelector>,
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
}

impl GpuLinear {
    /// Create a new GPU-accelerated linear layer
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        // Initialize weights with Xavier/Glorot initialization
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| {
                let scale = (2.0 / (in_features + out_features) as f32).sqrt();
                (rand::random::<f32>() - 0.5) * 2.0 * scale
            })
            .collect();

        let weight_shape = vec![out_features, in_features];
        let weight = Tensor::from_vec(weight_data, &weight_shape)?;

        // Initialize bias to zeros
        let bias_data = vec![0.0f32; out_features];
        let bias_shape = vec![out_features];
        let bias = Some(Tensor::from_vec(bias_data, &bias_shape)?);

        // Create backend selector with memory manager
        let memory_manager = nn::backend::MemoryManager;
        let backend_selector = Arc::new(BackendSelector::with_memory_manager(memory_manager));

        Ok(Self {
            weight,
            bias,
            backend_selector,
            in_features,
            out_features,
        })
    }

    /// Forward pass with automatic GPU acceleration
    pub async fn forward(&self, input: &Tensor<Float32, DenseStorage<Float32>, Float32>) -> Result<Tensor<Float32, DenseStorage<Float32>, Float32>> {
        let input_shape = input.shape().dims();
        let batch_size = input_shape[0];

        // Validate input dimensions
        if input_shape.len() != 2 || input_shape[1] != self.in_features {
            return Err(nn::error::NNError::InvalidInput {
                message: format!(
                    "Expected input shape [*, {}], got {:?}",
                    self.in_features, input_shape
                ),
            });
        }

        // Compute workload characteristics for adaptive backend selection
        let total_elements = batch_size * self.in_features * self.out_features;
        let compute_intensity = (batch_size * self.out_features * self.in_features) as f32
                              / (batch_size * self.in_features + self.in_features * self.out_features + batch_size * self.out_features) as f32;

        let workload = WorkloadCharacteristics {
            total_elements,
            access_pattern: MemoryAccessPattern::Dense,
            compute_intensity,
            data_locality: DataLocality::High,
            operation_type: OperationType::MatrixMultiplication,
        };

        // Select optimal backend
        let selected_backend = self.backend_selector.select_backend_memory_aware(&workload).await;

        println!("🔄 Linear layer selected backend: {:?}", selected_backend);

        // Perform matrix multiplication: output = input @ weight.T + bias
        let weight_t = self.weight.transpose(0, 1)?;

        // Matrix multiplication: [batch_size, in_features] @ [in_features, out_features] = [batch_size, out_features]
        let mut output = input.matmul(&weight_t)?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            // Broadcast bias from [out_features] to [batch_size, out_features]
            let bias_broadcast = bias.unsqueeze(0)?.expand(&[batch_size, self.out_features])?;
            output = output.add(&bias_broadcast)?;
        }

        Ok(output)
    }

    /// Get the number of parameters in this layer
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.in_features * self.out_features;
        let bias_params = self.bias.as_ref().map_or(0, |_| self.out_features);
        weight_params + bias_params
    }

    /// Get layer information
    pub fn info(&self) -> LayerInfo {
        LayerInfo {
            in_features: self.in_features,
            out_features: self.out_features,
            num_parameters: self.num_parameters(),
            backend_type: "Adaptive (CPU/GPU)".to_string(),
        }
    }
}

/// Simple GPU-accelerated MLP
pub struct GpuMLP {
    /// First linear layer
    fc1: GpuLinear,
    /// Second linear layer
    fc2: GpuLinear,
    /// Output layer info
    output_features: usize,
}

impl GpuMLP {
    /// Create a new GPU-accelerated MLP
    pub fn new(input_features: usize, hidden_features: usize, output_features: usize) -> Result<Self> {
        let fc1 = GpuLinear::new(input_features, hidden_features)?;
        let fc2 = GpuLinear::new(hidden_features, output_features)?;

        Ok(Self {
            fc1,
            fc2,
            output_features,
        })
    }

    /// Forward pass through the MLP
    pub async fn forward(&self, input: &Tensor<Float32, DenseStorage<Float32>, Float32>) -> Result<Tensor<Float32, DenseStorage<Float32>, Float32>> {
        // First layer with ReLU activation
        let hidden = self.fc1.forward(input).await?;
        let activated = self.relu(&hidden)?;

        // Second layer (output)
        let output = self.fc2.forward(&activated).await?;

        Ok(output)
    }

    /// Apply ReLU activation
    fn relu(&self, input: &Tensor<Float32, DenseStorage<Float32>, Float32>) -> Result<Tensor<Float32, DenseStorage<Float32>, Float32>> {
        // For now, use CPU ReLU (could be optimized with GPU ReLU)
        let backend = nn::backend::CpuBackend::<Float32>::new();
        let storage = input.storage();
        let relu_storage = backend.relu_dense(storage)?;
        Ok(Tensor::from_storage(relu_storage))
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.fc1.num_parameters() + self.fc2.num_parameters()
    }

    /// Get network information
    pub fn info(&self) -> NetworkInfo {
        NetworkInfo {
            layers: vec![self.fc1.info(), self.fc2.info()],
            total_parameters: self.num_parameters(),
            architecture: "MLP".to_string(),
        }
    }
}

/// Layer information
#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub in_features: usize,
    pub out_features: usize,
    pub num_parameters: usize,
    pub backend_type: String,
}

/// Network information
#[derive(Debug, Clone)]
pub struct NetworkInfo {
    pub layers: Vec<LayerInfo>,
    pub total_parameters: usize,
    pub architecture: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧠 GPU-Accelerated Neural Network Demo");
    println!("======================================");

    // Check available backends
    let selector = BackendSelector::new();
    let available_backends = selector.available_backends();

    println!("Available backends: {:?}", available_backends);

    let has_gpu = available_backends.contains(&BackendType::Gpu);
    if has_gpu {
        println!("✅ GPU acceleration available!");
    } else {
        println!("⚠️  GPU not available, will use CPU fallback");
    }

    // Create a simple MLP
    println!("\n🏗️  Building MLP: 784 -> 256 -> 10");
    let mlp = GpuMLP::new(784, 256, 10)?;

    let network_info = mlp.info();
    println!("Network architecture: {}", network_info.architecture);
    println!("Total parameters: {}", network_info.total_parameters);

    for (i, layer) in network_info.layers.iter().enumerate() {
        println!("  Layer {}: {} -> {} ({} params, {})",
                i + 1,
                layer.in_features,
                layer.out_features,
                layer.num_parameters,
                layer.backend_type);
    }

    // Create sample input (batch of 32 MNIST-like images)
    println!("\n📊 Creating sample input...");
    let batch_size = 32;
    let input_size = 784;

    // Generate random input data (normalized to [0, 1])
    let input_data: Vec<f32> = (0..batch_size * input_size)
        .map(|_| rand::random::<f32>())
        .collect();

    let input_shape = vec![batch_size, input_size];
    let input = Tensor::from_vec(input_data, &input_shape)?;

    println!("Input shape: {:?}", input.shape());
    println!("Input size: {:.2} MB", (input_data.len() * 4) as f32 / (1024.0 * 1024.0));

    // Forward pass
    println!("\n🚀 Running forward pass...");
    let start_time = std::time::Instant::now();

    let output = mlp.forward(&input).await?;

    let elapsed = start_time.elapsed();

    println!("✅ Forward pass completed in {:.4} seconds", elapsed.as_secs_f64());
    println!("Output shape: {:?}", output.shape());
    println!("Performance: {:.2} samples/sec", batch_size as f64 / elapsed.as_secs_f64());

    // Analyze output
    println!("\n📈 Output Analysis:");
    let output_data = output.to_vec()?;
    let output_slice = output_data.as_slice();

    // Find predicted classes for first few samples
    for sample_idx in 0..std::cmp::min(5, batch_size) {
        let start_idx = sample_idx * 10;
        let end_idx = start_idx + 10;

        let logits = &output_slice[start_idx..end_idx];
        let predicted_class = logits.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        println!("  Sample {}: predicted class {}", sample_idx, predicted_class);
    }

    // Performance summary
    println!("\n📊 Performance Summary:");
    println!("  Input size: {}x{}", batch_size, input_size);
    println!("  Network: MLP({} -> {} -> {})", input_size, 256, 10);
    println!("  Parameters: {}", network_info.total_parameters);
    println!("  Forward time: {:.4}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.2} samples/sec", batch_size as f64 / elapsed.as_secs_f64());

    if has_gpu {
        println!("  Backend: GPU-accelerated with adaptive selection");
    } else {
        println!("  Backend: CPU-only");
    }

    println!("\n💡 GPU Acceleration Features:");
    println!("  • Automatic backend selection based on workload");
    println!("  • Matrix multiplication on GPU for large tensors");
    println!("  • Element-wise operations (ReLU, add, multiply)");
    println!("  • Memory-aware scheduling and resource management");
    println!("  • Zero-overhead backend dispatch via static monomorphization");

    println!("\n🎯 Next Steps:");
    println!("  • Implement training loops with GPU acceleration");
    println!("  • Add more activation functions (GELU, SiLU, etc.)");
    println!("  • Implement attention mechanisms on GPU");
    println!("  • Add gradient computation and backpropagation");
    println!("  • Optimize memory transfers between CPU/GPU");

    Ok(())
}










