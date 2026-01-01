//! ONNX Ecosystem Expansion Demo
//!
//! This example demonstrates the expanded ONNX support in Coeus,
//! including export capabilities for Conv2D, BatchNorm2d, activation functions,
//! and pooling operations.

use backend::CpuBackend;
use dtype::float::Float32;
use nn::{AvgPool2d, BatchNorm2d, Conv2D, MaxPool2d, Module, OnnxExporter, ReLU, Sequential};
use storage::DenseStorage;
use tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Coeus ONNX Ecosystem Expansion Demo");
    println!("=====================================");

    // Create a simple CNN model
    let mut model = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();

    // Add Conv2D layer
    let conv = Conv2D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        3,
        64,
        (3, 3),
        None,
        None,
        Some(true),
    )?;
    model.add_module("conv1".to_string(), Box::new(conv));

    // Add BatchNorm2d
    let batchnorm =
        BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(64, 1e-5, 0.1)?;
    model.add_module("bn1".to_string(), Box::new(batchnorm));

    // Add ReLU activation
    let relu = ReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
    model.add_module("relu1".to_string(), Box::new(relu));

    // Add MaxPool2d
    let maxpool = MaxPool2d::new((2, 2), Some((2, 2)), (0, 0));
    model.add_module("maxpool1".to_string(), Box::new(maxpool));

    // Add AvgPool2d
    let avgpool = AvgPool2d::new((2, 2), Some((2, 2)), (0, 0));
    model.add_module("avgpool1".to_string(), Box::new(avgpool));

    println!("✅ Created CNN model with:");
    println!("   - Conv2D (3→64 channels, 3×3 kernel)");
    println!("   - BatchNorm2d (64 features)");
    println!("   - ReLU activation");
    println!("   - MaxPool2d (2×2 kernel, stride 2)");
    println!("   - AvgPool2d (2×2 kernel, stride 2)");

    // Create input tensor for export
    let input_shape = [1, 3, 32, 32]; // [batch, channels, height, width]
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&input_shape)?;

    println!("\n📤 Exporting model to ONNX format...");

    // Export to ONNX
    let mut exporter = OnnxExporter::new();
    let onnx_bytes = exporter.export(&model, &[3, 32, 32])?;

    println!("✅ Successfully exported model to ONNX!");
    println!("   - ONNX data size: {} bytes", onnx_bytes.len());

    // Parse and display ONNX structure
    let onnx_model: serde_json::Value = serde_json::from_slice(&onnx_bytes)?;
    if let Some(graph) = onnx_model.get("graph") {
        if let Some(nodes) = graph.get("nodes").and_then(|n| n.as_array()) {
            println!("   - Graph contains {} nodes:", nodes.len());
            for (i, node) in nodes.iter().enumerate() {
                if let Some(op_type) = node.get("op_type").and_then(|o| o.as_str()) {
                    println!("     {}. {}", i + 1, op_type);
                }
            }
        }

        if let Some(initializers) = graph.get("initializers").and_then(|i| i.as_array()) {
            println!("   - {} weight tensors initialized", initializers.len());
        }
    }

    println!("\n🎯 ONNX Ecosystem Features Demonstrated:");
    println!("   ✓ Conv2D layer export");
    println!("   ✓ BatchNorm2d layer export");
    println!("   ✓ Activation function export");
    println!("   ✓ MaxPool2d layer export");
    println!("   ✓ AvgPool2d layer export");
    println!("   ✓ JSON-based ONNX serialization");
    println!("   ✓ Operator coverage expansion");

    println!("\n🚀 Ready for ONNX Runtime integration!");
    println!("   (Future enhancement: protobuf support, ONNX Runtime inference)");

    Ok(())
}
