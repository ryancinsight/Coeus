//! Quantization Crate Usage Examples
//! 
//! This example demonstrates how to use the new quantization crate for:
//! - Symmetric and asymmetric quantization
//! - Different calibration methods
//! - Fake quantization for training
//! - Quantized operations

use coeus_quantization::{
    algorithms::{SymmetricQuantizer, AsymmetricQuantizer, DynamicQuantizer},
    calibration::{EntropyCalibrator, PercentileCalibrator, MSECalibrator},
    fake_quantize::{FakeQuantizeLinear, FakeQuantizeConfig, ObserverType},
    types::{QInt8, QInt4, ScaleZeroPoint},
    kernels::{quantized_add, quantized_matmul},
};
use coeus_tensor::Tensor;
use coeus_backend::CpuBackend;
use coeus_storage::DenseStorage;
use coeus_dtype::float::Float32;

type CpuTensor = Tensor<CpuBackend, DenseStorage<Float32>, Float32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Coeus Quantization Crate Examples ===\n");

    // Example 1: Symmetric Quantization
    symmetric_quantization_example()?;
    
    // Example 2: Asymmetric Quantization
    asymmetric_quantization_example()?;
    
    // Example 3: Dynamic Quantization
    dynamic_quantization_example()?;
    
    // Example 4: Calibration Methods
    calibration_methods_example()?;
    
    // Example 5: Fake Quantization for Training
    fake_quantization_example()?;
    
    // Example 6: Quantized Operations
    quantized_operations_example()?;
    
    // Example 7: Different Precision Levels
    precision_levels_example()?;

    println!("All quantization examples completed successfully!");
    Ok(())
}

fn symmetric_quantization_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Symmetric Quantization Example");
    println!("==================================");
    
    // Create quantizer for 8-bit symmetric quantization
    let quantizer = SymmetricQuantizer::<Float32>::new(8)?;
    
    // Create sample data for calibration
    let calibration_data = CpuTensor::from_vec(
        vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0, 0.5, -0.5], 
        &[8]
    )?;
    
    // Calibrate to determine scale
    let scale = quantizer.calibrate(&calibration_data)?;
    println!("Calibrated scale: {:.6}", scale);
    
    // Quantize input tensor
    let input = CpuTensor::from_vec(vec![1.5, -2.3, 0.8, -0.1], &[4])?;
    println!("Original values: {:?}", input.data().as_slice());
    
    let quantized = quantizer.quantize(&input, scale)?;
    println!("Quantized values: {:?}", quantized.data().as_slice());
    
    // Dequantize back to float
    let dequantized = quantizer.dequantize(&quantized, scale)?;
    println!("Dequantized values: {:?}", dequantized.data().as_slice());
    
    // Calculate quantization error
    let error = calculate_mse(&input, &dequantized)?;
    println!("Quantization MSE: {:.6}\n", error);
    
    Ok(())
}

fn asymmetric_quantization_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Asymmetric Quantization Example");
    println!("===================================");
    
    // Create quantizer for 8-bit asymmetric quantization
    let quantizer = AsymmetricQuantizer::<Float32>::new(8)?;
    
    // Create sample data with asymmetric range
    let calibration_data = CpuTensor::from_vec(
        vec![0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
        &[8]
    )?;
    
    // Calibrate to get scale and zero_point
    let (scale, zero_point) = quantizer.calibrate(&calibration_data)?;
    println!("Calibrated scale: {:.6}, zero_point: {}", scale, zero_point);
    
    // Quantize input tensor
    let input = CpuTensor::from_vec(vec![0.2, 1.8, 3.5, 5.9], &[4])?;
    println!("Original values: {:?}", input.data().as_slice());
    
    let quantized = quantizer.quantize(&input, scale, zero_point)?;
    println!("Quantized values: {:?}", quantized.data().as_slice());
    
    // Dequantize back to float
    let dequantized = quantizer.dequantize(&quantized, scale, zero_point)?;
    println!("Dequantized values: {:?}", dequantized.data().as_slice());
    
    // Calculate quantization error
    let error = calculate_mse(&input, &dequantized)?;
    println!("Quantization MSE: {:.6}\n", error);
    
    Ok(())
}

fn dynamic_quantization_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Dynamic Quantization Example");
    println!("================================");
    
    // Create dynamic quantizer (calibrates at runtime)
    let quantizer = DynamicQuantizer::<Float32>::new(8)?;
    
    // Quantize with automatic calibration
    let input = CpuTensor::from_vec(vec![0.1, 2.5, -1.8, 4.2, -3.1], &[5])?;
    println!("Original values: {:?}", input.data().as_slice());
    
    let quantized = quantizer.quantize_dynamic(&input)?;
    println!("Quantized values: {:?}", quantized.data().as_slice());
    
    let dequantized = quantizer.dequantize_dynamic(&quantized)?;
    println!("Dequantized values: {:?}", dequantized.data().as_slice());
    
    // Calculate quantization error
    let error = calculate_mse(&input, &dequantized)?;
    println!("Dynamic quantization MSE: {:.6}\n", error);
    
    Ok(())
}

fn calibration_methods_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Calibration Methods Comparison");
    println!("=================================");
    
    // Create sample data with outliers
    let calibration_data = CpuTensor::from_vec(
        vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 10.0], // 10.0 is an outlier
        &[8]
    )?;
    
    // Entropy-based calibration (KL divergence)
    let entropy_calibrator = EntropyCalibrator::new(2048)?; // 2048 histogram bins
    let entropy_scale = entropy_calibrator.calibrate(&calibration_data, 8)?;
    println!("Entropy calibration scale: {:.6}", entropy_scale);
    
    // Percentile-based calibration (handles outliers)
    let percentile_calibrator = PercentileCalibrator::new(99.9)?; // 99.9th percentile
    let percentile_scale = percentile_calibrator.calibrate(&calibration_data, 8)?;
    println!("Percentile calibration scale: {:.6}", percentile_scale);
    
    // MSE-based calibration (minimizes reconstruction error)
    let mse_calibrator = MSECalibrator::new();
    let mse_scale = mse_calibrator.calibrate(&calibration_data, 8)?;
    println!("MSE calibration scale: {:.6}", mse_scale);
    
    // Compare quantization quality
    let test_data = CpuTensor::from_vec(vec![1.0, 1.2, 1.4, 1.6], &[4])?;
    
    let quantizer = SymmetricQuantizer::<Float32>::new(8)?;
    
    let entropy_result = quantizer.dequantize(
        &quantizer.quantize(&test_data, entropy_scale)?, 
        entropy_scale
    )?;
    let entropy_error = calculate_mse(&test_data, &entropy_result)?;
    
    let percentile_result = quantizer.dequantize(
        &quantizer.quantize(&test_data, percentile_scale)?, 
        percentile_scale
    )?;
    let percentile_error = calculate_mse(&test_data, &percentile_result)?;
    
    let mse_result = quantizer.dequantize(
        &quantizer.quantize(&test_data, mse_scale)?, 
        mse_scale
    )?;
    let mse_error = calculate_mse(&test_data, &mse_result)?;
    
    println!("Calibration method comparison:");
    println!("  Entropy MSE: {:.6}", entropy_error);
    println!("  Percentile MSE: {:.6}", percentile_error);
    println!("  MSE-based MSE: {:.6}\n", mse_error);
    
    Ok(())
}

fn fake_quantization_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Fake Quantization for Training");
    println!("==================================");
    
    // Configure fake quantization
    let config = FakeQuantizeConfig {
        bits: 8,
        symmetric: true,
        per_channel: false,
        observer_type: ObserverType::MinMax,
    };
    
    // Create fake quantization layer
    let fake_quant = FakeQuantizeLinear::new(config)?;
    
    // Simulate training scenario
    let weight = CpuTensor::from_vec(
        vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6], 
        &[2, 3]
    )?;
    println!("Original weight: {:?}", weight.data().as_slice());
    
    // Apply fake quantization (simulates quantization without precision loss)
    let fake_quantized_weight = fake_quant.forward(&weight)?;
    println!("Fake quantized weight: {:?}", fake_quantized_weight.data().as_slice());
    
    // In real training, gradients would flow through fake quantization
    println!("Fake quantization preserves gradient flow during training");
    println!("Actual quantization would be applied only during inference\n");
    
    Ok(())
}

fn quantized_operations_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Quantized Operations");
    println!("=======================");
    
    // Create quantized tensors
    let a = CpuTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = CpuTensor::from_vec(vec![0.5, 1.5, 2.5, 3.5], &[2, 2])?;
    
    let quantizer = SymmetricQuantizer::<Float32>::new(8)?;
    let scale_a = quantizer.calibrate(&a)?;
    let scale_b = quantizer.calibrate(&b)?;
    
    let a_quantized = QInt8::from_tensor(&a, scale_a, 0)?;
    let b_quantized = QInt8::from_tensor(&b, scale_b, 0)?;
    
    println!("Tensor A: {:?}", a.data().as_slice());
    println!("Tensor B: {:?}", b.data().as_slice());
    
    // Quantized addition
    let result_params = ScaleZeroPoint::for_add(scale_a, 0, scale_b, 0);
    let sum_quantized = quantized_add(&a_quantized, &b_quantized, result_params)?;
    let sum_dequantized = sum_quantized.to_tensor(result_params.scale, result_params.zero_point)?;
    
    println!("Quantized addition result: {:?}", sum_dequantized.data().as_slice());
    
    // Compare with float addition
    let float_sum = &a + &b;
    let add_error = calculate_mse(&float_sum, &sum_dequantized)?;
    println!("Quantized addition MSE: {:.6}", add_error);
    
    // Quantized matrix multiplication
    let matmul_params = ScaleZeroPoint::for_matmul(scale_a, 0, scale_b, 0);
    let matmul_quantized = quantized_matmul(&a_quantized, &b_quantized, matmul_params)?;
    let matmul_dequantized = matmul_quantized.to_tensor(matmul_params.scale, matmul_params.zero_point)?;
    
    println!("Quantized matmul result: {:?}", matmul_dequantized.data().as_slice());
    
    // Compare with float matmul
    let float_matmul = a.matmul(&b)?;
    let matmul_error = calculate_mse(&float_matmul, &matmul_dequantized)?;
    println!("Quantized matmul MSE: {:.6}\n", matmul_error);
    
    Ok(())
}

fn precision_levels_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("7. Different Precision Levels");
    println!("=============================");
    
    let input = CpuTensor::from_vec(vec![1.234, -2.567, 3.891, -4.123], &[4])?;
    println!("Original values: {:?}", input.data().as_slice());
    
    // 4-bit quantization (8x compression)
    let qint4_tensor = QInt4::from_tensor(&input, 0.1, 0)?;
    let qint4_reconstructed = qint4_tensor.to_tensor(0.1, 0)?;
    let qint4_error = calculate_mse(&input, &qint4_reconstructed)?;
    println!("4-bit quantization:");
    println!("  Reconstructed: {:?}", qint4_reconstructed.data().as_slice());
    println!("  MSE: {:.6}", qint4_error);
    println!("  Memory savings: 8x");
    
    // 8-bit quantization (4x compression)
    let qint8_tensor = QInt8::from_tensor(&input, 0.05, 0)?;
    let qint8_reconstructed = qint8_tensor.to_tensor(0.05, 0)?;
    let qint8_error = calculate_mse(&input, &qint8_reconstructed)?;
    println!("8-bit quantization:");
    println!("  Reconstructed: {:?}", qint8_reconstructed.data().as_slice());
    println!("  MSE: {:.6}", qint8_error);
    println!("  Memory savings: 4x");
    
    // 16-bit quantization (2x compression)
    let qint16_tensor = QInt16::from_tensor(&input, 0.01, 0)?;
    let qint16_reconstructed = qint16_tensor.to_tensor(0.01, 0)?;
    let qint16_error = calculate_mse(&input, &qint16_reconstructed)?;
    println!("16-bit quantization:");
    println!("  Reconstructed: {:?}", qint16_reconstructed.data().as_slice());
    println!("  MSE: {:.6}", qint16_error);
    println!("  Memory savings: 2x");
    
    println!("\nPrecision vs Compression Trade-off:");
    println!("  4-bit: Highest compression, lowest precision");
    println!("  8-bit: Good balance of compression and precision");
    println!("  16-bit: Lower compression, higher precision\n");
    
    Ok(())
}

// Helper function to calculate Mean Squared Error
fn calculate_mse(a: &CpuTensor, b: &CpuTensor) -> Result<f32, Box<dyn std::error::Error>> {
    let diff = a - b;
    let squared = &diff * &diff;
    let sum: f32 = squared.data().as_slice().iter().sum();
    Ok(sum / squared.data().len() as f32)
}

// Additional helper types (these would be defined in the actual quantization crate)
use coeus_quantization::types::{QInt4, QInt16};

// Mock implementations for demonstration (actual implementations would be in quantization crate)
impl QInt4 {
    pub fn from_tensor(tensor: &CpuTensor, scale: f32, zero_point: i32) -> Result<Self, Box<dyn std::error::Error>> {
        // Mock implementation
        Ok(QInt4 { /* fields */ })
    }
    
    pub fn to_tensor(&self, scale: f32, zero_point: i32) -> Result<CpuTensor, Box<dyn std::error::Error>> {
        // Mock implementation - return original tensor for demo
        CpuTensor::zeros(&[4])
    }
}

impl QInt16 {
    pub fn from_tensor(tensor: &CpuTensor, scale: f32, zero_point: i32) -> Result<Self, Box<dyn std::error::Error>> {
        // Mock implementation
        Ok(QInt16 { /* fields */ })
    }
    
    pub fn to_tensor(&self, scale: f32, zero_point: i32) -> Result<CpuTensor, Box<dyn std::error::Error>> {
        // Mock implementation - return original tensor for demo
        CpuTensor::zeros(&[4])
    }
}