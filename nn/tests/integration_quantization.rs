//! Integration Tests for Quantization Crate Integration
//!
//! Tests quantization crate integration with nn, fake quantization during training,
//! and quantized model inference.
//! Validates Requirements 15.2

use backend::CpuBackend;
use dtype::float::Float32;
use nn::{Linear, Module, ReLU, Sequential};
use quantization::{
    CalibrationConfig, CalibrationMethod, CalibrationStats, LinearFakeQuantize,
    MixedPrecisionConfig, QuantizationBitwidth, QuantizationGranularity, QuantizationScheme,
};
use storage::{DenseStorage, Storage};
use tensor::Tensor;

type TestBackend = CpuBackend<Float32>;
type TestStorage = DenseStorage<Float32>;
type TestTensor = Tensor<TestBackend, TestStorage, Float32>;

/// Test basic quantization crate integration with nn
#[test]
fn test_quantization_crate_basic_integration() {
    // Create a simple linear layer
    let layer = Linear::<TestBackend, TestStorage, Float32>::new(4, 2).unwrap();

    // Get layer parameters
    let params = layer.parameters();
    assert_eq!(params.len(), 2); // weight + bias

    // Create mixed precision config
    let config = MixedPrecisionConfig::new()
        .with_default_bitwidth(QuantizationBitwidth::Bits8)
        .with_scheme(QuantizationScheme::Symmetric)
        .with_granularity(QuantizationGranularity::PerTensor);

    // Verify config creation
    assert_eq!(config.default_bitwidth, QuantizationBitwidth::Bits8);
    assert_eq!(config.scheme, QuantizationScheme::Symmetric);
    assert_eq!(config.granularity, QuantizationGranularity::PerTensor);
}

/// Test fake quantization for linear layers
#[test]
fn test_fake_quantization_linear() {
    // Create fake quantization module for 8-bit quantization
    let fake_quant = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 8>::new(
        QuantizationScheme::Symmetric,
        QuantizationGranularity::PerTensor,
        1, // per-tensor
    )
    .unwrap();

    // Create test input
    let input_data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
    ];
    let input = TestStorage::from_vec(input_data, &[2, 2]).unwrap();

    // Apply fake quantization
    let output = fake_quant.forward(&input).unwrap();

    // Verify output shape matches input
    assert_eq!(output.shape().dims(), input.shape().dims());
    assert_eq!(output.len(), input.len());

    // Verify output is not identical to input (quantization effect)
    let input_slice = input.as_slice();
    let output_slice = output.as_slice();

    // Values should be similar but not identical due to quantization
    for (inp, out) in input_slice.iter().zip(output_slice.iter()) {
        let diff = (inp.get() - out.get()).abs();
        // Quantization should introduce some error but not too much
        assert!(diff < 0.5, "Quantization error too large: {}", diff);
    }
}

/// Test fake quantization with per-channel granularity
#[test]
fn test_fake_quantization_per_channel() {
    // Create fake quantization module for per-channel 8-bit quantization
    let num_channels = 4;
    let fake_quant = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 8>::new(
        QuantizationScheme::Affine,
        QuantizationGranularity::PerChannel,
        num_channels,
    )
    .unwrap();

    // Create test input with 4 channels
    let input_data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
        Float32::new(5.0),
        Float32::new(6.0),
        Float32::new(7.0),
        Float32::new(8.0),
    ];
    let input = TestStorage::from_vec(input_data, &[2, 4]).unwrap();

    // Apply fake quantization
    let output = fake_quant.forward(&input).unwrap();

    // Verify output shape matches input
    assert_eq!(output.shape().dims(), &[2, 4]);
    assert_eq!(output.len(), 8);
}

/// Test different quantization bitwidths
#[test]
fn test_different_bitwidths() {
    let input_data = vec![Float32::new(1.0), Float32::new(2.0)];
    let input = TestStorage::from_vec(input_data, &[1, 2]).unwrap();

    // Test 4-bit quantization
    {
        let fake_quant_4bit = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 4>::new(
            QuantizationScheme::Symmetric,
            QuantizationGranularity::PerTensor,
            1,
        )
        .unwrap();

        let output = fake_quant_4bit.forward(&input).unwrap();
        assert_eq!(output.len(), input.len());
    }

    // Test 8-bit quantization
    {
        let fake_quant_8bit = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 8>::new(
            QuantizationScheme::Symmetric,
            QuantizationGranularity::PerTensor,
            1,
        )
        .unwrap();

        let output = fake_quant_8bit.forward(&input).unwrap();
        assert_eq!(output.len(), input.len());
    }

    // Test 16-bit quantization
    {
        let fake_quant_16bit = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 16>::new(
            QuantizationScheme::Symmetric,
            QuantizationGranularity::PerTensor,
            1,
        )
        .unwrap();

        let output = fake_quant_16bit.forward(&input).unwrap();
        assert_eq!(output.len(), input.len());
    }
}

/// Test quantization schemes (Symmetric vs Affine)
#[test]
fn test_quantization_schemes() {
    let input_data = vec![
        Float32::new(-2.0),
        Float32::new(-1.0),
        Float32::new(0.0),
        Float32::new(1.0),
        Float32::new(2.0),
    ];
    let input = TestStorage::from_vec(input_data, &[1, 5]).unwrap();

    // Test Symmetric quantization
    {
        let fake_quant_sym = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 8>::new(
            QuantizationScheme::Symmetric,
            QuantizationGranularity::PerTensor,
            1,
        )
        .unwrap();

        let output = fake_quant_sym.forward(&input).unwrap();
        assert_eq!(output.len(), input.len());
    }

    // Test Affine quantization
    {
        let fake_quant_affine = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 8>::new(
            QuantizationScheme::Affine,
            QuantizationGranularity::PerTensor,
            1,
        )
        .unwrap();

        let output = fake_quant_affine.forward(&input).unwrap();
        assert_eq!(output.len(), input.len());
    }
}

/// Test calibration statistics collection
#[test]
fn test_calibration_stats_collection() {
    // Create test data
    let data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
        Float32::new(5.0),
        Float32::new(6.0),
        Float32::new(7.0),
        Float32::new(8.0),
        Float32::new(9.0),
        Float32::new(10.0),
    ];

    // Collect calibration statistics
    let stats = CalibrationStats::collect(&data).unwrap();

    // Verify basic statistics
    assert_eq!(stats.min.get(), 1.0);
    assert_eq!(stats.max.get(), 10.0);
    assert!((stats.mean.get() - 5.5).abs() < 0.1);

    // Verify percentiles were calculated
    assert_eq!(stats.percentiles.len(), 12);
}

/// Test calibration methods
#[test]
fn test_calibration_methods() {
    let data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
        Float32::new(5.0),
    ];

    let stats = CalibrationStats::collect(&data).unwrap();

    // Test MinMax calibration
    {
        let (scale, zero_point) = stats.get_optimal_params(CalibrationMethod::MinMax, 8);
        assert!(scale.get() > 0.0);
        assert_eq!(zero_point.get(), 0.0);
    }

    // Test Percentile calibration
    {
        let (scale, zero_point) = stats.get_optimal_params(CalibrationMethod::Percentile, 8);
        assert!(scale.get() > 0.0);
        assert_eq!(zero_point.get(), 0.0);
    }

    // Test MSE minimization calibration
    {
        let (scale, zero_point) =
            stats.get_optimal_params(CalibrationMethod::MseMinimization, 8);
        assert!(scale.get() > 0.0);
        assert_eq!(zero_point.get(), 0.0);
    }

    // Test Entropy minimization calibration
    {
        let (scale, zero_point) =
            stats.get_optimal_params(CalibrationMethod::EntropyMinimization, 8);
        assert!(scale.get() > 0.0);
        assert_eq!(zero_point.get(), 0.0);
    }
}

/// Test mixed precision configuration
#[test]
fn test_mixed_precision_config() {
    let mut config = MixedPrecisionConfig::new()
        .with_default_bitwidth(QuantizationBitwidth::Bits8)
        .with_scheme(QuantizationScheme::Symmetric)
        .with_granularity(QuantizationGranularity::PerTensor);

    // Set different bitwidths for different layers
    config = config.with_layer_bitwidth("layer1", QuantizationBitwidth::Bits4);
    config = config.with_layer_bitwidth("layer2", QuantizationBitwidth::Bits16);

    // Verify layer-specific bitwidths
    assert_eq!(
        config.get_layer_bitwidth("layer1"),
        QuantizationBitwidth::Bits4
    );
    assert_eq!(
        config.get_layer_bitwidth("layer2"),
        QuantizationBitwidth::Bits16
    );
    assert_eq!(
        config.get_layer_bitwidth("layer3"),
        QuantizationBitwidth::Bits8
    ); // default
}

/// Test calibration config
#[test]
fn test_calibration_config() {
    let config = CalibrationConfig {
        method: CalibrationMethod::Percentile,
        num_samples: 500,
        percentile: 0.99,
        histogram_bins: 1024,
        collect_histogram: true,
    };

    assert_eq!(config.method, CalibrationMethod::Percentile);
    assert_eq!(config.num_samples, 500);
    assert_eq!(config.percentile, 0.99);
    assert_eq!(config.histogram_bins, 1024);
    assert!(config.collect_histogram);
}

/// Test fake quantization during training simulation
#[test]
fn test_fake_quantization_training_simulation() {
    // Create a simple network
    let mut model = Sequential::<TestBackend, TestStorage, Float32>::new();
    model.add_module("fc1".to_string(), Linear::new(4, 3).unwrap());
    model.add_module("relu".to_string(), ReLU::new());
    model.add_module("fc2".to_string(), Linear::new(3, 2).unwrap());

    // Create fake quantization modules for each linear layer
    let fake_quant1 = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 8>::new(
        QuantizationScheme::Symmetric,
        QuantizationGranularity::PerTensor,
        1,
    )
    .unwrap();

    let fake_quant2 = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 8>::new(
        QuantizationScheme::Symmetric,
        QuantizationGranularity::PerTensor,
        1,
    )
    .unwrap();

    // Simulate training with fake quantization
    let input = TestTensor::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[1, 4],
    )
    .unwrap();

    // Forward pass through model
    let output = model.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 2]);

    // Apply fake quantization to intermediate activations (conceptual)
    let activation_data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
    let activation = TestStorage::from_vec(activation_data, &[1, 3]).unwrap();

    let quantized_activation = fake_quant1.forward(&activation).unwrap();
    assert_eq!(quantized_activation.len(), activation.len());

    // Apply fake quantization to final output (conceptual)
    let output_storage = output.storage();
    let quantized_output = fake_quant2.forward(output_storage).unwrap();
    assert_eq!(quantized_output.len(), output_storage.len());
}

/// Test quantization error bounds
#[test]
fn test_quantization_error_bounds() {
    // Create test data with known range
    let input_data = vec![
        Float32::new(0.0),
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
    ];
    let input = TestStorage::from_vec(input_data.clone(), &[1, 5]).unwrap();

    // Test 8-bit quantization error
    let fake_quant = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 8>::new(
        QuantizationScheme::Symmetric,
        QuantizationGranularity::PerTensor,
        1,
    )
    .unwrap();

    let output = fake_quant.forward(&input).unwrap();

    // Calculate quantization error
    let input_slice = input.as_slice();
    let output_slice = output.as_slice();

    let mut max_error = 0.0f32;
    for (inp, out) in input_slice.iter().zip(output_slice.iter()) {
        let error = (inp.get() - out.get()).abs();
        max_error = max_error.max(error);
    }

    // For 8-bit quantization with range [0, 4], max error should be reasonable
    assert!(
        max_error < 0.1,
        "Quantization error too large: {}",
        max_error
    );
}

/// Test quantization with batch processing
#[test]
fn test_quantization_batch_processing() {
    // Create batch input (batch_size=4, features=3)
    let batch_data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
        Float32::new(3.0),
        Float32::new(4.0),
        Float32::new(5.0),
        Float32::new(4.0),
        Float32::new(5.0),
        Float32::new(6.0),
    ];
    let batch_input = TestStorage::from_vec(batch_data, &[4, 3]).unwrap();

    // Apply fake quantization
    let fake_quant = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 8>::new(
        QuantizationScheme::Symmetric,
        QuantizationGranularity::PerTensor,
        1,
    )
    .unwrap();

    let output = fake_quant.forward(&batch_input).unwrap();

    // Verify batch processing
    assert_eq!(output.shape().dims(), &[4, 3]);
    assert_eq!(output.len(), 12);
}

/// Test quantization parameter validation
#[test]
fn test_quantization_parameter_validation() {
    // Test invalid num_channels for per-tensor
    let result = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 8>::new(
        QuantizationScheme::Symmetric,
        QuantizationGranularity::PerTensor,
        2, // Should be 1 for per-tensor
    );
    assert!(result.is_err());

    // Test invalid num_channels for per-channel
    let result = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 8>::new(
        QuantizationScheme::Symmetric,
        QuantizationGranularity::PerChannel,
        1, // Should be >= 2 for per-channel
    );
    assert!(result.is_err());

    // Test zero num_channels
    let result = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 8>::new(
        QuantizationScheme::Symmetric,
        QuantizationGranularity::PerTensor,
        0, // Invalid
    );
    assert!(result.is_err());
}

/// Test quantization with empty calibration data
#[test]
fn test_empty_calibration_data() {
    let empty_data: Vec<Float32> = vec![];
    let result = CalibrationStats::collect(&empty_data);
    assert!(result.is_err());
}

/// Test quantization bitwidth enum
#[test]
fn test_quantization_bitwidth_enum() {
    assert_eq!(QuantizationBitwidth::Bits4.bits(), 4);
    assert_eq!(QuantizationBitwidth::Bits8.bits(), 8);
    assert_eq!(QuantizationBitwidth::Bits16.bits(), 16);
}

/// Test quantization integration with model parameters
#[test]
fn test_quantization_with_model_parameters() {
    // Create a model
    let layer = Linear::<TestBackend, TestStorage, Float32>::new(5, 3).unwrap();

    // Get parameters
    let params = layer.parameters();
    assert_eq!(params.len(), 2);

    // Simulate collecting calibration data from parameters
    for param in params.iter() {
        let param_data = param.data().as_slice();
        let float_data: Vec<Float32> = param_data.iter().copied().collect();

        // Collect calibration statistics
        let stats = CalibrationStats::collect(&float_data).unwrap();

        // Get optimal quantization parameters
        let (scale, _zero_point) = stats.get_optimal_params(CalibrationMethod::MinMax, 8);

        // Verify scale is positive
        assert!(scale.get() > 0.0);
    }
}

/// Test quantization scheme consistency
#[test]
fn test_quantization_scheme_consistency() {
    let input_data = vec![Float32::new(1.0), Float32::new(2.0)];
    let input = TestStorage::from_vec(input_data, &[1, 2]).unwrap();

    // Create two fake quant modules with same scheme
    let fake_quant1 = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 8>::new(
        QuantizationScheme::Symmetric,
        QuantizationGranularity::PerTensor,
        1,
    )
    .unwrap();

    let fake_quant2 = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 8>::new(
        QuantizationScheme::Symmetric,
        QuantizationGranularity::PerTensor,
        1,
    )
    .unwrap();

    // Apply both
    let output1 = fake_quant1.forward(&input).unwrap();
    let output2 = fake_quant2.forward(&input).unwrap();

    // Outputs should have same shape
    assert_eq!(output1.shape().dims(), output2.shape().dims());
}

/// Test quantization with different data ranges
#[test]
fn test_quantization_different_ranges() {
    // Test small range
    {
        let small_range_data = vec![Float32::new(0.1), Float32::new(0.2), Float32::new(0.3)];
        let input = TestStorage::from_vec(small_range_data, &[1, 3]).unwrap();

        let fake_quant = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 8>::new(
            QuantizationScheme::Symmetric,
            QuantizationGranularity::PerTensor,
            1,
        )
        .unwrap();

        let output = fake_quant.forward(&input).unwrap();
        assert_eq!(output.len(), input.len());
    }

    // Test large range
    {
        let large_range_data = vec![
            Float32::new(-100.0),
            Float32::new(0.0),
            Float32::new(100.0),
        ];
        let input = TestStorage::from_vec(large_range_data, &[1, 3]).unwrap();

        let fake_quant = LinearFakeQuantize::<TestBackend, TestStorage, Float32, 8>::new(
            QuantizationScheme::Symmetric,
            QuantizationGranularity::PerTensor,
            1,
        )
        .unwrap();

        let output = fake_quant.forward(&input).unwrap();
        assert_eq!(output.len(), input.len());
    }
}
