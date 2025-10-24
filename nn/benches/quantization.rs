//! Mixed Precision Quantization Benchmarks
//!
//! Comprehensive benchmarks evaluating mixed precision quantization performance,
//! calibration method accuracy, and memory efficiency across different bitwidths.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;

use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;
#[cfg(feature = "quantized")]
use coeus_nn::quantization::{
    CalibrationConfig, CalibrationMethod, CalibrationPipeline, MixedPrecisionConfig,
    MixedPrecisionQuantizedLinear, QuantizationBitwidth, QuantizationGranularity,
    QuantizationScheme,
};
use coeus_nn::{Linear, Module};
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;

/// Create random tensor with specified shape and distribution
fn random_tensor(
    shape: &[usize],
    mean: f32,
    std: f32,
) -> Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> {
    let mut rng = rand::thread_rng();
    let size: usize = shape.iter().product();
    let normal = rand_distr::Normal::new(mean, std).unwrap();

    let data: Vec<Float32> = (0..size)
        .map(|_| Float32::new(rng.sample(normal)))
        .collect();

    Tensor::from_vec(data, shape).unwrap()
}

/// Create tensor with outliers for calibration testing
fn random_tensor_with_outliers(
    shape: &[usize],
    outlier_prob: f64,
) -> Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> {
    let mut rng = rand::thread_rng();
    let size: usize = shape.iter().product();

    let data: Vec<Float32> = (0..size)
        .map(|_| {
            if rng.gen_bool(outlier_prob) {
                // Generate outlier
                let sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
                Float32::new(sign * rng.gen_range(10.0..100.0))
            } else {
                // Generate normal data
                Float32::new(rng.gen_range(-2.0..2.0))
            }
        })
        .collect();

    Tensor::from_vec(data, shape).unwrap()
}

/// Benchmark different calibration methods accuracy
#[cfg(feature = "quantized")]
fn bench_calibration_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("calibration_methods");

    let tensor_sizes = [1000, 10000, 100000];
    let methods = [
        ("MinMax", CalibrationMethod::MinMax),
        ("Percentile", CalibrationMethod::Percentile),
        ("MSE", CalibrationMethod::MseMinimization),
        ("Entropy", CalibrationMethod::EntropyMinimization),
    ];

    for &size in &tensor_sizes {
        let input_tensor = random_tensor_with_outliers(&[size], 0.01); // 1% outliers

        for (method_name, method) in &methods {
            group.bench_function(
                BenchmarkId::new(format!("{}_{}", method_name, size), size),
                |b| {
                    b.iter(|| {
                        let mut pipeline = CalibrationPipeline::new(CalibrationConfig {
                            method: *method,
                            num_samples: 100,
                            percentile: 0.999,
                            histogram_bins: 2048,
                            collect_histogram: false,
                        });

                        pipeline
                            .add_calibration_data("test_layer", &input_tensor)
                            .unwrap();
                        let (scale, zero_point) =
                            pipeline.get_optimal_params("test_layer", 8).unwrap();
                        black_box((scale, zero_point));
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark mixed precision layer forward pass performance
#[cfg(feature = "quantized")]
fn bench_mixed_precision_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_precision_forward");

    let configs = vec![
        ("4bit", QuantizationBitwidth::Bits4),
        ("8bit", QuantizationBitwidth::Bits8),
        ("16bit", QuantizationBitwidth::Bits16),
    ];

    let input_sizes = [784, 2048, 4096]; // Different embedding dimensions

    for input_size in input_sizes {
        let input = random_tensor(&[32, input_size], 0.0, 1.0); // batch_size=32

        for (config_name, bitwidth) in &configs {
            let config = MixedPrecisionConfig::new()
                .with_default_bitwidth(*bitwidth)
                .with_scheme(QuantizationScheme::Affine)
                .with_granularity(QuantizationGranularity::PerTensor)
                .with_calibration(CalibrationConfig::default());

            // Create quantized layer (output_size = input_size // 2 for typical reduction)
            let layer = MixedPrecisionQuantizedLinear::new(
                CpuBackend::new(),
                random_tensor(&[input_size, input_size / 2], 0.0, 0.1),
                Float32::new(1.0),
                Float32::new(0.0), // scale, zero_point
                None,              // no bias
                Float32::new(1.0),
                Float32::new(0.0), // input scale/zero
                Float32::new(1.0),
                Float32::new(0.0), // output scale/zero
                QuantizationScheme::Affine,
                format!("{}_{}", config_name, input_size),
                &config,
            )
            .unwrap();

            group.bench_function(
                BenchmarkId::new(format!("{}_{}", config_name, input_size), input_size),
                |b| {
                    b.iter(|| {
                        let output = black_box(layer.forward(&input).unwrap());
                        black_box(output);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark memory usage of different quantization bitwidths
#[cfg(feature = "quantized")]
fn bench_quantization_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_memory");

    let tensor_sizes = [1000, 10000, 100000, 1000000];
    let bitwidths = [
        ("4bit", QuantizationBitwidth::Bits4),
        ("8bit", QuantizationBitwidth::Bits8),
        ("16bit", QuantizationBitwidth::Bits16),
    ];

    for &size in &tensor_sizes {
        let original_tensor = random_tensor(&[size], 0.0, 1.0);

        for (bw_name, bitwidth) in &bitwidths {
            group.bench_function(
                BenchmarkId::new(format!("{}_{}", bw_name, size), size),
                |b| {
                    b.iter(|| {
                        // Measure quantized storage creation time (proxy for memory allocation)
                        let config = MixedPrecisionConfig::new().with_default_bitwidth(*bitwidth);
                        let layer = MixedPrecisionQuantizedLinear::new(
                            CpuBackend::new(),
                            original_tensor.clone(),
                            Float32::new(1.0),
                            Float32::new(0.0),
                            None,
                            Float32::new(1.0),
                            Float32::new(0.0),
                            Float32::new(1.0),
                            Float32::new(0.0),
                            QuantizationScheme::Affine,
                            format!("memory_test_{}", size),
                            &config,
                        )
                        .unwrap();
                        black_box(layer);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark calibration pipeline statistics collection
#[cfg(feature = "quantized")]
fn bench_calibration_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("calibration_statistics");

    let tensor_sizes = [1000, 10000, 100000];
    let sample_counts = [10, 100, 1000];

    for &size in &tensor_sizes {
        let base_tensor = random_tensor_with_outliers(&[size], 0.01);

        for &samples in &sample_counts {
            group.bench_function(
                BenchmarkId::new(format!("stats_{}_{}", size, samples), size),
                |b| {
                    b.iter(|| {
                        let config = CalibrationConfig {
                            method: CalibrationMethod::Percentile,
                            num_samples: samples,
                            percentile: 0.999,
                            histogram_bins: 2048,
                            collect_histogram: true,
                        };

                        let mut pipeline = CalibrationPipeline::new(config);

                        // Add multiple calibration samples
                        for i in 0..samples.min(10) {
                            // Limit to avoid excessive benchmark time
                            let sample_tensor = if i == 0 {
                                base_tensor.clone()
                            } else {
                                random_tensor_with_outliers(&[size], 0.01)
                            };
                            pipeline
                                .add_calibration_data("test_layer", &sample_tensor)
                                .unwrap();
                        }

                        let summary = pipeline.get_summary();
                        black_box(summary);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark end-to-end model accuracy with different quantization schemes
#[cfg(feature = "quantized")]
fn bench_quantization_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_accuracy");

    // Create a simple 2-layer MLP for accuracy testing
    let input_size = 784;
    let hidden_size = 256;
    let output_size = 10;
    let batch_size = 32;

    let input = random_tensor(&[batch_size, input_size], 0.0, 1.0);

    let configs = vec![
        ("FP32_Baseline", None),
        (
            "8bit_Affine",
            Some(
                MixedPrecisionConfig::new()
                    .with_default_bitwidth(QuantizationBitwidth::Bits8)
                    .with_scheme(QuantizationScheme::Affine),
            ),
        ),
        (
            "4bit_Symmetric",
            Some(
                MixedPrecisionConfig::new()
                    .with_default_bitwidth(QuantizationBitwidth::Bits4)
                    .with_scheme(QuantizationScheme::Symmetric),
            ),
        ),
    ];

    for (config_name, mp_config) in configs {
        group.bench_function(config_name, |b| {
            b.iter(|| {
                let result = if let Some(config) = mp_config.clone() {
                    // Create quantized layers
                    let layer1 = MixedPrecisionQuantizedLinear::new(
                        CpuBackend::new(),
                        random_tensor(&[input_size, hidden_size], 0.0, 0.1),
                        Float32::new(1.0),
                        Float32::new(0.0),
                        Some(random_tensor(&[hidden_size], 0.0, 0.1)),
                        Float32::new(1.0),
                        Float32::new(0.0),
                        Float32::new(1.0),
                        Float32::new(0.0),
                        config.scheme,
                        "layer1".to_string(),
                        &config,
                    )
                    .unwrap();

                    let layer2 = MixedPrecisionQuantizedLinear::new(
                        CpuBackend::new(),
                        random_tensor(&[hidden_size, output_size], 0.0, 0.1),
                        Float32::new(1.0),
                        Float32::new(0.0),
                        Some(random_tensor(&[output_size], 0.0, 0.1)),
                        Float32::new(1.0),
                        Float32::new(0.0),
                        Float32::new(1.0),
                        Float32::new(0.0),
                        config.scheme,
                        "layer2".to_string(),
                        &config,
                    )
                    .unwrap();

                    // Forward pass through both layers
                    let hidden = layer1.forward(&input).unwrap();
                    let output = layer2.forward(&hidden).unwrap();
                    output
                } else {
                    // FP32 baseline - use regular Linear layers
                    use coeus_nn::Linear;
                    let layer1 =
                        Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                            input_size,
                            hidden_size,
                        )
                        .unwrap();
                    let layer2 =
                        Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                            hidden_size,
                            output_size,
                        )
                        .unwrap();

                    let hidden = layer1.forward(&input).unwrap();
                    let output = layer2.forward(&hidden).unwrap();
                    output
                };

                black_box(result);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "quantized")]
criterion_group!(
    benches,
    bench_calibration_methods,
    bench_mixed_precision_forward,
    bench_quantization_memory_usage,
    bench_calibration_statistics,
    bench_quantization_accuracy
);

#[cfg(feature = "quantized")]
criterion_main!(benches);

// Placeholder for when quantized feature is not enabled
#[cfg(not(feature = "quantized"))]
fn main() {
    println!("Quantization benchmarks require the 'quantized' feature to be enabled");
    println!("Run with: cargo bench --features quantized --bench quantization");
}
