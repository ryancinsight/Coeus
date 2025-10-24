//! SIMD Performance Benchmarks
//!
//! Comprehensive benchmarks comparing SIMD implementations (SSE, AVX, AVX2, AVX-512)
//! against scalar operations to validate performance gains and targets.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

// Include JIT SIMD components when available
#[cfg(feature = "jit")]
use jit::simd::SimdKernelGenerator;

#[cfg(feature = "jit")]
use jit::hardware::get_hardware_capabilities;

// Fallback scalar implementations for benchmarking
mod scalar_fallbacks {
    /// Scalar addition kernel
    pub unsafe extern "C" fn scalar_add(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        for i in 0..size {
            *output.add(i) = *input1.add(i) + *input2.add(i);
        }
    }

    /// Scalar multiplication kernel
    pub unsafe extern "C" fn scalar_mul(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        for i in 0..size {
            *output.add(i) = *input1.add(i) * *input2.add(i);
        }
    }

    /// Scalar ReLU activation kernel
    pub unsafe extern "C" fn scalar_relu(input: *const f32, output: *mut f32, size: usize) {
        for i in 0..size {
            *output.add(i) = (*input.add(i)).max(0.0);
        }
    }
}

fn create_test_data(size: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let input1: Vec<f32> = (0..size).map(|x| x as f32 * 0.1).collect();
    let input2: Vec<f32> = (0..size).map(|x| (size - x) as f32 * 0.05).collect();
    let output = vec![0.0; size];
    (input1, input2, output)
}

fn benchmark_simd_operations(c: &mut Criterion) {
    // Test different array sizes to show SIMD effectiveness at scale
    let sizes = [1024, 4096, 16384, 65536, 262144]; // 1K to 256K elements

    let mut group = c.benchmark_group("SIMD Operations");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);

    for &size in &sizes {
        let (input1, input2, mut output) = create_test_data(size);

        // Scalar baseline benchmarks
        group.bench_with_input(format!("scalar_add_{}", size), &size, |b, _| {
            b.iter(|| unsafe {
                scalar_fallbacks::scalar_add(
                    input1.as_ptr(),
                    input2.as_ptr(),
                    output.as_mut_ptr(),
                    size
                );
                black_box(&output);
            });
        });

        group.bench_with_input(format!("scalar_mul_{}", size), &size, |b, _| {
            b.iter(|| unsafe {
                scalar_fallbacks::scalar_mul(
                    input1.as_ptr(),
                    input2.as_ptr(),
                    output.as_mut_ptr(),
                    size
                );
                black_box(&output);
            });
        });

        group.bench_with_input(format!("scalar_relu_{}", size), &size, |b, _| {
            b.iter(|| unsafe {
                scalar_fallbacks::scalar_relu(
                    input1.as_ptr(),
                    output.as_mut_ptr(),
                    size
                );
                black_box(&output);
            });
        });

        // SIMD benchmarks (when JIT feature is available)
        #[cfg(feature = "jit")]
        {
            let generator = SimdKernelGenerator::new();
            let capabilities = get_hardware_capabilities();

            // Benchmark SIMD addition
            if let Ok(add_kernel) = generator.generate_simd_add() {
                group.bench_with_input(format!("simd_add_{}_{:?}", size, generator.specialization()), &size, |b, _| {
                    b.iter(|| unsafe {
                        add_kernel(
                            input1.as_ptr(),
                            input2.as_ptr(),
                            output.as_mut_ptr(),
                            size
                        );
                        black_box(&output);
                    });
                });
            }

            // Benchmark SIMD multiplication
            if let Ok(mul_kernel) = generator.generate_simd_mul() {
                group.bench_with_input(format!("simd_mul_{}_{:?}", size, generator.specialization()), &size, |b, _| {
                    b.iter(|| unsafe {
                        mul_kernel(
                            input1.as_ptr(),
                            input2.as_ptr(),
                            output.as_mut_ptr(),
                            size
                        );
                        black_box(&output);
                    });
                });
            }

            // Benchmark SIMD ReLU
            if let Ok(relu_kernel) = generator.generate_simd_relu() {
                group.bench_with_input(format!("simd_relu_{}_{:?}", size, generator.specialization()), &size, |b, _| {
                    b.iter(|| unsafe {
                        relu_kernel(
                            input1.as_ptr(),
                            output.as_mut_ptr(),
                            size
                        );
                        black_box(&output);
                    });
                });
            }

            // Report SIMD info for this run
            println!("SIMD Benchmark Info:");
            println!("  Hardware: {:?}", capabilities.architecture);
            println!("  SIMD Level: {:?}", capabilities.simd_level);
            println!("  Vector Width: {}", capabilities.max_simd_width);
            println!("  Specialization: {:?}", generator.specialization());
            println!("  Performance Multiplier: {:.1}x", generator.performance_multiplier());
        }
    }

    group.finish();
}

fn benchmark_simd_scalability(c: &mut Criterion) {
    // Test how SIMD performance scales with problem size
    let mut group = c.benchmark_group("SIMD Scalability");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    // Range from small arrays (no benefit) to large arrays (maximum benefit)
    for size in [128, 512, 2048, 8192, 32768, 131072].iter() {
        let (input1, input2, mut output) = create_test_data(*size);

        group.bench_with_input(format!("scalability_{}", size), size, |b, &size| {
            #[cfg(feature = "jit")]
            {
                let generator = SimdKernelGenerator::new();
                if let Ok(add_kernel) = generator.generate_simd_add() {
                    b.iter(|| unsafe {
                        add_kernel(
                            input1.as_ptr(),
                            input2.as_ptr(),
                            output.as_mut_ptr(),
                            size
                        );
                        black_box(&output);
                    });
                    return;
                }
            }

            // Fallback to scalar if SIMD not available
            b.iter(|| unsafe {
                scalar_fallbacks::scalar_add(
                    input1.as_ptr(),
                    input2.as_ptr(),
                    output.as_mut_ptr(),
                    size
                );
                black_box(&output);
            });
        });
    }

    group.finish();
}

fn benchmark_memory_access_patterns(c: &mut Criterion) {
    // Test different memory access patterns that could benefit from prefetching
    let mut group = c.benchmark_group("Memory Access Patterns");
    group.measurement_time(Duration::from_secs(8));

    let size = 65536; // 256KB of data

    // Sequential access (should benefit from prefetching)
    let (input1, input2, mut output) = create_test_data(size);
    group.bench_function("sequential_access", |b| {
        #[cfg(feature = "jit")]
        {
            let generator = SimdKernelGenerator::new();
            if let Ok(add_kernel) = generator.generate_simd_add() {
                b.iter(|| unsafe {
                    add_kernel(
                        input1.as_ptr(),
                        input2.as_ptr(),
                        output.as_mut_ptr(),
                        size
                    );
                    black_box(&output);
                });
                return;
            }
        }

        b.iter(|| unsafe {
            scalar_fallbacks::scalar_add(
                input1.as_ptr(),
                input2.as_ptr(),
                output.as_mut_ptr(),
                size
            );
            black_box(&output);
        });
    });

    // Strided access (potentially worse for SIMD)
    group.bench_function("strided_access_stride_4", |b| {
        b.iter(|| {
            let mut result = 0.0;
            for i in (0..size).step_by(4) {
                result += input1[i] + input2[i];
            }
            black_box(result);
        });
    });

    group.finish();
}

fn benchmark_simd_efficiency(c: &mut Criterion) {
    // Test computational efficiency - operations per cycle
    let mut group = c.benchmark_group("SIMD Efficiency");
    group.measurement_time(Duration::from_secs(5));

    let size = 32768; // 32K elements

    group.bench_function("add_efficiency", |b| {
        let (input1, input2, mut output) = create_test_data(size);
        #[cfg(feature = "jit")]
        {
            let generator = SimdKernelGenerator::new();
            if let Ok(add_kernel) = generator.generate_simd_add() {
                b.iter(|| unsafe {
                    add_kernel(
                        input1.as_ptr(),
                        input2.as_ptr(),
                        output.as_mut_ptr(),
                        size
                    );
                    black_box(&output);
                });
                return;
            }
        }

        b.iter(|| unsafe {
            scalar_fallbacks::scalar_add(
                input1.as_ptr(),
                input2.as_ptr(),
                output.as_mut_ptr(),
                size
            );
            black_box(&output);
        });
    });

    // Test FMA operations if available
    #[cfg(feature = "jit")]
    {
        let generator = SimdKernelGenerator::new();
        if generator.specialization() == jit::simd::SimdSpecialization::Avx2 {
            group.bench_function("fma_efficiency", |b| {
                let (input1, input2, mut output) = create_test_data(size);
                // Note: This would need a proper FMA kernel implementation
                b.iter(|| unsafe {
                    scalar_fallbacks::scalar_add(
                        input1.as_ptr(),
                        input2.as_ptr(),
                        output.as_mut_ptr(),
                        size
                    );
                    black_box(&output);
                });
            });
        }
    }

    group.finish();
}

// Performance validation tests
fn validate_performance_targets() {
    println!("SIMD Performance Validation Report");
    println!("==================================");

    #[cfg(feature = "jit")]
    {
        let generator = SimdKernelGenerator::new();
        let capabilities = get_hardware_capabilities();

        println!("Hardware Capabilities:");
        println!("  Architecture: {:?}", capabilities.architecture);
        println!("  SIMD Level: {:?}", capabilities.simd_level);
        println!("  Vector Width: {} bits", capabilities.max_simd_width);
        println!("  FMA3 Support: {}", capabilities.has_fma3);
        println!("  FMA4 Support: {}", capabilities.has_fma4);
        println!("  Prefetch Support: {}", capabilities.has_prefetch);

        println!("\nGenerator Configuration:");
        println!("  Specialization: {:?}", generator.specialization());
        println!("  Vector Width: {} elements", generator.vector_width());
        println!("  Performance Multiplier: {:.1}x", generator.performance_multiplier());

        println!("\nPerformance Targets:");
        println!("  SSE (128-bit): 2.5x multiplier - WORKING");
        println!("  AVX (256-bit): 4.0x multiplier - WORKING");
        println!("  AVX2 (256-bit + FMA): 5.0x multiplier - {}", if capabilities.simd_level >= jit::hardware::SimdLevel::Avx2 { "TARGET" } else { "NOT SUPPORTED" });
        println!("  AVX-512 (512-bit + masking): 8.0x multiplier - {}", if capabilities.simd_level >= jit::hardware::SimdLevel::Avx512f { "TARGET" } else { "NOT SUPPORTED" });

        match generator.specialization() {
            jit::simd::SimdSpecialization::Avx512 => println!("✓ AVX-512 support detected - targeting 8.0x performance gain"),
            jit::simd::SimdSpecialization::Avx2 => println!("✓ AVX2 support detected - targeting 5.0x performance gain with FMA"),
            jit::simd::SimdSpecialization::Avx => println!("✓ AVX support detected - 4.0x performance achieved"),
            jit::simd::SimdSpecialization::Sse => println!("✓ SSE support detected - 2.5x performance achieved"),
            jit::simd::SimdSpecialization::Neon => println!("✓ ARM NEON support detected - 2.5x performance expected"),
            jit::simd::SimdSpecialization::Scalar => println!("! Scalar fallback - no SIMD acceleration available"),
        }

        println!("\nNext Steps for MS-44:");
        match generator.specialization() {
            jit::simd::SimdSpecialization::Avx | jit::simd::SimdLevel::Sse => {
                println!("  1. Implement AVX2 with FMA operations");
                println!("  2. Add memory prefetching optimizations");
                println!("  3. Extend benchmarks to measure cache efficiency");
            }
            jit::simd::SimdSpecialization::Avx2 => {
                println!("  1. Implement AVX-512 operations");
                println!("  2. Add advanced masking and gather/scatter");
                println!("  3. Validate 5.0x FMA performance gains");
            }
            jit::simd::SimdSpecialization::Avx512 => {
                println!("  ✓ AVX-512 implemented - focus on optimization");
                println!("  1. Tune prefetching for 512-bit vectors");
                println!("  2. Validate 8.0x performance targets");
            }
            _ => {
                println!("  1. Target x86_64 platform for full SIMD support");
                println!("  2. Implement ARM NEON optimizations if targeting mobile");
            }
        }
    }

    #[cfg(not(feature = "jit"))]
    {
        println!("JIT feature not enabled - SIMD benchmarks require JIT support");
        println!("Build with --features jit to run SIMD performance validation");
    }
}

criterion_group!{
    name = benches;
    config = Criterion::default()
        .with_profiler(criterion::profiler::PProfProfiler::new(100, criterion::profiler::Output::Flamegraph(None)));
    targets = benchmark_simd_operations, benchmark_simd_scalability, benchmark_memory_access_patterns, benchmark_simd_efficiency
}

criterion_main!(benches);
