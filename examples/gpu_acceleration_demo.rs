//! GPU Acceleration Demo for Coeus
//!
//! This example demonstrates GPU acceleration capabilities in the Coeus framework,
//! showing performance improvements for matrix operations and neural network computations.

use std::time::Instant;
use nn::backend::{Backend, BackendSelector, BackendType, WorkloadCharacteristics, OperationType, MemoryAccessPattern, DataLocality};
use nn::tensor::{Tensor, ops::creation};
use nn::dtype::float::Float32;
use nn::storage::DenseStorage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Coeus GPU Acceleration Demo");
    println!("================================");

    // Check available backends
    let selector = BackendSelector::new();
    let available_backends = selector.available_backends();

    println!("Available backends: {:?}", available_backends);

    if !available_backends.contains(&BackendType::Gpu) {
        println!("⚠️  No GPU backend available. GPU acceleration disabled.");
        println!("   Make sure you have GPU drivers installed and wgpu feature enabled.");
        return Ok(());
    }

    println!("✅ GPU backend detected and available!");

    // Create test data
    println!("\n📊 Creating test data...");

    // Create matrices for matmul benchmark
    let m = 512;
    let k = 1024;
    let n = 512;

    let lhs_data: Vec<f32> = (0..m*k).map(|i| (i as f32).sin()).collect();
    let rhs_data: Vec<f32> = (0..k*n).map(|i| (i as f32).cos()).collect();

    let lhs_shape = vec![m, k];
    let rhs_shape = vec![k, n];

    let lhs = DenseStorage::from_vec(lhs_data.clone(), &lhs_shape)?;
    let rhs = DenseStorage::from_vec(rhs_data.clone(), &rhs_shape)?;

    println!("   Matrix A: {}x{}", m, k);
    println!("   Matrix B: {}x{}", k, n);
    println!("   Result:   {}x{}", m, n);

    // Benchmark CPU vs GPU performance
    println!("\n⚡ Performance Benchmark: Matrix Multiplication");

    // CPU benchmark
    println!("   Testing CPU backend...");
    let cpu_workload = WorkloadCharacteristics {
        total_elements: m * k * n,
        access_pattern: MemoryAccessPattern::Dense,
        compute_intensity: (m * n * k) as f32 / (m * k + k * n + m * n) as f32,
        data_locality: DataLocality::High,
        operation_type: OperationType::MatrixMultiplication,
    };

    let cpu_backend = selector.select_backend(&cpu_workload);
    println!("   Selected CPU backend: {:?}", cpu_backend);

    let start = Instant::now();
    let cpu_result = match cpu_backend {
        BackendType::Cpu => {
            let backend = nn::backend::CpuBackend::<Float32>::new();
            backend.matmul_dense(&lhs, &rhs)?
        }
        _ => unreachable!("Should select CPU"),
    };
    let cpu_time = start.elapsed();

    println!("   CPU time: {:.4} seconds", cpu_time.as_secs_f64());

    // GPU benchmark
    println!("   Testing GPU backend...");
    let gpu_workload = WorkloadCharacteristics {
        total_elements: m * k * n,
        access_pattern: MemoryAccessPattern::Dense,
        compute_intensity: (m * n * k) as f32 / (m * k + k * n + m * n) as f32,
        data_locality: DataLocality::High,
        operation_type: OperationType::MatrixMultiplication,
    };

    let gpu_backend = selector.select_backend(&gpu_workload);
    println!("   Selected GPU backend: {:?}", gpu_backend);

    let start = Instant::now();
    let gpu_result = match gpu_backend {
        BackendType::Gpu => {
            #[cfg(feature = "gpu")]
            {
                let backend = nn::backend::GpuBackend::<Float32>::new().await?;
                backend.matmul_dense(&lhs, &rhs)?
            }
            #[cfg(not(feature = "gpu"))]
            {
                unreachable!("GPU feature not enabled")
            }
        }
        _ => {
            println!("   GPU not selected, falling back to CPU");
            let backend = nn::backend::CpuBackend::<Float32>::new();
            backend.matmul_dense(&lhs, &rhs)?
        }
    };
    let gpu_time = start.elapsed();

    println!("   GPU time: {:.4} seconds", gpu_time.as_secs_f64());

    // Calculate speedup
    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
    println!("   Speedup: {:.2}x", speedup);

    // Verify results are similar (within numerical precision)
    println!("\n🔍 Verifying results...");
    let cpu_slice = cpu_result.as_slice();
    let gpu_slice = gpu_result.as_slice();

    if cpu_slice.len() == gpu_slice.len() {
        let mut max_diff = 0.0f32;
        let mut total_diff = 0.0f32;

        for (cpu_val, gpu_val) in cpu_slice.iter().zip(gpu_slice.iter()) {
            let diff = (cpu_val.get() - gpu_val.get()).abs();
            max_diff = max_diff.max(diff);
            total_diff += diff;
        }

        let avg_diff = total_diff / cpu_slice.len() as f32;

        println!("   Maximum difference: {:.2e}", max_diff);
        println!("   Average difference: {:.2e}", avg_diff);

        if max_diff < 1e-3 {
            println!("   ✅ Results match within acceptable tolerance!");
        } else {
            println!("   ⚠️  Results differ significantly - possible implementation issue");
        }
    } else {
        println!("   ❌ Result shapes don't match!");
        println!("      CPU result shape: {:?}", cpu_result.shape());
        println!("      GPU result shape: {:?}", gpu_result.shape());
    }

    // Element-wise operations benchmark
    println!("\n🔢 Element-wise Operations Benchmark");

    let size = 1_000_000;
    let lhs_vec: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
    let rhs_vec: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).cos()).collect();

    let lhs_dense = DenseStorage::from_vec(lhs_vec.clone(), &[size])?;
    let rhs_dense = DenseStorage::from_vec(rhs_vec.clone(), &[size])?;

    // Addition benchmark
    println!("   Addition ({} elements):", size);

    let cpu_start = Instant::now();
    let cpu_add = match cpu_backend {
        BackendType::Cpu => {
            let backend = nn::backend::CpuBackend::<Float32>::new();
            backend.add_dense(&lhs_dense, &rhs_dense)?
        }
        _ => unreachable!(),
    };
    let cpu_add_time = cpu_start.elapsed();

    let gpu_start = Instant::now();
    let gpu_add = match gpu_backend {
        BackendType::Gpu => {
            #[cfg(feature = "gpu")]
            {
                let backend = nn::backend::GpuBackend::<Float32>::new().await?;
                backend.add_dense(&lhs_dense, &rhs_dense)?
            }
            #[cfg(not(feature = "gpu"))]
            {
                unreachable!("GPU feature not enabled")
            }
        }
        _ => {
            let backend = nn::backend::CpuBackend::<Float32>::new();
            backend.add_dense(&lhs_dense, &rhs_dense)?
        }
    };
    let gpu_add_time = gpu_start.elapsed();

    let add_speedup = cpu_add_time.as_secs_f64() / gpu_add_time.as_secs_f64();
    println!("      CPU: {:.4}s, GPU: {:.4}s, Speedup: {:.2}x", cpu_add_time.as_secs_f64(), gpu_add_time.as_secs_f64(), add_speedup);

    // ReLU benchmark
    println!("   ReLU activation ({} elements):", size);

    let cpu_start = Instant::now();
    let cpu_relu = match cpu_backend {
        BackendType::Cpu => {
            let backend = nn::backend::CpuBackend::<Float32>::new();
            backend.relu_dense(&lhs_dense)?
        }
        _ => unreachable!(),
    };
    let cpu_relu_time = cpu_start.elapsed();

    let gpu_start = Instant::now();
    let gpu_relu = match gpu_backend {
        BackendType::Gpu => {
            #[cfg(feature = "gpu")]
            {
                let backend = nn::backend::GpuBackend::<Float32>::new().await?;
                backend.relu_dense(&lhs_dense)?
            }
            #[cfg(not(feature = "gpu"))]
            {
                unreachable!("GPU feature not enabled")
            }
        }
        _ => {
            let backend = nn::backend::CpuBackend::<Float32>::new();
            backend.relu_dense(&lhs_dense)?
        }
    };
    let gpu_relu_time = gpu_start.elapsed();

    let relu_speedup = cpu_relu_time.as_secs_f64() / gpu_relu_time.as_secs_f64();
    println!("      CPU: {:.4}s, GPU: {:.4}s, Speedup: {:.2}x", cpu_relu_time.as_secs_f64(), gpu_relu_time.as_secs_f64(), relu_speedup);

    // Summary
    println!("\n📈 Summary:");
    println!("   Matrix Multiplication: {:.2}x speedup", speedup);
    println!("   Element-wise Addition: {:.2}x speedup", add_speedup);
    println!("   ReLU Activation: {:.2}x speedup", relu_speedup);

    let avg_speedup = (speedup + add_speedup + relu_speedup) / 3.0;
    println!("   Average speedup: {:.2}x", avg_speedup);

    if avg_speedup > 2.0 {
        println!("   🎉 Excellent GPU acceleration achieved!");
    } else if avg_speedup > 1.2 {
        println!("   ✅ Good GPU acceleration achieved.");
    } else {
        println!("   ⚠️  Limited GPU acceleration - may need optimization.");
    }

    println!("\n💡 Tips for maximizing GPU performance:");
    println!("   • Use larger matrices (>1000x1000) for best matmul performance");
    println!("   • GPU overhead is higher for small operations");
    println!("   • Ensure GPU drivers are up to date");
    println!("   • Consider memory bandwidth for large tensors");

    Ok(())
}










