//! GPU FFT Performance Benchmark
//!
//! Compares GPU-accelerated FFT performance against CPU implementation.
//! Demonstrates the efficiency gains from GPU parallel processing.

use std::time::Instant;
use std::sync::Arc;
use backend::gpu::GpuBackend;
use audio::{Fft, GpuFft};

/// Run comprehensive FFT performance benchmarks
async fn run_fft_benchmarks() {
    println!("🎵 GPU FFT Performance Benchmark");
    println!("=================================");

    // Test sizes covering audio processing ranges
    let sizes = vec![256, 512, 1024, 2048, 4096, 8192];

    for &size in &sizes {
        println!("\n--- FFT Size: {} ---", size);

        // Generate test signal (combination of sinusoids)
        let signal: Vec<f32> = (0..size)
            .map(|i| {
                let t = i as f32 / size as f32;
                (2.0 * std::f32::consts::PI * 5.0 * t).sin() * 0.5 +
                (2.0 * std::f32::consts::FRAC_PI_2 * 10.0 * t).sin() * 0.3
            })
            .collect();

        // CPU Benchmark
        let cpu_start = Instant::now();
        let mut cpu_fft = Fft::new(size).unwrap();
        let mut cpu_times = Vec::new();

        for _ in 0..100 { // 100 iterations for more stable measurement
            let iter_start = Instant::now();
            let _result = cpu_fft.forward_real_simple(&signal).unwrap();
            cpu_times.push(iter_start.elapsed());
        }

        let cpu_total = cpu_start.elapsed();
        let cpu_avg = cpu_times.iter().sum::<std::time::Duration>() / cpu_times.len() as u32;

        println!("CPU FFT:   {:.2} ms (avg per transform)",
                 cpu_avg.as_secs_f64() * 1000.0);

        // GPU Benchmark (if available)
        if let Ok(backend) = GpuBackend::new().await {
            let backend = Arc::new(backend);
            let gpu_fft = GpuFft::new(backend, size).unwrap();

            let gpu_start = Instant::now();
            let mut gpu_times = Vec::new();

            for _ in 0..100 {
                let iter_start = Instant::now();
                let _result = gpu_fft.forward_real(&signal).await.unwrap();
                gpu_times.push(iter_start.elapsed());
            }

            let gpu_total = gpu_start.elapsed();
            let gpu_avg = gpu_times.iter().sum::<std::time::Duration>() / gpu_times.len() as u32;

            println!("GPU FFT:   {:.2} ms (avg per transform)",
                     gpu_avg.as_secs_f64() * 1000.0);

            let speedup = cpu_avg.as_secs_f64() / gpu_avg.as_secs_f64();
            println!("Speedup:   {:.1}x", speedup);

            // Verify correctness on first iteration
            let cpu_result = cpu_fft.forward_real_simple(&signal).unwrap();
            let gpu_result = gpu_fft.forward_real(&signal).await.unwrap();

            let max_diff = cpu_result.iter()
                .zip(&gpu_result)
                .map(|(c, g)| ((c.re - g[0]).abs()).max((c.im - g[1]).abs()))
                .fold(0.0f32, f32::max);

            println!("Accuracy:  {:.2e} max error", max_diff);
        } else {
            println!("GPU FFT:   Not available");
        }
    }

    println!("\n--- Round-trip Test ---");
    // Test perfect reconstruction (forward + inverse)
    let test_size = 1024;
    let original: Vec<f32> = (0..test_size)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    if let Ok(backend) = GpuBackend::new().await {
        let backend = Arc::new(backend);
        let gpu_fft = GpuFft::new(backend, test_size).unwrap();

        let freq_data = gpu_fft.forward_real(&original).await.unwrap();
        let reconstructed = gpu_fft.inverse_complex(&freq_data).await.unwrap();

        let rmse: f32 = original.iter()
            .zip(&reconstructed)
            .map(|(o, r)| (o - r).powi(2))
            .sum::<f32>()
            .sqrt()
            / original.len() as f32;

        println!("Round-trip RMSE: {:.2e}", rmse);

        let max_diff = original.iter()
            .zip(&reconstructed)
            .map(|(o, r)| (o - r).abs())
            .fold(0.0f32, f32::max);

        println!("Round-trip Max Error: {:.2e}", max_diff);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_fft_benchmarks().await;
    Ok(())
}

