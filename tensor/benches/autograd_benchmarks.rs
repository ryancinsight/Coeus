//! Performance benchmarks for autograd operations
//!
//! This module provides focused benchmarks for autograd gradient computation
//! to establish performance baselines post-architectural fixes.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use coeus_tensor::{CpuBackend, Tensor};

/// Benchmark simple gradient computation (x²)
pub fn bench_simple_autograd(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_autograd");

    let sizes = [10, 100, 1000];

    for &size in &sizes {
        group.bench_with_input(format!("size_{}", size), &size, |b, &size| {
            let data = vec![1.0; size];
            let mut tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            tensor.set_requires_grad(true);

            let result = (&tensor * &tensor).unwrap(); // x²

            b.iter(|| {
                let mut result_copy = result.clone();
                let _: () = result_copy.backward().unwrap();
                black_box(());
            });
        });
    }
    group.finish();
}

/// Benchmark complex gradient computation with multiple operations
pub fn bench_complex_autograd(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_autograd");

    let sizes = [10, 50, 100];

    for &size in &sizes {
        group.bench_with_input(format!("size_{}", size), &size, |b, &size| {
            let data = vec![1.0; size];
            let mut x = Tensor::from_vec(CpuBackend::default(), data.clone(), vec![size]).unwrap();
            let mut y = Tensor::from_vec(CpuBackend::default(), data.clone(), vec![size]).unwrap();
            let mut z = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();

            x.set_requires_grad(true);
            y.set_requires_grad(true);
            z.set_requires_grad(true);

            // Complex expression: x² * y + sin(z)
            let x_squared = (&x * &x).unwrap();
            let x_squared_y = (&x_squared * &y).unwrap();
            let sin_z = z.sin().unwrap();
            let result = (&x_squared_y + &sin_z).unwrap();

            b.iter(|| {
                let mut result_copy = result.clone();
                let _: () = result_copy.backward().unwrap();
                black_box(());
            });
        });
    }
    group.finish();
}

/// Benchmark activation function gradients
pub fn bench_activation_gradients(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_gradients");

    let sizes = [100, 1000, 10000];

    for &size in &sizes {
        // ReLU gradient
        group.bench_with_input(format!("relu_grad_{}", size), &size, |b, &size| {
            let data = vec![1.0; size];
            let mut tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            tensor.set_requires_grad(true);

            let result = tensor.relu().unwrap();

            b.iter(|| {
                let mut result_copy = result.clone();
                let _: () = result_copy.backward().unwrap();
                black_box(());
            });
        });

        // Sigmoid gradient
        group.bench_with_input(format!("sigmoid_grad_{}", size), &size, |b, &size| {
            let data = vec![0.0; size];
            let mut tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            tensor.set_requires_grad(true);

            let result = tensor.sigmoid().unwrap();

            b.iter(|| {
                let mut result_copy = result.clone();
                let _: () = result_copy.backward().unwrap();
                black_box(());
            });
        });

        // Tanh gradient
        group.bench_with_input(format!("tanh_grad_{}", size), &size, |b, &size| {
            let data = vec![0.0; size];
            let mut tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            tensor.set_requires_grad(true);

            let result = tensor.tanh().unwrap();

            b.iter(|| {
                let mut result_copy = result.clone();
                let _: () = result_copy.backward().unwrap();
                black_box(());
            });
        });
    }
    group.finish();
}

/// Benchmark mathematical operation gradients
pub fn bench_math_gradients(c: &mut Criterion) {
    let mut group = c.benchmark_group("math_gradients");

    let sizes = [100, 1000, 10000];

    for &size in &sizes {
        // Exponential gradient
        group.bench_with_input(format!("exp_grad_{}", size), &size, |b, &size| {
            let data = vec![0.5; size];
            let mut tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            tensor.set_requires_grad(true);

            let result = tensor.exp().unwrap();

            b.iter(|| {
                let mut result_copy = result.clone();
                let _: () = result_copy.backward().unwrap();
                black_box(());
            });
        });

        // Logarithm gradient
        group.bench_with_input(format!("log_grad_{}", size), &size, |b, &size| {
            let data = vec![1.0; size];
            let mut tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            tensor.set_requires_grad(true);

            let result = tensor.log().unwrap();

            b.iter(|| {
                let mut result_copy = result.clone();
                let _: () = result_copy.backward().unwrap();
                black_box(());
            });
        });

        // Power gradient
        group.bench_with_input(format!("pow_grad_{}", size), &size, |b, &size| {
            let data = vec![2.0; size];
            let mut tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            tensor.set_requires_grad(true);

            let result = tensor.pow(3.0).unwrap();

            b.iter(|| {
                let mut result_copy = result.clone();
                let _: () = result_copy.backward().unwrap();
                black_box(());
            });
        });

        // Trigonometric gradients
        group.bench_with_input(format!("sin_grad_{}", size), &size, |b, &size| {
            let data = vec![1.0; size];
            let mut tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            tensor.set_requires_grad(true);

            let result = tensor.sin().unwrap();

            b.iter(|| {
                let mut result_copy = result.clone();
                let _: () = result_copy.backward().unwrap();
                black_box(());
            });
        });

        group.bench_with_input(format!("cos_grad_{}", size), &size, |b, &size| {
            let data = vec![1.0; size];
            let mut tensor = Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap();
            tensor.set_requires_grad(true);

            let result = tensor.cos().unwrap();

            b.iter(|| {
                let mut result_copy = result.clone();
                let _: () = result_copy.backward().unwrap();
                black_box(());
            });
        });
    }
    group.finish();
}

/// Benchmark chain rule with deep computation graphs
pub fn bench_chain_rule(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain_rule");

    let depths = [5, 10, 20];

    for &depth in &depths {
        group.bench_with_input(format!("depth_{}", depth), &depth, |b, &depth| {
            let mut x: Tensor<f64, CpuBackend> = Tensor::scalar(1.0);
            x.set_requires_grad(true);

            // Create a chain: x -> x² -> (x²)² -> ... for given depth
            let mut current = x.clone();
            for _ in 0..depth {
                current = (&current * &current).unwrap();
            }

            b.iter(|| {
                let mut result_copy = current.clone();
                let _: () = result_copy.backward().unwrap();
                black_box(());
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_simple_autograd,
    bench_complex_autograd,
    bench_activation_gradients,
    bench_math_gradients,
    bench_chain_rule,
);
criterion_main!(benches);
