use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

#[test]
fn test_wgpu_silu_parity() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let input_data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let input_cpu = Tensor::<f32, SequentialBackend>::from_slice([5], &input_data);
    let input_gpu = input_cpu.to_backend_on(&seq, &wgpu_b);

    let var_cpu = coeus_autograd::Var::new(input_cpu, true);
    let var_gpu = coeus_autograd::Var::new(input_gpu, true);

    let out_cpu = coeus_nn::silu(&var_cpu);
    let out_gpu = coeus_nn::silu(&var_gpu);

    let out_gpu_cpu = out_gpu.tensor.to_backend_on(&wgpu_b, &seq);
    let out_cpu_slice = out_cpu.tensor.as_slice();
    let out_gpu_slice = out_gpu_cpu.as_slice();

    for i in 0..5 {
        assert!((out_cpu_slice[i] - out_gpu_slice[i]).abs() < 1e-5);
    }

    out_cpu
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    out_gpu
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    let grad_cpu = var_cpu.grad().unwrap();
    let grad_gpu = var_gpu.grad().unwrap();
    let grad_gpu_cpu = grad_gpu.to_backend_on(&wgpu_b, &seq);

    let grad_cpu_slice = grad_cpu.as_slice();
    let grad_gpu_slice = grad_gpu_cpu.as_slice();

    for i in 0..5 {
        assert!((grad_cpu_slice[i] - grad_gpu_slice[i]).abs() < 1e-5);
    }
}

#[test]
fn test_wgpu_mish_parity() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let input_data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let input_cpu = Tensor::<f32, SequentialBackend>::from_slice([5], &input_data);
    let input_gpu = input_cpu.to_backend_on(&seq, &wgpu_b);

    let var_cpu = coeus_autograd::Var::new(input_cpu, true);
    let var_gpu = coeus_autograd::Var::new(input_gpu, true);

    let out_cpu = coeus_nn::mish(&var_cpu);
    let out_gpu = coeus_nn::mish(&var_gpu);

    let out_gpu_cpu = out_gpu.tensor.to_backend_on(&wgpu_b, &seq);
    let out_cpu_slice = out_cpu.tensor.as_slice();
    let out_gpu_slice = out_gpu_cpu.as_slice();

    for i in 0..5 {
        assert!((out_cpu_slice[i] - out_gpu_slice[i]).abs() < 1e-5);
    }

    out_cpu
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    out_gpu
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    let grad_cpu = var_cpu.grad().unwrap();
    let grad_gpu = var_gpu.grad().unwrap();
    let grad_gpu_cpu = grad_gpu.to_backend_on(&wgpu_b, &seq);

    let grad_cpu_slice = grad_cpu.as_slice();
    let grad_gpu_slice = grad_gpu_cpu.as_slice();

    for i in 0..5 {
        assert!((grad_cpu_slice[i] - grad_gpu_slice[i]).abs() < 1e-5);
    }
}

#[test]
fn test_wgpu_elu_parity() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let input_data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let input_cpu = Tensor::<f32, SequentialBackend>::from_slice([5], &input_data);
    let input_gpu = input_cpu.to_backend_on(&seq, &wgpu_b);

    let var_cpu = coeus_autograd::Var::new(input_cpu, true);
    let var_gpu = coeus_autograd::Var::new(input_gpu, true);

    let out_cpu = coeus_nn::elu(&var_cpu);
    let out_gpu = coeus_nn::elu(&var_gpu);

    let out_gpu_cpu = out_gpu.tensor.to_backend_on(&wgpu_b, &seq);
    let out_cpu_slice = out_cpu.tensor.as_slice();
    let out_gpu_slice = out_gpu_cpu.as_slice();

    for i in 0..5 {
        assert!((out_cpu_slice[i] - out_gpu_slice[i]).abs() < 1e-5);
    }

    out_cpu
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    out_gpu
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    let grad_cpu = var_cpu.grad().unwrap();
    let grad_gpu = var_gpu.grad().unwrap();
    let grad_gpu_cpu = grad_gpu.to_backend_on(&wgpu_b, &seq);

    let grad_cpu_slice = grad_cpu.as_slice();
    let grad_gpu_slice = grad_gpu_cpu.as_slice();

    for i in 0..5 {
        assert!((grad_cpu_slice[i] - grad_gpu_slice[i]).abs() < 1e-5);
    }
}
