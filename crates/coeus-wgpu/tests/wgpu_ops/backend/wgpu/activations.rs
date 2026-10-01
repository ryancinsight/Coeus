use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

#[test]
fn cosine_similarity_dispatches_with_wgpu_parity() {
    let cpu = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let x1_cpu = Tensor::<f32, SequentialBackend>::from_slice([2, 2], &[0.0, 0.0, 2.0, 1.0])
        .expect("invariant: test backend operation succeeds");
    let x2_cpu = Tensor::<f32, SequentialBackend>::from_slice([2, 2], &[1.0, 0.0, 1.0, 0.0])
        .expect("invariant: test backend operation succeeds");
    let x1_wgpu = x1_cpu
        .to_backend_on(&cpu, &wgpu)
        .expect("invariant: test backend operation succeeds");
    let x2_wgpu = x2_cpu
        .to_backend_on(&cpu, &wgpu)
        .expect("invariant: test backend operation succeeds");

    let x1_cpu =
        coeus_autograd::Var::new(x1_cpu, true).expect("invariant: test backend operation succeeds");
    let x2_cpu =
        coeus_autograd::Var::new(x2_cpu, true).expect("invariant: test backend operation succeeds");
    let x1_wgpu = coeus_autograd::Var::new(x1_wgpu, true)
        .expect("invariant: test backend operation succeeds");
    let x2_wgpu = coeus_autograd::Var::new(x2_wgpu, true)
        .expect("invariant: test backend operation succeeds");
    let cpu_output = coeus_autograd::cosine_similarity(&x1_cpu, &x2_cpu, 1, 0.5)
        .expect("invariant: test backend operation succeeds");
    let wgpu_output = coeus_autograd::cosine_similarity(&x1_wgpu, &x2_wgpu, 1, 0.5)
        .expect("invariant: test backend operation succeeds");

    cpu_output
        .backward()
        .expect("CPU cosine backward must succeed");
    wgpu_output
        .backward()
        .expect("WGPU cosine backward must succeed");

    let wgpu_result = wgpu_output
        .tensor
        .to_backend_on(&wgpu, &cpu)
        .expect("invariant: test backend operation succeeds");
    let wgpu_x1_gradient = x1_wgpu
        .grad()
        .expect("tracked WGPU x1 gradient")
        .to_backend_on(&wgpu, &cpu)
        .expect("invariant: test backend operation succeeds");
    let wgpu_x2_gradient = x2_wgpu
        .grad()
        .expect("tracked WGPU x2 gradient")
        .to_backend_on(&wgpu, &cpu)
        .expect("invariant: test backend operation succeeds");
    let cpu_x1_gradient = x1_cpu.grad().expect("tracked CPU x1 gradient");
    let cpu_x2_gradient = x2_cpu.grad().expect("tracked CPU x2 gradient");

    for (operation, expected, actual) in [
        (
            "forward",
            cpu_output.tensor.as_slice(),
            wgpu_result.as_slice(),
        ),
        (
            "x1 gradient",
            cpu_x1_gradient.as_slice(),
            wgpu_x1_gradient.as_slice(),
        ),
        (
            "x2 gradient",
            cpu_x2_gradient.as_slice(),
            wgpu_x2_gradient.as_slice(),
        ),
    ] {
        assert_eq!(expected.len(), actual.len());
        for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
            assert!(
                (expected - actual).abs() <= 8.0 * f32::EPSILON,
                "{operation}[{index}]: expected {expected}, got {actual}"
            );
        }
    }
}

#[test]
fn test_wgpu_silu_parity() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let input_data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let input_cpu = Tensor::<f32, SequentialBackend>::from_slice([5], &input_data)
        .expect("invariant: test backend operation succeeds");
    let input_gpu = input_cpu
        .to_backend_on(&seq, &wgpu_b)
        .expect("invariant: test backend operation succeeds");

    let var_cpu = coeus_autograd::Var::new(input_cpu, true)
        .expect("invariant: test backend operation succeeds");
    let var_gpu = coeus_autograd::Var::new(input_gpu, true)
        .expect("invariant: test backend operation succeeds");

    let out_cpu = coeus_nn::silu(&var_cpu).expect("invariant: test backend operation succeeds");
    let out_gpu = coeus_nn::silu(&var_gpu).expect("invariant: test backend operation succeeds");

    let out_gpu_cpu = out_gpu
        .tensor
        .to_backend_on(&wgpu_b, &seq)
        .expect("invariant: test backend operation succeeds");
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
    let grad_gpu_cpu = grad_gpu
        .to_backend_on(&wgpu_b, &seq)
        .expect("invariant: test backend operation succeeds");

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
    let input_cpu = Tensor::<f32, SequentialBackend>::from_slice([5], &input_data)
        .expect("invariant: test backend operation succeeds");
    let input_gpu = input_cpu
        .to_backend_on(&seq, &wgpu_b)
        .expect("invariant: test backend operation succeeds");

    let var_cpu = coeus_autograd::Var::new(input_cpu, true)
        .expect("invariant: test backend operation succeeds");
    let var_gpu = coeus_autograd::Var::new(input_gpu, true)
        .expect("invariant: test backend operation succeeds");

    let out_cpu = coeus_nn::mish(&var_cpu).expect("invariant: test backend operation succeeds");
    let out_gpu = coeus_nn::mish(&var_gpu).expect("invariant: test backend operation succeeds");

    let out_gpu_cpu = out_gpu
        .tensor
        .to_backend_on(&wgpu_b, &seq)
        .expect("invariant: test backend operation succeeds");
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
    let grad_gpu_cpu = grad_gpu
        .to_backend_on(&wgpu_b, &seq)
        .expect("invariant: test backend operation succeeds");

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
    let input_cpu = Tensor::<f32, SequentialBackend>::from_slice([5], &input_data)
        .expect("invariant: test backend operation succeeds");
    let input_gpu = input_cpu
        .to_backend_on(&seq, &wgpu_b)
        .expect("invariant: test backend operation succeeds");

    let var_cpu = coeus_autograd::Var::new(input_cpu, true)
        .expect("invariant: test backend operation succeeds");
    let var_gpu = coeus_autograd::Var::new(input_gpu, true)
        .expect("invariant: test backend operation succeeds");

    let out_cpu = coeus_nn::elu(&var_cpu).expect("invariant: test backend operation succeeds");
    let out_gpu = coeus_nn::elu(&var_gpu).expect("invariant: test backend operation succeeds");

    let out_gpu_cpu = out_gpu
        .tensor
        .to_backend_on(&wgpu_b, &seq)
        .expect("invariant: test backend operation succeeds");
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
    let grad_gpu_cpu = grad_gpu
        .to_backend_on(&wgpu_b, &seq)
        .expect("invariant: test backend operation succeeds");

    let grad_cpu_slice = grad_cpu.as_slice();
    let grad_gpu_slice = grad_gpu_cpu.as_slice();

    for i in 0..5 {
        assert!((grad_cpu_slice[i] - grad_gpu_slice[i]).abs() < 1e-5);
    }
}
