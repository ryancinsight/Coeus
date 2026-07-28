use super::*;

#[test]
fn test_cuda_parity_max_pool1d_forward_and_backward() {
    let Some((sequential, cuda)) = backends() else {
        return;
    };
    let input = Tensor::from_slice([1, 1, 5], &[1.0, 4.0, 2.0, 5.0, 3.0]);
    let grad_out = Tensor::from_slice([1, 1, 5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut expected = Tensor::<f32, SequentialBackend>::zeros([1, 1, 5]);
    let expected_layout = expected.layout().clone();
    sequential
        .max_pool1d(
            input.storage(),
            input.layout(),
            3,
            1,
            1,
            1,
            expected.storage_mut(),
            &expected_layout,
        )
        .expect("sequential max_pool1d dispatch");
    let mut expected_gradient = Tensor::<f32, SequentialBackend>::zeros([1, 1, 5]);
    let expected_gradient_layout = expected_gradient.layout().clone();
    sequential
        .max_pool1d_backward(
            grad_out.storage(),
            grad_out.layout(),
            input.storage(),
            input.layout(),
            3,
            1,
            1,
            1,
            expected_gradient.storage_mut(),
            &expected_gradient_layout,
        )
        .expect("sequential max_pool1d backward dispatch");

    let device_input = to_gpu(&input, &sequential, &cuda);
    let device_grad_out = to_gpu(&grad_out, &sequential, &cuda);
    let mut actual = Tensor::<f32, CudaBackend>::zeros_on([1, 1, 5], &cuda);
    let actual_layout = actual.layout().clone();
    cuda.max_pool1d(
        device_input.storage(),
        device_input.layout(),
        3,
        1,
        1,
        1,
        actual.storage_mut(),
        &actual_layout,
    )
    .expect("CUDA max_pool1d dispatch");
    let mut actual_gradient = Tensor::<f32, CudaBackend>::zeros_on([1, 1, 5], &cuda);
    let actual_gradient_layout = actual_gradient.layout().clone();
    cuda.max_pool1d_backward(
        device_grad_out.storage(),
        device_grad_out.layout(),
        device_input.storage(),
        device_input.layout(),
        3,
        1,
        1,
        1,
        actual_gradient.storage_mut(),
        &actual_gradient_layout,
    )
    .expect("CUDA max_pool1d backward dispatch");

    assert_eq!(expected.as_slice(), &[4.0, 4.0, 5.0, 5.0, 5.0]);
    assert_parity_tol(
        "max_pool1d",
        expected.as_slice(),
        to_cpu(&actual, &cuda, &sequential).as_slice(),
        CUDA_TOL,
    );
    assert_parity_tol(
        "max_pool1d_backward",
        expected_gradient.as_slice(),
        to_cpu(&actual_gradient, &cuda, &sequential).as_slice(),
        CUDA_TOL,
    );
}

#[test]
fn test_cuda_parity_avg_pool1d_forward_and_backward() {
    let Some((sequential, cuda)) = backends() else {
        return;
    };
    let input = Tensor::from_slice([1, 1, 5], &[1.0, 4.0, 2.0, 5.0, 3.0]);
    let grad_out = Tensor::from_slice([1, 1, 5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut expected = Tensor::<f32, SequentialBackend>::zeros([1, 1, 5]);
    let expected_layout = expected.layout().clone();
    sequential
        .avg_pool1d(
            input.storage(),
            input.layout(),
            3,
            1,
            1,
            1,
            expected.storage_mut(),
            &expected_layout,
        )
        .expect("sequential avg_pool1d dispatch");
    let mut expected_gradient = Tensor::<f32, SequentialBackend>::zeros([1, 1, 5]);
    let expected_gradient_layout = expected_gradient.layout().clone();
    sequential
        .avg_pool1d_backward(
            grad_out.storage(),
            grad_out.layout(),
            3,
            1,
            1,
            1,
            expected_gradient.storage_mut(),
            &expected_gradient_layout,
        )
        .expect("sequential avg_pool1d backward dispatch");

    let device_input = to_gpu(&input, &sequential, &cuda);
    let device_grad_out = to_gpu(&grad_out, &sequential, &cuda);
    let mut actual = Tensor::<f32, CudaBackend>::zeros_on([1, 1, 5], &cuda);
    let actual_layout = actual.layout().clone();
    cuda.avg_pool1d(
        device_input.storage(),
        device_input.layout(),
        3,
        1,
        1,
        1,
        actual.storage_mut(),
        &actual_layout,
    )
    .expect("CUDA avg_pool1d dispatch");
    let mut actual_gradient = Tensor::<f32, CudaBackend>::zeros_on([1, 1, 5], &cuda);
    let actual_gradient_layout = actual_gradient.layout().clone();
    cuda.avg_pool1d_backward(
        device_grad_out.storage(),
        device_grad_out.layout(),
        3,
        1,
        1,
        1,
        actual_gradient.storage_mut(),
        &actual_gradient_layout,
    )
    .expect("CUDA avg_pool1d backward dispatch");

    assert_parity_tol(
        "avg_pool1d",
        expected.as_slice(),
        to_cpu(&actual, &cuda, &sequential).as_slice(),
        CUDA_TOL,
    );
    assert_parity_tol(
        "avg_pool1d_backward",
        expected_gradient.as_slice(),
        to_cpu(&actual_gradient, &cuda, &sequential).as_slice(),
        CUDA_TOL,
    );
}

#[test]
fn test_cuda_parity_max_pool2d() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data: Vec<f32> = (0..2 * 2 * 4 * 4).map(|x| x as f32 * 0.1).collect();
    let x = Tensor::from_slice(vec![2, 2, 4, 4], &data);

    let mut cpu_out = Tensor::<f32, SequentialBackend>::zeros(vec![2, 2, 2, 2]);
    let cpu_out_layout = cpu_out.layout().clone();
    s.max_pool2d(
        x.storage(),
        x.layout(),
        2,
        2,
        0,
        1,
        cpu_out.storage_mut(),
        &cpu_out_layout,
    )
    .expect("invariant: validated CPU max_pool2d dispatch must succeed");

    let xg = to_gpu(&x, &s, &c);
    let mut gpu_out = Tensor::<f32, CudaBackend>::zeros_on(vec![2, 2, 2, 2], &c);
    let gpu_out_layout = gpu_out.layout().clone();
    c.max_pool2d(
        xg.storage(),
        xg.layout(),
        2,
        2,
        0,
        1,
        gpu_out.storage_mut(),
        &gpu_out_layout,
    )
    .expect("invariant: validated CUDA max_pool2d dispatch must succeed");

    assert_parity_tol(
        "max_pool2d",
        cpu_out.as_slice(),
        to_cpu(&gpu_out, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

#[test]
fn test_cuda_parity_avg_pool2d() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data: Vec<f32> = (0..2 * 2 * 4 * 4).map(|x| x as f32 * 0.1).collect();
    let x = Tensor::from_slice(vec![2, 2, 4, 4], &data);

    let mut cpu_out = Tensor::<f32, SequentialBackend>::zeros(vec![2, 2, 2, 2]);
    let cpu_out_layout = cpu_out.layout().clone();
    s.avg_pool2d(
        x.storage(),
        x.layout(),
        2,
        2,
        0,
        1,
        cpu_out.storage_mut(),
        &cpu_out_layout,
    )
    .expect("invariant: validated CPU avg_pool2d dispatch must succeed");

    let xg = to_gpu(&x, &s, &c);
    let mut gpu_out = Tensor::<f32, CudaBackend>::zeros_on(vec![2, 2, 2, 2], &c);
    let gpu_out_layout = gpu_out.layout().clone();
    c.avg_pool2d(
        xg.storage(),
        xg.layout(),
        2,
        2,
        0,
        1,
        gpu_out.storage_mut(),
        &gpu_out_layout,
    )
    .expect("invariant: validated CUDA avg_pool2d dispatch must succeed");

    assert_parity_tol(
        "avg_pool2d",
        cpu_out.as_slice(),
        to_cpu(&gpu_out, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

#[test]
fn test_cuda_parity_max_pool2d_backward() {
    let Some((s, c)) = backends() else {
        return;
    };
    // Non-monotonic data so the argmax routing of the gradient is exercised.
    let data: Vec<f32> = (0..2 * 2 * 4 * 4)
        .map(|i| ((i * 7 + 3) % 13) as f32)
        .collect();
    let x = Tensor::from_slice(vec![2, 2, 4, 4], &data);
    let grad_out: Vec<f32> = (0..2 * 2 * 2 * 2).map(|i| i as f32 * 0.5 + 1.0).collect();
    let go = Tensor::from_slice(vec![2, 2, 2, 2], &grad_out);

    let mut gi_c = Tensor::<f32, SequentialBackend>::zeros(vec![2, 2, 4, 4]);
    let gi_l = gi_c.layout().clone();
    s.max_pool2d_backward(
        go.storage(),
        go.layout(),
        x.storage(),
        x.layout(),
        2,
        2,
        0,
        1,
        gi_c.storage_mut(),
        &gi_l,
    )
    .expect("invariant: validated CPU max_pool2d backward dispatch must succeed");

    let xg = to_gpu(&x, &s, &c);
    let gog = to_gpu(&go, &s, &c);
    let mut gi_g = Tensor::<f32, CudaBackend>::zeros_on(vec![2, 2, 4, 4], &c);
    c.max_pool2d_backward(
        gog.storage(),
        gog.layout(),
        xg.storage(),
        xg.layout(),
        2,
        2,
        0,
        1,
        gi_g.storage_mut(),
        &gi_l,
    )
    .expect("invariant: validated CUDA max_pool2d backward dispatch must succeed");

    assert_parity_tol(
        "max_pool2d_bwd",
        gi_c.as_slice(),
        to_cpu(&gi_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

#[test]
fn test_cuda_parity_avg_pool2d_backward() {
    let Some((s, c)) = backends() else {
        return;
    };
    // avg-pool backward distributes grad_out uniformly over each window and
    // needs no input values, so only grad_out is supplied.
    let grad_out: Vec<f32> = (0..2 * 2 * 2 * 2).map(|i| i as f32 * 0.5 + 1.0).collect();
    let go = Tensor::from_slice(vec![2, 2, 2, 2], &grad_out);

    let mut gi_c = Tensor::<f32, SequentialBackend>::zeros(vec![2, 2, 4, 4]);
    let gi_l = gi_c.layout().clone();
    s.avg_pool2d_backward(
        go.storage(),
        go.layout(),
        2,
        2,
        0,
        1,
        gi_c.storage_mut(),
        &gi_l,
    )
    .expect("invariant: validated CPU avg_pool2d backward dispatch must succeed");

    let gog = to_gpu(&go, &s, &c);
    let mut gi_g = Tensor::<f32, CudaBackend>::zeros_on(vec![2, 2, 4, 4], &c);
    c.avg_pool2d_backward(
        gog.storage(),
        gog.layout(),
        2,
        2,
        0,
        1,
        gi_g.storage_mut(),
        &gi_l,
    )
    .expect("invariant: validated CUDA avg_pool2d backward dispatch must succeed");

    assert_parity_tol(
        "avg_pool2d_bwd",
        gi_c.as_slice(),
        to_cpu(&gi_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

// Optimizer step (AdamW).
