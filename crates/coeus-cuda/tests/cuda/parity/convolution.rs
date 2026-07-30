use super::*;

#[test]
fn test_cuda_parity_conv1d_forward() {
    let Some((s, c)) = backends() else {
        return;
    };
    let (batch, in_c, len, out_c, ksize) = (2, 3, 8, 4, 3);
    let input: Vec<f32> = (0..batch * in_c * len)
        .map(|x| x as f32 * 0.05 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..out_c * in_c * ksize)
        .map(|x| x as f32 * 0.1 - 1.8)
        .collect();
    let bias: Vec<f32> = (0..out_c).map(|x| x as f32 * 0.2 - 0.3).collect();

    let in_t = Tensor::from_slice(vec![batch, in_c, len], &input);
    let w_t = Tensor::from_slice(vec![out_c, in_c, ksize], &weight);
    let b_t = Tensor::from_slice(vec![out_c], &bias);
    let out_len = len - ksize + 1;

    let mut cpu_out = Tensor::<f32, SequentialBackend>::zeros(vec![batch, out_c, out_len]);
    let cpu_out_layout = cpu_out.layout().clone();
    s.conv1d(
        in_t.storage(),
        in_t.layout(),
        w_t.storage(),
        w_t.layout(),
        Some(b_t.storage()),
        1,
        0,
        1,
        cpu_out.storage_mut(),
        &cpu_out_layout,
    )
    .expect("CPU conv1d dispatch");

    let in_g = to_gpu(&in_t, &s, &c);
    let w_g = to_gpu(&w_t, &s, &c);
    let b_g = to_gpu(&b_t, &s, &c);
    let mut gpu_out = Tensor::<f32, CudaBackend>::zeros_on(vec![batch, out_c, out_len], &c);
    let gpu_out_layout = gpu_out.layout().clone();
    c.conv1d(
        in_g.storage(),
        in_g.layout(),
        w_g.storage(),
        w_g.layout(),
        Some(b_g.storage()),
        1,
        0,
        1,
        gpu_out.storage_mut(),
        &gpu_out_layout,
    )
    .expect("CUDA conv1d dispatch");

    assert_parity_tol(
        "conv1d_fwd",
        cpu_out.as_slice(),
        to_cpu(&gpu_out, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}

#[test]
fn test_cuda_parity_conv2d_forward() {
    let Some((s, c)) = backends() else {
        return;
    };
    let (batch, in_c, h, ww, out_c, kh, kw) = (2, 2, 5, 5, 3, 3, 3);
    let input: Vec<f32> = (0..batch * in_c * h * ww)
        .map(|x| x as f32 * 0.05 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..out_c * in_c * kh * kw)
        .map(|x| x as f32 * 0.1 - 1.5)
        .collect();
    let bias: Vec<f32> = (0..out_c).map(|x| x as f32 * 0.2 - 0.1).collect();

    let in_t = Tensor::from_slice(vec![batch, in_c, h, ww], &input);
    let wt = Tensor::from_slice(vec![out_c, in_c, kh, kw], &weight);
    let bt = Tensor::from_slice(vec![out_c], &bias);
    let oh = h - kh + 1;
    let ow = ww - kw + 1;

    let mut cpu_out = Tensor::<f32, SequentialBackend>::zeros(vec![batch, out_c, oh, ow]);
    let cpu_out_layout = cpu_out.layout().clone();
    s.conv2d(
        in_t.storage(),
        in_t.layout(),
        wt.storage(),
        wt.layout(),
        Some(bt.storage()),
        1,
        0,
        1,
        cpu_out.storage_mut(),
        &cpu_out_layout,
    )
    .expect("CPU conv2d dispatch");

    let in_g = to_gpu(&in_t, &s, &c);
    let wg = to_gpu(&wt, &s, &c);
    let bg = to_gpu(&bt, &s, &c);
    let mut gpu_out = Tensor::<f32, CudaBackend>::zeros_on(vec![batch, out_c, oh, ow], &c);
    let gpu_out_layout = gpu_out.layout().clone();
    c.conv2d(
        in_g.storage(),
        in_g.layout(),
        wg.storage(),
        wg.layout(),
        Some(bg.storage()),
        1,
        0,
        1,
        gpu_out.storage_mut(),
        &gpu_out_layout,
    )
    .expect("CUDA conv2d dispatch");

    assert_parity_tol(
        "conv2d_fwd",
        cpu_out.as_slice(),
        to_cpu(&gpu_out, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}

#[test]
fn test_cuda_parity_conv2d_backward() {
    let Some((s, c)) = backends() else {
        return;
    };
    let (n, in_c, h, w, out_c, kh, kw) = (2, 2, 5, 5, 3, 3, 3);
    let oh = h - kh + 1;
    let ow = w - kw + 1;

    let input: Vec<f32> = (0..n * in_c * h * w)
        .map(|x| x as f32 * 0.05 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..out_c * in_c * kh * kw)
        .map(|x| x as f32 * 0.1 - 1.5)
        .collect();
    let grad_out: Vec<f32> = (0..n * out_c * oh * ow)
        .map(|x| x as f32 * 0.03 - 0.4)
        .collect();

    let in_t = Tensor::from_slice(vec![n, in_c, h, w], &input);
    let w_t = Tensor::from_slice(vec![out_c, in_c, kh, kw], &weight);
    let go_t = Tensor::from_slice(vec![n, out_c, oh, ow], &grad_out);

    // CPU reference gradients.
    let mut gi_c = Tensor::<f32, SequentialBackend>::zeros(vec![n, in_c, h, w]);
    let mut gw_c = Tensor::<f32, SequentialBackend>::zeros(vec![out_c, in_c, kh, kw]);
    let mut gb_c = Tensor::<f32, SequentialBackend>::zeros(vec![out_c]);
    let gi_l = gi_c.layout().clone();
    let gw_l = gw_c.layout().clone();
    s.conv2d_backward(
        go_t.storage(),
        go_t.layout(),
        in_t.storage(),
        in_t.layout(),
        w_t.storage(),
        w_t.layout(),
        Some(gi_c.storage_mut()),
        &gi_l,
        Some(gw_c.storage_mut()),
        &gw_l,
        Some(gb_c.storage_mut()),
        1,
        0,
        1,
    )
    .expect("CPU conv2d backward dispatch");

    // CUDA gradients.
    let in_g = to_gpu(&in_t, &s, &c);
    let w_g = to_gpu(&w_t, &s, &c);
    let go_g = to_gpu(&go_t, &s, &c);
    let mut gi_g = Tensor::<f32, CudaBackend>::zeros_on(vec![n, in_c, h, w], &c);
    let mut gw_g = Tensor::<f32, CudaBackend>::zeros_on(vec![out_c, in_c, kh, kw], &c);
    let mut gb_g = Tensor::<f32, CudaBackend>::zeros_on(vec![out_c], &c);
    c.conv2d_backward(
        go_g.storage(),
        go_g.layout(),
        in_g.storage(),
        in_g.layout(),
        w_g.storage(),
        w_g.layout(),
        Some(gi_g.storage_mut()),
        &gi_l,
        Some(gw_g.storage_mut()),
        &gw_l,
        Some(gb_g.storage_mut()),
        1,
        0,
        1,
    )
    .expect("CUDA conv2d backward dispatch");

    assert_parity_tol(
        "conv2d_bwd_grad_input",
        gi_c.as_slice(),
        to_cpu(&gi_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
    assert_parity_tol(
        "conv2d_bwd_grad_weight",
        gw_c.as_slice(),
        to_cpu(&gw_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
    assert_parity_tol(
        "conv2d_bwd_grad_bias",
        gb_c.as_slice(),
        to_cpu(&gb_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}

#[test]
fn test_cuda_parity_conv3d_forward() {
    let Some((s, c)) = backends() else {
        return;
    };
    let (n, in_c, d, h, w, out_c, kd, kh, kw) = (2, 2, 4, 4, 4, 3, 2, 2, 2);
    let od = d - kd + 1;
    let oh = h - kh + 1;
    let ow = w - kw + 1;

    let input: Vec<f32> = (0..n * in_c * d * h * w)
        .map(|x| x as f32 * 0.05 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..out_c * in_c * kd * kh * kw)
        .map(|x| x as f32 * 0.1 - 1.5)
        .collect();
    let bias: Vec<f32> = (0..out_c).map(|x| x as f32 * 0.2 - 0.3).collect();

    let in_t = Tensor::from_slice(vec![n, in_c, d, h, w], &input);
    let w_t = Tensor::from_slice(vec![out_c, in_c, kd, kh, kw], &weight);
    let b_t = Tensor::from_slice(vec![out_c], &bias);

    let mut out_s = Tensor::<f32, SequentialBackend>::zeros(vec![n, out_c, od, oh, ow]);
    let out_l = out_s.layout().clone();
    s.conv3d(
        in_t.storage(),
        in_t.layout(),
        w_t.storage(),
        w_t.layout(),
        Some(b_t.storage()),
        1,
        0,
        1,
        out_s.storage_mut(),
        &out_l,
    )
    .expect("CPU conv3d dispatch");

    let in_g = to_gpu(&in_t, &s, &c);
    let w_g = to_gpu(&w_t, &s, &c);
    let b_g = to_gpu(&b_t, &s, &c);
    let mut out_g = Tensor::<f32, CudaBackend>::zeros_on(vec![n, out_c, od, oh, ow], &c);
    c.conv3d(
        in_g.storage(),
        in_g.layout(),
        w_g.storage(),
        w_g.layout(),
        Some(b_g.storage()),
        1,
        0,
        1,
        out_g.storage_mut(),
        &out_l,
    )
    .expect("CUDA conv3d dispatch");

    assert_parity_tol(
        "conv3d_fwd",
        out_s.as_slice(),
        to_cpu(&out_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}

#[test]
fn test_cuda_parity_conv3d_backward() {
    let Some((s, c)) = backends() else {
        return;
    };
    let (n, in_c, d, h, w, out_c, kd, kh, kw) = (2, 2, 4, 4, 4, 3, 2, 2, 2);
    let od = d - kd + 1;
    let oh = h - kh + 1;
    let ow = w - kw + 1;

    let input: Vec<f32> = (0..n * in_c * d * h * w)
        .map(|x| x as f32 * 0.05 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..out_c * in_c * kd * kh * kw)
        .map(|x| x as f32 * 0.1 - 1.5)
        .collect();
    let grad_out: Vec<f32> = (0..n * out_c * od * oh * ow)
        .map(|x| x as f32 * 0.03 - 0.4)
        .collect();

    let in_t = Tensor::from_slice(vec![n, in_c, d, h, w], &input);
    let w_t = Tensor::from_slice(vec![out_c, in_c, kd, kh, kw], &weight);
    let go_t = Tensor::from_slice(vec![n, out_c, od, oh, ow], &grad_out);

    let mut gi_c = Tensor::<f32, SequentialBackend>::zeros(vec![n, in_c, d, h, w]);
    let mut gw_c = Tensor::<f32, SequentialBackend>::zeros(vec![out_c, in_c, kd, kh, kw]);
    let mut gb_c = Tensor::<f32, SequentialBackend>::zeros(vec![out_c]);
    let gi_l = gi_c.layout().clone();
    let gw_l = gw_c.layout().clone();
    s.conv3d_backward(
        go_t.storage(),
        go_t.layout(),
        in_t.storage(),
        in_t.layout(),
        w_t.storage(),
        w_t.layout(),
        Some(gi_c.storage_mut()),
        &gi_l,
        Some(gw_c.storage_mut()),
        &gw_l,
        Some(gb_c.storage_mut()),
        1,
        0,
        1,
    )
    .expect("CPU conv3d backward dispatch");

    let in_g = to_gpu(&in_t, &s, &c);
    let w_g = to_gpu(&w_t, &s, &c);
    let go_g = to_gpu(&go_t, &s, &c);
    let mut gi_g = Tensor::<f32, CudaBackend>::zeros_on(vec![n, in_c, d, h, w], &c);
    let mut gw_g = Tensor::<f32, CudaBackend>::zeros_on(vec![out_c, in_c, kd, kh, kw], &c);
    let mut gb_g = Tensor::<f32, CudaBackend>::zeros_on(vec![out_c], &c);
    c.conv3d_backward(
        go_g.storage(),
        go_g.layout(),
        in_g.storage(),
        in_g.layout(),
        w_g.storage(),
        w_g.layout(),
        Some(gi_g.storage_mut()),
        &gi_l,
        Some(gw_g.storage_mut()),
        &gw_l,
        Some(gb_g.storage_mut()),
        1,
        0,
        1,
    )
    .expect("CUDA conv3d backward dispatch");

    assert_parity_tol(
        "conv3d_bwd_grad_input",
        gi_c.as_slice(),
        to_cpu(&gi_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
    assert_parity_tol(
        "conv3d_bwd_grad_weight",
        gw_c.as_slice(),
        to_cpu(&gw_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
    assert_parity_tol(
        "conv3d_bwd_grad_bias",
        gb_c.as_slice(),
        to_cpu(&gb_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}

// Pooling.
