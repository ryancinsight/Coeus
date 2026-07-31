use super::*;

#[test]
fn test_cuda_parity_conv_transpose1d() {
    let Some((s, c)) = backends() else {
        return;
    };
    // input [n, c_in, l], weight [c_in, c_out, k]
    let (n, c_in, l, c_out, k) = (2, 3, 5, 4, 3);
    let (stride, padding, output_padding, dilation) = (2usize, 1usize, 0usize, 1usize);
    let l_out = (l - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1;

    let input: Vec<f32> = (0..n * c_in * l).map(|x| x as f32 * 0.05 - 0.7).collect();
    let weight: Vec<f32> = (0..c_in * c_out * k)
        .map(|x| x as f32 * 0.1 - 0.6)
        .collect();
    let bias: Vec<f32> = (0..c_out).map(|x| x as f32 * 0.25 - 0.3).collect();

    let in_t = Tensor::from_slice(vec![n, c_in, l], &input);
    let w_t = Tensor::from_slice(vec![c_in, c_out, k], &weight);
    let b_t = Tensor::from_slice(vec![c_out], &bias);

    let mut out_s = Tensor::<f32, SequentialBackend>::zeros(vec![n, c_out, l_out]);
    let out_l = out_s.layout().clone();
    s.conv_transpose1d(
        in_t.storage(),
        in_t.layout(),
        w_t.storage(),
        w_t.layout(),
        Some(b_t.storage()),
        stride,
        padding,
        output_padding,
        dilation,
        out_s.storage_mut(),
        &out_l,
    )
    .expect("CPU transposed conv1d dispatch");

    let in_g = to_gpu(&in_t, &s, &c);
    let w_g = to_gpu(&w_t, &s, &c);
    let b_g = to_gpu(&b_t, &s, &c);
    let mut out_g = Tensor::<f32, CudaBackend>::zeros_on(vec![n, c_out, l_out], &c);
    c.conv_transpose1d(
        in_g.storage(),
        in_g.layout(),
        w_g.storage(),
        w_g.layout(),
        Some(b_g.storage()),
        stride,
        padding,
        output_padding,
        dilation,
        out_g.storage_mut(),
        &out_l,
    )
    .expect("CUDA transposed conv1d dispatch");

    assert_parity_tol(
        "conv_transpose1d",
        out_s.as_slice(),
        to_cpu(&out_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}

#[test]
fn test_cuda_parity_conv_transpose2d() {
    let Some((s, c)) = backends() else {
        return;
    };
    // input [n, c_in, h, w], weight [c_in, c_out, kh, kw]
    let (n, c_in, h, w, c_out, kh, kw) = (2, 2, 4, 4, 3, 3, 3);
    let (stride, padding, output_padding, dilation) = (2usize, 1usize, 1usize, 1usize);
    let h_out = (h - 1) * stride - 2 * padding + dilation * (kh - 1) + output_padding + 1;
    let w_out = (w - 1) * stride - 2 * padding + dilation * (kw - 1) + output_padding + 1;

    let input: Vec<f32> = (0..n * c_in * h * w)
        .map(|x| x as f32 * 0.03 - 0.5)
        .collect();
    let weight: Vec<f32> = (0..c_in * c_out * kh * kw)
        .map(|x| x as f32 * 0.07 - 0.4)
        .collect();
    let bias: Vec<f32> = (0..c_out).map(|x| x as f32 * 0.2 - 0.2).collect();

    let in_t = Tensor::from_slice(vec![n, c_in, h, w], &input);
    let wt_t = Tensor::from_slice(vec![c_in, c_out, kh, kw], &weight);
    let b_t = Tensor::from_slice(vec![c_out], &bias);

    let mut out_s = Tensor::<f32, SequentialBackend>::zeros(vec![n, c_out, h_out, w_out]);
    let out_l = out_s.layout().clone();
    s.conv_transpose2d(
        in_t.storage(),
        in_t.layout(),
        wt_t.storage(),
        wt_t.layout(),
        Some(b_t.storage()),
        stride,
        padding,
        output_padding,
        dilation,
        out_s.storage_mut(),
        &out_l,
    )
    .expect("CPU transposed conv2d dispatch");

    let in_g = to_gpu(&in_t, &s, &c);
    let w_g = to_gpu(&wt_t, &s, &c);
    let b_g = to_gpu(&b_t, &s, &c);
    let mut out_g = Tensor::<f32, CudaBackend>::zeros_on(vec![n, c_out, h_out, w_out], &c);
    c.conv_transpose2d(
        in_g.storage(),
        in_g.layout(),
        w_g.storage(),
        w_g.layout(),
        Some(b_g.storage()),
        stride,
        padding,
        output_padding,
        dilation,
        out_g.storage_mut(),
        &out_l,
    )
    .expect("CUDA transposed conv2d dispatch");

    assert_parity_tol(
        "conv_transpose2d",
        out_s.as_slice(),
        to_cpu(&out_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}

#[test]
fn test_cuda_parity_conv_transpose1d_backward() {
    let Some((s, c)) = backends() else {
        return;
    };

    let input = [0.5f32, -0.25, 0.75];
    let weight = [0.7f32, -0.4];
    let seed = [1.0f32, -0.5, 0.25, 2.0];

    let input_cpu = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice([1, 1, 3], &input),
        true,
    );
    let weight_cpu = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice([1, 1, 2], &weight),
        true,
    );
    let out_cpu =
        coeus_ops::conv_transpose1d(&input_cpu.tensor, &weight_cpu.tensor, None, 1, 0, 0, 1, &s)
            .expect("CPU transposed convolution forward");
    let tracked_cpu =
        coeus_autograd::conv_transpose1d(&input_cpu, &weight_cpu, &None, out_cpu, 1, 0, 0, 1);
    tracked_cpu
        .backward_with_seed(Tensor::<f32, SequentialBackend>::from_slice(
            [1, 1, 4],
            &seed,
        ))
        .expect("invariant: valid autograd fixture completes backward");

    let input_gpu = Var::new(
        Tensor::<f32, CudaBackend>::from_slice_on([1, 1, 3], &input, &c),
        true,
    );
    let weight_gpu = Var::new(
        Tensor::<f32, CudaBackend>::from_slice_on([1, 1, 2], &weight, &c),
        true,
    );
    let out_gpu =
        coeus_ops::conv_transpose1d(&input_gpu.tensor, &weight_gpu.tensor, None, 1, 0, 0, 1, &c)
            .expect("CUDA transposed convolution forward");
    let tracked_gpu =
        coeus_autograd::conv_transpose1d(&input_gpu, &weight_gpu, &None, out_gpu, 1, 0, 0, 1);
    tracked_gpu
        .backward_with_seed(Tensor::<f32, CudaBackend>::from_slice_on(
            [1, 1, 4],
            &seed,
            &c,
        ))
        .expect("invariant: valid autograd fixture completes backward");

    assert_parity_tol(
        "conv_transpose1d_backward_input",
        input_cpu.grad().unwrap().as_slice(),
        to_cpu(&input_gpu.grad().unwrap(), &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
    assert_parity_tol(
        "conv_transpose1d_backward_weight",
        weight_cpu.grad().unwrap().as_slice(),
        to_cpu(&weight_gpu.grad().unwrap(), &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}

#[test]
fn test_cuda_parity_conv_transpose2d_backward() {
    let Some((s, c)) = backends() else {
        return;
    };

    let input = [0.5f32, -0.25, 0.75, 1.25];
    let weight = [0.6f32, -0.2, 0.3, -0.5];
    let seed: Vec<f32> = (0..9).map(|x| x as f32 * 0.2 - 0.7).collect();

    let input_cpu = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice([1, 1, 2, 2], &input),
        true,
    );
    let weight_cpu = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice([1, 1, 2, 2], &weight),
        true,
    );
    let out_cpu =
        coeus_ops::conv_transpose2d(&input_cpu.tensor, &weight_cpu.tensor, None, 1, 0, 0, 1, &s)
            .expect("CPU transposed convolution forward");
    let tracked_cpu =
        coeus_autograd::conv_transpose2d(&input_cpu, &weight_cpu, &None, out_cpu, 1, 0, 0, 1);
    tracked_cpu
        .backward_with_seed(Tensor::<f32, SequentialBackend>::from_slice(
            [1, 1, 3, 3],
            &seed,
        ))
        .expect("invariant: valid autograd fixture completes backward");

    let input_gpu = Var::new(
        Tensor::<f32, CudaBackend>::from_slice_on([1, 1, 2, 2], &input, &c),
        true,
    );
    let weight_gpu = Var::new(
        Tensor::<f32, CudaBackend>::from_slice_on([1, 1, 2, 2], &weight, &c),
        true,
    );
    let out_gpu =
        coeus_ops::conv_transpose2d(&input_gpu.tensor, &weight_gpu.tensor, None, 1, 0, 0, 1, &c)
            .expect("CUDA transposed convolution forward");
    let tracked_gpu =
        coeus_autograd::conv_transpose2d(&input_gpu, &weight_gpu, &None, out_gpu, 1, 0, 0, 1);
    tracked_gpu
        .backward_with_seed(Tensor::<f32, CudaBackend>::from_slice_on(
            [1, 1, 3, 3],
            &seed,
            &c,
        ))
        .expect("invariant: valid autograd fixture completes backward");

    assert_parity_tol(
        "conv_transpose2d_backward_input",
        input_cpu.grad().unwrap().as_slice(),
        to_cpu(&input_gpu.grad().unwrap(), &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
    assert_parity_tol(
        "conv_transpose2d_backward_weight",
        weight_cpu.grad().unwrap().as_slice(),
        to_cpu(&weight_gpu.grad().unwrap(), &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}
