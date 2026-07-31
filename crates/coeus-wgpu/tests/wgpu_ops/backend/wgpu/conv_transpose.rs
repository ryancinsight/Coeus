// On-device WGPU transposed convolution forward parity vs the CPU reference.
//
// The WGSL gather kernels must match the verified CPU scatter reference
// (`BackendOps::conv_transpose1d`/`2d`) element-wise within an accumulation
// tolerance (f32 sum-order differs between gather and scatter).

use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_ops::ConvOps;
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

/// Accumulating-op tolerance: gather (device) vs scatter (CPU) sum-order over
/// `c_in * k` terms; f32 roundoff plus reorder, bounded well under this.
const TOL: f32 = 1e-3;

fn assert_close(label: &str, gpu: &[f32], cpu: &[f32]) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
    for (i, (&g, &c)) in gpu.iter().zip(cpu).enumerate() {
        let tol = TOL * (1.0 + c.abs());
        assert!(
            (g - c).abs() <= tol,
            "{label}[{i}]: GPU={g}, CPU={c}, tol {tol}",
        );
    }
}

#[test]
fn test_wgpu_conv_transpose1d() {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let (n, c_in, l, c_out, k) = (2, 3, 5, 4, 3);
    let (stride, padding, output_padding, dilation) = (2usize, 1usize, 0usize, 1usize);
    let l_out = (l - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1;

    let input: Vec<f32> = (0..n * c_in * l).map(|x| x as f32 * 0.05 - 0.7).collect();
    let weight: Vec<f32> = (0..c_in * c_out * k)
        .map(|x| x as f32 * 0.1 - 0.6)
        .collect();
    let bias: Vec<f32> = (0..c_out).map(|x| x as f32 * 0.25 - 0.3).collect();

    let in_c = Tensor::<f32, SequentialBackend>::from_slice([n, c_in, l], &input);
    let w_c = Tensor::<f32, SequentialBackend>::from_slice([c_in, c_out, k], &weight);
    let b_c = Tensor::<f32, SequentialBackend>::from_slice([c_out], &bias);

    let mut out_c = Tensor::<f32, SequentialBackend>::zeros([n, c_out, l_out]);
    let out_l = out_c.layout().clone();
    seq.conv_transpose1d(
        in_c.storage(),
        in_c.layout(),
        w_c.storage(),
        w_c.layout(),
        Some(b_c.storage()),
        stride,
        padding,
        output_padding,
        dilation,
        out_c.storage_mut(),
        &out_l,
    )
    .expect("CPU transposed conv1d dispatch");

    let in_g = in_c.to_backend_on(&seq, &wgpu);
    let w_g = w_c.to_backend_on(&seq, &wgpu);
    let b_g = b_c.to_backend_on(&seq, &wgpu);
    let mut out_g = Tensor::<f32, WgpuBackend>::zeros_on([n, c_out, l_out], &wgpu);
    wgpu.conv_transpose1d(
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
    .expect("WGPU transposed conv1d dispatch");

    assert_close(
        "conv_transpose1d",
        out_g.to_backend_on(&wgpu, &seq).as_slice(),
        out_c.as_slice(),
    );
}

#[test]
fn test_wgpu_conv_transpose2d() {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
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

    let in_c = Tensor::<f32, SequentialBackend>::from_slice([n, c_in, h, w], &input);
    let wt_c = Tensor::<f32, SequentialBackend>::from_slice([c_in, c_out, kh, kw], &weight);
    let b_c = Tensor::<f32, SequentialBackend>::from_slice([c_out], &bias);

    let mut out_c = Tensor::<f32, SequentialBackend>::zeros([n, c_out, h_out, w_out]);
    let out_l = out_c.layout().clone();
    seq.conv_transpose2d(
        in_c.storage(),
        in_c.layout(),
        wt_c.storage(),
        wt_c.layout(),
        Some(b_c.storage()),
        stride,
        padding,
        output_padding,
        dilation,
        out_c.storage_mut(),
        &out_l,
    )
    .expect("CPU transposed conv2d dispatch");

    let in_g = in_c.to_backend_on(&seq, &wgpu);
    let w_g = wt_c.to_backend_on(&seq, &wgpu);
    let b_g = b_c.to_backend_on(&seq, &wgpu);
    let mut out_g = Tensor::<f32, WgpuBackend>::zeros_on([n, c_out, h_out, w_out], &wgpu);
    wgpu.conv_transpose2d(
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
    .expect("WGPU transposed conv2d dispatch");

    assert_close(
        "conv_transpose2d",
        out_g.to_backend_on(&wgpu, &seq).as_slice(),
        out_c.as_slice(),
    );
}

#[test]
fn test_wgpu_conv_transpose1d_backward_matches_cpu_autograd() {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();

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
    let out_cpu = coeus_ops::conv_transpose1d(
        &input_cpu.tensor,
        &weight_cpu.tensor,
        None,
        1,
        0,
        0,
        1,
        &seq,
    )
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
        Tensor::<f32, WgpuBackend>::from_slice_on([1, 1, 3], &input, &wgpu),
        true,
    );
    let weight_gpu = Var::new(
        Tensor::<f32, WgpuBackend>::from_slice_on([1, 1, 2], &weight, &wgpu),
        true,
    );
    let out_gpu = coeus_ops::conv_transpose1d(
        &input_gpu.tensor,
        &weight_gpu.tensor,
        None,
        1,
        0,
        0,
        1,
        &wgpu,
    )
    .expect("WGPU transposed convolution forward");
    let tracked_gpu =
        coeus_autograd::conv_transpose1d(&input_gpu, &weight_gpu, &None, out_gpu, 1, 0, 0, 1);
    tracked_gpu
        .backward_with_seed(Tensor::<f32, WgpuBackend>::from_slice_on(
            [1, 1, 4],
            &seed,
            &wgpu,
        ))
        .expect("invariant: valid autograd fixture completes backward");

    let input_grad_gpu = input_gpu.grad().unwrap().to_backend_on(&wgpu, &seq);
    let weight_grad_gpu = weight_gpu.grad().unwrap().to_backend_on(&wgpu, &seq);
    assert_close(
        "conv_transpose1d_backward_input",
        input_grad_gpu.as_slice(),
        input_cpu.grad().unwrap().as_slice(),
    );
    assert_close(
        "conv_transpose1d_backward_weight",
        weight_grad_gpu.as_slice(),
        weight_cpu.grad().unwrap().as_slice(),
    );
}

#[test]
fn test_wgpu_conv_transpose2d_backward_matches_cpu_autograd() {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();

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
    let out_cpu = coeus_ops::conv_transpose2d(
        &input_cpu.tensor,
        &weight_cpu.tensor,
        None,
        1,
        0,
        0,
        1,
        &seq,
    )
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
        Tensor::<f32, WgpuBackend>::from_slice_on([1, 1, 2, 2], &input, &wgpu),
        true,
    );
    let weight_gpu = Var::new(
        Tensor::<f32, WgpuBackend>::from_slice_on([1, 1, 2, 2], &weight, &wgpu),
        true,
    );
    let out_gpu = coeus_ops::conv_transpose2d(
        &input_gpu.tensor,
        &weight_gpu.tensor,
        None,
        1,
        0,
        0,
        1,
        &wgpu,
    )
    .expect("WGPU transposed convolution forward");
    let tracked_gpu =
        coeus_autograd::conv_transpose2d(&input_gpu, &weight_gpu, &None, out_gpu, 1, 0, 0, 1);
    tracked_gpu
        .backward_with_seed(Tensor::<f32, WgpuBackend>::from_slice_on(
            [1, 1, 3, 3],
            &seed,
            &wgpu,
        ))
        .expect("invariant: valid autograd fixture completes backward");

    let input_grad_gpu = input_gpu.grad().unwrap().to_backend_on(&wgpu, &seq);
    let weight_grad_gpu = weight_gpu.grad().unwrap().to_backend_on(&wgpu, &seq);
    assert_close(
        "conv_transpose2d_backward_input",
        input_grad_gpu.as_slice(),
        input_cpu.grad().unwrap().as_slice(),
    );
    assert_close(
        "conv_transpose2d_backward_weight",
        weight_grad_gpu.as_slice(),
        weight_cpu.grad().unwrap().as_slice(),
    );
}
