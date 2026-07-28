use coeus_core::{ComputeBackend, CpuAddressableStorage, SequentialBackend};
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

const TOL: f32 = 1e-4;

#[derive(Clone, Copy, Debug)]
struct Conv3dCase {
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    depth: usize,
    height: usize,
    width: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv3dCase {
    fn input_shape(self) -> Vec<usize> {
        vec![
            self.batch,
            self.in_channels,
            self.depth,
            self.height,
            self.width,
        ]
    }

    fn weight_shape(self) -> Vec<usize> {
        vec![
            self.out_channels,
            self.in_channels,
            self.kernel,
            self.kernel,
            self.kernel,
        ]
    }

    fn output_extent(self, input_extent: usize) -> usize {
        (input_extent + 2 * self.padding - self.dilation * (self.kernel - 1) - 1) / self.stride + 1
    }

    fn output_shape(self) -> Vec<usize> {
        vec![
            self.batch,
            self.out_channels,
            self.output_extent(self.depth),
            self.output_extent(self.height),
            self.output_extent(self.width),
        ]
    }
}

fn patterned_values(len: usize, scale: f32, bias: f32) -> Vec<f32> {
    (0..len).map(|x| x as f32 * scale + bias).collect()
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&res, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (res - exp).abs();
        assert!(
            diff < TOL,
            "{label}[{i}]: actual={res:.6} expected={exp:.6} diff={diff:.2e}"
        );
    }
}

fn assert_forward_matches_cpu(case: Conv3dCase) {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let input_shape = case.input_shape();
    let weight_shape = case.weight_shape();
    let output_shape = case.output_shape();
    let input_data = patterned_values(input_shape.iter().product(), 0.05, -0.7);
    let weight_data = patterned_values(weight_shape.iter().product(), 0.03, -0.4);
    let bias_data = patterned_values(case.out_channels, 0.2, -0.1);
    let out_len = output_shape.iter().product();

    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(input_shape, &input_data)
        .expect("construct tensor");
    let weight_seq = Tensor::<f32, SequentialBackend>::from_slice(weight_shape, &weight_data)
        .expect("construct tensor");
    let bias_seq =
        Tensor::<f32, SequentialBackend>::from_slice(vec![case.out_channels], &bias_data)
            .expect("construct tensor");

    let input_wgpu = input_seq.to_backend_on(&seq, &wgpu_b).expect("transfer tensor");
    let weight_wgpu = weight_seq.to_backend_on(&seq, &wgpu_b).expect("transfer tensor");
    let bias_wgpu = bias_seq.to_backend_on(&seq, &wgpu_b).expect("transfer tensor");

    let out_layout = coeus_core::Layout::new(output_shape.into());
    let mut out_wgpu_storage = wgpu_b.allocate::<f32>(out_len).expect("allocate tensor storage");

    coeus_ops::ConvOps::conv3d(
        &wgpu_b,
        input_wgpu.storage(),
        input_wgpu.layout(),
        weight_wgpu.storage(),
        weight_wgpu.layout(),
        Some(bias_wgpu.storage()),
        case.stride,
        case.padding,
        case.dilation,
        &mut out_wgpu_storage,
        &out_layout,
    )
    .expect("execute WGPU convolution");

    let out_tensor_wgpu =
        Tensor::<f32, WgpuBackend>::from_raw_parts(out_wgpu_storage, out_layout.clone());
    let out_wgpu_cpu = out_tensor_wgpu.to_backend_on(&wgpu_b, &seq).expect("transfer tensor");

    let mut out_expected_storage = seq.allocate::<f32>(out_len).expect("allocate tensor storage");
    coeus_ops::ConvOps::conv3d(
        &seq,
        input_seq.storage(),
        input_seq.layout(),
        weight_seq.storage(),
        weight_seq.layout(),
        Some(bias_seq.storage()),
        case.stride,
        case.padding,
        case.dilation,
        &mut out_expected_storage,
        &out_layout,
    )
    .expect("execute CPU convolution");
    let out_expected =
        Tensor::<f32, SequentialBackend>::from_raw_parts(out_expected_storage, out_layout);

    assert_close(
        "conv3d_forward",
        out_wgpu_cpu.as_slice(),
        out_expected.as_slice(),
    );
}

fn assert_backward_matches_cpu(case: Conv3dCase) {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let input_shape = case.input_shape();
    let weight_shape = case.weight_shape();
    let output_shape = case.output_shape();
    let input_len = input_shape.iter().product();
    let weight_len = weight_shape.iter().product();
    let grad_out_len = output_shape.iter().product();

    let grad_out_data = patterned_values(grad_out_len, 0.04, 0.2);
    let input_data = patterned_values(input_len, 0.05, -0.7);
    let weight_data = patterned_values(weight_len, 0.03, -0.4);

    let grad_out_seq = Tensor::<f32, SequentialBackend>::from_slice(output_shape, &grad_out_data)
        .expect("construct tensor");
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(input_shape.clone(), &input_data)
        .expect("construct tensor");
    let weight_seq =
        Tensor::<f32, SequentialBackend>::from_slice(weight_shape.clone(), &weight_data)
            .expect("construct tensor");

    let grad_out_wgpu = grad_out_seq.to_backend_on(&seq, &wgpu_b).expect("transfer tensor");
    let input_wgpu = input_seq.to_backend_on(&seq, &wgpu_b).expect("transfer tensor");
    let weight_wgpu = weight_seq.to_backend_on(&seq, &wgpu_b).expect("transfer tensor");

    let mut gi_wgpu = wgpu_b.allocate::<f32>(input_len).expect("allocate tensor storage");
    wgpu_b
        .fill(&mut gi_wgpu, 0.0)
        .expect("fill gradient storage");
    let mut gw_wgpu = wgpu_b.allocate::<f32>(weight_len).expect("allocate tensor storage");
    wgpu_b
        .fill(&mut gw_wgpu, 0.0)
        .expect("fill gradient storage");
    let mut gb_wgpu = wgpu_b.allocate::<f32>(case.out_channels).expect("allocate tensor storage");
    wgpu_b
        .fill(&mut gb_wgpu, 0.0)
        .expect("fill gradient storage");

    let gi_layout = coeus_core::Layout::new(input_shape.into());
    let gw_layout = coeus_core::Layout::new(weight_shape.into());
    let gb_layout = coeus_core::Layout::new(vec![case.out_channels].into());

    coeus_ops::ConvOps::conv3d_backward(
        &wgpu_b,
        grad_out_wgpu.storage(),
        grad_out_wgpu.layout(),
        input_wgpu.storage(),
        input_wgpu.layout(),
        weight_wgpu.storage(),
        weight_wgpu.layout(),
        Some(&mut gi_wgpu),
        &gi_layout,
        Some(&mut gw_wgpu),
        &gw_layout,
        Some(&mut gb_wgpu),
        case.stride,
        case.padding,
        case.dilation,
    )
    .expect("execute WGPU convolution backward");

    let mut gi_expected = seq.allocate::<f32>(input_len).expect("allocate tensor storage");
    seq.fill(&mut gi_expected, 0.0)
        .expect("fill gradient storage");
    let mut gw_expected = seq.allocate::<f32>(weight_len).expect("allocate tensor storage");
    seq.fill(&mut gw_expected, 0.0)
        .expect("fill gradient storage");
    let mut gb_expected = seq.allocate::<f32>(case.out_channels).expect("allocate tensor storage");
    seq.fill(&mut gb_expected, 0.0)
        .expect("fill gradient storage");

    coeus_ops::ConvOps::conv3d_backward(
        &seq,
        grad_out_seq.storage(),
        grad_out_seq.layout(),
        input_seq.storage(),
        input_seq.layout(),
        weight_seq.storage(),
        weight_seq.layout(),
        Some(&mut gi_expected),
        &gi_layout,
        Some(&mut gw_expected),
        &gw_layout,
        Some(&mut gb_expected),
        case.stride,
        case.padding,
        case.dilation,
    )
    .expect("execute CPU convolution backward");

    let gi_wgpu_tensor = Tensor::<f32, WgpuBackend>::from_raw_parts(gi_wgpu, gi_layout);
    let gi_wgpu_cpu = gi_wgpu_tensor.to_backend_on(&wgpu_b, &seq).expect("transfer tensor");
    assert_close(
        "conv3d_grad_input",
        gi_wgpu_cpu.as_slice(),
        gi_expected.as_slice(),
    );

    let gw_wgpu_tensor = Tensor::<f32, WgpuBackend>::from_raw_parts(gw_wgpu, gw_layout);
    let gw_wgpu_cpu = gw_wgpu_tensor.to_backend_on(&wgpu_b, &seq).expect("transfer tensor");
    assert_close(
        "conv3d_grad_weight",
        gw_wgpu_cpu.as_slice(),
        gw_expected.as_slice(),
    );

    let gb_wgpu_tensor = Tensor::<f32, WgpuBackend>::from_raw_parts(gb_wgpu, gb_layout);
    let gb_wgpu_cpu = gb_wgpu_tensor.to_backend_on(&wgpu_b, &seq).expect("transfer tensor");
    assert_close(
        "conv3d_grad_bias",
        gb_wgpu_cpu.as_slice(),
        gb_expected.as_slice(),
    );
}

#[test]
fn test_wgpu_conv3d() {
    assert_forward_matches_cpu(Conv3dCase {
        batch: 1,
        in_channels: 1,
        out_channels: 1,
        depth: 3,
        height: 3,
        width: 3,
        kernel: 2,
        stride: 1,
        padding: 0,
        dilation: 1,
    });
}

#[test]
fn test_wgpu_conv3d_stride_padding_dilation_matches_cpu() {
    for case in [
        Conv3dCase {
            batch: 2,
            in_channels: 2,
            out_channels: 3,
            depth: 4,
            height: 4,
            width: 4,
            kernel: 2,
            stride: 2,
            padding: 1,
            dilation: 1,
        },
        Conv3dCase {
            batch: 1,
            in_channels: 2,
            out_channels: 2,
            depth: 5,
            height: 5,
            width: 5,
            kernel: 2,
            stride: 1,
            padding: 0,
            dilation: 2,
        },
    ] {
        assert_forward_matches_cpu(case);
    }
}

#[test]
fn test_wgpu_conv3d_backward() {
    assert_backward_matches_cpu(Conv3dCase {
        batch: 1,
        in_channels: 1,
        out_channels: 1,
        depth: 3,
        height: 3,
        width: 3,
        kernel: 2,
        stride: 1,
        padding: 0,
        dilation: 1,
    });
}

#[test]
fn test_wgpu_conv3d_backward_stride_padding_dilation_matches_cpu() {
    for case in [
        Conv3dCase {
            batch: 2,
            in_channels: 2,
            out_channels: 3,
            depth: 4,
            height: 4,
            width: 4,
            kernel: 2,
            stride: 2,
            padding: 1,
            dilation: 1,
        },
        Conv3dCase {
            batch: 1,
            in_channels: 2,
            out_channels: 2,
            depth: 5,
            height: 5,
            width: 5,
            kernel: 2,
            stride: 1,
            padding: 0,
            dilation: 2,
        },
    ] {
        assert_backward_matches_cpu(case);
    }
}
