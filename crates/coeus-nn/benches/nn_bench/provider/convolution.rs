//! convolution benchmarks.

use super::*;

/// Native-provider Conv1d comparison for `[n, ch, len]` input through
/// `Conv1d(ch -> ch, k, stride 1, no pad, no bias)`.
pub(crate) fn bench_conv1d_shape(
    c: &mut Criterion,
    label: &str,
    n: usize,
    ch: usize,
    len: usize,
    k: usize,
) {
    let conv_seq = Conv1d::<f32, SequentialBackend>::new(ch, ch, k, false);
    let conv_moirai = Conv1d::<f32, MoiraiBackend>::new(ch, ch, k, false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![n, ch, len]),
        false,
    );
    let x_moirai = Var::new(Tensor::<f32, MoiraiBackend>::ones(vec![n, ch, len]), false);

    let mut group = c.benchmark_group(label);
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                conv_seq
                    .forward(black_box(&x_seq))
                    .expect("valid convolution benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                conv_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid convolution benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_conv1d_forward(c: &mut Criterion) {
    bench_conv1d_shape(c, "Coeus — Conv1d forward (8x32x256, k3)", 8, 32, 256, 3);
}

pub(crate) fn bench_conv1d_forward_backward(c: &mut Criterion) {
    // Conv1d forward + backward: [8, 32, 256], k3, no bias.
    const FB1_N: usize = 8;
    const FB1_C: usize = 32;
    const FB1_L: usize = 256;
    const FB1_K: usize = 3;

    let input_data: Vec<f32> = (0..(FB1_N * FB1_C * FB1_L))
        .map(|i| (i as f32 * 0.003).sin())
        .collect();

    let conv_seq = Conv1d::<f32, SequentialBackend>::new(FB1_C, FB1_C, FB1_K, false);
    let conv_moirai = Conv1d::<f32, MoiraiBackend>::new(FB1_C, FB1_C, FB1_K, false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![FB1_N, FB1_C, FB1_L], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![FB1_N, FB1_C, FB1_L], &input_data),
        true,
    );

    let mut group = c.benchmark_group("Coeus — Conv1d forward+backward (8x32x256, k3, no bias)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            conv_seq.zero_grad();
            x_seq.zero_grad();
            let out = conv_seq
                .forward(black_box(&x_seq))
                .expect("valid convolution benchmark input");
            coeus_autograd::sum(&out)
                .backward()
                .expect("invariant: valid autograd fixture completes backward");
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            conv_moirai.zero_grad();
            x_moirai.zero_grad();
            let out = conv_moirai
                .forward(black_box(&x_moirai))
                .expect("valid convolution benchmark input");
            coeus_autograd::sum(&out)
                .backward()
                .expect("invariant: valid autograd fixture completes backward");
        })
    });
    group.finish();
}

/// Native-provider Conv2d comparison for a square `[n, ch, hw, hw]`
/// input through `Conv2d(ch -> ch, k, stride 1, no pad, no bias)`. Bias is
/// disabled for both providers.
pub(crate) fn bench_conv2d_shape(
    c: &mut Criterion,
    label: &str,
    n: usize,
    ch: usize,
    hw: usize,
    k: usize,
) {
    let conv_seq = Conv2d::<f32, SequentialBackend>::new(ch, ch, k, false);
    let conv_moirai = Conv2d::<f32, MoiraiBackend>::new(ch, ch, k, false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![n, ch, hw, hw]),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::ones(vec![n, ch, hw, hw]),
        false,
    );

    let mut group = c.benchmark_group(label);
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                conv_seq
                    .forward(black_box(&x_seq))
                    .expect("valid convolution benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                conv_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid convolution benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_conv2d_forward(c: &mut Criterion) {
    // Small shape: per-output-row input band c_in*kh*w = 16*3*32 ≈ 6 KB is
    // L1-resident, so the AXPY kernel is bounded by call/dispatch overhead.
    bench_conv2d_shape(c, "Coeus — Conv2d forward (8x16x32x32, k3)", 8, 16, 32, 3);
    // Large shape: band c_in*kh*w = 128*3*32 ≈ 49 KB spills L1, so cross-channel
    // input reuse (each input window feeds every output channel) is the lever —
    // this is the regime that would justify a channel-batched (axpy_rows) tile.
    bench_conv2d_shape(c, "Coeus — Conv2d forward (2x128x32x32, k3)", 2, 128, 32, 3);
}

/// Native-provider Conv3d comparison for cubic `[n, ch, d, h, w]`
/// input through `Conv3d(ch -> ch, k, stride 1, no pad, no bias)`.
pub(crate) fn bench_conv3d_shape(
    c: &mut Criterion,
    label: &str,
    n: usize,
    ch: usize,
    dim: usize,
    k: usize,
) {
    let conv_seq = Conv3d::<f32, SequentialBackend>::new(ch, ch, k, false);
    let conv_moirai = Conv3d::<f32, MoiraiBackend>::new(ch, ch, k, false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![n, ch, dim, dim, dim]),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::ones(vec![n, ch, dim, dim, dim]),
        false,
    );

    let mut group = c.benchmark_group(label);
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                conv_seq
                    .forward(black_box(&x_seq))
                    .expect("valid convolution benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                conv_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid convolution benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_conv3d_forward(c: &mut Criterion) {
    bench_conv3d_shape(c, "Coeus — Conv3d forward (2x8x16x16x16, k3)", 2, 8, 16, 3);
}

pub(crate) fn bench_conv_transpose1d_forward(c: &mut Criterion) {
    // ConvTranspose1d: [B=4, C_in=32, L=16] → [B=4, C_out=16, L_out=32]
    const CT_B: usize = 4;
    const CT_CIN: usize = 32;
    const CT_COUT: usize = 16;
    const CT_L: usize = 16;
    let input_data: Vec<f32> = (0..(CT_B * CT_CIN * CT_L))
        .map(|i| (i as f32 * 0.007).sin())
        .collect();
    let ct_seq = ConvTranspose1d::<f32, SequentialBackend>::with_params(
        CT_CIN, CT_COUT, 2, 2, 0, 0, 1, true,
    );
    let ct_moirai =
        ConvTranspose1d::<f32, MoiraiBackend>::with_params(CT_CIN, CT_COUT, 2, 2, 0, 0, 1, true);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![CT_B, CT_CIN, CT_L], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![CT_B, CT_CIN, CT_L], &input_data),
        false,
    );
    let mut group =
        c.benchmark_group("Coeus — ConvTranspose1d forward (4x32x16, cin32→cout16 k2 s2)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                ct_seq
                    .forward(black_box(&x_seq))
                    .expect("valid transposed convolution benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                ct_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid transposed convolution benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_conv2d_forward_backward(c: &mut Criterion) {
    // Conv2d forward + backward: [4, 32, 16, 16], k3, no bias.
    const FB_N: usize = 4;
    const FB_C: usize = 32;
    const FB_HW: usize = 16;
    const FB_K: usize = 3;

    let input_data: Vec<f32> = (0..(FB_N * FB_C * FB_HW * FB_HW))
        .map(|i| (i as f32 * 0.007).sin())
        .collect();

    // Coeus: Conv2d with tracked Var.
    let conv_seq = Conv2d::<f32, SequentialBackend>::new(FB_C, FB_C, FB_K, false);
    let conv_moirai = Conv2d::<f32, MoiraiBackend>::new(FB_C, FB_C, FB_K, false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![FB_N, FB_C, FB_HW, FB_HW], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![FB_N, FB_C, FB_HW, FB_HW], &input_data),
        true,
    );

    let mut group = c.benchmark_group("Coeus — Conv2d forward+backward (4x32x16x16, k3, no bias)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            conv_seq.zero_grad();
            x_seq.zero_grad();
            let out = conv_seq
                .forward(black_box(&x_seq))
                .expect("valid convolution benchmark input");
            coeus_autograd::sum(&out)
                .backward()
                .expect("invariant: valid autograd fixture completes backward");
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            conv_moirai.zero_grad();
            x_moirai.zero_grad();
            let out = conv_moirai
                .forward(black_box(&x_moirai))
                .expect("valid convolution benchmark input");
            coeus_autograd::sum(&out)
                .backward()
                .expect("invariant: valid autograd fixture completes backward");
        })
    });
    group.finish();
}

pub(crate) fn bench_conv_transpose3d_forward(c: &mut Criterion) {
    // ConvTranspose3d: [B=2, C_in=8, D=4, H=4, W=4] → [B=2, C_out=4, D=8, H=8, W=8]
    const CT3_B: usize = 2;
    const CT3_CIN: usize = 8;
    const CT3_COUT: usize = 4;
    const CT3_D: usize = 4;
    const CT3_H: usize = 4;
    const CT3_W: usize = 4;
    let input_data: Vec<f32> = (0..(CT3_B * CT3_CIN * CT3_D * CT3_H * CT3_W))
        .map(|i| (i as f32 * 0.013).sin())
        .collect();

    let ct3_seq = ConvTranspose3d::<f32, SequentialBackend>::with_params(
        CT3_CIN, CT3_COUT, 2, 2, 0, 0, 1, true,
    );
    let ct3_moirai =
        ConvTranspose3d::<f32, MoiraiBackend>::with_params(CT3_CIN, CT3_COUT, 2, 2, 0, 0, 1, true);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(
            vec![CT3_B, CT3_CIN, CT3_D, CT3_H, CT3_W],
            &input_data,
        ),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(
            vec![CT3_B, CT3_CIN, CT3_D, CT3_H, CT3_W],
            &input_data,
        ),
        false,
    );
    let mut group =
        c.benchmark_group("Coeus — ConvTranspose3d forward (2x8x4x4x4, cin8→cout4 k2 s2)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                ct3_seq
                    .forward(black_box(&x_seq))
                    .expect("valid three-dimensional transposed convolution benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                ct3_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid three-dimensional transposed convolution benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_conv2d_fwd_bwd(c: &mut Criterion) {
    // Conv2d (8,16,k=3) forward+backward: [4,8,16,16]
    const N: usize = 4;
    const C_IN: usize = 8;
    const C_OUT: usize = 16;
    const K: usize = 3;
    const H: usize = 16;
    const W: usize = 16;
    let inp_data: Vec<f32> = (0..(N * C_IN * H * W))
        .map(|i| (i as f32 * 0.002).cos())
        .collect();

    let inp_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![N, C_IN, H, W], &inp_data),
        true,
    );
    let inp_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![N, C_IN, H, W], &inp_data),
        true,
    );
    let conv_seq = coeus_nn::Conv2d::<f32, SequentialBackend>::new(C_IN, C_OUT, K, false);
    let conv_moirai = coeus_nn::Conv2d::<f32, MoiraiBackend>::new(C_IN, C_OUT, K, false);
    use coeus_nn::Module;

    let mut group = c.benchmark_group("Coeus - Conv2d(8,16,k=3) fwd+bwd (4x8x16x16)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let out = conv_seq
                .forward(black_box(&inp_seq))
                .expect("valid convolution benchmark input");
            black_box(out)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let out = conv_moirai
                .forward(black_box(&inp_moirai))
                .expect("valid convolution benchmark input");
            black_box(out)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.finish();
}

pub(crate) fn bench_conv1d2_forward(c: &mut Criterion) {
    // Conv1d(16,32,k=3): [8,16,64] — second conv1d row with different shape
    const N1: usize = 8;
    const C_IN1: usize = 16;
    const C_OUT1: usize = 32;
    const K1: usize = 3;
    const L1: usize = 64;
    let inp_data: Vec<f32> = (0..(N1 * C_IN1 * L1))
        .map(|i| (i as f32 * 0.002).cos())
        .collect();
    let inp_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![N1, C_IN1, L1], &inp_data),
        false,
    );
    let inp_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![N1, C_IN1, L1], &inp_data),
        false,
    );
    let conv_seq = coeus_nn::Conv1d::<f32, SequentialBackend>::new(C_IN1, C_OUT1, K1, false);
    let conv_moirai = coeus_nn::Conv1d::<f32, MoiraiBackend>::new(C_IN1, C_OUT1, K1, false);
    use coeus_nn::Module;
    let mut group = c.benchmark_group("Coeus - Conv1d(16,32,k=3) fwd (8x16x64)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                conv_seq
                    .forward(black_box(&inp_seq))
                    .expect("valid convolution benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                conv_moirai
                    .forward(black_box(&inp_moirai))
                    .expect("valid convolution benchmark input"),
            )
        })
    });
    group.finish();
}
