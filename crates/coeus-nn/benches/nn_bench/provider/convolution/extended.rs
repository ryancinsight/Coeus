use super::*;

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
        Tensor::<f32, SequentialBackend>::from_slice(vec![N1, C_IN1, L1], &inp_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let inp_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![N1, C_IN1, L1], &inp_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let conv_seq = coeus_nn::Conv1d::<f32, SequentialBackend>::new(C_IN1, C_OUT1, K1, false)
        .expect("invariant: test operation succeeds");
    let conv_moirai = coeus_nn::Conv1d::<f32, MoiraiBackend>::new(C_IN1, C_OUT1, K1, false)
        .expect("invariant: test operation succeeds");
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

pub(crate) fn bench_conv_transpose2d_forward(c: &mut Criterion) {
    // ConvTranspose2d: [B=4, C_in=32, 16, 16] → [B=4, C_out=16, 32, 32]
    const CT_B: usize = 4;
    const CT_CIN: usize = 32;
    const CT_COUT: usize = 16;
    const CT_HW: usize = 16;
    let input_data: Vec<f32> = (0..(CT_B * CT_CIN * CT_HW * CT_HW))
        .map(|i| (i as f32 * 0.007).sin())
        .collect();
    let ct_seq = coeus_nn::ConvTranspose2d::<f32, SequentialBackend>::with_params(
        CT_CIN, CT_COUT, 2, 2, 0, 0, 1, true,
    )
    .expect("valid sequential transposed convolution benchmark fixture");
    let ct_moirai = coeus_nn::ConvTranspose2d::<f32, MoiraiBackend>::with_params(
        CT_CIN, CT_COUT, 2, 2, 0, 0, 1, true,
    )
    .expect("valid parallel transposed convolution benchmark fixture");
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![CT_B, CT_CIN, CT_HW, CT_HW], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![CT_B, CT_CIN, CT_HW, CT_HW], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    use coeus_nn::Module;
    let mut group =
        c.benchmark_group("Coeus — ConvTranspose2d forward (4x32x16x16, cin32→cout16 k2 s2)");
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

pub(crate) fn bench_depthwise_conv3d_forward(c: &mut Criterion) {
    // DepthwiseConv3d: [N=2, C=8, 16, 16, 16], k3, pad 1, no bias.
    const DW_N: usize = 2;
    const DW_C: usize = 8;
    const DW_D: usize = 16;
    let input_data: Vec<f32> = (0..(DW_N * DW_C * DW_D * DW_D * DW_D))
        .map(|i| (i as f32 * 0.003).sin())
        .collect();
    let dw_seq = coeus_nn::DepthwiseConv3d::<f32, SequentialBackend>::new(DW_C, 3, 1, false)
        .expect("valid sequential depthwise convolution benchmark fixture");
    let dw_moirai = coeus_nn::DepthwiseConv3d::<f32, MoiraiBackend>::new(DW_C, 3, 1, false)
        .expect("valid parallel depthwise convolution benchmark fixture");
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(
            vec![DW_N, DW_C, DW_D, DW_D, DW_D],
            &input_data,
        )
        .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![DW_N, DW_C, DW_D, DW_D, DW_D], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    use coeus_nn::Module;
    let mut group = c.benchmark_group("Coeus — DepthwiseConv3d forward (2x8x16x16x16, k3)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                dw_seq
                    .forward(black_box(&x_seq))
                    .expect("valid depthwise convolution benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                dw_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid depthwise convolution benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_unfold1d_forward(c: &mut Criterion) {
    // Unfold1d: [4, 16, 64], k3, s1 → [4, 48, 62].
    const UF_N: usize = 4;
    const UF_C: usize = 16;
    const UF_L: usize = 64;
    let input_data: Vec<f32> = (0..(UF_N * UF_C * UF_L))
        .map(|i| (i as f32 * 0.005).sin())
        .collect();
    let uf_seq = coeus_nn::Unfold1d::<f32, SequentialBackend>::new(3, 1, 0, 1);
    let uf_moirai = coeus_nn::Unfold1d::<f32, MoiraiBackend>::new(3, 1, 0, 1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![UF_N, UF_C, UF_L], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![UF_N, UF_C, UF_L], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    use coeus_nn::Module;
    let mut group = c.benchmark_group("Coeus — Unfold1d forward (4x16x64, k3 s1)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                uf_seq
                    .forward(black_box(&x_seq))
                    .expect("valid unfold benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                uf_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid unfold benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_unfold2d_forward(c: &mut Criterion) {
    // Unfold2d: [4, 16, 32, 32], k3, s1 → [4, 144, 900].
    const UF_N: usize = 4;
    const UF_C: usize = 16;
    const UF_HW: usize = 32;
    let input_data: Vec<f32> = (0..(UF_N * UF_C * UF_HW * UF_HW))
        .map(|i| (i as f32 * 0.005).sin())
        .collect();
    let uf_seq = coeus_nn::Unfold2d::<f32, SequentialBackend>::new(3, 1, 0, 1);
    let uf_moirai = coeus_nn::Unfold2d::<f32, MoiraiBackend>::new(3, 1, 0, 1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![UF_N, UF_C, UF_HW, UF_HW], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![UF_N, UF_C, UF_HW, UF_HW], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    use coeus_nn::Module;
    let mut group = c.benchmark_group("Coeus — Unfold2d forward (4x16x32x32, k3 s1)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                uf_seq
                    .forward(black_box(&x_seq))
                    .expect("valid unfold benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                uf_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid unfold benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_fold1d_forward(c: &mut Criterion) {
    // Fold1d: [4, 48, 62] → [4, 16, 64], k3, s1.
    const F_N: usize = 4;
    const F_CK: usize = 48;
    const F_LOUT: usize = 62;
    let input_data: Vec<f32> = (0..(F_N * F_CK * F_LOUT))
        .map(|i| (i as f32 * 0.005).sin())
        .collect();
    let f_seq = coeus_nn::Fold1d::<f32, SequentialBackend>::new(64, 3, 1, 0, 1);
    let f_moirai = coeus_nn::Fold1d::<f32, MoiraiBackend>::new(64, 3, 1, 0, 1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![F_N, F_CK, F_LOUT], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![F_N, F_CK, F_LOUT], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    use coeus_nn::Module;
    let mut group = c.benchmark_group("Coeus — Fold1d forward (4x48x62 → 4x16x64)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                f_seq
                    .forward(black_box(&x_seq))
                    .expect("valid fold benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                f_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid fold benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_fold2d_forward(c: &mut Criterion) {
    // Fold2d: [4, 144, 900] → [4, 16, 32, 32], k3, s1.
    const F_N: usize = 4;
    const F_CKK: usize = 144;
    const F_LOUT: usize = 900;
    let input_data: Vec<f32> = (0..(F_N * F_CKK * F_LOUT))
        .map(|i| (i as f32 * 0.005).sin())
        .collect();
    let f_seq = coeus_nn::Fold2d::<f32, SequentialBackend>::new(32, 32, 3, 1, 0, 1);
    let f_moirai = coeus_nn::Fold2d::<f32, MoiraiBackend>::new(32, 32, 3, 1, 0, 1);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![F_N, F_CKK, F_LOUT], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![F_N, F_CKK, F_LOUT], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    use coeus_nn::Module;
    let mut group = c.benchmark_group("Coeus — Fold2d forward (4x144x900 → 4x16x32x32)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                f_seq
                    .forward(black_box(&x_seq))
                    .expect("valid fold benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                f_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid fold benchmark input"),
            )
        })
    });
    group.finish();
}
