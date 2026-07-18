//! dense benchmarks.

use super::*;

pub(crate) fn bench_linear_forward(c: &mut Criterion) {
    // Coeus: same dims; forward builds the autograd graph (production path).
    let lin_seq = Linear::<f32, SequentialBackend>::new(FEATURES, FEATURES, true);
    let lin_moirai = Linear::<f32, MoiraiBackend>::new(FEATURES, FEATURES, true);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![BATCH, FEATURES]),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::ones(vec![BATCH, FEATURES]),
        false,
    );

    let mut group = c.benchmark_group("Coeus — Linear forward (128x256 -> 256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(lin_seq.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(lin_moirai.forward(black_box(&x_moirai))))
    });
    group.finish();
}

/// Bilinear feature-interaction comparison across native providers.
pub(crate) fn bench_bilinear_forward(c: &mut Criterion) {
    // Bilinear(in1=64, in2=64, out=32) forward on batch=128 two-input feature
    // interaction: out[n,k] = x1[n,:] @ W[k,:,:] @ x2[n,:].T + b[k].
    const BL_BATCH: usize = 128;
    const BL_IN1: usize = 64;
    const BL_IN2: usize = 64;
    const BL_OUT: usize = 32;

    let x1_data: Vec<f32> = (0..(BL_BATCH * BL_IN1))
        .map(|i| (i as f32 * 0.003).sin())
        .collect();
    let x2_data: Vec<f32> = (0..(BL_BATCH * BL_IN2))
        .map(|i| (i as f32 * 0.0027).cos())
        .collect();

    let bl_seq = Bilinear::<f32, SequentialBackend>::new(BL_IN1, BL_IN2, BL_OUT, true);
    let bl_moirai = Bilinear::<f32, MoiraiBackend>::new(BL_IN1, BL_IN2, BL_OUT, true);
    let x1_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BL_BATCH, BL_IN1], &x1_data),
        false,
    );
    let x2_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BL_BATCH, BL_IN2], &x2_data),
        false,
    );
    let x1_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BL_BATCH, BL_IN1], &x1_data),
        false,
    );
    let x2_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BL_BATCH, BL_IN2], &x2_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — Bilinear forward (batch128, in1=64 in2=64 out=32)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(bl_seq.bilinear_forward(black_box(&x1_seq), black_box(&x2_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(bl_moirai.bilinear_forward(black_box(&x1_moirai), black_box(&x2_moirai)))
        })
    });
    group.finish();
}

pub(crate) fn bench_linear_forward_backward(c: &mut Criterion) {
    // Full autograd cycle — forward + sum-loss + backward — for Linear
    // uses its Autodiff<provider> backend over the same manual linear expression.

    let lin_seq = Linear::<f32, SequentialBackend>::new(FEATURES, FEATURES, true);
    let lin_moirai = Linear::<f32, MoiraiBackend>::new(FEATURES, FEATURES, true);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![BATCH, FEATURES]),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::ones(vec![BATCH, FEATURES]),
        true,
    );

    let mut group = c.benchmark_group("Coeus — Linear forward+backward (128x256 -> 256)");
    // Coeus accumulates grads into the leaf Vars; zero them each iteration so the
    // measured backward work is identical across iterations (zero_grad is O(params)).
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            lin_seq.zero_grad();
            x_seq.zero_grad();
            coeus_autograd::sum(&lin_seq.forward(black_box(&x_seq))).backward();
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            lin_moirai.zero_grad();
            x_moirai.zero_grad();
            coeus_autograd::sum(&lin_moirai.forward(black_box(&x_moirai))).backward();
        })
    });
    group.finish();
}

pub(crate) fn bench_dropout_forward(c: &mut Criterion) {
    // Dropout eval-mode forward (no masking, p=0.5): [128, 256].
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0023).sin())
        .collect();

    // Coeus: Dropout is not generic over B; uses f32 Var with default backend.
    let mut do_layer = Dropout::new(0.5);
    do_layer.set_training(false);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — Dropout eval forward (128x256, p=0.5)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(do_layer.forward(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(do_layer.forward(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_linear_fwd_bwd(c: &mut Criterion) {
    // Linear(256,512) fwd+bwd: [128,256]
    const IN_F: usize = FEATURES;
    const OUT_F: usize = 512;
    let inp_data: Vec<f32> = (0..(BATCH * IN_F))
        .map(|i| (i as f32 * 0.002).sin())
        .collect();
    let inp_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, IN_F], &inp_data),
        true,
    );
    let inp_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, IN_F], &inp_data),
        true,
    );
    let lin_seq = coeus_nn::Linear::<f32, SequentialBackend>::new(IN_F, OUT_F, false);
    let lin_moirai = coeus_nn::Linear::<f32, MoiraiBackend>::new(IN_F, OUT_F, false);
    let mut group = c.benchmark_group("Coeus - Linear(256,512) fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = lin_seq.forward(black_box(&inp_seq));
            black_box(o).backward()
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = lin_moirai.forward(black_box(&inp_moirai));
            black_box(o).backward()
        })
    });
    group.finish();
}
