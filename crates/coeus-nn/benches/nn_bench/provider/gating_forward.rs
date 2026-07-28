//! gating forward benchmarks.

use super::*;

pub(crate) fn bench_sigmoid_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let mut group = c.benchmark_group("Coeus — Sigmoid forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(sigmoid(black_box(&x_seq)).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(sigmoid(black_box(&x_moirai)).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_tanh_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let mut group = c.benchmark_group("Coeus — Tanh forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(tanh(black_box(&x_seq)).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(tanh(black_box(&x_moirai)).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_silu_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let mut group = c.benchmark_group("Coeus — SiLU forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(silu(black_box(&x_seq)).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(silu(black_box(&x_moirai)).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_mish_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus — Mish forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_nn::mish(black_box(&x_seq)).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_nn::mish(black_box(&x_moirai)).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_log_sigmoid_forward(c: &mut Criterion) {
    // LogSigmoid = log(sigmoid(x)) = -softplus(-x).
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus — LogSigmoid forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_nn::log_sigmoid(black_box(&x_seq)).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_nn::log_sigmoid(black_box(&x_moirai)).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_softplus_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus — Softplus forward (128x256, beta=1)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_nn::softplus(black_box(&x_seq)).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_nn::softplus(black_box(&x_moirai)).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_softplus_activation(c: &mut Criterion) {
    // softplus: [128, 256] — F.softplus(x, beta=1, threshold=20).
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0037).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let mut group = c.benchmark_group("Coeus — softplus forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_nn::softplus(black_box(&x_seq)).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_nn::softplus(black_box(&x_moirai)).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_tanh2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - tanh fwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::tanh(black_box(&x_seq)).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::tanh(black_box(&x_moirai)).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_sigmoid2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - sigmoid fwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::sigmoid(black_box(&x_seq)).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::sigmoid(black_box(&x_moirai)).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_atanh_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.001).sin() * 0.9)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - atanh forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::atanh(black_box(&x_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::atanh(black_box(&x_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_silu2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - silu2 fwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::silu(black_box(&x_seq)).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::silu(black_box(&x_moirai)).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_hardsigmoid2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 5.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - hardsigmoid fwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::hardsigmoid(black_box(&x_seq)).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::hardsigmoid(black_box(&x_moirai)).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_softsign_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - softsign forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::softsign(&x_seq).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::softsign(&x_moirai).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_hardsigmoid_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 5.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - hardsigmoid forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::hardsigmoid(&x_seq).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::hardsigmoid(&x_moirai).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_softplus2_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - softplus2 forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::softplus(&x_seq).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::softplus(&x_moirai).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_hardswish_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.002).sin() * 4.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - hardswish forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::hardswish(&x_seq).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::hardswish(&x_moirai).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_hardtanh_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - hardtanh(-1,1) forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::hardtanh(&x_seq, -1.0, 1.0).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::hardtanh(&x_moirai, -1.0, 1.0).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_tanh3_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| i as f32 * 0.002 - 1.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - tanh3 forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::tanh(black_box(&x_seq)).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::tanh(black_box(&x_moirai)).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_sigmoid3_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| i as f32 * 0.002 - 1.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - sigmoid3 forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::sigmoid(black_box(&x_seq)).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::sigmoid(black_box(&x_moirai)).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_tanh4_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| i as f32 * 0.0025 - 1.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - tanh4 forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::tanh(black_box(&x_seq)).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::tanh(black_box(&x_moirai)).expect("run operation")))
    });
    group.finish();
}

pub(crate) fn bench_sigmoid4_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| i as f32 * 0.0025 - 1.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let mut group = c.benchmark_group("Coeus - sigmoid4 forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::sigmoid(black_box(&x_seq)).expect("run operation")))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::sigmoid(black_box(&x_moirai)).expect("run operation")))
    });
    group.finish();
}
