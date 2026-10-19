use super::*;

fn regression_vars(
    data: &[f32],
    n: usize,
    d: usize,
) -> (
    coeus_autograd::Var<f32, SequentialBackend>,
    coeus_autograd::Var<f32, MoiraiBackend>,
) {
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![n, d], data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![n, d], data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    (x_seq, x_moirai)
}

pub(crate) fn bench_l1_loss(c: &mut Criterion) {
    // L1 loss on [128, 256] prediction vs target.
    const N: usize = 128;
    const D: usize = 256;

    let pred_data: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0041).sin()).collect();
    let target_data: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0077).cos()).collect();
    let (p_seq, p_moirai) = regression_vars(&pred_data, N, D);
    let (t_seq, t_moirai) = regression_vars(&target_data, N, D);

    let mut group = c.benchmark_group("Coeus — L1Loss (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::l1_loss(black_box(&p_seq), black_box(&t_seq))
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::l1_loss(black_box(&p_moirai), black_box(&t_moirai))
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_smooth_l1_loss(c: &mut Criterion) {
    // Smooth-L1 loss on [128, 256] prediction vs target, beta 1.0.
    const N: usize = 128;
    const D: usize = 256;

    let pred_data: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0041).sin()).collect();
    let target_data: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0077).cos()).collect();
    let (p_seq, p_moirai) = regression_vars(&pred_data, N, D);
    let (t_seq, t_moirai) = regression_vars(&target_data, N, D);

    let mut group = c.benchmark_group("Coeus — SmoothL1Loss (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::smooth_l1_loss(black_box(&p_seq), black_box(&t_seq), 1.0)
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::smooth_l1_loss(black_box(&p_moirai), black_box(&t_moirai), 1.0)
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_poisson_nll_loss(c: &mut Criterion) {
    // Poisson NLL on [128, 256] input vs non-negative target.
    const N: usize = 128;
    const D: usize = 256;

    let input_data: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0041).sin()).collect();
    let target_data: Vec<f32> = (0..(N * D))
        .map(|i| 1.0 + (i as f32 * 0.0077).cos())
        .collect();
    let (x_seq, x_moirai) = regression_vars(&input_data, N, D);
    let (t_seq, t_moirai) = regression_vars(&target_data, N, D);

    let mut group = c.benchmark_group("Coeus — PoissonNLLLoss (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::poisson_nll(black_box(&x_seq), black_box(&t_seq))
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::poisson_nll(black_box(&x_moirai), black_box(&t_moirai))
                    .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_gaussian_nll_loss(c: &mut Criterion) {
    // Gaussian NLL on [128, 256] input/target/variance, full=false.
    const N: usize = 128;
    const D: usize = 256;

    let input_data: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0041).sin()).collect();
    let target_data: Vec<f32> = (0..(N * D)).map(|i| (i as f32 * 0.0077).cos()).collect();
    let var_data: Vec<f32> = (0..(N * D))
        .map(|i| 0.5 + 0.25 * (i as f32 * 0.0033).sin())
        .collect();
    let (x_seq, x_moirai) = regression_vars(&input_data, N, D);
    let (t_seq, t_moirai) = regression_vars(&target_data, N, D);
    let (v_seq, v_moirai) = regression_vars(&var_data, N, D);

    let mut group = c.benchmark_group("Coeus — GaussianNLLLoss (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::gaussian_nll_loss(
                    black_box(&x_seq),
                    black_box(&t_seq),
                    black_box(&v_seq),
                    false,
                )
                .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                coeus_nn::gaussian_nll_loss(
                    black_box(&x_moirai),
                    black_box(&t_moirai),
                    black_box(&v_moirai),
                    false,
                )
                .expect("invariant: benchmark shapes are valid"),
            )
        })
    });
    group.finish();
}
