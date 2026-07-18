//! indexing benchmarks.

use super::*;

pub(crate) fn bench_diff_forward(c: &mut Criterion) {
    // torch.diff(x, n=1) — first-order discrete difference along last dim.
    let input_data: Vec<f32> = (0..BATCH * FEATURES)
        .map(|i| (i as f32 * 0.0023).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus — diff(n=1) forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::diff(black_box(&x_seq), 1, 1)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::diff(black_box(&x_moirai), 1, 1)))
    });
    group.finish();
}

pub(crate) fn bench_tril_forward(c: &mut Criterion) {
    // tril: [256, 256] lower-triangular mask, diagonal=0.
    const SZ: usize = 256;
    let input_data: Vec<f32> = (0..(SZ * SZ)).map(|i| i as f32 * 0.001).collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![SZ, SZ], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![SZ, SZ], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — tril forward (256x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::tril(black_box(&x_seq), 0)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::tril(black_box(&x_moirai), 0)))
    });
    group.finish();
}

pub(crate) fn bench_topk_forward(c: &mut Criterion) {
    // topk k=16 along dim=1 on [128, 256].
    const TOPK: usize = 16;
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.007).sin())
        .collect();
    let x_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data);
    let x_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data);

    let mut group = c.benchmark_group("Coeus — topk(k=16) forward (128x256, dim=1)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_ops::topk(black_box(&x_seq), TOPK, 1, true)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_ops::topk(black_box(&x_moirai), TOPK, 1, true)))
    });
    group.finish();
}

pub(crate) fn bench_roll_forward(c: &mut Criterion) {
    // roll shift=32 along dim=1 on [128, 256].
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0031).sin())
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — roll(shift=32,dim=1) forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_autograd::roll(
                black_box(&x_seq),
                &[32isize],
                &[1usize],
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_autograd::roll(
                black_box(&x_moirai),
                &[32isize],
                &[1usize],
            ))
        })
    });
    group.finish();
}

pub(crate) fn bench_gather_forward(c: &mut Criterion) {
    // gather dim=1 on [128, 256] with indices covering full column range.
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.0017).sin())
        .collect();
    let idx_f32_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| ((i * 7 + 13) % FEATURES) as f32)
        .collect();

    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let idx_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &idx_f32_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let idx_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &idx_f32_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — gather forward (128x256, dim=1)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_autograd::gather(
                black_box(&x_seq),
                1,
                black_box(&idx_seq),
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_autograd::gather(
                black_box(&x_moirai),
                1,
                black_box(&idx_moirai),
            ))
        })
    });
    group.finish();
}

pub(crate) fn bench_clamp_forward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - clamp(-1,1) forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_autograd::clamp(black_box(&x_seq), -1.0, 1.0)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_autograd::clamp(black_box(&x_moirai), -1.0, 1.0)))
    });
    group.finish();
}

pub(crate) fn bench_scatter_add_forward(c: &mut Criterion) {
    // scatter_add dim=1 on [128,256] with random indices covering full range
    const S: usize = BATCH;
    const D: usize = FEATURES;
    let src_data: Vec<f32> = (0..(S * D)).map(|i| (i as f32 * 0.001).sin()).collect();
    let idx_f32: Vec<f32> = (0..(S * D)).map(|i| ((i * 7 + 13) % D) as f32).collect();
    let base_data = vec![0.0f32; S * D];
    let base_seq =
        coeus_tensor::Tensor::<f32, SequentialBackend>::from_slice(vec![S, D], &base_data);
    let idx_seq = coeus_tensor::Tensor::<f32, SequentialBackend>::from_slice(vec![S, D], &idx_f32);
    let src_seq = coeus_tensor::Tensor::<f32, SequentialBackend>::from_slice(vec![S, D], &src_data);
    let base_moirai =
        coeus_tensor::Tensor::<f32, MoiraiBackend>::from_slice(vec![S, D], &base_data);
    let idx_moirai = coeus_tensor::Tensor::<f32, MoiraiBackend>::from_slice(vec![S, D], &idx_f32);
    let src_moirai = coeus_tensor::Tensor::<f32, MoiraiBackend>::from_slice(vec![S, D], &src_data);

    let mut group = c.benchmark_group("Coeus - scatter_add forward (128x256, dim=1)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_ops::scatter_add(
                black_box(&base_seq),
                1,
                black_box(&idx_seq),
                black_box(&src_seq),
                &SequentialBackend,
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::scatter_add(
                black_box(&base_moirai),
                1,
                black_box(&idx_moirai),
                black_box(&src_moirai),
                &MoiraiBackend,
            ))
        })
    });
    group.finish();
}

pub(crate) fn bench_argmax2_forward(c: &mut Criterion) {
    // argmax dim=1 on [128,256]
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.007).sin())
        .collect();
    let x_seq = coeus_tensor::Tensor::<f32, SequentialBackend>::from_slice(
        vec![BATCH, FEATURES],
        &input_data,
    );
    let x_moirai =
        coeus_tensor::Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data);

    let mut group = c.benchmark_group("Coeus - argmax(dim=1) forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_ops::argmax(black_box(&x_seq), 1)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_ops::argmax(black_box(&x_moirai), 1)))
    });
    group.finish();
}

pub(crate) fn bench_topk2_forward(c: &mut Criterion) {
    // topk k=32 on [128,256] dim=1
    const K2: usize = 32;
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.009).cos())
        .collect();
    let x_seq = coeus_tensor::Tensor::<f32, SequentialBackend>::from_slice(
        vec![BATCH, FEATURES],
        &input_data,
    );
    let x_moirai =
        coeus_tensor::Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data);

    let mut group = c.benchmark_group("Coeus - topk(k=32,dim=1) forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_ops::topk(black_box(&x_seq), K2, 1, true)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(coeus_ops::topk(black_box(&x_moirai), K2, 1, true)))
    });
    group.finish();
}

pub(crate) fn bench_where_cond_forward(c: &mut Criterion) {
    let cond_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let a_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin())
        .collect();
    let b_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.005).cos())
        .collect();
    let cond_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &cond_data),
        false,
    );
    let a_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &a_data),
        false,
    );
    let b_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &b_data),
        false,
    );
    let cond_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &cond_data),
        false,
    );
    let a_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &a_data),
        false,
    );
    let b_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &b_data),
        false,
    );
    let mut group = c.benchmark_group("Coeus - where_cond forward (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_autograd::where_cond(
                black_box(&cond_seq),
                black_box(&a_seq),
                black_box(&b_seq),
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_autograd::where_cond(
                black_box(&cond_moirai),
                black_box(&a_moirai),
                black_box(&b_moirai),
            ))
        })
    });
    group.finish();
}

pub(crate) fn bench_flip_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES)).map(|i| i as f32 * 0.01).collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - flip(axis=1) fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::flip(black_box(&x_seq), 1);
            black_box(o).backward()
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::flip(black_box(&x_moirai), 1);
            black_box(o).backward()
        })
    });
    group.finish();
}

pub(crate) fn bench_permute_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES)).map(|i| i as f32 * 0.01).collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - permute([1,0]) fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::permute(black_box(&x_seq), &[1, 0]);
            black_box(o).backward()
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::permute(black_box(&x_moirai), &[1, 0]);
            black_box(o).backward()
        })
    });
    group.finish();
}

pub(crate) fn bench_tile_backward(c: &mut Criterion) {
    let sz = 64usize;
    let input_data: Vec<f32> = (0..(sz * sz)).map(|i| i as f32 * 0.01).collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![sz, sz], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![sz, sz], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - tile([2,2]) fwd+bwd (64x64)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::tile(black_box(&x_seq), &[2, 2]);
            black_box(o).backward()
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::tile(black_box(&x_moirai), &[2, 2]);
            black_box(o).backward()
        })
    });
    group.finish();
}

pub(crate) fn bench_clamp_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.004).sin() * 3.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - clamp(-1,1) fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::clamp(black_box(&x_seq), -1.0, 1.0);
            black_box(o).backward()
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::clamp(black_box(&x_moirai), -1.0, 1.0);
            black_box(o).backward()
        })
    });
    group.finish();
}

pub(crate) fn bench_sort_backward(c: &mut Criterion) {
    let input_data: Vec<f32> = (0..(BATCH * FEATURES))
        .map(|i| (i as f32 * 0.003).sin() * 2.0)
        .collect();
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, FEATURES], &input_data),
        true,
    );
    let mut group = c.benchmark_group("Coeus - sort(dim=1) fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let (o, _) = coeus_autograd::sort(black_box(&x_seq), 1, false);
            black_box(o).backward()
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let (o, _) = coeus_autograd::sort(black_box(&x_moirai), 1, false);
            black_box(o).backward()
        })
    });
    group.finish();
}
