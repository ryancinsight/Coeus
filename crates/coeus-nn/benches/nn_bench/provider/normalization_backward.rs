//! normalization backward benchmarks.

use super::*;

pub(crate) fn bench_norm_backward(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("Coeus - L2_norm fwd+bwd (128x256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            let o = coeus_autograd::norm(black_box(&x_seq));
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            let o = coeus_autograd::norm(black_box(&x_moirai));
            black_box(o)
                .backward()
                .expect("invariant: valid autograd fixture completes backward")
        })
    });
    group.finish();
}
