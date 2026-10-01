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
