use super::*;

pub(crate) fn bench_gru_cell_forward(c: &mut Criterion) {
    // GRUCell one step (batch=4, in=64 hidden=128).
    const RNN_BATCH: usize = 4;
    const RNN_IN: usize = 64;
    const RNN_H: usize = 128;

    let input_data: Vec<f32> = (0..(RNN_BATCH * RNN_IN))
        .map(|index| (index as f32 * 0.0029).cos())
        .collect();
    let hidden_data: Vec<f32> = (0..(RNN_BATCH * RNN_H))
        .map(|index| (index as f32 * 0.0017).sin())
        .collect();
    let cell_seq = coeus_nn::GRUCell::<f32, SequentialBackend>::new(RNN_IN, RNN_H)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let cell_moirai = coeus_nn::GRUCell::<f32, MoiraiBackend>::new(RNN_IN, RNN_H)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![RNN_BATCH, RNN_IN], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let h_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![RNN_BATCH, RNN_H], &hidden_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![RNN_BATCH, RNN_IN], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let h_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![RNN_BATCH, RNN_H], &hidden_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — GRUCell forward (batch=4, in=64 hidden=128)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(cell_seq.step(black_box(&x_seq), black_box(&h_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(cell_moirai.step(black_box(&x_moirai), black_box(&h_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_lstm_cell_forward(c: &mut Criterion) {
    // LSTMCell one step (batch=4, in=64 hidden=128).
    const RNN_BATCH: usize = 4;
    const RNN_IN: usize = 64;
    const RNN_H: usize = 128;

    let input_data: Vec<f32> = (0..(RNN_BATCH * RNN_IN))
        .map(|index| (index as f32 * 0.0029).cos())
        .collect();
    let hidden_data: Vec<f32> = (0..(RNN_BATCH * RNN_H))
        .map(|index| (index as f32 * 0.0017).sin())
        .collect();
    let cell_data: Vec<f32> = (0..(RNN_BATCH * RNN_H))
        .map(|index| (index as f32 * 0.0011).cos())
        .collect();
    let cell_seq = coeus_nn::LSTMCell::<f32, SequentialBackend>::new(RNN_IN, RNN_H)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let cell_moirai = coeus_nn::LSTMCell::<f32, MoiraiBackend>::new(RNN_IN, RNN_H)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![RNN_BATCH, RNN_IN], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let h_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![RNN_BATCH, RNN_H], &hidden_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let c_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![RNN_BATCH, RNN_H], &cell_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![RNN_BATCH, RNN_IN], &input_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let h_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![RNN_BATCH, RNN_H], &hidden_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let c_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![RNN_BATCH, RNN_H], &cell_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let mut group = c.benchmark_group("Coeus — LSTMCell forward (batch=4, in=64 hidden=128)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(cell_seq.step(black_box(&x_seq), black_box(&h_seq), black_box(&c_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(cell_moirai.step(
                black_box(&x_moirai),
                black_box(&h_moirai),
                black_box(&c_moirai),
            ))
        })
    });
    group.finish();
}
