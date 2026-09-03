//! sequence benchmarks.

use super::*;

#[path = "sequence/embedding.rs"]
mod embedding;

pub(crate) use embedding::{bench_embedding_forward, bench_embeddingbag_sum};

pub(crate) fn bench_sequential_composition_forward(c: &mut Criterion) {
    const BATCH: usize = 128;
    const INPUT: usize = 256;
    const HIDDEN: usize = 512;
    const OUTPUT: usize = 256;

    let mut dynamic_sequential = Sequential::<f32, SequentialBackend>::new();
    dynamic_sequential
        .add(
            Linear::new(INPUT, HIDDEN, true)
                .expect("invariant: the fixture's layer dimensions are non-zero"),
        )
        .add(ReLU)
        .add(
            Linear::new(HIDDEN, OUTPUT, true)
                .expect("invariant: the fixture's layer dimensions are non-zero"),
        );
    let mut static_sequential = Linear::<f32, SequentialBackend>::new(INPUT, HIDDEN, true)
        .expect("invariant: the fixture's layer dimensions are non-zero")
        .append(ReLU)
        .append(
            Linear::new(HIDDEN, OUTPUT, true)
                .expect("invariant: the fixture's layer dimensions are non-zero"),
        );
    static_sequential.load_parameters(&dynamic_sequential.parameters());

    let mut dynamic_moirai = Sequential::<f32, MoiraiBackend>::new();
    dynamic_moirai
        .add(
            Linear::new(INPUT, HIDDEN, true)
                .expect("invariant: the fixture's layer dimensions are non-zero"),
        )
        .add(ReLU)
        .add(
            Linear::new(HIDDEN, OUTPUT, true)
                .expect("invariant: the fixture's layer dimensions are non-zero"),
        );
    let mut static_moirai = Linear::<f32, MoiraiBackend>::new(INPUT, HIDDEN, true)
        .expect("invariant: the fixture's layer dimensions are non-zero")
        .append(ReLU)
        .append(
            Linear::new(HIDDEN, OUTPUT, true)
                .expect("invariant: the fixture's layer dimensions are non-zero"),
        );
    static_moirai.load_parameters(&dynamic_moirai.parameters());

    let input_data: Vec<f32> = (0..(BATCH * INPUT))
        .map(|index| (index as f32 * 0.0013).sin())
        .collect();
    let input_sequential = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice([BATCH, INPUT], &input_data),
        false,
    );
    let input_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice([BATCH, INPUT], &input_data),
        false,
    );

    let dynamic_sequential_output = dynamic_sequential
        .forward(&input_sequential)
        .expect("valid dynamic sequential benchmark input");
    let static_sequential_output = static_sequential
        .forward(&input_sequential)
        .expect("valid static sequential benchmark input");
    assert_eq!(
        dynamic_sequential_output.tensor.as_slice(),
        static_sequential_output.tensor.as_slice(),
        "dynamic and static Sequential composition must be value-equivalent"
    );

    let dynamic_moirai_output = dynamic_moirai
        .forward(&input_moirai)
        .expect("valid dynamic Moirai sequential benchmark input");
    let static_moirai_output = static_moirai
        .forward(&input_moirai)
        .expect("valid static Moirai sequential benchmark input");
    assert_eq!(
        dynamic_moirai_output.tensor.as_slice(),
        static_moirai_output.tensor.as_slice(),
        "dynamic and static Moirai composition must be value-equivalent"
    );
    drop((
        dynamic_sequential_output,
        static_sequential_output,
        dynamic_moirai_output,
        static_moirai_output,
    ));

    let mut group = c.benchmark_group("Coeus — sequential composition forward (128x256)");
    group.bench_function("Sequential backend, dynamic modules", |b| {
        b.iter(|| {
            black_box(
                dynamic_sequential
                    .forward(black_box(&input_sequential))
                    .expect("valid dynamic sequential benchmark input"),
            )
        })
    });
    group.bench_function("Sequential backend, static modules", |b| {
        b.iter(|| {
            black_box(
                static_sequential
                    .forward(black_box(&input_sequential))
                    .expect("valid static sequential benchmark input"),
            )
        })
    });
    group.bench_function("Moirai backend, dynamic modules", |b| {
        b.iter(|| {
            black_box(
                dynamic_moirai
                    .forward(black_box(&input_moirai))
                    .expect("valid dynamic Moirai sequential benchmark input"),
            )
        })
    });
    group.bench_function("Moirai backend, static modules", |b| {
        b.iter(|| {
            black_box(
                static_moirai
                    .forward(black_box(&input_moirai))
                    .expect("valid static Moirai sequential benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_lstm_forward(c: &mut Criterion) {
    // LSTM sequence forward on [batch=4, seq=32, input=64] → hidden=128.
    // Modest size keeps the run tractable while exercising the full unroll path.
    const LSTM_BATCH: usize = 4;
    const LSTM_SEQ: usize = 32;
    const LSTM_IN: usize = 64;
    const LSTM_H: usize = 128;

    let input_data: Vec<f32> = (0..(LSTM_BATCH * LSTM_SEQ * LSTM_IN))
        .map(|i| (i as f32 * 0.0017).cos())
        .collect();

    // Coeus: Lstm::new(input_size, hidden_size).expect("invariant: the fixture's layer dimensions are non-zero").
    let lstm_seq = Lstm::<f32, SequentialBackend>::new(LSTM_IN, LSTM_H)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let lstm_moirai = Lstm::<f32, MoiraiBackend>::new(LSTM_IN, LSTM_H)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(
            vec![LSTM_BATCH, LSTM_SEQ, LSTM_IN],
            &input_data,
        ),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![LSTM_BATCH, LSTM_SEQ, LSTM_IN], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — LSTM forward (4x32 seq, in=64 hidden=128)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                lstm_seq
                    .forward(black_box(&x_seq))
                    .expect("valid LSTM benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                lstm_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid LSTM benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_gru_forward(c: &mut Criterion) {
    // GRU sequence forward on [batch=4, seq=32, input=64] → hidden=128.
    // Same shape as the LSTM row so the recurrent-family comparison is direct.
    // GRU has 3 gates vs LSTM's 4, but the same compute shape (one projection per
    // timestep); the unroll loop + cat/reshape output stacking costs dominate.
    const GRU_BATCH: usize = 4;
    const GRU_SEQ: usize = 32;
    const GRU_IN: usize = 64;
    const GRU_H: usize = 128;

    let input_data: Vec<f32> = (0..(GRU_BATCH * GRU_SEQ * GRU_IN))
        .map(|i| (i as f32 * 0.0023).sin())
        .collect();

    // Coeus: Gru::new(input_size, hidden_size).expect("invariant: the fixture's layer dimensions are non-zero").
    let gru_seq = CoeusGru::<f32, SequentialBackend>::new(GRU_IN, GRU_H)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let gru_moirai = CoeusGru::<f32, MoiraiBackend>::new(GRU_IN, GRU_H)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![GRU_BATCH, GRU_SEQ, GRU_IN], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![GRU_BATCH, GRU_SEQ, GRU_IN], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — GRU forward (4x32 seq, in=64 hidden=128)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                gru_seq
                    .forward(black_box(&x_seq))
                    .expect("valid GRU benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                gru_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid GRU benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_rnn_forward(c: &mut Criterion) {
    // CPU execution policies on the same sequence workload as LSTM and GRU.
    const RNN_BATCH: usize = 4;
    const RNN_SEQ: usize = 32;
    const RNN_IN: usize = 64;
    const RNN_H: usize = 128;

    let input_data: Vec<f32> = (0..(RNN_BATCH * RNN_SEQ * RNN_IN))
        .map(|index| (index as f32 * 0.0011).sin())
        .collect();
    let rnn_seq = Rnn::<f32, SequentialBackend>::new(RNN_IN, RNN_H, RnnNonlinearity::Tanh)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let rnn_moirai = Rnn::<f32, MoiraiBackend>::new(RNN_IN, RNN_H, RnnNonlinearity::Tanh)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![RNN_BATCH, RNN_SEQ, RNN_IN], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![RNN_BATCH, RNN_SEQ, RNN_IN], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — vanilla RNN forward (4x32 seq, in=64 hidden=128)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                rnn_seq
                    .forward(black_box(&x_seq))
                    .expect("valid recurrent layer benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                rnn_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid recurrent layer benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_rnn_cell_forward(c: &mut Criterion) {
    // isolates one recurrent step from sequence unrolling and output stacking.
    const RNN_BATCH: usize = 4;
    const RNN_IN: usize = 64;
    const RNN_H: usize = 128;

    let input_data: Vec<f32> = (0..(RNN_BATCH * RNN_IN))
        .map(|index| (index as f32 * 0.0029).cos())
        .collect();
    let hidden_data: Vec<f32> = (0..(RNN_BATCH * RNN_H))
        .map(|index| (index as f32 * 0.0017).sin())
        .collect();
    let cell_seq = RNNCell::<f32, SequentialBackend>::new(RNN_IN, RNN_H, RnnNonlinearity::Tanh)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let cell_moirai = RNNCell::<f32, MoiraiBackend>::new(RNN_IN, RNN_H, RnnNonlinearity::Tanh)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![RNN_BATCH, RNN_IN], &input_data),
        false,
    );
    let h_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![RNN_BATCH, RNN_H], &hidden_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![RNN_BATCH, RNN_IN], &input_data),
        false,
    );
    let h_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![RNN_BATCH, RNN_H], &hidden_data),
        false,
    );

    let mut group =
        c.benchmark_group("Coeus — vanilla RNNCell forward (batch=4, in=64 hidden=128)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(cell_seq.step(black_box(&x_seq), black_box(&h_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(cell_moirai.step(black_box(&x_moirai), black_box(&h_moirai))))
    });
    group.finish();
}

pub(crate) fn bench_bidirectional_rnn_forward(c: &mut Criterion) {
    const RNN_BATCH: usize = 4;
    const RNN_SEQ: usize = 32;
    const RNN_IN: usize = 64;
    const RNN_H: usize = 128;

    let input_data: Vec<f32> = (0..(RNN_BATCH * RNN_SEQ * RNN_IN))
        .map(|index| (index as f32 * 0.0009).cos())
        .collect();
    let bi_seq = Bidirectional::new(
        Rnn::<f32, SequentialBackend>::new(RNN_IN, RNN_H, RnnNonlinearity::Tanh)
            .expect("invariant: the fixture's layer dimensions are non-zero"),
        Rnn::<f32, SequentialBackend>::new(RNN_IN, RNN_H, RnnNonlinearity::Tanh)
            .expect("invariant: the fixture's layer dimensions are non-zero"),
    );
    let bi_moirai = Bidirectional::new(
        Rnn::<f32, MoiraiBackend>::new(RNN_IN, RNN_H, RnnNonlinearity::Tanh)
            .expect("invariant: the fixture's layer dimensions are non-zero"),
        Rnn::<f32, MoiraiBackend>::new(RNN_IN, RNN_H, RnnNonlinearity::Tanh)
            .expect("invariant: the fixture's layer dimensions are non-zero"),
    );
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![RNN_BATCH, RNN_SEQ, RNN_IN], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![RNN_BATCH, RNN_SEQ, RNN_IN], &input_data),
        false,
    );

    let mut group =
        c.benchmark_group("Coeus — bidirectional RNN forward (4x32 seq, in=64 hidden=128)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                bi_seq
                    .forward(black_box(&x_seq))
                    .expect("valid bidirectional recurrent layer benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                bi_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid bidirectional recurrent layer benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_swiglu_forward(c: &mut Criterion) {
    // SwiGLU forward on [batch=32, d_input=256] → d_output=512 — the FFN-style
    // projection shape. Two parallel d_input→d_output linear projections plus a
    // SiLU gate and an element-wise product; the two matmuls dominate.
    const SG_BATCH: usize = 32;
    const SG_IN: usize = 256;
    const SG_OUT: usize = 512;

    let input_data: Vec<f32> = (0..(SG_BATCH * SG_IN))
        .map(|i| (i as f32 * 0.0017).sin())
        .collect();

    // Coeus: SwiGlu::new(d_input, d_output, bias=false).expect("invariant: the fixture's layer dimensions are non-zero").
    let sg_seq = SwiGlu::<f32, SequentialBackend>::new(SG_IN, SG_OUT, false)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let sg_moirai = SwiGlu::<f32, MoiraiBackend>::new(SG_IN, SG_OUT, false)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![SG_BATCH, SG_IN], &input_data),
        false,
    );
    let x_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![SG_BATCH, SG_IN], &input_data),
        false,
    );

    let mut group = c.benchmark_group("Coeus — SwiGLU forward (32 batch, in=256 out=512)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                sg_seq
                    .forward(black_box(&x_seq))
                    .expect("valid SwiGLU benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                sg_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid SwiGLU benchmark input"),
            )
        })
    });
    group.finish();
}
