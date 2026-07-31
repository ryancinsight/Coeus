//! sequence benchmarks.

use super::*;

pub(crate) fn bench_embedding_forward(c: &mut Criterion) {
    // Embedding lookup on [batch=2, seq=16] into [vocab=4096, d_model=256].
    // forward path used by the module via `forward_indices`.
    const EMB_BATCH: usize = 2;
    const EMB_SEQ: usize = 16;
    const EMB_VOCAB: usize = 4096;
    const EMB_DIM: usize = 256;

    let indices: [[i32; EMB_SEQ]; EMB_BATCH] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    ];
    let idx_data: Vec<f32> = indices
        .iter()
        .flat_map(|row| row.iter())
        .map(|&v| v as f32)
        .collect();

    let emb_seq = Embedding::<f32, SequentialBackend>::new(EMB_VOCAB, EMB_DIM);
    let emb_moirai = Embedding::<f32, MoiraiBackend>::new(EMB_VOCAB, EMB_DIM);
    let idx_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![EMB_BATCH, EMB_SEQ], &idx_data);
    let idx_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![EMB_BATCH, EMB_SEQ], &idx_data);

    let mut group = c.benchmark_group("Coeus — Embedding lookup forward (2x16, vocab=4096, d=256)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(emb_seq.forward_indices(black_box(&idx_seq))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(emb_moirai.forward_indices(black_box(&idx_moirai))))
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

    // Coeus: Lstm::new(input_size, hidden_size).
    let lstm_seq = Lstm::<f32, SequentialBackend>::new(LSTM_IN, LSTM_H);
    let lstm_moirai = Lstm::<f32, MoiraiBackend>::new(LSTM_IN, LSTM_H);
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

    // Coeus: Gru::new(input_size, hidden_size).
    let gru_seq = CoeusGru::<f32, SequentialBackend>::new(GRU_IN, GRU_H);
    let gru_moirai = CoeusGru::<f32, MoiraiBackend>::new(GRU_IN, GRU_H);
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
    let rnn_seq = Rnn::<f32, SequentialBackend>::new(RNN_IN, RNN_H, RnnNonlinearity::Tanh);
    let rnn_moirai = Rnn::<f32, MoiraiBackend>::new(RNN_IN, RNN_H, RnnNonlinearity::Tanh);
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
    let cell_seq = RNNCell::<f32, SequentialBackend>::new(RNN_IN, RNN_H, RnnNonlinearity::Tanh);
    let cell_moirai = RNNCell::<f32, MoiraiBackend>::new(RNN_IN, RNN_H, RnnNonlinearity::Tanh);
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
        Rnn::<f32, SequentialBackend>::new(RNN_IN, RNN_H, RnnNonlinearity::Tanh),
        Rnn::<f32, SequentialBackend>::new(RNN_IN, RNN_H, RnnNonlinearity::Tanh),
    );
    let bi_moirai = Bidirectional::new(
        Rnn::<f32, MoiraiBackend>::new(RNN_IN, RNN_H, RnnNonlinearity::Tanh),
        Rnn::<f32, MoiraiBackend>::new(RNN_IN, RNN_H, RnnNonlinearity::Tanh),
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

    // Coeus: SwiGlu::new(d_input, d_output, bias=false).
    let sg_seq = SwiGlu::<f32, SequentialBackend>::new(SG_IN, SG_OUT, false);
    let sg_moirai = SwiGlu::<f32, MoiraiBackend>::new(SG_IN, SG_OUT, false);
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

pub(crate) fn bench_embeddingbag_sum(c: &mut Criterion) {
    // EmbeddingBag sum/mean-mode forward: 16 bags × 100 tokens each, vocab=200, dim=64.
    const EB_VOCAB: usize = 200;
    const EB_DIM: usize = 64;
    const EB_BAGS: usize = 16;
    const EB_BAG_SIZE: usize = 100;

    // Build deterministic indices: each bag cycles through vocab.
    let flat_indices: Vec<usize> = (0..(EB_BAGS * EB_BAG_SIZE)).map(|i| i % EB_VOCAB).collect();
    let offsets: Vec<usize> = (0..EB_BAGS).map(|b| b * EB_BAG_SIZE).collect();

    // Coeus EmbeddingBag.
    let eb_seq =
        EmbeddingBag::<f32, SequentialBackend>::new(EB_VOCAB, EB_DIM, EmbeddingBagMode::Sum);
    let eb_moirai =
        EmbeddingBag::<f32, MoiraiBackend>::new(EB_VOCAB, EB_DIM, EmbeddingBagMode::Sum);
    let eb_mean_seq =
        EmbeddingBag::<f32, SequentialBackend>::new(EB_VOCAB, EB_DIM, EmbeddingBagMode::Mean);
    let eb_mean_moirai =
        EmbeddingBag::<f32, MoiraiBackend>::new(EB_VOCAB, EB_DIM, EmbeddingBagMode::Mean);

    let mut group = c.benchmark_group(
        "Coeus — EmbeddingBag reductions (16 bags × 100 tokens, vocab=200 dim=64)",
    );
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                eb_seq.forward_with_offsets(black_box(&flat_indices), Some(black_box(&offsets))),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                eb_moirai.forward_with_offsets(black_box(&flat_indices), Some(black_box(&offsets))),
            )
        })
    });
    group.bench_function("Coeus Sequential mean", |b| {
        b.iter(|| {
            black_box(
                eb_mean_seq
                    .forward_with_offsets(black_box(&flat_indices), Some(black_box(&offsets))),
            )
        })
    });
    group.bench_function("Coeus Moirai mean", |b| {
        b.iter(|| {
            black_box(
                eb_mean_moirai
                    .forward_with_offsets(black_box(&flat_indices), Some(black_box(&offsets))),
            )
        })
    });
    group.finish();
}
