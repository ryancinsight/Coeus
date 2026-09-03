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
