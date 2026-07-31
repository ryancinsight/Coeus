//! attention benchmarks.

use super::*;

pub(crate) fn bench_mha_forward(c: &mut Criterion) {
    // Self-attention forward on a realistic transformer block:
    // [batch=8, seq=64, d_model=256] with 8 heads (d_head=32).
    const B: usize = 8;
    const SEQ: usize = 64;
    const D: usize = 256;
    const H: usize = 8;

    let mha_seq = MultiHeadAttention::<f32, SequentialBackend, H, NullMask>::new(D, true);
    let mha_moirai = MultiHeadAttention::<f32, MoiraiBackend, H, NullMask>::new(D, true);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![B, SEQ, D]),
        false,
    );
    let x_moirai = Var::new(Tensor::<f32, MoiraiBackend>::ones(vec![B, SEQ, D]), false);

    let mut group = c.benchmark_group("Coeus — MHA self-attn forward (8x64x256, 8 heads)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                mha_seq
                    .forward(black_box(&x_seq))
                    .expect("valid multi-head attention benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                mha_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid multi-head attention benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_mha_cross_attention_forward(c: &mut Criterion) {
    // Cross-attention forward with deliberately unequal query/memory lengths
    // ([batch=8, q_seq=32, d_model=256] attending over memory seq=64, 8 heads):
    // the resulting non-square q_seq x memory_seq score matrix is the path
    // self-attention cannot exercise, so the two rows are not redundant.
    const B: usize = 8;
    const QUERY_SEQ: usize = 32;
    const MEMORY_SEQ: usize = 64;
    const D: usize = 256;
    const H: usize = 8;
    let query_data: Vec<f32> = (0..(B * QUERY_SEQ * D))
        .map(|index| (index as f32 * 0.0013).sin())
        .collect();
    let memory_data: Vec<f32> = (0..(B * MEMORY_SEQ * D))
        .map(|index| (index as f32 * 0.0007).cos())
        .collect();
    let mha_seq = MultiHeadAttention::<f32, SequentialBackend, H, NullMask>::new(D, true);
    let mha_moirai = MultiHeadAttention::<f32, MoiraiBackend, H, NullMask>::new(D, true);
    let query_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![B, QUERY_SEQ, D], &query_data),
        false,
    );
    let memory_seq = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![B, MEMORY_SEQ, D], &memory_data),
        false,
    );
    let query_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![B, QUERY_SEQ, D], &query_data),
        false,
    );
    let memory_moirai = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![B, MEMORY_SEQ, D], &memory_data),
        false,
    );

    let mut group = c.benchmark_group(
        "Coeus — MHA cross-attn forward (query 8x32x256, memory 8x64x256, 8 heads)",
    );
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(mha_seq.forward_cross(
                black_box(&query_seq),
                black_box(&memory_seq),
                black_box(&memory_seq),
                None,
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(mha_moirai.forward_cross(
                black_box(&query_moirai),
                black_box(&memory_moirai),
                black_box(&memory_moirai),
                None,
            ))
        })
    });
    group.finish();
}

pub(crate) fn bench_transformer_encoder_forward(c: &mut Criterion) {
    // One Pre-LN transformer encoder layer (self-attn + FFN + 2 LayerNorms +
    // a single-layer encoder with norm-first and dropout disabled to match.
    const B: usize = 8;
    const SEQ: usize = 64;
    const D: usize = 256;
    const D_FF: usize = 1024;
    const H: usize = 8;

    let enc_seq = TransformerEncoderLayer::<f32, SequentialBackend, H, NullMask>::new(D, D_FF, 0.0);
    let enc_moirai = TransformerEncoderLayer::<f32, MoiraiBackend, H, NullMask>::new(D, D_FF, 0.0);
    let x_seq = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![B, SEQ, D]),
        false,
    );
    let x_moirai = Var::new(Tensor::<f32, MoiraiBackend>::ones(vec![B, SEQ, D]), false);

    let mut group =
        c.benchmark_group("Coeus — Transformer encoder layer forward (8x64x256, d_ff=1024)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(
                enc_seq
                    .forward(black_box(&x_seq))
                    .expect("valid transformer encoder benchmark input"),
            )
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(
                enc_moirai
                    .forward(black_box(&x_moirai))
                    .expect("valid transformer encoder benchmark input"),
            )
        })
    });
    group.finish();
}

pub(crate) fn bench_sdp_attention_forward(c: &mut Criterion) {
    // Scaled dot-product attention: [4, 8, 64] Q×K×V (4 heads, seq=8, d_k=64).
    // The output alloc uses alloc_on: every position written by the kernel.
    const SA_B: usize = 4;
    const SA_S: usize = 8;
    const SA_D: usize = 64;
    let q_data: Vec<f32> = (0..(SA_B * SA_S * SA_D))
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let v_data: Vec<f32> = (0..(SA_B * SA_S * SA_D))
        .map(|i| (i as f32 * 0.001).cos())
        .collect();

    let backend_seq = SequentialBackend;
    let q_seq =
        coeus_tensor::Tensor::<f32, SequentialBackend>::from_slice(vec![SA_B, SA_S, SA_D], &q_data);
    let k_seq = q_seq.clone();
    let v_seq =
        coeus_tensor::Tensor::<f32, SequentialBackend>::from_slice(vec![SA_B, SA_S, SA_D], &v_data);

    let backend_moirai = MoiraiBackend;
    let q_moirai =
        coeus_tensor::Tensor::<f32, MoiraiBackend>::from_slice(vec![SA_B, SA_S, SA_D], &q_data);
    let k_moirai = q_moirai.clone();
    let v_moirai =
        coeus_tensor::Tensor::<f32, MoiraiBackend>::from_slice(vec![SA_B, SA_S, SA_D], &v_data);

    let scale = 1.0f32 / (SA_D as f32).sqrt();

    let mut group = c.benchmark_group("Coeus — sdp_attention forward (4x8x64)");
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_ops::scaled_dot_product_attention(
                black_box(&q_seq),
                black_box(&k_seq),
                black_box(&v_seq),
                None,
                false,
                scale,
                &backend_seq,
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::scaled_dot_product_attention(
                black_box(&q_moirai),
                black_box(&k_moirai),
                black_box(&v_moirai),
                None,
                false,
                scale,
                &backend_moirai,
            ))
        })
    });
    group.finish();
}
