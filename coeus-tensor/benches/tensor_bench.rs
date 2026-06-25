use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_tensor::Tensor;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use leto::Array;

// ── Burn NdArray (dev/bench only) ─────────────────────────────────────────
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Tensor as BurnTensor, TensorData};
type BurnB = NdArray<f32>;

fn bench_elementwise_add(c: &mut Criterion) {
    let size = 1024;
    let shape = vec![size, size];

    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();

    let a_seq = Tensor::<f32, SequentialBackend>::ones(shape.clone());
    let b_seq = Tensor::<f32, SequentialBackend>::ones(shape.clone());

    let a_moirai = Tensor::<f32, MoiraiBackend>::ones(shape.clone());
    let b_moirai = Tensor::<f32, MoiraiBackend>::ones(shape.clone());

    let mut group = c.benchmark_group("Elementwise Add (1024x1024)");

    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_ops::add(
                black_box(&a_seq),
                black_box(&b_seq),
                black_box(&seq_backend),
            ));
        })
    });

    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::add(
                black_box(&a_moirai),
                black_box(&b_moirai),
                black_box(&moirai_backend),
            ));
        })
    });

    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    // 256x256 matmul to keep bench times reasonable
    let m = 256;
    let k = 256;
    let n = 256;

    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();

    let a_seq = Tensor::<f32, SequentialBackend>::ones(vec![m, k]);
    let b_seq = Tensor::<f32, SequentialBackend>::ones(vec![k, n]);

    let a_moirai = Tensor::<f32, MoiraiBackend>::ones(vec![m, k]);
    let b_moirai = Tensor::<f32, MoiraiBackend>::ones(vec![k, n]);

    // Leto and direct layouts
    let a_leto = Array::from_shape_vec([m, k], vec![1.0f32; m * k]).unwrap();
    let b_leto = Array::from_shape_vec([k, n], vec![1.0f32; k * n]).unwrap();

    let coeus_layout_a = coeus_core::Layout::new(vec![m, k].into());
    let coeus_layout_b = coeus_core::Layout::new(vec![k, n].into());
    let coeus_layout_out = coeus_core::Layout::new(vec![m, n].into());
    let coeus_a = vec![1.0f32; m * k];
    let coeus_b = vec![1.0f32; k * n];

    let mut group = c.benchmark_group("Matrix Multiplication (256x256)");

    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_ops::matmul(
                black_box(&a_seq),
                black_box(&b_seq),
                black_box(&seq_backend),
            ));
        })
    });

    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::matmul(
                black_box(&a_moirai),
                black_box(&b_moirai),
                black_box(&moirai_backend),
            ));
        })
    });

    group.bench_function("Leto direct", |b| {
        b.iter_batched(
            || Array::zeros([m, n]),
            |mut out| {
                leto_ops::matmul(
                    black_box(&a_leto.view()),
                    black_box(&b_leto.view()),
                    &mut out.view_mut(),
                )
                .unwrap();
                black_box(out);
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("Coeus-Leto dispatch", |b| {
        b.iter_batched(
            || vec![0.0f32; m * n],
            |mut out| {
                coeus_leto::matmul_into(
                    black_box(&coeus_layout_a),
                    black_box(&coeus_a),
                    black_box(&coeus_layout_b),
                    black_box(&coeus_b),
                    black_box(&coeus_layout_out),
                    black_box(&mut out),
                )
                .unwrap();
                black_box(out);
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.finish();
}

// ── Burn NdArray comparison benchmarks ────────────────────────────────────

fn bench_burn_elementwise_add(c: &mut Criterion) {
    let size = 1024;
    let device = NdArrayDevice::default();
    let data: Vec<f32> = vec![1.0f32; size * size];

    let a_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [size, size]), &device);
    let b_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [size, size]), &device);

    let mut group = c.benchmark_group("Burn vs Coeus — Elementwise Add (1024x1024)");

    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(a_burn.clone() + b_burn.clone()))
    });

    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let a_seq = Tensor::<f32, SequentialBackend>::ones(vec![size, size]);
    let b_seq = Tensor::<f32, SequentialBackend>::ones(vec![size, size]);
    let a_moirai = Tensor::<f32, MoiraiBackend>::ones(vec![size, size]);
    let b_moirai = Tensor::<f32, MoiraiBackend>::ones(vec![size, size]);

    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_ops::add(
                black_box(&a_seq),
                black_box(&b_seq),
                black_box(&seq_backend),
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::add(
                black_box(&a_moirai),
                black_box(&b_moirai),
                black_box(&moirai_backend),
            ))
        })
    });

    group.finish();
}

fn bench_burn_matmul(c: &mut Criterion) {
    let (m, k, n) = (256, 256, 256);
    let device = NdArrayDevice::default();
    let a_data: Vec<f32> = (0..m * k).map(|x| x as f32 * 0.001).collect();
    let b_data: Vec<f32> = (0..k * n).map(|x| x as f32 * 0.001).collect();

    let a_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(a_data.clone(), [m, k]), &device);
    let b_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(b_data.clone(), [k, n]), &device);

    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![m, k], &a_data);
    let b_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![k, n], &b_data);
    let a_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![m, k], &a_data);
    let b_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![k, n], &b_data);

    let mut group = c.benchmark_group("Burn vs Coeus — Matmul (256x256)");

    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(a_burn.clone().matmul(b_burn.clone())))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_ops::matmul(
                black_box(&a_seq),
                black_box(&b_seq),
                black_box(&seq_backend),
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::matmul(
                black_box(&a_moirai),
                black_box(&b_moirai),
                black_box(&moirai_backend),
            ))
        })
    });

    group.finish();
}

fn bench_burn_relu(c: &mut Criterion) {
    let size = 1024;
    let device = NdArrayDevice::default();
    let data: Vec<f32> = (0..size * size).map(|x| x as f32 * 0.01 - 5.0).collect();

    let x_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [size, size]), &device);
    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let x_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![size, size], &data);
    let x_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![size, size], &data);

    let mut group = c.benchmark_group("Burn vs Coeus — ReLU (1024x1024)");

    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn::tensor::activation::relu(x_burn.clone())))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_ops::relu(black_box(&x_seq), black_box(&seq_backend))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::relu(
                black_box(&x_moirai),
                black_box(&moirai_backend),
            ))
        })
    });

    group.finish();
}

fn bench_burn_gelu(c: &mut Criterion) {
    let size = 1024;
    let device = NdArrayDevice::default();
    let data: Vec<f32> = (0..size * size).map(|x| x as f32 * 0.01 - 5.0).collect();

    let x_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [size, size]), &device);
    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let x_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![size, size], &data);
    let x_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![size, size], &data);

    let mut group = c.benchmark_group("Burn vs Coeus — GELU (1024x1024)");

    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn::tensor::activation::gelu(x_burn.clone())))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(coeus_ops::gelu(black_box(&x_seq), black_box(&seq_backend))))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::gelu(
                black_box(&x_moirai),
                black_box(&moirai_backend),
            ))
        })
    });

    group.finish();
}

fn bench_burn_sum(c: &mut Criterion) {
    let size = 1024;
    let device = NdArrayDevice::default();
    let data: Vec<f32> = vec![1.0f32; size * size];

    let x_burn: BurnTensor<BurnB, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [size, size]), &device);
    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let x_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![size, size], &data);
    let x_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![size, size], &data);

    let mut group = c.benchmark_group("Burn vs Coeus — Sum axis=1 (1024x1024)");

    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(x_burn.clone().sum_dim(1)))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_ops::sum_axis(
                black_box(&x_seq),
                1,
                black_box(&seq_backend),
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::sum_axis(
                black_box(&x_moirai),
                1,
                black_box(&moirai_backend),
            ))
        })
    });

    group.finish();
}

fn bench_burn_conv2d(c: &mut Criterion) {
    use burn::tensor::module::conv2d as burn_conv2d;
    use burn::tensor::ops::ConvOptions;
    use burn::tensor::Tensor as BT;
    use burn::tensor::TensorData;
    use coeus_ops::BackendOps;

    // [batch=1, c_in=4, h=16, w=16] × [c_out=8, c_in=4, kh=3, kw=3]
    const BATCH: usize = 1;
    const C_IN: usize = 4;
    const H: usize = 16;
    const W: usize = 16;
    const C_OUT: usize = 8;
    const KH: usize = 3;
    const H_OUT: usize = H - KH + 1;

    let device = NdArrayDevice::default();
    let x_data: Vec<f32> = (0..BATCH * C_IN * H * W).map(|i| i as f32 * 0.01).collect();
    let w_data: Vec<f32> = (0..C_OUT * C_IN * KH * KH)
        .map(|i| i as f32 * 0.001)
        .collect();

    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let x_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, C_IN, H, W], &x_data);
    let w_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![C_OUT, C_IN, KH, KH], &w_data);
    let x_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, C_IN, H, W], &x_data);
    let w_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![C_OUT, C_IN, KH, KH], &w_data);
    let x_b: BT<BurnB, 4> = BT::from_data(
        TensorData::new(x_data.clone(), [BATCH, C_IN, H, W]),
        &device,
    );
    let w_b: BT<BurnB, 4> = BT::from_data(
        TensorData::new(w_data.clone(), [C_OUT, C_IN, KH, KH]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — Conv2d (1×4×16×16, k=3)");

    group.bench_function("Burn NdArray", |b| {
        b.iter(|| {
            black_box(burn_conv2d(
                x_b.clone(),
                w_b.clone(),
                None,
                ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
            ))
        })
    });
    group.bench_function("Coeus Sequential", |ben| {
        let mut out = Tensor::<f32, SequentialBackend>::zeros(vec![BATCH, C_OUT, H_OUT, H_OUT]);
        ben.iter(|| {
            let out_l = out.layout().clone();
            seq_backend.conv2d(
                x_seq.storage(),
                x_seq.layout(),
                w_seq.storage(),
                w_seq.layout(),
                None,
                1, // stride
                0, // padding
                1, // dilation
                out.storage_mut(),
                &out_l,
            );
            black_box(());
        })
    });
    group.bench_function("Coeus Moirai", |ben| {
        let mut out = Tensor::<f32, MoiraiBackend>::zeros(vec![BATCH, C_OUT, H_OUT, H_OUT]);
        ben.iter(|| {
            let out_l = out.layout().clone();
            moirai_backend.conv2d(
                x_moirai.storage(),
                x_moirai.layout(),
                w_moirai.storage(),
                w_moirai.layout(),
                None,
                1, // stride
                0, // padding
                1, // dilation
                out.storage_mut(),
                &out_l,
            );
            black_box(());
        })
    });

    group.finish();
}

fn bench_burn_conv_transpose2d(c: &mut Criterion) {
    use burn::tensor::module::conv_transpose2d as burn_conv_transpose2d;
    use burn::tensor::ops::ConvTransposeOptions;
    use burn::tensor::Tensor as BT;
    use burn::tensor::TensorData;

    // input [batch=1, c_in=4, h=16, w=16] × weight [c_in=4, c_out=8, kh=3, kw=3]
    // (transposed convention: c_in first, matching Burn). stride 2 (upsampling).
    const BATCH: usize = 1;
    const C_IN: usize = 4;
    const H: usize = 16;
    const W: usize = 16;
    const C_OUT: usize = 8;
    const K: usize = 3;
    const STRIDE: usize = 2;
    const PAD: usize = 1;
    const OUT_PAD: usize = 1;
    const DIL: usize = 1;

    let device = NdArrayDevice::default();
    let x_data: Vec<f32> = (0..BATCH * C_IN * H * W).map(|i| i as f32 * 0.01).collect();
    let w_data: Vec<f32> = (0..C_IN * C_OUT * K * K)
        .map(|i| i as f32 * 0.001)
        .collect();

    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let x_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, C_IN, H, W], &x_data);
    let w_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![C_IN, C_OUT, K, K], &w_data);
    let x_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, C_IN, H, W], &x_data);
    let w_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![C_IN, C_OUT, K, K], &w_data);
    let x_b: BT<BurnB, 4> = BT::from_data(
        TensorData::new(x_data.clone(), [BATCH, C_IN, H, W]),
        &device,
    );
    let w_b: BT<BurnB, 4> = BT::from_data(
        TensorData::new(w_data.clone(), [C_IN, C_OUT, K, K]),
        &device,
    );

    let mut group = c.benchmark_group("Burn vs Coeus — ConvTranspose2d (1×4×16×16, k=3, s=2)");

    group.bench_function("Burn NdArray", |b| {
        b.iter(|| {
            black_box(burn_conv_transpose2d(
                x_b.clone(),
                w_b.clone(),
                None,
                ConvTransposeOptions::new(
                    [STRIDE, STRIDE],
                    [PAD, PAD],
                    [OUT_PAD, OUT_PAD],
                    [DIL, DIL],
                    1,
                ),
            ))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(coeus_ops::conv_transpose2d(
                black_box(&x_seq),
                black_box(&w_seq),
                None,
                STRIDE,
                PAD,
                OUT_PAD,
                DIL,
                black_box(&seq_backend),
            ));
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(coeus_ops::conv_transpose2d(
                black_box(&x_moirai),
                black_box(&w_moirai),
                None,
                STRIDE,
                PAD,
                OUT_PAD,
                DIL,
                black_box(&moirai_backend),
            ));
        })
    });

    group.finish();
}

fn bench_burn_max_pool2d(c: &mut Criterion) {
    use burn::tensor::module::max_pool2d as burn_max_pool2d;
    use burn::tensor::Tensor as BT;
    use burn::tensor::TensorData;
    use coeus_ops::BackendOps;

    // [batch=1, c=8, h=32, w=32], kernel=2, stride=2 -> [1, 8, 16, 16].
    const BATCH: usize = 1;
    const C: usize = 8;
    const H: usize = 32;
    const W: usize = 32;
    const K: usize = 2;
    const STRIDE: usize = 2;
    const H_OUT: usize = (H - K) / STRIDE + 1;

    let device = NdArrayDevice::default();
    let x_data: Vec<f32> = (0..BATCH * C * H * W)
        .map(|i| (i as f32 * 0.05).sin())
        .collect();

    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let x_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, C, H, W], &x_data);
    let x_moirai = Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, C, H, W], &x_data);
    let x_b: BT<BurnB, 4> =
        BT::from_data(TensorData::new(x_data.clone(), [BATCH, C, H, W]), &device);

    let mut group = c.benchmark_group("Burn vs Coeus — MaxPool2d (1×8×32×32, k=2, s=2)");

    group.bench_function("Burn NdArray", |b| {
        b.iter(|| {
            black_box(burn_max_pool2d(
                x_b.clone(),
                [K, K],
                [STRIDE, STRIDE],
                [0, 0],
                [1, 1],
            ))
        })
    });
    group.bench_function("Coeus Sequential", |ben| {
        let mut out = Tensor::<f32, SequentialBackend>::zeros(vec![BATCH, C, H_OUT, H_OUT]);
        ben.iter(|| {
            let out_l = out.layout().clone();
            seq_backend.max_pool2d(
                x_seq.storage(),
                x_seq.layout(),
                K,
                STRIDE,
                0,
                1,
                out.storage_mut(),
                &out_l,
            );
            black_box(());
        })
    });
    group.bench_function("Coeus Moirai", |ben| {
        let mut out = Tensor::<f32, MoiraiBackend>::zeros(vec![BATCH, C, H_OUT, H_OUT]);
        ben.iter(|| {
            let out_l = out.layout().clone();
            moirai_backend.max_pool2d(
                x_moirai.storage(),
                x_moirai.layout(),
                K,
                STRIDE,
                0,
                1,
                out.storage_mut(),
                &out_l,
            );
            black_box(());
        })
    });

    group.finish();
}

fn bench_burn_softmax(c: &mut Criterion) {
    use burn::tensor::activation::softmax as burn_softmax;
    use burn::tensor::Tensor as BT;
    use burn::tensor::TensorData;
    use coeus_autograd::Var;
    use coeus_nn::softmax;

    // [rows=256, cols=1024], softmax over the last dim. Inputs use a
    // requires_grad=false Var so the forward path builds no backward node —
    // a fair forward-only comparison against Burn's NdArray (non-autodiff).
    const ROWS: usize = 256;
    const COLS: usize = 1024;
    let device = NdArrayDevice::default();
    let data: Vec<f32> = (0..ROWS * COLS).map(|i| (i as f32 * 0.001).sin()).collect();

    let xv = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![ROWS, COLS], &data),
        false,
    );
    let xv_m = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![ROWS, COLS], &data),
        false,
    );
    let xb: BT<BurnB, 2> = BT::from_data(TensorData::new(data.clone(), [ROWS, COLS]), &device);

    let mut group = c.benchmark_group("Burn vs Coeus — Softmax (256×1024, dim=-1)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(burn_softmax(xb.clone(), 1)))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(softmax(black_box(&xv), 1)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(softmax(black_box(&xv_m), 1)))
    });
    group.finish();
}

fn bench_burn_attention(c: &mut Criterion) {
    use burn::tensor::activation::softmax as burn_softmax;
    use burn::tensor::Tensor as BT;
    use burn::tensor::TensorData;
    use coeus_ops::scaled_dot_product_attention;

    // Scaled dot-product attention computed identically on both sides:
    //   scores = (Q·Kᵀ)·scale -> softmax(-1) -> ·V.
    // Burn expresses it with batched matmul + softmax; Coeus uses its fused SDP.
    // q/k/v: [batch*heads, seq, dim].
    const B: usize = 8;
    const SEQ: usize = 64;
    const D: usize = 32;
    let scale = (D as f32).powf(-0.5);
    let device = NdArrayDevice::default();
    let q_data: Vec<f32> = (0..B * SEQ * D)
        .map(|i| ((i as f32 + 1.0) * 0.013).sin())
        .collect();
    let k_data: Vec<f32> = (0..B * SEQ * D)
        .map(|i| ((i as f32 + 3.0) * 0.017).cos())
        .collect();
    let v_data: Vec<f32> = (0..B * SEQ * D)
        .map(|i| ((i as f32 + 5.0) * 0.011).sin())
        .collect();

    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();
    let q_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![B, SEQ, D], &q_data);
    let k_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![B, SEQ, D], &k_data);
    let v_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![B, SEQ, D], &v_data);
    let q_m = Tensor::<f32, MoiraiBackend>::from_slice(vec![B, SEQ, D], &q_data);
    let k_m = Tensor::<f32, MoiraiBackend>::from_slice(vec![B, SEQ, D], &k_data);
    let v_m = Tensor::<f32, MoiraiBackend>::from_slice(vec![B, SEQ, D], &v_data);
    let qb: BT<BurnB, 3> = BT::from_data(TensorData::new(q_data.clone(), [B, SEQ, D]), &device);
    let kb: BT<BurnB, 3> = BT::from_data(TensorData::new(k_data.clone(), [B, SEQ, D]), &device);
    let vb: BT<BurnB, 3> = BT::from_data(TensorData::new(v_data.clone(), [B, SEQ, D]), &device);

    let mut group = c.benchmark_group("Burn vs Coeus — SDP Attention (8×64×32)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| {
            let scores = qb
                .clone()
                .matmul(kb.clone().swap_dims(1, 2))
                .mul_scalar(scale);
            let attn = burn_softmax(scores, 2);
            black_box(attn.matmul(vb.clone()))
        })
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| {
            black_box(scaled_dot_product_attention(
                black_box(&q_seq),
                black_box(&k_seq),
                black_box(&v_seq),
                None,
                false,
                scale,
                black_box(&seq_backend),
            ))
        })
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| {
            black_box(scaled_dot_product_attention(
                black_box(&q_m),
                black_box(&k_m),
                black_box(&v_m),
                None,
                false,
                scale,
                black_box(&moirai_backend),
            ))
        })
    });
    group.finish();
}

fn bench_burn_layernorm(c: &mut Criterion) {
    use burn::nn::{LayerNorm as BurnLN, LayerNormConfig};
    use burn::tensor::Tensor as BT;
    use burn::tensor::TensorData;
    use coeus_autograd::Var;
    use coeus_nn::{LayerNorm, Module};

    const BATCH: usize = 4;
    const SEQ: usize = 64;
    const FEAT: usize = 128;
    let device = NdArrayDevice::default();
    let data: Vec<f32> = (0..BATCH * SEQ * FEAT)
        .map(|i| (i as f32 * 0.001) % 3.0 - 1.5)
        .collect();

    let xv = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![BATCH, SEQ, FEAT], &data),
        false,
    );
    let xv_m = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice(vec![BATCH, SEQ, FEAT], &data),
        false,
    );
    let ln_seq = LayerNorm::<f32, SequentialBackend>::new(FEAT, 1e-5);
    let ln_moirai = LayerNorm::<f32, MoiraiBackend>::new(FEAT, 1e-5);
    let ln_burn: BurnLN<BurnB> = LayerNormConfig::new(FEAT).init(&device);
    let xb: BT<BurnB, 3> =
        BT::from_data(TensorData::new(data.clone(), [BATCH, SEQ, FEAT]), &device);

    let mut group = c.benchmark_group("Burn vs Coeus — LayerNorm (4×64×128)");
    group.bench_function("Burn NdArray", |b| {
        b.iter(|| black_box(ln_burn.forward(xb.clone())))
    });
    group.bench_function("Coeus Sequential", |b| {
        b.iter(|| black_box(ln_seq.forward(&xv)))
    });
    group.bench_function("Coeus Moirai", |b| {
        b.iter(|| black_box(ln_moirai.forward(&xv_m)))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_elementwise_add,
    bench_matmul,
    bench_burn_elementwise_add,
    bench_burn_matmul,
    bench_burn_relu,
    bench_burn_gelu,
    bench_burn_sum,
    bench_burn_conv2d,
    bench_burn_conv_transpose2d,
    bench_burn_max_pool2d,
    bench_burn_softmax,
    bench_burn_attention,
    bench_burn_layernorm,
);
criterion_main!(benches);
