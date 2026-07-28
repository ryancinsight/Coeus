//! Differential verification for CPU attention backward contiguous dot seams.
//!
//! `dA = dO @ V^T` and each softmax-backward row product are contiguous row
//! dot products, so CPU backward routes those reductions through
//! `Scalar::dot_slice` (`hermes_simd::dot` for native floats). The reference
//! below keeps an independent scalar formulation of the same equations.

use coeus_core::{CpuAddressableStorage, MoiraiBackend, SequentialBackend};
use coeus_ops::{scaled_dot_product_attention, scaled_dot_product_attention_backward, BackendOps};
use coeus_tensor::Tensor;

const BATCH: usize = 1;
const SEQ_Q: usize = 3;
const SEQ_K: usize = 17;
const D_K: usize = 5;
const D_V: usize = 17;

fn idx3(b: usize, i: usize, j: usize, dim1: usize, dim2: usize) -> usize {
    b * dim1 * dim2 + i * dim2 + j
}

fn reference_backward(
    grad_out: &[f32],
    query: &[f32],
    key: &[f32],
    value: &[f32],
    attn_weights: &[f32],
    scale: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut grad_q = vec![0.0; BATCH * SEQ_Q * D_K];
    let mut grad_k = vec![0.0; BATCH * SEQ_K * D_K];
    let mut grad_v = vec![0.0; BATCH * SEQ_K * D_V];

    for b in 0..BATCH {
        for j in 0..SEQ_K {
            for l in 0..D_V {
                let mut acc = 0.0;
                for i in 0..SEQ_Q {
                    acc += attn_weights[idx3(b, i, j, SEQ_Q, SEQ_K)]
                        * grad_out[idx3(b, i, l, SEQ_Q, D_V)];
                }
                grad_v[idx3(b, j, l, SEQ_K, D_V)] += acc;
            }
        }

        let mut d_attn = vec![0.0; SEQ_Q * SEQ_K];
        for i in 0..SEQ_Q {
            for j in 0..SEQ_K {
                let mut acc = 0.0;
                for l in 0..D_V {
                    acc += grad_out[idx3(b, i, l, SEQ_Q, D_V)] * value[idx3(b, j, l, SEQ_K, D_V)];
                }
                d_attn[i * SEQ_K + j] = acc;
            }
        }

        let mut d_scores = vec![0.0; SEQ_Q * SEQ_K];
        for i in 0..SEQ_Q {
            let mut row_sum = 0.0;
            for j in 0..SEQ_K {
                row_sum += attn_weights[idx3(b, i, j, SEQ_Q, SEQ_K)] * d_attn[i * SEQ_K + j];
            }
            for j in 0..SEQ_K {
                d_scores[i * SEQ_K + j] =
                    attn_weights[idx3(b, i, j, SEQ_Q, SEQ_K)] * (d_attn[i * SEQ_K + j] - row_sum);
            }
        }

        for i in 0..SEQ_Q {
            for dk in 0..D_K {
                let mut acc = 0.0;
                for j in 0..SEQ_K {
                    acc += d_scores[i * SEQ_K + j] * key[idx3(b, j, dk, SEQ_K, D_K)];
                }
                grad_q[idx3(b, i, dk, SEQ_Q, D_K)] += acc * scale;
            }
        }

        for j in 0..SEQ_K {
            for dk in 0..D_K {
                let mut acc = 0.0;
                for i in 0..SEQ_Q {
                    acc += d_scores[i * SEQ_K + j] * query[idx3(b, i, dk, SEQ_Q, D_K)];
                }
                grad_k[idx3(b, j, dk, SEQ_K, D_K)] += acc * scale;
            }
        }
    }

    (grad_q, grad_k, grad_v)
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        // The implementation and reference differ only by reassociation in two
        // row dot products of length <= 17. The factor covers those reductions
        // and their subsequent native-precision products/additions.
        let tol = 256.0 * f32::EPSILON * (1.0 + want.abs());
        assert!(
            (got - want).abs() <= tol,
            "{label}[{index}]: got {got}, expected {want}, tol {tol}",
        );
    }
}

fn check_backend<B>(backend: &B)
where
    B: BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let query_data: Vec<f32> = (0..BATCH * SEQ_Q * D_K)
        .map(|i| ((i as f32 + 1.0) * 0.03125).sin())
        .collect();
    let key_data: Vec<f32> = (0..BATCH * SEQ_K * D_K)
        .map(|i| ((i as f32 + 2.0) * 0.046875).cos())
        .collect();
    let value_data: Vec<f32> = (0..BATCH * SEQ_K * D_V)
        .map(|i| (i as f32 % 11.0 - 5.0) * 0.0625)
        .collect();
    let grad_out_data: Vec<f32> = (0..BATCH * SEQ_Q * D_V)
        .map(|i| (i as f32 % 7.0 - 3.0) * 0.125)
        .collect();

    let query = Tensor::<f32, B>::from_slice_on([BATCH, SEQ_Q, D_K], &query_data, backend).expect("construct tensor");
    let key = Tensor::<f32, B>::from_slice_on([BATCH, SEQ_K, D_K], &key_data, backend).expect("construct tensor");
    let value = Tensor::<f32, B>::from_slice_on([BATCH, SEQ_K, D_V], &value_data, backend).expect("construct tensor");
    let grad_out = Tensor::<f32, B>::from_slice_on([BATCH, SEQ_Q, D_V], &grad_out_data, backend).expect("construct tensor");
    let scale = 0.25;

    let (_, attn_weights) =
        scaled_dot_product_attention(&query, &key, &value, None, false, scale, backend)
            .expect("run attention forward");

    let mut grad_q = Tensor::<f32, B>::zeros_on([BATCH, SEQ_Q, D_K], backend).expect("construct tensor");
    let mut grad_k = Tensor::<f32, B>::zeros_on([BATCH, SEQ_K, D_K], backend).expect("construct tensor");
    let mut grad_v = Tensor::<f32, B>::zeros_on([BATCH, SEQ_K, D_V], backend).expect("construct tensor");

    scaled_dot_product_attention_backward(
        &grad_out,
        &query,
        &key,
        &value,
        &attn_weights,
        scale,
        Some(&mut grad_q),
        Some(&mut grad_k),
        Some(&mut grad_v),
        backend,
    ).expect("run attention backward");

    let aw = attn_weights.as_slice();
    let (expected_q, expected_k, expected_v) = reference_backward(
        &grad_out_data,
        &query_data,
        &key_data,
        &value_data,
        aw,
        scale,
    );

    assert_close("grad_q", grad_q.as_slice(), &expected_q);
    assert_close("grad_k", grad_k.as_slice(), &expected_k);
    assert_close("grad_v", grad_v.as_slice(), &expected_v);
}

#[test]
fn sequential_attention_backward_matches_reference() {
    check_backend(&SequentialBackend);
}

#[test]
fn moirai_attention_backward_matches_reference() {
    check_backend(&MoiraiBackend);
}
