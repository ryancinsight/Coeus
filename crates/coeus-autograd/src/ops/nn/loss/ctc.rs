// ── CTC (Connectionist Temporal Classification) Loss ──────────────────────────
//
// PyTorch `F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
//                      blank=0, reduction='mean', zero_infinity=False)`.
//
// # Semantics
// - `log_probs`: `[T, N, C]` log-probability tensors (output of `log_softmax`)
// - `targets`: flat target sequence `[sum(target_lengths)]` OR `[N, S]`
// - `input_lengths`: `[N]` — valid frames per sample (<= T)
// - `target_lengths`: `[N]` — number of target labels per sample
// - `blank`: index of the blank label (default 0)
//
// # Algorithm (log-space DP)
// Extended target `l'` of length `2*S+1` interleaves blank tokens.
// Forward variable α(t, s) = log P(prefix t frames → collapsed sequence 0..s).
// Backward variable β(t, s) = log P(suffix T-t frames → collapsed sequence s..end).
// CTC negative log likelihood = -log[ α(T, 2S) + α(T, 2S-1) ]
//
// # Gradient
// dL/d_log_probs(t, k) = -(sum_s I[l'[s]==k] * α(t,s) * β(t,s) / exp(α(T,end)))
//                         / exp(log_probs(t,k))
// (see Graves 2006, eq. 16)

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

const NEG_INF: f64 = f64::NEG_INFINITY;

/// Autograd node for CTC loss.
pub struct CtcLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Gradient accumulator for the output loss scalar.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Log-probability input `[T, N, C]`.
    pub inputs: Vec<Var<T, B>>,
    /// Per-sample log α table: `[N][T*(2S+1)]` (ragged in S dimension).
    pub log_alpha: Vec<Vec<f64>>,
    /// Per-sample log β table: `[N][T*(2S+1)]`.
    pub log_beta: Vec<Vec<f64>>,
    /// Per-sample extended targets (blank-interleaved): `[N][2*S+1]`.
    pub ext_targets: Vec<Vec<usize>>,
    /// Per-sample valid frame counts.
    pub input_lengths: Vec<usize>,
    /// Per-sample target lengths.
    pub target_lengths: Vec<usize>,
    /// Log-probability host copy: flat `[T * N * C]`.
    pub log_probs_host: Vec<f64>,
    /// Blank label index.
    pub blank: usize,
    /// Number of time steps T.
    pub t_steps: usize,
    /// Batch size N.
    pub batch: usize,
    /// Number of classes C.
    pub num_classes: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for CtcLossNode<T, B> {
    fn op_name(&self) -> &'static str {
        "ctc_loss"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        let Some(Some(ref g)) = input_grads.get(0) else {
            return Ok(());
        };

        let t = self.t_steps;
        let n = self.batch;
        let c = self.num_classes;

        // Read upstream scalar gradient.
        let mut host_grad = [T::zero()];
        let temp;
        let g_cont = if grad_out.is_contiguous() && grad_out.layout().offset() == 0 {
            grad_out
        } else {
            temp = grad_out.to_contiguous_on(&backend)?;
            &temp
        };
        backend.copy_to_host(g_cont.storage(), &mut host_grad)?;
        let g_upstream = <T as Scalar>::to_f64(host_grad[0]);

        // Compute gradient for log_probs [T, N, C].
        let mut dx = vec![0.0f64; t * n * c];

        for ni in 0..n {
            let t_n = self.input_lengths[ni];
            let s_n = self.target_lengths[ni];
            let ls = 2 * s_n + 1;
            let ext = &self.ext_targets[ni];
            let alpha = &self.log_alpha[ni];
            let beta = &self.log_beta[ni];

            if t_n == 0 || s_n == 0 {
                continue;
            }

            // Compute log P(y|x) for this sample = log_sum_exp(alpha[T-1, 2S], alpha[T-1, 2S-1])
            let a_end1 = alpha[(t_n - 1) * ls + (ls - 1)];
            let a_end2 = if ls >= 2 {
                alpha[(t_n - 1) * ls + (ls - 2)]
            } else {
                NEG_INF
            };
            let log_p = log_sum_exp2(a_end1, a_end2);

            for ti in 0..t_n {
                for si in 0..ls {
                    let k = ext[si];
                    let a = alpha[ti * ls + si];
                    let b = beta[ti * ls + si];
                    if a == NEG_INF || b == NEG_INF {
                        continue;
                    }
                    let ab = a + b - log_p; // log(alpha * beta / P)
                    // Accumulate into dx[(ti, ni, k)]
                    let idx = ti * n * c + ni * c + k;
                    dx[idx] = log_sum_exp2_f64(dx[idx].ln(), ab).exp();
                }
            }

            // dL/d_log_probs(t,n,k) = -(accumulated / exp(log_probs(t,n,k))) * g_upstream / N
            for ti in 0..t_n {
                for ki in 0..c {
                    let idx = ti * n * c + ni * c + ki;
                    let lp = self.log_probs_host[ti * n * c + ni * c + ki];
                    // grad = -(1 / (N · L_n)) · (sum_s / p(t,k)) · g_upstream.
                    // The 1/L_n matches the forward reduction='mean', which
                    // normalizes each sample's NLL by its target length before
                    // the batch mean; sum_s is already in linear space.
                    dx[idx] = (-dx[idx] / lp.exp()) * g_upstream / (n as f64 * s_n as f64);
                }
            }
        }

        let dx_t: Vec<T> = dx.iter().map(|&v| T::from_f64(v)).collect();
        let grad_tensor = Tensor::from_slice_on([t, n, c], &dx_t, &backend)?;
        let gl = g.write();
        coeus_ops::add_assign(gl, &grad_tensor, &backend)?;

        Ok(())
    }
}

/// Numerically stable log(exp(a) + exp(b)).
#[inline]
fn log_sum_exp2(a: f64, b: f64) -> f64 {
    if a == NEG_INF {
        return b;
    }
    if b == NEG_INF {
        return a;
    }
    let mx = a.max(b);
    mx + ((a - mx).exp() + (b - mx).exp()).ln()
}

/// log_sum_exp2 operating on raw f64 values (not log-space inputs).
/// Here both values are treated as regular f64, with b in log-space but a is linear.
/// Actually just a convenience alias.
#[inline]
fn log_sum_exp2_f64(a: f64, b: f64) -> f64 {
    // Both are in log-space for accumulation.
    if a == f64::NEG_INFINITY || a.is_nan() {
        return b.exp();
    }
    // a is the previous accumulated sum (linear), b is a new log-space term.
    // We accumulate in linear: result = a + exp(b)
    a.exp() + b.exp()
}

/// CTC forward DP: returns (log_alpha, log_beta, loss) for one sample.
///
/// `log_probs_flat`: row-major `[T * C]` log-probability slice for this sample.
/// `ext`: extended target of length `2*S+1` (blank-interleaved).
/// `t_valid`: number of valid frames.
fn ctc_forward_one(
    log_probs_flat: &[f64], // [T_valid * C]
    ext: &[usize],          // [2*S+1]
    t_valid: usize,
    num_classes: usize,
) -> (Vec<f64>, Vec<f64>, f64) {
    let ls = ext.len();
    let t = t_valid;

    // ── Forward variables α ─────────────────────────────────────
    let mut alpha = vec![NEG_INF; t * ls];

    // Init t=0.
    if ls >= 1 {
        alpha[0] = log_probs_flat[ext[0]]; // blank
    }
    if ls >= 2 {
        alpha[1] = log_probs_flat[ext[1]]; // first label
    }

    for ti in 1..t {
        let base_lp = ti * num_classes;
        let prev = (ti - 1) * ls;
        let cur = ti * ls;

        for si in 0..ls {
            let k = ext[si];
            let lp = log_probs_flat[base_lp + k];

            // From same position.
            let mut a = alpha[prev + si];

            // From one position back.
            if si > 0 {
                a = log_sum_exp2(a, alpha[prev + si - 1]);
            }

            // From two positions back (skip blank when prev == blank-equivalent).
            if si > 1 && ext[si] != ext[si - 2] {
                a = log_sum_exp2(a, alpha[prev + si - 2]);
            }

            alpha[cur + si] = if a == NEG_INF { NEG_INF } else { a + lp };
        }
    }

    // CTC loss for this sample.
    let a_end1 = alpha[(t - 1) * ls + ls - 1];
    let a_end2 = if ls >= 2 {
        alpha[(t - 1) * ls + ls - 2]
    } else {
        NEG_INF
    };
    let log_p = log_sum_exp2(a_end1, a_end2);
    let loss = if log_p == NEG_INF { 0.0 } else { -log_p };

    // ── Backward variables β ─────────────────────────────────────
    let mut beta = vec![NEG_INF; t * ls];

    // Init t = T-1.
    if ls >= 1 {
        beta[(t - 1) * ls + ls - 1] = 0.0; // log(1)
    }
    if ls >= 2 {
        beta[(t - 1) * ls + ls - 2] = 0.0;
    }

    for ti in (0..t - 1).rev() {
        let next_base_lp = (ti + 1) * num_classes;
        let next = (ti + 1) * ls;
        let cur = ti * ls;

        for si in 0..ls {
            // From same position.
            let mut b = if beta[next + si] == NEG_INF {
                NEG_INF
            } else {
                beta[next + si] + log_probs_flat[next_base_lp + ext[si]]
            };

            // From one forward.
            if si + 1 < ls {
                let v = if beta[next + si + 1] == NEG_INF {
                    NEG_INF
                } else {
                    beta[next + si + 1] + log_probs_flat[next_base_lp + ext[si + 1]]
                };
                b = log_sum_exp2(b, v);
            }

            // From two forward (skip blank).
            if si + 2 < ls && ext[si] != ext[si + 2] {
                let v = if beta[next + si + 2] == NEG_INF {
                    NEG_INF
                } else {
                    beta[next + si + 2] + log_probs_flat[next_base_lp + ext[si + 2]]
                };
                b = log_sum_exp2(b, v);
            }

            beta[cur + si] = b;
        }
    }

    (alpha, beta, loss)
}

/// Tracked CTC loss.
///
/// # Arguments
/// - `log_probs`: `[T, N, C]` — log-probabilities from `log_softmax`.
/// - `targets`: flat target indices `[sum(target_lengths)]`.
/// - `input_lengths`: valid frame count per sample `[N]`.
/// - `target_lengths`: target sequence length per sample `[N]`.
/// - `blank`: blank class index (default 0).
///
/// Returns the mean CTC loss over the batch as a scalar `Var`.
pub fn ctc_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    log_probs: &Var<T, B>,
    targets: &[usize],
    input_lengths: &[usize],
    target_lengths: &[usize],
    blank: usize,
) -> Result<Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T>,
{
    let shape = log_probs.tensor.shape();
    assert_eq!(shape.len(), 3, "ctc_loss: log_probs must be [T, N, C]");
    let t_steps = shape[0];
    let batch = shape[1];
    let num_classes = shape[2];
    assert_eq!(
        input_lengths.len(),
        batch,
        "ctc_loss: input_lengths must have length N"
    );
    assert_eq!(
        target_lengths.len(),
        batch,
        "ctc_loss: target_lengths must have length N"
    );

    let backend = B::default();

    // Read log_probs to host.
    let lp_cont;
    let lp_raw = if log_probs.tensor.is_contiguous() && log_probs.tensor.layout().offset() == 0 {
        &log_probs.tensor
    } else {
        lp_cont = log_probs.tensor.to_contiguous_on(&backend)?;
        &lp_cont
    };
    let total = t_steps * batch * num_classes;
    let lp_host: Vec<f64> = {
        let mut v = vec![T::zero(); total];
        backend.copy_to_host(lp_raw.storage(), &mut v)?;
        v.iter().map(|&x| <T as Scalar>::to_f64(x)).collect()
    };

    // Per-sample DP.
    let mut log_alphas: Vec<Vec<f64>> = Vec::with_capacity(batch);
    let mut log_betas: Vec<Vec<f64>> = Vec::with_capacity(batch);
    let mut ext_targets_all: Vec<Vec<usize>> = Vec::with_capacity(batch);
    let mut total_loss = 0.0f64;
    let mut target_offset = 0usize;

    for ni in 0..batch {
        let t_n = input_lengths[ni].min(t_steps);
        let s_n = target_lengths[ni];
        let sample_targets = &targets[target_offset..target_offset + s_n];
        target_offset += s_n;

        // Build extended target: blank, label_0, blank, label_1, ..., blank
        let ls = 2 * s_n + 1;
        let mut ext = Vec::with_capacity(ls);
        for i in 0..s_n {
            ext.push(blank);
            ext.push(sample_targets[i]);
        }
        ext.push(blank);

        // Extract per-sample log-probs as [T_n * C].
        let mut sample_lp = vec![0.0f64; t_n * num_classes];
        for ti in 0..t_n {
            for ki in 0..num_classes {
                sample_lp[ti * num_classes + ki] =
                    lp_host[ti * batch * num_classes + ni * num_classes + ki];
            }
        }

        let (alpha, beta, loss) = if t_n > 0 && s_n > 0 {
            ctc_forward_one(&sample_lp, &ext, t_n, num_classes)
        } else {
            (vec![NEG_INF; ls.max(1)], vec![NEG_INF; ls.max(1)], 0.0)
        };

        // reduction='mean' (torch default): each sample's negative log-
        // likelihood is divided by its target length before the batch mean.
        total_loss += if s_n > 0 { loss / s_n as f64 } else { 0.0 };
        log_alphas.push(alpha);
        log_betas.push(beta);
        ext_targets_all.push(ext);
    }

    let mean_loss = total_loss / batch as f64;
    let loss_val = T::from_f64(mean_loss);
    let out_tensor = Tensor::from_slice_on([1], &[loss_val], &backend)?;

    let requires_grad = crate::grad_mode::should_track_var(log_probs);
    if !requires_grad {
        return Var::new(out_tensor, false);
    }

    let output_grad = Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend)?));
    let grad = Some(output_grad.clone());

    let node = CtcLossNode {
        output_grad,
        inputs: vec![log_probs.clone()],
        log_alpha: log_alphas,
        log_beta: log_betas,
        ext_targets: ext_targets_all,
        input_lengths: input_lengths.to_vec(),
        target_lengths: target_lengths.to_vec(),
        log_probs_host: lp_host,
        blank,
        t_steps,
        batch,
        num_classes,
    };
    let creator = Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>);

    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}
