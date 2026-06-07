use std::sync::{Arc, Mutex};
use coeus_core::{Scalar, Float, Storage};
use coeus_tensor::Tensor;
use crate::node::BackwardNode;
use crate::var::Var;

// ── Cross-Entropy Loss ──

pub struct CrossEntropyLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub targets: Vec<usize>,
    /// Softmax probabilities stored as Vec<T> — no f64 widening.
    pub probs: Vec<T>,
    pub n: usize,
    pub c: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for CrossEntropyLossNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "cross_entropy_loss"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.get(0) {
            let temp_grad;
            let grad_out_cont = if grad_out.is_contiguous() && grad_out.layout().offset() == 0 {
                grad_out
            } else {
                temp_grad = grad_out.to_contiguous_on(&backend);
                &temp_grad
            };
            let mut host_grad = [T::zero()];
            backend.copy_to_host(grad_out_cont.storage(), &mut host_grad);
            // Scale in T precision — no widening to f64
            let n_t = T::from_f64(self.n as f64);
            let grad_out_val = host_grad[0];
            let scale = grad_out_val / n_t;

            let mut d_logits = vec![T::zero(); self.n * self.c];
            for i in 0..self.n {
                let offset = i * self.c;
                let target_idx = self.targets[i];
                for j in 0..self.c {
                    let p = self.probs[offset + j];
                    let indicator = if j == target_idx { T::one() } else { T::zero() };
                    d_logits[offset + j] = (p - indicator) * scale;
                }
            }
            let grad_tensor = Tensor::from_slice_on([self.n, self.c], &d_logits, &backend);
            let mut gl = g.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &grad_tensor, &backend);
        }
    }
}

/// Tracked Cross-Entropy Loss.
/// Called from coeus-nn/src/loss.rs after host-side log-sum-exp computation.
/// `probs` must be Vec<T>, computed in T precision.
pub fn cross_entropy_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    logits: &Var<T, B>,
    targets: Vec<usize>,
    out_tensor: Tensor<T, B>,
    probs: Vec<T>,
    n: usize,
    c: usize,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = logits.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![logits.clone()];

        let node = CrossEntropyLossNode {
            output_grad,
            inputs,
            targets,
            probs,
            n,
            c,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var { tensor: out_tensor, grad, creator }
}

// ── Binary Cross-Entropy Loss ──

pub struct BinaryCrossEntropyNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    /// Clamped prediction values in [eps, 1-eps], stored as Vec<T>.
    pub probs: Vec<T>,
    /// Target values (0.0 or 1.0) stored as Vec<T>.
    pub targets: Vec<T>,
    pub n: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for BinaryCrossEntropyNode<T, B> {
    fn op_name(&self) -> &'static str { "binary_cross_entropy" }
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> { &self.output_grad }
    fn inputs(&self) -> &[Var<T, B>] { &self.inputs }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.get(0) {
            let mut host_grad = [T::zero()];
            let temp_grad;
            let grad_cont = if grad_out.is_contiguous() && grad_out.layout().offset() == 0 {
                grad_out
            } else {
                temp_grad = grad_out.to_contiguous_on(&backend);
                &temp_grad
            };
            backend.copy_to_host(grad_cont.storage(), &mut host_grad);
            let g_out = host_grad[0];
            let n_t = T::from_f64(self.n as f64);
            let scale = g_out / n_t;

            let mut d_pred = vec![T::zero(); self.n];
            for i in 0..self.n {
                let p = self.probs[i];
                let t = self.targets[i];
                let one = T::one();
                // d/dp = -(t/p - (1-t)/(1-p)) / n
                // Use T::zero() - x idiom since unary - may not be in scope for T
                d_pred[i] = (T::zero() - (t / p) + (one - t) / (one - p)) * scale;
            }
            let grad_tensor = Tensor::from_slice_on([self.n], &d_pred, &backend);
            let mut gl = g.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &grad_tensor, &backend);
        }
    }
}

/// Tracked Binary Cross-Entropy Loss.
/// pred: [N] probabilities (will be clamped internally), target: [N] float targets (0.0 or 1.0).
/// eps: numerical stability clamp (e.g., T::from_f64(1e-7)).
pub fn binary_cross_entropy<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
    eps: T,
) -> Var<T, B> {
    let backend = B::default();
    let shape = pred.tensor.shape();
    let n = shape[0];

    // Host-side computation for forward + clamp
    let pred_cont;
    let pred_raw = if pred.tensor.is_contiguous() && pred.tensor.layout().offset() == 0 {
        &pred.tensor
    } else {
        pred_cont = pred.tensor.to_contiguous_on(&backend);
        &pred_cont
    };
    let target_cont;
    let target_raw = if target.tensor.is_contiguous() && target.tensor.layout().offset() == 0 {
        &target.tensor
    } else {
        target_cont = target.tensor.to_contiguous_on(&backend);
        &target_cont
    };

    let pred_host: std::borrow::Cow<[T]> = if let Some(s) = pred_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(pred_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };
    let target_host: std::borrow::Cow<[T]> = if let Some(s) = target_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(target_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };

    let one_minus_eps = T::one() - eps;
    let mut probs = vec![T::zero(); n];
    let mut targets_t = vec![T::zero(); n];
    let mut loss_val = T::zero();
    let n_t = T::from_f64(n as f64);

    for i in 0..n {
        let p_raw = pred_host[i];
        // Clamp to [eps, 1-eps]
        let p = if p_raw < eps { eps } else if p_raw > one_minus_eps { one_minus_eps } else { p_raw };
        let t = target_host[i];
        probs[i] = p;
        targets_t[i] = t;
        // -(t * log(p) + (1-t) * log(1-p)) using T::zero()-x for negation
        loss_val = loss_val + (T::zero() - (t * p.log_op() + (T::one() - t) * (T::one() - p).log_op()));
    }
    loss_val = loss_val / n_t;

    let out_tensor = Tensor::from_slice_on([1], &[loss_val], &backend);
    let requires_grad = pred.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = BinaryCrossEntropyNode {
            output_grad,
            inputs: vec![pred.clone()],
            probs,
            targets: targets_t,
            n,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Var { tensor: out_tensor, grad, creator }
}

// ── NLL Loss ──

pub struct NllLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub targets: Vec<usize>,
    pub n: usize,
    pub c: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for NllLossNode<T, B> {
    fn op_name(&self) -> &'static str { "nll_loss" }
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> { &self.output_grad }
    fn inputs(&self) -> &[Var<T, B>] { &self.inputs }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.get(0) {
            let mut host_grad = [T::zero()];
            let temp_grad;
            let grad_cont = if grad_out.is_contiguous() && grad_out.layout().offset() == 0 {
                grad_out
            } else {
                temp_grad = grad_out.to_contiguous_on(&backend);
                &temp_grad
            };
            backend.copy_to_host(grad_cont.storage(), &mut host_grad);
            let g_out = host_grad[0];
            let n_t = T::from_f64(self.n as f64);
            // Use T::zero() - x idiom for negation
            let neg_scale = T::zero() - (g_out / n_t);

            let mut d_log = vec![T::zero(); self.n * self.c];
            for i in 0..self.n {
                d_log[i * self.c + self.targets[i]] = neg_scale;
            }
            let grad_tensor = Tensor::from_slice_on([self.n, self.c], &d_log, &backend);
            let mut gl = g.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &grad_tensor, &backend);
        }
    }
}

/// Tracked Negative Log-Likelihood Loss.
/// log_probs: [N, C] (already log-probabilities), targets: [N] class indices.
pub fn nll_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    log_probs: &Var<T, B>,
    targets: &[usize],
) -> Var<T, B> {
    let backend = B::default();
    let shape = log_probs.tensor.shape();
    let n = shape[0];
    let c = shape[1];
    assert_eq!(targets.len(), n);

    let cont;
    let log_raw = if log_probs.tensor.is_contiguous() && log_probs.tensor.layout().offset() == 0 {
        &log_probs.tensor
    } else {
        cont = log_probs.tensor.to_contiguous_on(&backend);
        &cont
    };

    let host: std::borrow::Cow<[T]> = if let Some(s) = log_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n * c])
    } else {
        let mut v = vec![T::zero(); n * c];
        backend.copy_to_host(log_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };

    let mut loss_val = T::zero();
    for i in 0..n {
        // T::zero() - x for negation
        loss_val = loss_val + (T::zero() - host[i * c + targets[i]]);
    }
    loss_val = loss_val / T::from_f64(n as f64);

    let out_tensor = Tensor::from_slice_on([1], &[loss_val], &backend);
    let requires_grad = log_probs.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = NllLossNode {
            output_grad,
            inputs: vec![log_probs.clone()],
            targets: targets.to_vec(),
            n,
            c,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Var { tensor: out_tensor, grad, creator }
}

// ── Huber Loss ──

pub struct HuberLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    /// Element-wise differences pred[i] - target[i], stored for backward.
    pub diffs: Vec<T>,
    pub delta: T,
    pub n: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for HuberLossNode<T, B> {
    fn op_name(&self) -> &'static str { "huber_loss" }
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> { &self.output_grad }
    fn inputs(&self) -> &[Var<T, B>] { &self.inputs }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.get(0) {
            let mut host_grad = [T::zero()];
            let temp_grad;
            let grad_cont = if grad_out.is_contiguous() && grad_out.layout().offset() == 0 {
                grad_out
            } else {
                temp_grad = grad_out.to_contiguous_on(&backend);
                &temp_grad
            };
            backend.copy_to_host(grad_cont.storage(), &mut host_grad);
            let g_out = host_grad[0];
            let n_t = T::from_f64(self.n as f64);
            let scale = g_out / n_t;
            let delta = self.delta;

            let mut d_pred = vec![T::zero(); self.n];
            for i in 0..self.n {
                let diff = self.diffs[i];
                // Huber grad: diff/delta clamped to [-1, 1]
                let raw = diff / delta;
                // Use T::zero() - T::one() for -1.0 since unary - not guaranteed on T
                let neg_one = T::zero() - T::one();
                let clamped = if raw > T::one() { T::one() } else if raw < neg_one { neg_one } else { raw };
                d_pred[i] = clamped * scale;
            }
            let grad_tensor = Tensor::from_slice_on([self.n], &d_pred, &backend);
            let mut gl = g.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &grad_tensor, &backend);
        }
    }
}

/// Tracked Huber (Smooth L1) Loss.
/// pred: [N], target: [N], delta: threshold.
pub fn huber_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
    delta: T,
) -> Var<T, B> {
    let backend = B::default();
    let n = pred.tensor.shape()[0];

    let p_cont;
    let p_raw = if pred.tensor.is_contiguous() && pred.tensor.layout().offset() == 0 {
        &pred.tensor
    } else {
        p_cont = pred.tensor.to_contiguous_on(&backend);
        &p_cont
    };
    let t_cont;
    let t_raw = if target.tensor.is_contiguous() && target.tensor.layout().offset() == 0 {
        &target.tensor
    } else {
        t_cont = target.tensor.to_contiguous_on(&backend);
        &t_cont
    };

    let p_host: std::borrow::Cow<[T]> = if let Some(s) = p_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(p_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };
    let t_host: std::borrow::Cow<[T]> = if let Some(s) = t_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(t_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };

    let half = T::from_f64(0.5);
    let mut diffs = vec![T::zero(); n];
    let mut loss_val = T::zero();
    for i in 0..n {
        let diff = p_host[i] - t_host[i];
        diffs[i] = diff;
        // abs_diff using T::zero() - diff for negation
        let abs_diff = if diff < T::zero() { T::zero() - diff } else { diff };
        let elem = if abs_diff <= delta {
            half * diff * diff / delta
        } else {
            abs_diff - half * delta
        };
        loss_val = loss_val + elem;
    }
    loss_val = loss_val / T::from_f64(n as f64);

    let out_tensor = Tensor::from_slice_on([1], &[loss_val], &backend);
    let requires_grad = pred.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = HuberLossNode {
            output_grad,
            inputs: vec![pred.clone()],
            diffs,
            delta,
            n,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Var { tensor: out_tensor, grad, creator }
}

// ── Cosine Embedding Loss ──

pub struct CosineEmbeddingLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub x1_host: Vec<T>,
    pub x2_host: Vec<T>,
    pub y: Vec<T>,
    pub margin: T,
    pub n: usize,
    pub d: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for CosineEmbeddingLossNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str { "cosine_embedding_loss" }
    #[inline]
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> { &self.output_grad }
    #[inline]
    fn inputs(&self) -> &[Var<T, B>] { &self.inputs }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        let need_g1 = input_grads.get(0).and_then(|g| g.as_ref()).is_some();
        let need_g2 = input_grads.get(1).and_then(|g| g.as_ref()).is_some();
        if !need_g1 && !need_g2 { return; }

        let mut host_grad = [T::zero()];
        let temp_grad;
        let grad_cont = if grad_out.is_contiguous() && grad_out.layout().offset() == 0 {
            grad_out
        } else {
            temp_grad = grad_out.to_contiguous_on(&backend);
            &temp_grad
        };
        backend.copy_to_host(grad_cont.storage(), &mut host_grad);
        let g_out = host_grad[0];
        let n_t = T::from_f64(self.n as f64);
        let scale = g_out / n_t;
        let eps = T::from_f64(1e-8);

        let mut dg1 = vec![T::zero(); self.n * self.d];
        let mut dg2 = vec![T::zero(); self.n * self.d];

        for i in 0..self.n {
            let offset = i * self.d;
            let mut dot = T::zero();
            let mut norm1_sq = T::zero();
            let mut norm2_sq = T::zero();
            for j in 0..self.d {
                let val1 = self.x1_host[offset + j];
                let val2 = self.x2_host[offset + j];
                dot = dot + val1 * val2;
                norm1_sq = norm1_sq + val1 * val1;
                norm2_sq = norm2_sq + val2 * val2;
            }
            let norm1 = norm1_sq.sqrt();
            let norm2 = norm2_sq.sqrt();
            let den = if norm1 * norm2 > eps { norm1 * norm2 } else { eps };
            let cos = dot / den;

            let y_val = self.y[i];
            let target_is_one = y_val == T::one();
            let w_i = if target_is_one {
                T::zero() - T::one()
            } else {
                if cos > self.margin { T::one() } else { T::zero() }
            };

            if w_i != T::zero() {
                let n1_sq_safe = if norm1_sq > eps { norm1_sq } else { eps };
                let n2_sq_safe = if norm2_sq > eps { norm2_sq } else { eps };

                for j in 0..self.d {
                    let val1 = self.x1_host[offset + j];
                    let val2 = self.x2_host[offset + j];
                    let g1_val = w_i * scale * (val2 - (dot / n1_sq_safe) * val1) / den;
                    dg1[offset + j] = g1_val;

                    let g2_val = w_i * scale * (val1 - (dot / n2_sq_safe) * val2) / den;
                    dg2[offset + j] = g2_val;
                }
            }
        }

        if let Some(Some(ref g)) = input_grads.get(0) {
            let grad_tensor = Tensor::from_slice_on([self.n, self.d], &dg1, &backend);
            let mut gl = g.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &grad_tensor, &backend);
        }
        if let Some(Some(ref g)) = input_grads.get(1) {
            let grad_tensor = Tensor::from_slice_on([self.n, self.d], &dg2, &backend);
            let mut gl = g.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &grad_tensor, &backend);
        }
    }
}

/// Tracked Cosine Embedding Loss.
/// x1: [N, D], x2: [N, D], y: [N] (elements 1 or -1).
pub fn cosine_embedding_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x1: &Var<T, B>,
    x2: &Var<T, B>,
    y: &[T],
    margin: T,
) -> Var<T, B> {
    let backend = B::default();
    let n = x1.tensor.shape()[0];
    let d = x1.tensor.shape()[1];
    assert_eq!(x2.tensor.shape(), x1.tensor.shape(), "cosine_embedding_loss: x1 and x2 must have same shape");
    assert_eq!(y.len(), n, "cosine_embedding_loss: y must have length equal to batch size");

    let x1_cont;
    let x1_raw = if x1.tensor.is_contiguous() && x1.tensor.layout().offset() == 0 {
        &x1.tensor
    } else {
        x1_cont = x1.tensor.to_contiguous_on(&backend);
        &x1_cont
    };
    let x2_cont;
    let x2_raw = if x2.tensor.is_contiguous() && x2.tensor.layout().offset() == 0 {
        &x2.tensor
    } else {
        x2_cont = x2.tensor.to_contiguous_on(&backend);
        &x2_cont
    };

    let numel = n * d;
    let x1_host: Vec<T> = if let Some(s) = x1_raw.storage().try_as_slice() {
        s[..numel].to_vec()
    } else {
        let mut v = vec![T::zero(); numel];
        backend.copy_to_host(x1_raw.storage(), &mut v);
        v
    };
    let x2_host: Vec<T> = if let Some(s) = x2_raw.storage().try_as_slice() {
        s[..numel].to_vec()
    } else {
        let mut v = vec![T::zero(); numel];
        backend.copy_to_host(x2_raw.storage(), &mut v);
        v
    };

    let eps = T::from_f64(1e-8);
    let mut loss_val = T::zero();
    for i in 0..n {
        let offset = i * d;
        let mut dot = T::zero();
        let mut norm1_sq = T::zero();
        let mut norm2_sq = T::zero();
        for j in 0..d {
            let val1 = x1_host[offset + j];
            let val2 = x2_host[offset + j];
            dot = dot + val1 * val2;
            norm1_sq = norm1_sq + val1 * val1;
            norm2_sq = norm2_sq + val2 * val2;
        }
        let norm1 = norm1_sq.sqrt();
        let norm2 = norm2_sq.sqrt();
        let den = if norm1 * norm2 > eps { norm1 * norm2 } else { eps };
        let cos = dot / den;
        let y_val = y[i];
        let target_is_one = y_val == T::one();
        let item_loss = if target_is_one {
            T::one() - cos
        } else {
            let diff = cos - margin;
            if diff > T::zero() { diff } else { T::zero() }
        };
        loss_val = loss_val + item_loss;
    }
    loss_val = loss_val / T::from_f64(n as f64);

    let out_tensor = Tensor::from_slice_on([1], &[loss_val], &backend);
    let requires_grad = x1.grad.is_some() || x2.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = CosineEmbeddingLossNode {
            output_grad,
            inputs: vec![x1.clone(), x2.clone()],
            x1_host,
            x2_host,
            y: y.to_vec(),
            margin,
            n,
            d,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var { tensor: out_tensor, grad, creator }
}
