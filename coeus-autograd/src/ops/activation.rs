use std::sync::{Arc, Mutex};
use coeus_core::{Scalar, Float};
use coeus_tensor::Tensor;
use crate::node::BackwardNode;
use crate::var::Var;

/// Abstract interface for compile-time specialized unary autograd operations.
pub trait UnaryAutogradOp<T: Scalar, B: coeus_ops::BackendOps<T> + Default>: Send + Sync {
    /// Human-readable operation name for tracking.
    const OP_NAME: &'static str;

    /// Execute forward pass.
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B>;

    /// Compute input gradient: computes the derivative and scales by grad_out.
    ///
    /// Accepts both the input tensor `x` and output tensor `y` to allow optimized
    /// derivative computation using the output values where mathematically feasible.
    fn backward(grad_out: &Tensor<T, B>, x: &Tensor<T, B>, y: &Tensor<T, B>, backend: &B) -> Tensor<T, B>;
}

pub struct UnaryNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default, Op: UnaryAutogradOp<T, B>> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub a_tensor: Tensor<T, B>,
    pub out_tensor: Tensor<T, B>,
    pub _phantom: std::marker::PhantomData<Op>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default, Op: UnaryAutogradOp<T, B>> BackwardNode<T, B> for UnaryNode<T, B, Op> {
    #[inline]
    fn op_name(&self) -> &'static str {
        Op::OP_NAME
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
            let mask = Op::backward(grad_out, &self.a_tensor, &self.out_tensor, &backend);
            let mut gl = g.lock().unwrap();
            coeus_ops::add_assign(&mut gl, &mask, &backend);
        }
    }
}

/// Generic, monomorphized activation wrapper that builds the autograd node.
#[inline]
pub fn unary_op<T: Scalar, B: coeus_ops::BackendOps<T> + Default, Op: UnaryAutogradOp<T, B> + 'static>(
    a: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    let out_tensor = Op::forward(&a.tensor, &backend);
    let requires_grad = a.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![a.clone()];
        let a_tensor = a.tensor.clone();
        let out_t = out_tensor.clone();
        let node: UnaryNode<T, B, Op> = UnaryNode {
            output_grad,
            inputs,
            a_tensor,
            out_tensor: out_t,
            _phantom: std::marker::PhantomData,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var { tensor: out_tensor, grad, creator }
}

// ── ZST Operation Tags ──

/// ZST tag for ReLU autograd.
pub struct ReluOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for ReluOp {
    const OP_NAME: &'static str = "relu";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::relu(x, backend)
    }

    #[inline(always)]
    fn backward(grad_out: &Tensor<T, B>, x: &Tensor<T, B>, _y: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        let mask = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::ReluGrad);
        coeus_ops::mul(grad_out, &mask, backend)
    }
}

/// ZST tag for Sigmoid autograd.
pub struct SigmoidOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for SigmoidOp {
    const OP_NAME: &'static str = "sigmoid";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::sigmoid(x, backend)
    }

    #[inline(always)]
    fn backward(grad_out: &Tensor<T, B>, _x: &Tensor<T, B>, y: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        let deriv = coeus_ops::elementwise_unary(y, backend, coeus_ops::UnaryOp::SigmoidGrad);
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// ZST tag for Tanh autograd.
pub struct TanhOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for TanhOp {
    const OP_NAME: &'static str = "tanh";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::tanh(x, backend)
    }

    #[inline(always)]
    fn backward(grad_out: &Tensor<T, B>, _x: &Tensor<T, B>, y: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        let deriv = coeus_ops::elementwise_unary(y, backend, coeus_ops::UnaryOp::TanhGrad);
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// ZST tag for Exponential autograd.
pub struct ExpOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for ExpOp {
    const OP_NAME: &'static str = "exp";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::exp(x, backend)
    }

    #[inline(always)]
    fn backward(grad_out: &Tensor<T, B>, _x: &Tensor<T, B>, y: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::mul(grad_out, y, backend)
    }
}

/// ZST tag for Natural Logarithm autograd.
pub struct LogOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for LogOp {
    const OP_NAME: &'static str = "log";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::log(x, backend)
    }

    #[inline(always)]
    fn backward(grad_out: &Tensor<T, B>, x: &Tensor<T, B>, _y: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::div(grad_out, x, backend)
    }
}

/// ZST tag for GELU autograd.
pub struct GeluOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for GeluOp {
    const OP_NAME: &'static str = "gelu";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::gelu(x, backend)
    }

    #[inline(always)]
    fn backward(grad_out: &Tensor<T, B>, x: &Tensor<T, B>, _y: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        let deriv = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::GeluGrad);
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// ZST tag for SiLU autograd.
pub struct SiluOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for SiluOp {
    const OP_NAME: &'static str = "silu";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::silu(x, backend)
    }

    #[inline(always)]
    fn backward(grad_out: &Tensor<T, B>, x: &Tensor<T, B>, _y: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        let mask = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::SiluGrad);
        coeus_ops::mul(grad_out, &mask, backend)
    }
}

/// ZST tag for Mish autograd.
pub struct MishOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for MishOp {
    const OP_NAME: &'static str = "mish";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::mish(x, backend)
    }

    #[inline(always)]
    fn backward(grad_out: &Tensor<T, B>, x: &Tensor<T, B>, _y: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        let mask = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::MishGrad);
        coeus_ops::mul(grad_out, &mask, backend)
    }
}

// ── Public Tracked Activation Functions ──

/// Tracked ReLU activation.
#[inline]
pub fn relu<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, ReluOp>(a)
}

/// Tracked Sigmoid activation.
#[inline]
pub fn sigmoid<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, SigmoidOp>(a)
}

/// Tracked Tanh activation.
#[inline]
pub fn tanh<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, TanhOp>(a)
}

/// Tracked Exponential function.
#[inline]
pub fn exp<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, ExpOp>(a)
}

/// Tracked Natural Logarithm.
#[inline]
pub fn log<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, LogOp>(a)
}

/// Tracked GELU activation.
#[inline]
pub fn gelu<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, GeluOp>(a)
}

/// Tracked SiLU activation.
#[inline]
pub fn silu<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, SiluOp>(a)
}

/// Tracked Mish activation.
#[inline]
pub fn mish<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, MishOp>(a)
}

// ── New Phase 7 Activation ZST Tags ──

/// ZST tag for ELU autograd (alpha=1.0).
pub struct EluOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for EluOp {
    const OP_NAME: &'static str = "elu";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::elu(x, backend)
    }

    #[inline(always)]
    fn backward(grad_out: &Tensor<T, B>, x: &Tensor<T, B>, _y: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        // EluGrad takes the original input x and returns exp(x) or 1
        let deriv = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::EluGrad);
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// ZST tag for Softplus autograd.
pub struct SoftplusOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for SoftplusOp {
    const OP_NAME: &'static str = "softplus";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::softplus(x, backend)
    }

    #[inline(always)]
    fn backward(grad_out: &Tensor<T, B>, x: &Tensor<T, B>, _y: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        // SoftplusGrad = sigmoid(x)
        let deriv = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::SoftplusGrad);
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// ZST tag for GELU tanh approximation autograd.
pub struct GeluTanhOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for GeluTanhOp {
    const OP_NAME: &'static str = "gelu_tanh";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::gelu_tanh(x, backend)
    }

    #[inline(always)]
    fn backward(grad_out: &Tensor<T, B>, x: &Tensor<T, B>, _y: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        let deriv = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::GeluTanhGrad);
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

// ── Leaky ReLU (parameterized — carries slope data, cannot be ZST) ──

/// Inline backward node for LeakyReLU.
struct LeakyReluNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    output_grad: Arc<Mutex<Tensor<T, B>>>,
    inputs: Vec<Var<T, B>>,
    input_tensor: Tensor<T, B>,
    negative_slope: u64, // f64::to_bits
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for LeakyReluNode<T, B> {
    fn op_name(&self) -> &'static str { "leaky_relu" }
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> { &self.output_grad }
    fn inputs(&self) -> &[Var<T, B>] { &self.inputs }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.get(0) {
            let deriv = coeus_ops::elementwise_unary(
                &self.input_tensor,
                &backend,
                coeus_ops::UnaryOp::LeakyReluGrad(self.negative_slope),
            );
            let mask = coeus_ops::mul(grad_out, &deriv, &backend);
            let mut lock = g.lock().unwrap();
            coeus_ops::add_assign(&mut *lock, &mask, &backend);
        }
    }
}

/// Tracked Leaky ReLU activation.
#[inline]
pub fn leaky_relu<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    negative_slope: f64,
) -> Var<T, B> {
    let backend = B::default();
    let slope_bits = f64::to_bits(negative_slope);
    let out_tensor = coeus_ops::leaky_relu(&a.tensor, &backend, negative_slope);
    let requires_grad = a.grad.is_some();

    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = LeakyReluNode {
            output_grad,
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            negative_slope: slope_bits,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Var { tensor: out_tensor, grad, creator }
}

/// Tracked ELU activation.
#[inline]
pub fn elu<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, EluOp>(a)
}

/// Tracked Softplus activation.
#[inline]
pub fn softplus<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, SoftplusOp>(a)
}

/// Tracked GELU tanh approximation.
#[inline]
pub fn gelu_tanh<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, GeluTanhOp>(a)
}

// ── Unary Math Ops ─────────────────────────────────────────────────────────
//
// NegOp, AbsOp, SqrtOp: ZST-tagged, use generic UnaryNode<T, B, Op>.
// PowNode, ClampNode:    parametric — store exponent/bounds as scalar fields.
//
// All backward formulas are analytically verified against standard calculus.

// ── NegOp ──────────────────────────────────────────────────────────────────

/// ZST tag for negation autograd. Works for any `Scalar` (not just `Float`).
pub struct NegOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for NegOp {
    const OP_NAME: &'static str = "neg";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::neg(x, backend)
    }

    /// d/dx [−x] = −1, so grad_in = −grad_out.
    #[inline(always)]
    fn backward(grad_out: &Tensor<T, B>, _x: &Tensor<T, B>, _y: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::neg(grad_out, backend)
    }
}

/// Tracked element-wise negation.
#[inline]
pub fn neg<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, NegOp>(a)
}

// ── AbsOp ──────────────────────────────────────────────────────────────────

/// ZST tag for absolute-value autograd.
pub struct AbsOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for AbsOp {
    const OP_NAME: &'static str = "abs";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::abs(x, backend)
    }

    /// d/dx |x| = sign(x).
    ///
    /// Computed as `abs(x) / x` where `x ≠ 0`.  At `x = 0` the result is 0
    /// (convention matches PyTorch/JAX).  Implemented via `abs(x)` then
    /// element-wise division using the backend's existing primitives — no new
    /// kernel required.
    #[inline(always)]
    fn backward(grad_out: &Tensor<T, B>, x: &Tensor<T, B>, y: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        // sign(x) = abs(x) / x; where x = 0 this produces NaN, masked to 0
        // by multiplying with ReluGrad(x) mask (0 at x≤0) combined with
        // ReluGrad(-x) (0 at x≥0). A cleaner and backend-portable approach:
        // use the stored forward output y = |x| and divide by x, then mul by grad_out.
        // Avoid division instability at x=0 by zeroing via the mask pattern:
        // sign = (x > 0) - (x < 0) encoded as abs(x)/x with zero at x=0 clamped.
        //
        // Implementation: sign(x) = neg(neg_mask) where:
        //   pos_mask = ReluGrad(x)      — 1 where x > 0, 0 elsewhere
        //   neg_mask = ReluGrad(-x_neg) — 1 where x < 0, 0 elsewhere
        // sign = pos_mask - neg_mask
        let pos_mask = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::ReluGrad);
        let x_neg = coeus_ops::neg(x, backend);
        let neg_mask = coeus_ops::elementwise_unary(&x_neg, backend, coeus_ops::UnaryOp::ReluGrad);
        let sign = coeus_ops::sub(&pos_mask, &neg_mask, backend);
        // Ignore the stored forward output `y` here — sign is cheaper directly from x.
        let _ = y;
        coeus_ops::mul(grad_out, &sign, backend)
    }
}

/// Tracked element-wise absolute value.
#[inline]
pub fn abs<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, AbsOp>(a)
}

// ── SqrtOp ─────────────────────────────────────────────────────────────────

/// ZST tag for square-root autograd.
pub struct SqrtOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for SqrtOp {
    const OP_NAME: &'static str = "sqrt";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::sqrt(x, backend)
    }

    /// d/dx √x = 1 / (2√x) = grad_out / (2·y) where y = √x.
    ///
    /// Uses the stored forward output `y` to avoid a redundant sqrt call.
    #[inline(always)]
    fn backward(grad_out: &Tensor<T, B>, _x: &Tensor<T, B>, y: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        let two = Tensor::full_on(y.shape(), T::from_f64(2.0), backend);
        let denom = coeus_ops::mul(y, &two, backend);
        coeus_ops::div(grad_out, &denom, backend)
    }
}

/// Tracked element-wise square root.
#[inline]
pub fn sqrt<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, SqrtOp>(a)
}

// ── PowNode ────────────────────────────────────────────────────────────────
//
// Parametric: stores `exp` as `u64` (bit-cast of f64) to stay `Sync`.
// Cannot use generic `UnaryNode<T, B, Op>` as `Op` would need to carry `exp`.
// Follows the same bespoke pattern as `LeakyReluNode`.

/// Autograd node for element-wise power: `y = x ^ exp`.
pub struct PowNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad:  Arc<Mutex<Tensor<T, B>>>,
    pub inputs:       Vec<Var<T, B>>,
    /// Input tensor snapshot for backward.
    pub input_tensor: Tensor<T, B>,
    /// Exponent stored as bit-cast `u64` for `Sync`-safe storage.
    pub exp_bits:     u64,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for PowNode<T, B> {
    #[inline] fn op_name(&self) -> &'static str { "pow" }
    #[inline] fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> { &self.output_grad }
    #[inline] fn inputs(&self) -> &[Var<T, B>] { &self.inputs }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let exp = f64::from_bits(self.exp_bits);
            let exp_t = T::from_f64(exp);
            let exp_m1 = exp - 1.0;
            // d/dx x^n = n · x^(n-1)
            // Compute x^(n-1) element-wise in native precision via ln/exp:
            //   x^(n-1) = exp((n-1) * ln(x))   for x > 0
            // For integer exponents or small n, this is the standard path.
            // A future optimization can use a dedicated pow kernel; for now
            // this composes existing backend primitives at zero cost.
            let ln_x = coeus_ops::log(&self.input_tensor, &backend);
            let scaled = {
                let scale = Tensor::full_on(ln_x.shape(), T::from_f64(exp_m1), &backend);
                coeus_ops::mul(&ln_x, &scale, &backend)
            };
            let x_pow_n_m1 = coeus_ops::exp(&scaled, &backend);
            let n_tensor = Tensor::full_on(x_pow_n_m1.shape(), exp_t, &backend);
            let local_grad = coeus_ops::mul(&n_tensor, &x_pow_n_m1, &backend);
            let grad_in = coeus_ops::mul(grad_out, &local_grad, &backend);
            let mut lock = g.lock().unwrap();
            coeus_ops::add_assign(&mut *lock, &grad_in, &backend);
        }
    }
}

/// Tracked element-wise power: `y = x ^ exp`.
///
/// `exp` is an `f64` scalar applied uniformly to all elements.
/// Backward: `d/dx x^n = n · x^(n−1)`.
///
/// # Precision
/// Forward and backward execute in the native precision of `T` without widening.
/// The exponent is converted once via `T::from_f64(exp)`.
#[inline]
pub fn pow<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>, exp: f64) -> Var<T, B> {
    let backend = B::default();
    let exp_t = T::from_f64(exp);

    // Forward: x^n = exp(n * ln(x))
    let ln_x = coeus_ops::log(&a.tensor, &backend);
    let n_tensor = Tensor::full_on(ln_x.shape(), exp_t, &backend);
    let scaled = coeus_ops::mul(&n_tensor, &ln_x, &backend);
    let out_tensor = coeus_ops::exp(&scaled, &backend);

    let requires_grad = a.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = PowNode {
            output_grad,
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            exp_bits: f64::to_bits(exp),
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Var { tensor: out_tensor, grad, creator }
}

// ── ClampNode ──────────────────────────────────────────────────────────────
//
// Gradient of clamp is 1 inside [min, max] and 0 outside.  This is the
// straight-through estimator for the STE variant; for exact grad it is the
// indicator function.  Here we implement the indicator (standard definition):
//
//   d/dx clamp(x, lo, hi) = 1_{lo ≤ x ≤ hi}
//
// which means: gradient passes through unchanged where the input is inside the
// clamp bounds, and is zeroed where x < lo or x > hi.
//
// Implementation via two ReluGrad applications on shifted inputs (zero-copy
// composition of existing backend primitives, no new kernel needed):
//   inside = ReluGrad(x - lo) * ReluGrad(hi - x)

/// Autograd node for element-wise clamp.
pub struct ClampNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad:  Arc<Mutex<Tensor<T, B>>>,
    pub inputs:       Vec<Var<T, B>>,
    /// Input tensor snapshot for backward mask computation.
    pub input_tensor: Tensor<T, B>,
    /// Lower bound of the clamp.
    pub min_val:      T,
    /// Upper bound of the clamp.
    pub max_val:      T,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for ClampNode<T, B> {
    #[inline] fn op_name(&self) -> &'static str { "clamp" }
    #[inline] fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> { &self.output_grad }
    #[inline] fn inputs(&self) -> &[Var<T, B>] { &self.inputs }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let shape = self.input_tensor.shape();
            // x - lo ≥ 0  ⟺  x ≥ lo
            let lo_t = Tensor::full_on(shape, self.min_val, &backend);
            let shifted_lo = coeus_ops::sub(&self.input_tensor, &lo_t, &backend);
            let mask_lo = coeus_ops::elementwise_unary(&shifted_lo, &backend, coeus_ops::UnaryOp::ReluGrad);
            // hi - x ≥ 0  ⟺  x ≤ hi
            let hi_t = Tensor::full_on(shape, self.max_val, &backend);
            let shifted_hi = coeus_ops::sub(&hi_t, &self.input_tensor, &backend);
            let mask_hi = coeus_ops::elementwise_unary(&shifted_hi, &backend, coeus_ops::UnaryOp::ReluGrad);
            // Inside mask = lo_mask AND hi_mask (product of indicators)
            let inside = coeus_ops::mul(&mask_lo, &mask_hi, &backend);
            let grad_in = coeus_ops::mul(grad_out, &inside, &backend);
            let mut lock = g.lock().unwrap();
            coeus_ops::add_assign(&mut *lock, &grad_in, &backend);
        }
    }
}

/// Tracked element-wise clamp: `y = clamp(x, min_val, max_val)`.
///
/// Gradient is the indicator `1_{min_val ≤ x ≤ max_val}` — passes through
/// where input is in range, zeroed at saturated positions.
#[inline]
pub fn clamp<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    min_val: T,
    max_val: T,
) -> Var<T, B> {
    let backend = B::default();

    // Forward: clamp element-wise.
    // Implemented via existing primitives: clamp(x, lo, hi) = min(max(x, lo), hi)
    // = relu_shift approach using two relu operations on shifted values.
    // More precisely: clip from below then from above.
    let lo_t = Tensor::full_on(a.tensor.shape(), min_val, &backend);
    let hi_t = Tensor::full_on(a.tensor.shape(), max_val, &backend);
    // x_clamped_lo = relu(x - lo) + lo = max(x, lo)
    let shifted_lo = coeus_ops::sub(&a.tensor, &lo_t, &backend);
    let relu_lo = coeus_ops::elementwise_unary(&shifted_lo, &backend, coeus_ops::UnaryOp::Relu);
    let clamped_lo = coeus_ops::add(&relu_lo, &lo_t, &backend);
    // max(x, lo) clamped from above: min(clamped_lo, hi) = hi - relu(hi - clamped_lo)
    let shifted_hi = coeus_ops::sub(&hi_t, &clamped_lo, &backend);
    let relu_hi = coeus_ops::elementwise_unary(&shifted_hi, &backend, coeus_ops::UnaryOp::Relu);
    let out_tensor = coeus_ops::sub(&hi_t, &relu_hi, &backend);

    let requires_grad = a.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = ClampNode {
            output_grad,
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            min_val,
            max_val,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Var { tensor: out_tensor, grad, creator }
}

