use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

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
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B>;
}

/// Autograd node for a generic unary activation operation.
pub struct UnaryNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default, Op: UnaryAutogradOp<T, B>> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Saved input tensor for backward computation.
    pub a_tensor: Tensor<T, B>,
    /// Saved output tensor for backward computation.
    pub out_tensor: Tensor<T, B>,
    /// Zero-sized phantom to bind the op type parameter.
    pub _phantom: std::marker::PhantomData<Op>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default, Op: UnaryAutogradOp<T, B>> BackwardNode<T, B>
    for UnaryNode<T, B, Op>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        Op::OP_NAME
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let mask = Op::backward(grad_out, &self.a_tensor, &self.out_tensor, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &mask, &backend)?;
        }
        Ok(())
    }
}

/// Generic, monomorphized activation wrapper that builds the autograd node.
#[inline]
pub fn unary_op<
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
    Op: UnaryAutogradOp<T, B> + 'static,
>(
    a: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    let out_tensor = Op::forward(&a.tensor, &backend);
    let saved_out_tensor = out_tensor.clone();
    let requires_grad = crate::grad_mode::should_track_var(a);
    Var::from_tracked_op(out_tensor, requires_grad, &backend, |output_grad| {
        let node: UnaryNode<T, B, Op> = UnaryNode {
            output_grad,
            inputs: vec![a.clone()],
            a_tensor: a.tensor.clone(),
            out_tensor: saved_out_tensor,
            _phantom: std::marker::PhantomData,
        };
        node
    })
}

/// Applies a compiled unary `backward` closure to the saved tensors.
///
/// The closure is routed through this function (rather than invoked directly in
/// the generic `impl`) so that its parameter types are fixed by the `Fn` bound.
/// A bare `closure(args)` leaves them to be inferred from the body, which
/// cannot resolve generic constructors such as `Tensor::full_on(.., T, ..)`
/// before the parameter types are known.
#[inline(always)]
pub(crate) fn unary_backward<T, B, F>(
    backward: F,
    grad_out: &Tensor<T, B>,
    x: &Tensor<T, B>,
    y: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
    F: Fn(&Tensor<T, B>, &Tensor<T, B>, &Tensor<T, B>, &B) -> Tensor<T, B>,
{
    backward(grad_out, x, y, backend)
}

/// Applies an existing unary-gradient kernel, then scales it by `grad_out`.
///
/// This captures the common `elementwise_unary(...Grad) -> mul(grad_out, deriv)`
/// pattern used by many activation backwards without introducing dynamic
/// dispatch or extra allocations beyond the derivative tensor they already
/// materialize today.
#[inline(always)]
pub(crate) fn backward_via_unary_derivative<T, B>(
    grad_out: &Tensor<T, B>,
    source: &Tensor<T, B>,
    backend: &B,
    grad_op: coeus_ops::UnaryOp,
) -> Tensor<T, B>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
{
    let deriv = coeus_ops::elementwise_unary(source, backend, grad_op).expect("elementwise_unary");
    coeus_ops::mul(grad_out, &deriv, backend)
}

/// Returns the zero gradient for non-differentiable unary ops.
#[inline(always)]
pub(crate) fn zero_unary_grad<T, B>(grad_out: &Tensor<T, B>, backend: &B) -> Tensor<T, B>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
{
    Tensor::zeros_on(grad_out.shape(), backend)
}

// ── Declarative unary-op boilerplate ────────────────────────────────────────
//
// Every shape-conforming unary autograd op is the same trio: a zero-sized tag,
// a `UnaryAutogradOp` impl whose `forward` is a same-named `coeus_ops` entry
// point, and a one-line `unary_op::<T, B, Tag>` wrapper. `unary_autograd!`
// emits all three from one invocation so they cannot drift apart.
//
// The default form bounds the scalar by `Float`; an op whose bound differs
// passes it in braces (`{Scalar}`, `{Scalar + FloatOps}`). The fourth argument
// is the `backward` body, invoked as `(grad_out, x, y, backend)` — so its
// parameters must be bound in that order (prefix unused ones with `_`).
//
// Ops that do not fit stay hand-written: `neg`/`abs` (bespoke bounds and
// bodies), the parametric `pow`/`clamp`/`leaky_relu`/`prelu` nodes, `selu`
// (composed forward), and the `ext.rs` parameter-free family (their forward is
// an `elementwise_unary` dispatch, not a same-named `coeus_ops` fn).
macro_rules! unary_autograd {
    // Internal: emit the tag, the `UnaryAutogradOp` impl, and the wrapper.
    (
        @emit [$($bound:tt)+] $(#[$meta:meta])*
        $op:ident, $name:literal, $fwd:ident, $back:expr
    ) => {
        #[doc = concat!("Zero-sized autograd tag for the `", $name, "` unary op.")]
        pub struct $op;

        impl<T: $($bound)+, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for $op {
            const OP_NAME: &'static str = $name;

            #[inline(always)]
            fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
                coeus_ops::$fwd(x, backend)
            }

            #[inline(always)]
            fn backward(
                grad_out: &Tensor<T, B>,
                x: &Tensor<T, B>,
                y: &Tensor<T, B>,
                backend: &B,
            ) -> Tensor<T, B> {
                $crate::ops::activation::unary_backward($back, grad_out, x, y, backend)
            }
        }

        $(#[$meta])*
        #[doc = concat!("Tracked element-wise `", $name, "`.")]
        #[must_use]
        #[inline]
        pub fn $fwd<T: $($bound)+, B: coeus_ops::BackendOps<T> + Default>(
            a: &Var<T, B>,
        ) -> Var<T, B> {
            unary_op::<T, B, $op>(a)
        }
    };

    // Public: default `T: Float` bound.
    ($(#[$meta:meta])* $op:ident, $name:literal, $fwd:ident, $back:expr) => {
        unary_autograd!(@emit [Float] $(#[$meta])* $op, $name, $fwd, $back);
    };

    // Public: explicit scalar bound, e.g. `{Scalar + FloatOps}`.
    ($(#[$meta:meta])* {$($bound:tt)+} $op:ident, $name:literal, $fwd:ident, $back:expr) => {
        unary_autograd!(@emit [$($bound)+] $(#[$meta])* $op, $name, $fwd, $back);
    };
}

// ── Leaf modules ──
/// Extended activation family (Hardtanh, Hardsigmoid, Hardswish, Hardshrink,
/// Softshrink, Softsign, Threshold, Celu). See `ext.rs` for subgradient
/// contracts at kink points.
pub mod ext;
/// GELU activation forward/backward nodes.
pub mod gelu;
/// Mathematical unary ops (abs, floor, round, sign, sqrt, etc.).
pub mod math;
/// ReLU-family activations (ReLU, LeakyReLU, ELU).
pub mod relu;
/// Sigmoid activation forward/backward.
pub mod sigmoid;
/// SiLU-family activations (SiLU, Mish, Softplus).
pub mod silu;
/// Tanh activation forward/backward.
pub mod tanh_act;
/// Trigonometric and exponential ops (sin, cos, exp, log).
pub mod trig;

// ── Re-exports ──
pub use gelu::{gelu, gelu_tanh, GeluOp, GeluTanhOp};
pub use math::{
    abs, ceil, clamp, floor, neg, pow, recip, round, sign, sqrt, trunc, AbsOp, CeilOp, ClampNode,
    FloorOp, NegOp, PowNode, RecipOp, RoundOp, SignOp, SqrtOp, TruncOp,
};
pub use relu::{elu, leaky_relu, prelu, relu, EluOp, ReluOp};
pub use relu::{selu, SeluOp};
pub use sigmoid::{sigmoid, SigmoidOp};
pub use silu::{mish, silu, softplus, MishOp, SiluOp, SoftplusOp};
pub use tanh_act::{tanh, TanhOp};
pub use trig::{
    acos, acosh, asin, asinh, atan, atanh, cos, cosh, erf, erfc, exp, exp2, expm1, lgamma_forward,
    log, log10, log1p, log2, sin, sinh, tan, AcosOp, AcoshOp, AsinOp, AsinhOp, AtanOp, AtanhOp,
    CosOp, CoshOp, ErfOp, ErfcOp, Exp2Op, ExpOp, Expm1Op, Log10Op, Log1pOp, Log2Op, LogOp, SinOp,
    SinhOp, TanOp,
};
// Extended-family re-exports (G-037).
pub use ext::{
    celu, hardshrink, hardsigmoid, hardswish, hardtanh, pack_pairs, softshrink, softsign,
    threshold, HardsigmoidOp, HardswishOp, SoftsignOp,
};




