use std::sync::{Arc, Mutex};
use coeus_core::{Scalar, Shape};
use coeus_tensor::Tensor;
use crate::var::Var;
use crate::backward::reduce_broadcast;
use crate::node::BackwardNode;

/// Abstract interface for compile-time specialized binary autograd operations.
pub trait BinaryAutogradOp<T: Scalar, B: coeus_ops::BackendOps<T> + Default>: Send + Sync {
    /// Human-readable operation name for tracking.
    const OP_NAME: &'static str;

    /// Execute forward pass.
    fn forward(a: &Tensor<T, B>, b: &Tensor<T, B>, backend: &B) -> Tensor<T, B>;

    /// Compute input gradients backward.
    fn backward(
        grad_out: &Tensor<T, B>,
        a: &Tensor<T, B>,
        b: &Tensor<T, B>,
        a_shape: &Shape,
        b_shape: &Shape,
        input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>],
        backend: &B,
    );
}

pub struct BinaryNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default, Op: BinaryAutogradOp<T, B>> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub a_tensor: Tensor<T, B>,
    pub b_tensor: Tensor<T, B>,
    pub a_shape: Shape,
    pub b_shape: Shape,
    pub _phantom: std::marker::PhantomData<Op>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default, Op: BinaryAutogradOp<T, B>> BackwardNode<T, B> for BinaryNode<T, B, Op> {
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
        Op::backward(grad_out, &self.a_tensor, &self.b_tensor, &self.a_shape, &self.b_shape, input_grads, &backend);
    }
}

/// Generic, monomorphized binary wrapper that builds the autograd node.
#[inline]
pub fn binary_op<T: Scalar, B: coeus_ops::BackendOps<T> + Default, Op: BinaryAutogradOp<T, B> + 'static>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    let out_tensor = Op::forward(&a.tensor, &b.tensor, &backend);
    let requires_grad = a.grad.is_some() || b.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![a.clone(), b.clone()];
        let a_shape: Shape = a.tensor.shape_cloned();
        let b_shape: Shape = b.tensor.shape_cloned();
        let a_tensor = a.tensor.clone();
        let b_tensor = b.tensor.clone();
        let node: BinaryNode<T, B, Op> = BinaryNode {
            output_grad,
            inputs,
            a_tensor,
            b_tensor,
            a_shape,
            b_shape,
            _phantom: std::marker::PhantomData,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var { tensor: out_tensor, grad, creator }
}

/// Abstract interface for compile-time specialized reduction autograd operations.
pub trait ReductionAutogradOp<T: Scalar, B: coeus_ops::BackendOps<T> + Default>: Send + Sync {
    /// Human-readable operation name for tracking.
    const OP_NAME: &'static str;

    /// Execute forward pass.
    fn forward(a: &Tensor<T, B>, param: Option<usize>, backend: &B) -> Tensor<T, B>;

    /// Return optional scaling tensor for backward propagation.
    fn scaler(a: &Tensor<T, B>, param: Option<usize>, backend: &B) -> Option<Tensor<T, B>>;
}

pub struct ReductionNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default, Op: ReductionAutogradOp<T, B>> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub a_shape: Shape,
    pub scaler_tensor: Option<Tensor<T, B>>,
    pub _phantom: std::marker::PhantomData<Op>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default, Op: ReductionAutogradOp<T, B>> BackwardNode<T, B> for ReductionNode<T, B, Op> {
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
            let grad_to_broadcast = if let Some(ref scaler) = self.scaler_tensor {
                coeus_ops::mul(grad_out, scaler, &backend)
            } else {
                grad_out.clone()
            };
            let broadcasted = grad_to_broadcast.broadcast(self.a_shape.clone());
            let mut gl = g.lock().unwrap();
            coeus_ops::add_assign(&mut gl, &broadcasted, &backend);
        }
    }
}

/// Generic, monomorphized reduction wrapper that builds the autograd node.
#[inline]
pub fn reduction_op<T: Scalar, B: coeus_ops::BackendOps<T> + Default, Op: ReductionAutogradOp<T, B> + 'static>(
    a: &Var<T, B>,
    param: Option<usize>,
) -> Var<T, B> {
    let backend = B::default();
    let out_tensor = Op::forward(&a.tensor, param, &backend);
    let requires_grad = a.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![a.clone()];
        let a_shape: Shape = a.tensor.shape_cloned();
        let scaler_tensor = Op::scaler(&a.tensor, param, &backend);
        let node: ReductionNode<T, B, Op> = ReductionNode {
            output_grad,
            inputs,
            a_shape,
            scaler_tensor,
            _phantom: std::marker::PhantomData,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var { tensor: out_tensor, grad, creator }
}

// ── ZST Operation Tags for Binary Ops ──

pub struct AddOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BinaryAutogradOp<T, B> for AddOp {
    const OP_NAME: &'static str = "add";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, b: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::add(a, b, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        _a: &Tensor<T, B>,
        _b: &Tensor<T, B>,
        a_shape: &Shape,
        b_shape: &Shape,
        input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>],
        backend: &B,
    ) {
        if let Some(Some(ref g)) = input_grads.get(0) {
            let mut gl = g.lock().unwrap();
            if grad_out.shape() == &a_shape[..] {
                coeus_ops::add_assign(&mut gl, grad_out, backend);
            } else {
                let reduced = reduce_broadcast(grad_out.clone(), a_shape);
                coeus_ops::add_assign(&mut gl, &reduced, backend);
            }
        }
        if let Some(Some(ref g)) = input_grads.get(1) {
            let mut gl = g.lock().unwrap();
            if grad_out.shape() == &b_shape[..] {
                coeus_ops::add_assign(&mut gl, grad_out, backend);
            } else {
                let reduced = reduce_broadcast(grad_out.clone(), b_shape);
                coeus_ops::add_assign(&mut gl, &reduced, backend);
            }
        }
    }
}

pub struct SubOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BinaryAutogradOp<T, B> for SubOp {
    const OP_NAME: &'static str = "sub";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, b: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::sub(a, b, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        _a: &Tensor<T, B>,
        _b: &Tensor<T, B>,
        a_shape: &Shape,
        b_shape: &Shape,
        input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>],
        backend: &B,
    ) {
        if let Some(Some(ref g)) = input_grads.get(0) {
            let mut gl = g.lock().unwrap();
            if grad_out.shape() == &a_shape[..] {
                coeus_ops::add_assign(&mut gl, grad_out, backend);
            } else {
                let reduced = reduce_broadcast(grad_out.clone(), a_shape);
                coeus_ops::add_assign(&mut gl, &reduced, backend);
            }
        }
        if let Some(Some(ref g)) = input_grads.get(1) {
            let mut gl = g.lock().unwrap();
            if grad_out.shape() == &b_shape[..] {
                coeus_ops::sub_assign(&mut gl, grad_out, backend);
            } else {
                let reduced = reduce_broadcast(grad_out.clone(), b_shape);
                coeus_ops::sub_assign(&mut gl, &reduced, backend);
            }
        }
    }
}

pub struct MulOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BinaryAutogradOp<T, B> for MulOp {
    const OP_NAME: &'static str = "mul";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, b: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::mul(a, b, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        a: &Tensor<T, B>,
        b: &Tensor<T, B>,
        _a_shape: &Shape,
        _b_shape: &Shape,
        input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>],
        backend: &B,
    ) {
        if let Some(Some(ref g)) = input_grads.get(0) {
            let prod = coeus_ops::mul(grad_out, b, backend);
            let mut gl = g.lock().unwrap();
            if prod.shape() == a.shape() {
                coeus_ops::add_assign(&mut gl, &prod, backend);
            } else {
                let reduced = reduce_broadcast(prod, a.shape());
                coeus_ops::add_assign(&mut gl, &reduced, backend);
            }
        }
        if let Some(Some(ref g)) = input_grads.get(1) {
            let prod = coeus_ops::mul(grad_out, a, backend);
            let mut gl = g.lock().unwrap();
            if prod.shape() == b.shape() {
                coeus_ops::add_assign(&mut gl, &prod, backend);
            } else {
                let reduced = reduce_broadcast(prod, b.shape());
                coeus_ops::add_assign(&mut gl, &reduced, backend);
            }
        }
    }
}

pub struct DivOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BinaryAutogradOp<T, B> for DivOp {
    const OP_NAME: &'static str = "div";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, b: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::div(a, b, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        a: &Tensor<T, B>,
        b: &Tensor<T, B>,
        _a_shape: &Shape,
        _b_shape: &Shape,
        input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>],
        backend: &B,
    ) {
        if let Some(Some(ref g)) = input_grads.get(0) {
            let grad_a = coeus_ops::div(grad_out, b, backend);
            let mut gl = g.lock().unwrap();
            if grad_a.shape() == a.shape() {
                coeus_ops::add_assign(&mut gl, &grad_a, backend);
            } else {
                let reduced = reduce_broadcast(grad_a, a.shape());
                coeus_ops::add_assign(&mut gl, &reduced, backend);
            }
        }
        if let Some(Some(ref g)) = input_grads.get(1) {
            let b_sq = coeus_ops::mul(b, b, backend);
            let grad_b_pos = coeus_ops::div(&coeus_ops::mul(grad_out, a, backend), &b_sq, backend);
            let mut gl = g.lock().unwrap();
            if grad_b_pos.shape() == b.shape() {
                coeus_ops::sub_assign(&mut gl, &grad_b_pos, backend);
            } else {
                let reduced = reduce_broadcast(grad_b_pos, b.shape());
                coeus_ops::sub_assign(&mut gl, &reduced, backend);
            }
        }
    }
}

// ── ZST Operation Tags for Reduction Ops ──

pub struct SumOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> ReductionAutogradOp<T, B> for SumOp {
    const OP_NAME: &'static str = "sum";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, _param: Option<usize>, backend: &B) -> Tensor<T, B> {
        let total = coeus_ops::sum(a, backend);
        Tensor::from_slice_on([1], &[total], backend)
    }

    #[inline(always)]
    fn scaler(_a: &Tensor<T, B>, _param: Option<usize>, _backend: &B) -> Option<Tensor<T, B>> {
        None
    }
}

pub struct SumAxisOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> ReductionAutogradOp<T, B> for SumAxisOp {
    const OP_NAME: &'static str = "sum_axis";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, param: Option<usize>, backend: &B) -> Tensor<T, B> {
        coeus_ops::sum_axis(a, param.unwrap(), backend)
    }

    #[inline(always)]
    fn scaler(_a: &Tensor<T, B>, _param: Option<usize>, _backend: &B) -> Option<Tensor<T, B>> {
        None
    }
}

pub struct MeanOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> ReductionAutogradOp<T, B> for MeanOp {
    const OP_NAME: &'static str = "mean";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, _param: Option<usize>, backend: &B) -> Tensor<T, B> {
        let total = coeus_ops::sum(a, backend);
        let n = a.numel() as f64;
        Tensor::from_slice_on([1], &[total / T::from_f64(n)], backend)
    }

    #[inline(always)]
    fn scaler(a: &Tensor<T, B>, _param: Option<usize>, backend: &B) -> Option<Tensor<T, B>> {
        let n = a.numel() as f64;
        Some(Tensor::full_on([1], T::from_f64(1.0 / n), backend))
    }
}

pub struct MeanAxisOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> ReductionAutogradOp<T, B> for MeanAxisOp {
    const OP_NAME: &'static str = "mean_axis";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, param: Option<usize>, backend: &B) -> Tensor<T, B> {
        coeus_ops::mean_axis(a, param.unwrap(), backend)
    }

    #[inline(always)]
    fn scaler(a: &Tensor<T, B>, param: Option<usize>, backend: &B) -> Option<Tensor<T, B>> {
        let axis_len = a.shape()[param.unwrap()] as f64;
        Some(Tensor::full_on([1], T::from_f64(1.0 / axis_len), backend))
    }
}

// ── Public Tracked Arithmetic and Reduction Functions ──

/// Tracked element-wise addition.
#[inline]
pub fn add<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>, b: &Var<T, B>) -> Var<T, B> {
    binary_op::<T, B, AddOp>(a, b)
}

/// Tracked element-wise subtraction.
#[inline]
pub fn sub<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>, b: &Var<T, B>) -> Var<T, B> {
    binary_op::<T, B, SubOp>(a, b)
}

/// Tracked element-wise multiplication.
#[inline]
pub fn mul<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>, b: &Var<T, B>) -> Var<T, B> {
    binary_op::<T, B, MulOp>(a, b)
}

/// Tracked element-wise division.
#[inline]
pub fn div<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>, b: &Var<T, B>) -> Var<T, B> {
    binary_op::<T, B, DivOp>(a, b)
}

/// Tracked sum reduction of all elements.
#[inline]
pub fn sum<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    reduction_op::<T, B, SumOp>(a, None)
}

/// Tracked mean reduction of all elements.
#[inline]
pub fn mean<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    reduction_op::<T, B, MeanOp>(a, None)
}

/// Tracked sum reduction along an axis.
#[inline]
pub fn sum_axis<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>, axis: usize) -> Var<T, B> {
    reduction_op::<T, B, SumAxisOp>(a, Some(axis))
}

/// Tracked mean reduction along an axis.
#[inline]
pub fn mean_axis<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>, axis: usize) -> Var<T, B> {
    reduction_op::<T, B, MeanAxisOp>(a, Some(axis))
}

// ── Scalar Arithmetic ───────────────────────────────────────────────────────
//
// `scalar_mul` and `scalar_add` are convenience wrappers that apply a scalar
// value to every element of a `Var` without requiring the caller to construct a
// broadcast tensor.  Internally each creates a single-element non-tracked `Var`
// from the scalar and delegates to the existing `binary_op::<MulOp/AddOp>`
// path, which handles broadcasting. The scalar `Var` carries no gradient, so
// the backward pass only accumulates into the LHS gradient — identical
// semantics to PyTorch's `tensor * scalar`.
//
// Zero-cost: `Tensor::full_on([1], s, &backend)` allocates one element; the
// subsequent broadcast in `binary_op` is stride-based and copies no data.

/// Tracked element-wise multiply by a scalar.
///
/// Equivalent to `mul(x, Var::new(Tensor::full_on([1], s), false))` but
/// expressed without the call-site boilerplate.
#[inline]
pub fn scalar_mul<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    s: T,
) -> Var<T, B> {
    let backend = B::default();
    let scalar_tensor = Tensor::full_on([1], s, &backend);
    let scalar_var = Var::new(scalar_tensor, false);
    binary_op::<T, B, MulOp>(x, &scalar_var)
}

/// Tracked element-wise add by a scalar.
///
/// Equivalent to `add(x, Var::new(Tensor::full_on([1], s), false))` but
/// expressed without the call-site boilerplate.
#[inline]
pub fn scalar_add<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    s: T,
) -> Var<T, B> {
    let backend = B::default();
    let scalar_tensor = Tensor::full_on([1], s, &backend);
    let scalar_var = Var::new(scalar_tensor, false);
    binary_op::<T, B, AddOp>(x, &scalar_var)
}

/// Tracked element-wise subtraction by a scalar (x - s).
#[inline]
pub fn scalar_sub<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    s: T,
) -> Var<T, B> {
    let backend = B::default();
    let scalar_tensor = Tensor::full_on([1], s, &backend);
    let scalar_var = Var::new(scalar_tensor, false);
    binary_op::<T, B, SubOp>(x, &scalar_var)
}

/// Tracked element-wise division by a scalar (x / s).
#[inline]
pub fn scalar_div<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    s: T,
) -> Var<T, B> {
    let backend = B::default();
    let scalar_tensor = Tensor::full_on([1], s, &backend);
    let scalar_var = Var::new(scalar_tensor, false);
    binary_op::<T, B, DivOp>(x, &scalar_var)
}

