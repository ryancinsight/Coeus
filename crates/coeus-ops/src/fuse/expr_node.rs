use crate::fuse::op_tags::{BinaryOpTag, UnaryOpTag};
use coeus_core::{BackendError, ComputeBackend, Layout, Scalar, Shape, Storage};
use coeus_tensor::broadcast::broadcast_shapes;
use coeus_tensor::Tensor;
use std::cell::RefCell;
use std::collections::HashMap;

/// Cached tensor data and layout for CPU evaluation of device-resident tensors.
pub struct CachedTensor<T> {
    /// Flat data buffer.
    pub data: Vec<T>,
    /// Tensor layout descriptor.
    pub layout: Layout,
}

thread_local! {
    /// Thread-local cache mapping tensor pointers to CPU-cached data for fused evaluation.
    pub static CPU_EVAL_CACHE: RefCell<HashMap<usize, Box<dyn std::any::Any>>> = RefCell::new(HashMap::new());
}

/// A node in the fused expression DAG.
///
/// Implementors describe how to collect input tensors, emit a WGSL shader
/// fragment, and evaluate the node on the CPU.
pub trait ExprNode<T: Scalar, B: ComputeBackend>: Send + Sync {
    /// Collect all input tensors borrowed by this node.
    fn collect_inputs<'expression>(&'expression self, list: &mut Vec<&'expression Tensor<T, B>>);
    /// Emit a WGSL expression string referencing the inputs by index.
    fn to_shader_expr(&self, input_map: &HashMap<*const Tensor<T, B>, usize>) -> String;

    /// Evaluates the expression node on the CPU at the specified coordinates.
    ///
    /// # Safety
    /// The caller must ensure that the coordinates are within bounds for the expression shape,
    /// and that the underlying input tensors are still valid.
    unsafe fn eval_cpu(&self, coords: &[usize]) -> T;
    /// Returns the output shape of this node, or `None` for scalar nodes.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when child shapes cannot be broadcast.
    fn shape(&self) -> Result<Option<Shape>, BackendError>;

    /// Returns `true` if this node is contiguous and matches `out_shape`.
    fn is_contiguous_and_same_shape(&self, out_shape: &[usize]) -> bool;
    /// Evaluates the expression node on the CPU at a flat index for contiguous tensors.
    ///
    /// # Safety
    /// The caller must ensure that the index is within flat offset bounds.
    unsafe fn eval_cpu_flat(&self, idx: usize) -> T;
}

/// A borrowed tensor leaf in the fused expression DAG.
pub struct TensorRef<'tensor, T: Scalar, B: ComputeBackend>(&'tensor Tensor<T, B>);

impl<T: Scalar, B: ComputeBackend> Clone for TensorRef<'_, T, B> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Scalar, B: ComputeBackend> Copy for TensorRef<'_, T, B> {}

impl<T: Scalar, B: ComputeBackend> ExprNode<T, B> for TensorRef<'_, T, B> {
    fn collect_inputs<'expression>(&'expression self, list: &mut Vec<&'expression Tensor<T, B>>) {
        if !list.iter().any(|tensor| std::ptr::eq(*tensor, self.0)) {
            list.push(self.0);
        }
    }

    fn to_shader_expr(&self, input_map: &HashMap<*const Tensor<T, B>, usize>) -> String {
        let ptr = std::ptr::from_ref(self.0);
        let idx = input_map.get(&ptr).expect("Input tensor not found in map");
        format!("val_{}", idx)
    }

    unsafe fn eval_cpu(&self, coords: &[usize]) -> T {
        let tensor = self.0;
        if let Some(slice) = tensor.storage().try_as_slice() {
            let layout = tensor.layout();
            let shape = layout.shape();
            let strides = layout.strides();
            let ndim = layout.ndim();
            let out_ndim = coords.len();

            let mut off = layout.offset();
            let diff = out_ndim.saturating_sub(ndim);
            for d in diff..out_ndim {
                let ad = d - diff;
                if shape[ad] > 1 {
                    off += coords[d] * strides[ad];
                }
            }
            slice[off]
        } else {
            CPU_EVAL_CACHE.with(|cache| {
                let cache_ref = cache.borrow();
                let any_ref = cache_ref
                    .get(&(std::ptr::from_ref(self.0) as usize))
                    .expect("Device tensor not cached for CPU evaluation");
                if let Some(cached) = any_ref.downcast_ref::<CachedTensor<T>>() {
                    let layout = &cached.layout;
                    let shape = layout.shape();
                    let strides = layout.strides();
                    let ndim = layout.ndim();
                    let out_ndim = coords.len();

                    let mut off = layout.offset();
                    let diff = out_ndim.saturating_sub(ndim);
                    for d in diff..out_ndim {
                        let ad = d - diff;
                        if shape[ad] > 1 {
                            off += coords[d] * strides[ad];
                        }
                    }
                    cached.data[off]
                } else if let Some(slice) = any_ref.downcast_ref::<Vec<T>>() {
                    let layout = tensor.layout();
                    let shape = layout.shape();
                    let strides = layout.strides();
                    let ndim = layout.ndim();
                    let out_ndim = coords.len();

                    let mut off = layout.offset();
                    let diff = out_ndim.saturating_sub(ndim);
                    for d in diff..out_ndim {
                        let ad = d - diff;
                        if shape[ad] > 1 {
                            off += coords[d] * strides[ad];
                        }
                    }
                    slice[off]
                } else {
                    panic!("Incorrect type in cache");
                }
            })
        }
    }

    fn shape(&self) -> Result<Option<Shape>, BackendError> {
        Ok(Some(self.0.shape_cloned()))
    }

    fn is_contiguous_and_same_shape(&self, out_shape: &[usize]) -> bool {
        let layout = self.0.layout();
        layout.is_contiguous() && layout.shape() == out_shape
    }

    unsafe fn eval_cpu_flat(&self, idx: usize) -> T {
        let tensor = self.0;
        if let Some(slice) = tensor.storage().try_as_slice() {
            let layout = tensor.layout();
            let off = layout.offset() + idx;
            slice[off]
        } else {
            CPU_EVAL_CACHE.with(|cache| {
                let cache_ref = cache.borrow();
                let any_ref = cache_ref
                    .get(&(std::ptr::from_ref(self.0) as usize))
                    .expect("Device tensor not cached for CPU evaluation");
                if let Some(cached) = any_ref.downcast_ref::<CachedTensor<T>>() {
                    let layout = &cached.layout;
                    let off = layout.offset() + idx;
                    cached.data[off]
                } else if let Some(slice) = any_ref.downcast_ref::<Vec<T>>() {
                    let layout = tensor.layout();
                    let off = layout.offset() + idx;
                    slice[off]
                } else {
                    panic!("Incorrect type in cache");
                }
            })
        }
    }
}

#[derive(Clone, Copy)]
/// A scalar constant leaf node in the fused expression DAG.
pub struct ScalarVal<T: Scalar>(pub T);

impl<T: Scalar, B: ComputeBackend> ExprNode<T, B> for ScalarVal<T> {
    fn collect_inputs<'expression>(&'expression self, _list: &mut Vec<&'expression Tensor<T, B>>) {}

    fn to_shader_expr(&self, _input_map: &HashMap<*const Tensor<T, B>, usize>) -> String {
        let val = <T as Scalar>::to_f64(self.0);
        if val.is_infinite() {
            if val.is_sign_positive() {
                "3.40282347e+38".to_string()
            } else {
                "-3.40282347e+38".to_string()
            }
        } else if val.is_nan() {
            "0.0".to_string()
        } else {
            let s = format!("{:?}", val);
            if s.contains('.') || s.contains('e') {
                s
            } else {
                format!("{}.0", s)
            }
        }
    }

    unsafe fn eval_cpu(&self, _coords: &[usize]) -> T {
        self.0
    }

    fn shape(&self) -> Result<Option<Shape>, BackendError> {
        Ok(None)
    }

    fn is_contiguous_and_same_shape(&self, _out_shape: &[usize]) -> bool {
        true
    }

    unsafe fn eval_cpu_flat(&self, _idx: usize) -> T {
        self.0
    }
}

#[derive(Clone, Copy)]
/// A unary operation node in the fused expression DAG.
pub struct UnaryExpr<Op, Child> {
    /// The operation tag.
    pub op: Op,
    /// The child sub-expression.
    pub child: Child,
}

impl<Op: UnaryOpTag<T> + Send, Child: ExprNode<T, B>, T: Scalar, B: ComputeBackend> ExprNode<T, B>
    for UnaryExpr<Op, Child>
{
    fn collect_inputs<'expression>(&'expression self, list: &mut Vec<&'expression Tensor<T, B>>) {
        self.child.collect_inputs(list);
    }

    fn to_shader_expr(&self, input_map: &HashMap<*const Tensor<T, B>, usize>) -> String {
        let child_str = self.child.to_shader_expr(input_map);
        Op::wgsl_expr(&child_str)
    }

    unsafe fn eval_cpu(&self, coords: &[usize]) -> T {
        let val = self.child.eval_cpu(coords);
        Op::apply(val)
    }

    fn shape(&self) -> Result<Option<Shape>, BackendError> {
        self.child.shape()
    }

    fn is_contiguous_and_same_shape(&self, out_shape: &[usize]) -> bool {
        self.child.is_contiguous_and_same_shape(out_shape)
    }

    unsafe fn eval_cpu_flat(&self, idx: usize) -> T {
        let val = self.child.eval_cpu_flat(idx);
        Op::apply(val)
    }
}

#[derive(Clone, Copy)]
/// A binary operation node in the fused expression DAG.
pub struct BinaryExpr<Op: BinaryOpTag, Left, Right> {
    /// The operation tag.
    pub op: Op,
    /// The left-hand sub-expression.
    pub left: Left,
    /// The right-hand sub-expression.
    pub right: Right,
}

impl<
        Op: BinaryOpTag + Send,
        Left: ExprNode<T, B>,
        Right: ExprNode<T, B>,
        T: Scalar,
        B: ComputeBackend,
    > ExprNode<T, B> for BinaryExpr<Op, Left, Right>
{
    fn collect_inputs<'expression>(&'expression self, list: &mut Vec<&'expression Tensor<T, B>>) {
        self.left.collect_inputs(list);
        self.right.collect_inputs(list);
    }

    fn to_shader_expr(&self, input_map: &HashMap<*const Tensor<T, B>, usize>) -> String {
        let left_str = self.left.to_shader_expr(input_map);
        let right_str = self.right.to_shader_expr(input_map);
        format!("(({}) {} ({}))", left_str, Op::WGSL_SYMBOL, right_str)
    }

    unsafe fn eval_cpu(&self, coords: &[usize]) -> T {
        let left_val = self.left.eval_cpu(coords);
        let right_val = self.right.eval_cpu(coords);
        Op::apply(left_val, right_val)
    }

    fn shape(&self) -> Result<Option<Shape>, BackendError> {
        let left_shape = self.left.shape()?;
        let right_shape = self.right.shape()?;
        match (left_shape, right_shape) {
            (Some(l), Some(r)) => {
                let out = broadcast_shapes(&l, &r).ok_or_else(|| {
                    BackendError::IncompatibleBroadcast {
                        operation: "fused expression",
                        from: l.to_vec(),
                        to: r.to_vec(),
                    }
                })?;
                Ok(Some(out))
            }
            (Some(l), None) => Ok(Some(l)),
            (None, Some(r)) => Ok(Some(r)),
            (None, None) => Ok(None),
        }
    }

    fn is_contiguous_and_same_shape(&self, out_shape: &[usize]) -> bool {
        self.left.is_contiguous_and_same_shape(out_shape)
            && self.right.is_contiguous_and_same_shape(out_shape)
    }

    unsafe fn eval_cpu_flat(&self, idx: usize) -> T {
        let left_val = self.left.eval_cpu_flat(idx);
        let right_val = self.right.eval_cpu_flat(idx);
        Op::apply(left_val, right_val)
    }
}

#[derive(Clone, Copy)]
/// A wrapper around an expression node, providing operator overloading and builder methods.
pub struct Expr<E>(pub E);

impl<E: ExprNode<T, B>, T: Scalar, B: ComputeBackend> ExprNode<T, B> for Expr<E> {
    #[inline(always)]
    fn collect_inputs<'expression>(&'expression self, list: &mut Vec<&'expression Tensor<T, B>>) {
        self.0.collect_inputs(list);
    }

    #[inline(always)]
    fn to_shader_expr(&self, input_map: &HashMap<*const Tensor<T, B>, usize>) -> String {
        self.0.to_shader_expr(input_map)
    }

    #[inline(always)]
    unsafe fn eval_cpu(&self, coords: &[usize]) -> T {
        self.0.eval_cpu(coords)
    }

    #[inline(always)]
    fn shape(&self) -> Result<Option<Shape>, BackendError> {
        self.0.shape()
    }

    #[inline(always)]
    fn is_contiguous_and_same_shape(&self, out_shape: &[usize]) -> bool {
        self.0.is_contiguous_and_same_shape(out_shape)
    }

    #[inline(always)]
    unsafe fn eval_cpu_flat(&self, idx: usize) -> T {
        self.0.eval_cpu_flat(idx)
    }
}

/// Extension trait that converts a [`Tensor`] into a fused expression leaf.
pub trait TensorExprExt<T: Scalar, B: ComputeBackend> {
    /// Create a fused expression leaf referencing this tensor.
    fn expr(&self) -> Expr<TensorRef<'_, T, B>>;
}

impl<T: Scalar, B: ComputeBackend> TensorExprExt<T, B> for Tensor<T, B> {
    #[inline(always)]
    fn expr(&self) -> Expr<TensorRef<'_, T, B>> {
        Expr(TensorRef(self))
    }
}

/// Create a fused expression scalar constant.
#[inline(always)]
pub fn scalar<T: Scalar, B: ComputeBackend>(val: T) -> Expr<ScalarVal<T>> {
    Expr(ScalarVal(val))
}
