use crate::fuse::op_tags::{BinaryOpTag, UnaryOpTag};
use coeus_core::{ComputeBackend, Layout, Scalar, Shape, Storage};
use coeus_tensor::broadcast::broadcast_shapes;
use coeus_tensor::Tensor;
use std::cell::RefCell;
use std::collections::HashMap;

pub struct CachedTensor<T> {
    pub data: Vec<T>,
    pub layout: Layout,
}

thread_local! {
    pub static CPU_EVAL_CACHE: RefCell<HashMap<usize, Box<dyn std::any::Any>>> = RefCell::new(HashMap::new());
}

pub trait ExprNode<T: Scalar, B: ComputeBackend>: 'static + Send + Sync {
    fn collect_inputs(&self, list: &mut Vec<*const Tensor<T, B>>);
    fn to_shader_expr(&self, input_map: &HashMap<*const Tensor<T, B>, usize>) -> String;

    /// Evaluates the expression node on the CPU at the specified coordinates.
    ///
    /// # Safety
    /// The caller must ensure that the coordinates are within bounds for the expression shape,
    /// and that the underlying input tensors are still valid.
    unsafe fn eval_cpu(&self, coords: &[usize]) -> T;
    fn shape(&self) -> Option<Shape>;

    fn is_contiguous_and_same_shape(&self, out_shape: &[usize]) -> bool;
    /// Evaluates the expression node on the CPU at a flat index for contiguous tensors.
    ///
    /// # Safety
    /// The caller must ensure that the index is within flat offset bounds.
    unsafe fn eval_cpu_flat(&self, idx: usize) -> T;
}

pub struct TensorRef<T: Scalar, B: ComputeBackend>(pub *const Tensor<T, B>);

unsafe impl<T: Scalar, B: ComputeBackend> Send for TensorRef<T, B> {}
unsafe impl<T: Scalar, B: ComputeBackend> Sync for TensorRef<T, B> {}

impl<T: Scalar, B: ComputeBackend> Clone for TensorRef<T, B> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Scalar, B: ComputeBackend> Copy for TensorRef<T, B> {}

impl<T: Scalar, B: ComputeBackend> ExprNode<T, B> for TensorRef<T, B> {
    fn collect_inputs(&self, list: &mut Vec<*const Tensor<T, B>>) {
        let ptr = self.0;
        if !list.contains(&ptr) {
            list.push(ptr);
        }
    }

    fn to_shader_expr(&self, input_map: &HashMap<*const Tensor<T, B>, usize>) -> String {
        let ptr = self.0;
        let idx = input_map.get(&ptr).expect("Input tensor not found in map");
        format!("val_{}", idx)
    }

    unsafe fn eval_cpu(&self, coords: &[usize]) -> T {
        let tensor = &*self.0;
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
                    .get(&(self.0 as usize))
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

    fn shape(&self) -> Option<Shape> {
        unsafe { Some((*self.0).shape_cloned()) }
    }

    fn is_contiguous_and_same_shape(&self, out_shape: &[usize]) -> bool {
        unsafe {
            let tensor = &*self.0;
            let layout = tensor.layout();
            layout.is_contiguous() && layout.shape() == out_shape
        }
    }

    unsafe fn eval_cpu_flat(&self, idx: usize) -> T {
        let tensor = &*self.0;
        if let Some(slice) = tensor.storage().try_as_slice() {
            let layout = tensor.layout();
            let off = layout.offset() + idx;
            slice[off]
        } else {
            CPU_EVAL_CACHE.with(|cache| {
                let cache_ref = cache.borrow();
                let any_ref = cache_ref
                    .get(&(self.0 as usize))
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
pub struct ScalarVal<T: Scalar>(pub T);

impl<T: Scalar, B: ComputeBackend> ExprNode<T, B> for ScalarVal<T> {
    fn collect_inputs(&self, _list: &mut Vec<*const Tensor<T, B>>) {}

    fn to_shader_expr(&self, _input_map: &HashMap<*const Tensor<T, B>, usize>) -> String {
        let val = self.0.to_f64();
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

    fn shape(&self) -> Option<Shape> {
        None
    }

    fn is_contiguous_and_same_shape(&self, _out_shape: &[usize]) -> bool {
        true
    }

    unsafe fn eval_cpu_flat(&self, _idx: usize) -> T {
        self.0
    }
}

#[derive(Clone, Copy)]
pub struct UnaryExpr<Op, Child> {
    pub op: Op,
    pub child: Child,
}

impl<Op: UnaryOpTag<T> + Send, Child: ExprNode<T, B>, T: Scalar, B: ComputeBackend> ExprNode<T, B>
    for UnaryExpr<Op, Child>
{
    fn collect_inputs(&self, list: &mut Vec<*const Tensor<T, B>>) {
        self.child.collect_inputs(list);
    }

    fn to_shader_expr(&self, input_map: &HashMap<*const Tensor<T, B>, usize>) -> String {
        let child_str = self.child.to_shader_expr(input_map);
        Op::WGSL_TEMPLATE.replace("{}", &child_str)
    }

    unsafe fn eval_cpu(&self, coords: &[usize]) -> T {
        let val = self.child.eval_cpu(coords);
        Op::apply(val)
    }

    fn shape(&self) -> Option<Shape> {
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
pub struct BinaryExpr<Op: BinaryOpTag, Left, Right> {
    pub op: Op,
    pub left: Left,
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
    fn collect_inputs(&self, list: &mut Vec<*const Tensor<T, B>>) {
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

    fn shape(&self) -> Option<Shape> {
        let left_shape = self.left.shape();
        let right_shape = self.right.shape();
        match (left_shape, right_shape) {
            (Some(l), Some(r)) => {
                let out =
                    broadcast_shapes(&l, &r).expect("Incompatible shapes in fused expression");
                Some(out)
            }
            (Some(l), None) => Some(l),
            (None, Some(r)) => Some(r),
            (None, None) => None,
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
pub struct Expr<E>(pub E);

impl<E: ExprNode<T, B>, T: Scalar, B: ComputeBackend> ExprNode<T, B> for Expr<E> {
    #[inline(always)]
    fn collect_inputs(&self, list: &mut Vec<*const Tensor<T, B>>) {
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
    fn shape(&self) -> Option<Shape> {
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

pub trait TensorExprExt<T: Scalar, B: ComputeBackend> {
    fn expr(&self) -> Expr<TensorRef<T, B>>;
}

impl<T: Scalar, B: ComputeBackend> TensorExprExt<T, B> for Tensor<T, B> {
    #[inline(always)]
    fn expr(&self) -> Expr<TensorRef<T, B>> {
        Expr(TensorRef(self as *const Tensor<T, B>))
    }
}

#[inline(always)]
pub fn scalar<T: Scalar, B: ComputeBackend>(val: T) -> Expr<ScalarVal<T>> {
    Expr(ScalarVal(val))
}
