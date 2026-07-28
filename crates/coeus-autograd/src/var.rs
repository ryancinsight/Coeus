// ── Differentiable variable ──

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use coeus_core::{BackendError, ComputeBackend, MoiraiBackend, Scalar};
use coeus_tensor::Tensor;
use std::collections::HashSet;
use std::sync::Arc;

/// A differentiable variable wrapping a tensor and its gradient accumulator.
///
/// # Examples
///
/// Build a small computation graph, run reverse-mode autodiff, and read back the
/// leaf gradients. For `y = x^2` with `x = [3, 4]`, the analytic gradient is
/// `dy/dx = 2x = [6, 8]`.
///
/// ```
/// use coeus_autograd::Var;
/// use coeus_core::MoiraiBackend;
/// use coeus_tensor::Tensor;
///
/// let x = Var::<f32, MoiraiBackend>::new(
///     Tensor::from_slice([2], &[3.0, 4.0]).expect("construct tensor"),
///     true,
/// ).expect("construct variable");
/// let y = coeus_autograd::mul(&x, &x).expect("multiply variables"); // y = x * x
/// y.backward().expect("backward propagation");
/// let grad = x.grad().unwrap();
/// assert!((grad.as_slice()[0] - 6.0).abs() < 1e-5); // 2 * 3
/// assert!((grad.as_slice()[1] - 8.0).abs() < 1e-5); // 2 * 4
/// ```
#[derive(Clone)]
pub struct Var<T: Scalar, B: ComputeBackend + Default = MoiraiBackend> {
    /// The tensor value.
    pub tensor: Tensor<T, B>,
    /// Accumulated gradient (None if not tracking).
    pub grad: Option<Arc<GradBuffer<T, B>>>,
    /// The operation that created this variable (None for leaf nodes).
    pub creator: Option<Arc<dyn BackwardNode<T, B>>>,
}

impl<T: Scalar, B: ComputeBackend + Default> Var<T, B> {
    /// Create a new leaf variable.
    ///
    /// When `requires_grad` is `true` the leaf allocates a gradient buffer that
    /// accumulates gradients during [`Var::backward`]. A leaf constructed with
    /// `requires_grad = false` is a constant: it carries no gradient state.
    ///
    /// # Examples
    ///
    /// A leaf with `requires_grad = true` accumulates gradients; one with
    /// `requires_grad = false` reports `None`.
    ///
    /// ```
    /// use coeus_autograd::Var;
    /// use coeus_core::MoiraiBackend;
    /// use coeus_tensor::Tensor;
    ///
    /// let x = Var::<f32, MoiraiBackend>::new(
    ///     Tensor::from_slice([2], &[1.0, 2.0]).expect("construct tensor"),
    ///     true,
    /// ).expect("construct variable");
    /// let c = Var::<f32, MoiraiBackend>::new(
    ///     Tensor::from_slice([2], &[10.0, 20.0]).expect("construct tensor"),
    ///     false,
    /// ).expect("construct variable");
    /// let y = coeus_autograd::add(&x, &c).expect("add variables");
    /// y.backward().expect("backward propagation");
    /// assert!(x.grad().is_some()); // tracked leaf: gradient present
    /// assert!(c.grad().is_none()); // constant leaf: no gradient state
    /// ```
    #[inline]
    pub fn new(tensor: Tensor<T, B>, requires_grad: bool) -> Result<Self, B::Error> {
        let grad = if requires_grad {
            Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
                tensor.shape(),
                &B::default(),
            )?)))
        } else {
            None
        };
        Ok(Self {
            tensor,
            grad,
            creator: None,
        })
    }

    /// Create an intermediate variable (result of an op).
    #[inline]
    pub fn with_creator(
        tensor: Tensor<T, B>,
        grad: Option<Arc<GradBuffer<T, B>>>,
        creator: Arc<dyn BackwardNode<T, B>>,
    ) -> Self {
        Self {
            tensor,
            grad,
            creator: Some(creator),
        }
    }

    /// Read the accumulated gradient.
    ///
    /// Returns `None` for constants (leaves built with `requires_grad = false`)
    /// and for intermediate variables produced under a no-grad scope. After
    /// [`Var::backward`], a tracked leaf holds the sum of gradients pushed to it.
    ///
    /// # Examples
    ///
    /// For `y = x + 1` summed to a scalar, `dy/dx = 1` for every element.
    ///
    /// ```
    /// use coeus_autograd::Var;
    /// use coeus_core::MoiraiBackend;
    /// use coeus_tensor::Tensor;
    ///
    /// let x = Var::<f32, MoiraiBackend>::new(
    ///     Tensor::from_slice([3], &[1.0, 2.0, 3.0]).expect("construct tensor"),
    ///     true,
    /// ).expect("construct variable");
    /// let one = Var::<f32, MoiraiBackend>::new(
    ///     Tensor::from_slice([3], &[1.0; 3]).expect("construct tensor"),
    ///     false,
    /// ).expect("construct variable");
    /// let y = coeus_autograd::add(&x, &one).expect("add variables");
    /// let loss = coeus_autograd::sum(&y).expect("sum variables");
    /// loss.backward().expect("backward propagation");
    /// let grad = x.grad().unwrap();
    /// assert!((grad.as_slice()[0] - 1.0).abs() < 1e-5);
    /// assert!((grad.as_slice()[1] - 1.0).abs() < 1e-5);
    /// assert!((grad.as_slice()[2] - 1.0).abs() < 1e-5);
    /// ```
    #[inline]
    pub fn grad(&self) -> Option<Tensor<T, B>> {
        self.grad.as_ref().map(|g| g.clone_tensor())
    }

    /// Set the gradient tensor.
    #[inline]
    pub fn set_grad(&self, grad: Tensor<T, B>) {
        if let Some(ref g) = self.grad {
            *g.write() = grad;
        }
    }

    /// Zero the accumulated gradient.
    ///
    /// Useful when a leaf is reused across multiple backward passes: call
    /// `zero_grad` between passes so gradients accumulate from a clean slate
    /// rather than summing across passes.
    ///
    /// # Examples
    ///
    /// Running backward twice without `zero_grad` accumulates; calling
    /// `zero_grad` between passes resets the buffer to zero.
    ///
    /// ```
    /// use coeus_autograd::Var;
    /// use coeus_core::MoiraiBackend;
    /// use coeus_tensor::Tensor;
    ///
    /// let x = Var::<f32, MoiraiBackend>::new(
    ///     Tensor::from_slice([2], &[2.0, 3.0]).expect("construct tensor"),
    ///     true,
    /// ).expect("construct variable");
    /// let y = coeus_autograd::sum(&x).expect("sum variables");
    ///
    /// y.backward().expect("backward propagation");
    /// let g1 = x.grad().unwrap();
    /// assert!((g1.as_slice()[0] - 1.0).abs() < 1e-5);
    ///
    /// // Second backward would accumulate; zero first.
    /// x.zero_grad().expect("zero gradient");
    /// assert!((x.grad().unwrap().as_slice()[0] - 0.0).abs() < 1e-5);
    /// ```
    #[inline]
    pub fn zero_grad(&self) -> Result<(), B::Error> {
        if let Some(ref g) = self.grad {
            let guard = g.write();
            let storage = guard.storage_mut()?;
            B::default().fill(storage, T::zero())?;
        }
        Ok(())
    }

    /// Run reverse-mode autodiff from this variable, seeding with `Tensor::ones`.
    ///
    /// Equivalent to `self.backward_with_seed(Tensor::ones(self.tensor.shape().to_vec()))`.
    ///
    /// # Examples
    ///
    /// For `y = x^2` summed to a scalar, `backward` seeds the scalar output with
    /// 1 and propagates `dy/dx = 2x` to the leaf.
    ///
    /// ```
    /// use coeus_autograd::Var;
    /// use coeus_core::MoiraiBackend;
    /// use coeus_tensor::Tensor;
    ///
    /// let x = Var::<f32, MoiraiBackend>::new(
    ///     Tensor::from_slice([2], &[3.0, 4.0]).expect("construct tensor"),
    ///     true,
    /// ).expect("construct variable");
    /// let y = coeus_autograd::mul(&x, &x).expect("multiply variables");
    /// let loss = coeus_autograd::sum(&y).expect("sum variables"); // scalar: y_0 + y_1
    /// loss.backward().expect("backward propagation");
    /// let grad = x.grad().unwrap();
    /// assert!((grad.as_slice()[0] - 6.0).abs() < 1e-5); // 2 * 3
    /// assert!((grad.as_slice()[1] - 8.0).abs() < 1e-5); // 2 * 4
    /// ```
    #[inline]
    pub fn backward(&self) -> Result<(), B::Error> {
        let seed = Tensor::ones_on(self.tensor.shape(), &B::default())?;
        self.backward_with_seed(seed)
    }

    /// Run reverse-mode autodiff from this variable, seeding with the given gradient.
    ///
    /// Sets `self.grad = seed` and propagates backwards through the computation graph.
    /// This enables testing with non-uniform upstream gradients (e.g. for softmax,
    /// where a uniform seed produces zero input gradient due to Jacobian row-sums).
    ///
    /// # Panics
    /// If `seed.shape()` does not match `self.tensor.shape()`.
    #[inline]
    pub fn backward_with_seed(&self, seed: Tensor<T, B>) -> Result<(), B::Error> {
        if seed.shape() != self.tensor.shape() {
            return Err(B::Error::from(BackendError::ShapeMismatch {
                operation: "backward_with_seed",
                lhs: seed.shape().to_vec(),
                rhs: self.tensor.shape().to_vec(),
            }));
        }

        // Topological sort via DFS
        let mut visited = HashSet::new();
        let mut order: Vec<Arc<dyn BackwardNode<T, B>>> = Vec::new();

        fn dfs<T: Scalar, B: ComputeBackend + Default>(
            node: &Arc<dyn BackwardNode<T, B>>,
            visited: &mut HashSet<*const ()>,
            order: &mut Vec<Arc<dyn BackwardNode<T, B>>>,
        ) {
            let ptr = Arc::as_ptr(node) as *const ();
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);
            for input in node.inputs() {
                if let Some(ref cr) = input.creator {
                    dfs(cr, visited, order);
                }
            }
            order.push(node.clone());
        }

        if let Some(ref creator) = self.creator {
            dfs(creator, &mut visited, &mut order);
        }

        // Seed with the provided gradient
        if let Some(ref g) = self.grad {
            *g.write() = seed;
        }

        // Propagate in reverse topological order
        for node in order.into_iter().rev() {
            let out_grad = node.output_grad().read().clone();
            let input_grads: Vec<Option<Arc<GradBuffer<T, B>>>> =
                node.inputs().iter().map(|v| v.grad.clone()).collect();
            node.backward(&out_grad, &input_grads)?;
        }
        Ok(())
    }
}
