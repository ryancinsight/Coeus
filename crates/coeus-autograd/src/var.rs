// ── Differentiable variable ──

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::autodiff_cache::ComputeGraphCache;
use coeus_core::{ComputeBackend, MoiraiBackend, Scalar};
use coeus_tensor::Tensor;
use std::collections::HashSet;
use std::sync::Arc;
use std::cell::RefCell;

thread_local! {
    static BACKWARD_CACHE: RefCell<ComputeGraphCache> = RefCell::new(ComputeGraphCache::new());
}

/// Get the thread-local backward pass cache for autodiff compilation.
pub fn get_backward_cache() -> ComputeGraphCache {
    BACKWARD_CACHE.with(|cache| cache.borrow().clone())
}

/// Reset the thread-local backward pass cache statistics.
pub fn reset_backward_cache_stats() {
    BACKWARD_CACHE.with(|cache| cache.borrow().reset_stats());
}

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
/// let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([2], &[3.0, 4.0]), true);
/// let y = coeus_autograd::mul(&x, &x); // y = x * x
/// y.backward().expect("invariant: valid autograd fixture completes backward");
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
    /// let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([2], &[1.0, 2.0]), true);
    /// let c = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([2], &[10.0, 20.0]), false);
    /// let y = coeus_autograd::add(&x, &c);
    /// y.backward().expect("invariant: valid autograd fixture completes backward");
    /// assert!(x.grad().is_some()); // tracked leaf: gradient present
    /// assert!(c.grad().is_none()); // constant leaf: no gradient state
    /// ```
    #[inline]
    pub fn new(tensor: Tensor<T, B>, requires_grad: bool) -> Self {
        let grad = if requires_grad {
            Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
                tensor.shape(),
                &B::default(),
            ))))
        } else {
            None
        };
        Self {
            tensor,
            grad,
            creator: None,
        }
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
    /// let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([3], &[1.0, 2.0, 3.0]), true);
    /// let one = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([3], &[1.0; 3]), false);
    /// let y = coeus_autograd::add(&x, &one);
    /// let loss = coeus_autograd::sum(&y);
    /// loss.backward().expect("invariant: valid autograd fixture completes backward");
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
    /// let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([2], &[2.0, 3.0]), true);
    /// let y = coeus_autograd::sum(&x);
    ///
    /// y.backward().expect("invariant: valid autograd fixture completes backward");
    /// let g1 = x.grad().unwrap();
    /// assert!((g1.as_slice()[0] - 1.0).abs() < 1e-5);
    ///
    /// // Second backward would accumulate; zero first.
    /// x.zero_grad();
    /// assert!((x.grad().unwrap().as_slice()[0] - 0.0).abs() < 1e-5);
    /// ```
    #[inline]
    pub fn zero_grad(&self) {
        if let Some(ref g) = self.grad {
            B::default().fill(g.write().storage_mut(), T::zero());
        }
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
    /// let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([2], &[3.0, 4.0]), true);
    /// let y = coeus_autograd::mul(&x, &x);
    /// let loss = coeus_autograd::sum(&y); // scalar: y_0 + y_1
    /// loss.backward().expect("invariant: valid autograd fixture completes backward");
    /// let grad = x.grad().unwrap();
    /// assert!((grad.as_slice()[0] - 6.0).abs() < 1e-5); // 2 * 3
    /// assert!((grad.as_slice()[1] - 8.0).abs() < 1e-5); // 2 * 4
    /// ```
    #[inline]
    ///
    /// # Errors
    ///
    /// Returns the backend error when gradient computation or accumulation
    /// cannot complete.
    pub fn backward(&self) -> Result<(), B::Error> {
        let seed = Tensor::ones_on(self.tensor.shape(), &B::default());
        self.backward_with_seed(seed)
    }

    /// Run reverse-mode autodiff from this variable, seeding with the given gradient.
    ///
    /// Sets `self.grad = seed` and propagates backwards through the computation graph.
    /// This enables testing with non-uniform upstream gradients (e.g. for softmax,
    /// where a uniform seed produces zero input gradient due to Jacobian row-sums).
    ///
    /// # Errors
    ///
    /// Returns the backend error when gradient computation or accumulation
    /// cannot complete.
    ///
    /// # Panics
    /// If `seed.shape()` does not match `self.tensor.shape()`.
    #[inline]
    pub fn backward_with_seed(&self, seed: Tensor<T, B>) -> Result<(), B::Error> {
        assert_eq!(
            seed.shape(),
            self.tensor.shape(),
            "backward_with_seed: seed shape {:?} must match tensor shape {:?}",
            seed.shape(),
            self.tensor.shape()
        );

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

#[cfg(test)]
mod tests {
    use super::Var;
    use crate::{BackwardNode, GradBuffer};
    use coeus_core::{BackendError, MoiraiBackend};
    use coeus_tensor::Tensor;
    use std::sync::Arc;

    struct FailingNode {
        output_grad: Arc<GradBuffer<f32, MoiraiBackend>>,
        inputs: Vec<Var<f32, MoiraiBackend>>,
    }

    impl BackwardNode<f32, MoiraiBackend> for FailingNode {
        fn op_name(&self) -> &'static str {
            "failing_test_node"
        }

        fn output_grad(&self) -> &Arc<GradBuffer<f32, MoiraiBackend>> {
            &self.output_grad
        }

        fn inputs(&self) -> &[Var<f32, MoiraiBackend>] {
            &self.inputs
        }

        fn backward(
            &self,
            _grad_out: &Tensor<f32, MoiraiBackend>,
            _input_grads: &[Option<Arc<GradBuffer<f32, MoiraiBackend>>>],
        ) -> Result<(), BackendError> {
            Err(BackendError::Storage {
                operation: "failing_test_node",
                reason: "injected gradient accumulation failure".to_owned(),
            })
        }
    }

    #[test]
    fn backward_returns_the_exact_node_error() {
        let tensor = Tensor::from_slice([1], &[1.0]);
        let output_grad = Arc::new(GradBuffer::new(Tensor::zeros([1])));
        let node = Arc::new(FailingNode {
            output_grad: Arc::clone(&output_grad),
            inputs: Vec::new(),
        });
        let output = Var::with_creator(tensor, Some(output_grad), node);

        assert_eq!(
            output.backward(),
            Err(BackendError::Storage {
                operation: "failing_test_node",
                reason: "injected gradient accumulation failure".to_owned(),
            })
        );
    }
}
