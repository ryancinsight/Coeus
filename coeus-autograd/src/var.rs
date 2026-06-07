// ── Differentiable variable ──

use std::sync::{Arc, Mutex};
use std::collections::HashSet;
use coeus_core::{Scalar, ComputeBackend, MoiraiBackend};
use coeus_tensor::Tensor;
use crate::node::BackwardNode;

/// A differentiable variable wrapping a tensor and its gradient accumulator.
#[derive(Clone)]
pub struct Var<T: Scalar, B: ComputeBackend + Default = MoiraiBackend> {
    /// The tensor value.
    pub tensor: Tensor<T, B>,
    /// Accumulated gradient (None if not tracking).
    pub grad: Option<Arc<Mutex<Tensor<T, B>>>>,
    /// The operation that created this variable (None for leaf nodes).
    pub creator: Option<Arc<dyn BackwardNode<T, B>>>,
}

impl<T: Scalar, B: ComputeBackend + Default> Var<T, B> {
    /// Create a new leaf variable.
    #[inline]
    pub fn new(tensor: Tensor<T, B>, requires_grad: bool) -> Self {
        let grad = if requires_grad {
            Some(Arc::new(Mutex::new(Tensor::zeros_on(tensor.shape(), &B::default()))))
        } else {
            None
        };
        Self { tensor, grad, creator: None }
    }

    /// Create an intermediate variable (result of an op).
    #[inline]
    pub fn with_creator(
        tensor: Tensor<T, B>,
        grad: Option<Arc<Mutex<Tensor<T, B>>>>,
        creator: Arc<dyn BackwardNode<T, B>>,
    ) -> Self {
        Self { tensor, grad, creator: Some(creator) }
    }

    /// Read the accumulated gradient.
    #[inline]
    pub fn grad(&self) -> Option<Tensor<T, B>> {
        self.grad.as_ref().map(|g| g.lock().unwrap().clone())
    }

    /// Set the gradient tensor.
    #[inline]
    pub fn set_grad(&self, grad: Tensor<T, B>) {
        if let Some(ref g) = self.grad {
            *g.lock().unwrap() = grad;
        }
    }

    /// Zero the accumulated gradient.
    #[inline]
    pub fn zero_grad(&self) {
        if let Some(ref g) = self.grad {
            let mut grad_ref = g.lock().unwrap();
            B::default().fill(grad_ref.storage_mut(), T::zero());
        }
    }

    /// Run reverse-mode autodiff from this variable, seeding with `Tensor::ones`.
    ///
    /// Equivalent to `self.backward_with_seed(Tensor::ones(self.tensor.shape().to_vec()))`.
    #[inline]
    pub fn backward(&self) {
        let seed = Tensor::ones_on(self.tensor.shape(), &B::default());
        self.backward_with_seed(seed);
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
    pub fn backward_with_seed(&self, seed: Tensor<T, B>) {
        assert_eq!(seed.shape(), self.tensor.shape(),
            "backward_with_seed: seed shape {:?} must match tensor shape {:?}",
            seed.shape(), self.tensor.shape());

        // Topological sort via DFS
        let mut visited = HashSet::new();
        let mut order: Vec<Arc<dyn BackwardNode<T, B>>> = Vec::new();

        fn dfs<T: Scalar, B: ComputeBackend + Default>(
            node: &Arc<dyn BackwardNode<T, B>>,
            visited: &mut HashSet<*const ()>,
            order: &mut Vec<Arc<dyn BackwardNode<T, B>>>,
        ) {
            let ptr = Arc::as_ptr(node) as *const ();
            if visited.contains(&ptr) { return; }
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
            *g.lock().unwrap() = seed;
        }

        // Propagate in reverse topological order
        for node in order.into_iter().rev() {
            let out_grad = node.output_grad().lock().unwrap().clone();
            let input_grads: Vec<Option<Arc<Mutex<Tensor<T, B>>>>> =
                node.inputs().iter().map(|v| v.grad.clone()).collect();
            node.backward(&out_grad, &input_grads);
        }
    }
}
