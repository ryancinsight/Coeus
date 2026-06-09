use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::{Arc, Mutex};

/// A simple, fast, deterministic pseudo-random number generator (Xorshift64).
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create a new generator with a seed. Seed must be non-zero.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1337 } else { seed },
        }
    }

    /// Draw next u64 value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Draw a float in [0.0, 1.0).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

pub struct DropoutNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub mask: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for DropoutNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "dropout"
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
            let prod = coeus_ops::mul(grad_out, &self.mask, &backend);
            let mut gl = g.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &prod, &backend);
        }
    }
}

/// Tracked Dropout.
pub fn dropout<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    p: f64,
    is_training: bool,
    seed: u64,
) -> Var<T, B> {
    if !is_training || p == 0.0 {
        return input.clone();
    }

    let scale = 1.0 / (1.0 - p);
    let rng = std::cell::RefCell::new(Xorshift64::new(seed));
    let shape = input.tensor.shape_cloned();

    let cpu_backend = coeus_core::MoiraiBackend::new();
    let mask_cpu =
        Tensor::<T, coeus_core::MoiraiBackend>::from_fn_on(shape.clone(), &cpu_backend, |_| {
            let r = rng.borrow_mut().next_f64();
            if r < p {
                T::zero()
            } else {
                T::from_f64(scale)
            }
        });

    let target_backend = B::default();
    let mask = mask_cpu.to_backend_on(&cpu_backend, &target_backend);
    let out_tensor = coeus_ops::mul(&input.tensor, &mask, &target_backend);

    let requires_grad = input.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(
            shape.clone(),
            &target_backend,
        ))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![input.clone()];
        let node = DropoutNode {
            output_grad,
            inputs,
            mask,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
