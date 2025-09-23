//! Differentiable operations for the autograd system

use super::TensorRef;
use coeus_dtype::Dtype;
use num_traits::{Float, Zero};

/// Trait for backward operations
pub trait BackwardOp<T: Dtype> {
    /// Compute gradients for inputs given output gradient
    #[allow(unused_variables)]
    fn backward(&self, inputs: &[&TensorRef<T>], output_grad: &TensorRef<T>) -> Vec<TensorRef<T>>;
}

/// Addition operation
pub struct AddOp;

impl<T: Dtype> BackwardOp<T> for AddOp {
    #[allow(unused_variables)]
    fn backward(&self, inputs: &[&TensorRef<T>], output_grad: &TensorRef<T>) -> Vec<TensorRef<T>> {
        // Gradient with respect to both inputs is the output gradient
        vec![output_grad.clone(), output_grad.clone()]
    }
}

/// Subtraction operation
pub struct SubOp;

impl<T: Dtype + std::ops::Neg<Output = T>> BackwardOp<T> for SubOp {
    #[allow(unused_variables)]
    fn backward(&self, inputs: &[&TensorRef<T>], output_grad: &TensorRef<T>) -> Vec<TensorRef<T>> {
        // Gradient w.r.t. first input: output_grad
        // Gradient w.r.t. second input: -output_grad
        vec![output_grad.clone(), output_grad.neg()]
    }
}

/// Multiplication operation
pub struct MulOp;

impl<T: Dtype> BackwardOp<T> for MulOp {
    fn backward(&self, inputs: &[&TensorRef<T>], output_grad: &TensorRef<T>) -> Vec<TensorRef<T>> {
        assert_eq!(inputs.len(), 2, "Multiplication requires 2 inputs");
        let a = &inputs[0];
        let b = &inputs[1];

        // Gradient w.r.t. first input: b * output_grad
        // Gradient w.r.t. second input: a * output_grad
        vec![b.mul(output_grad), a.mul(output_grad)]
    }
}

/// Division operation
pub struct DivOp;

impl<T: Dtype + std::ops::Neg<Output = T>> BackwardOp<T> for DivOp {
    fn backward(&self, inputs: &[&TensorRef<T>], output_grad: &TensorRef<T>) -> Vec<TensorRef<T>> {
        assert_eq!(inputs.len(), 2, "Division requires 2 inputs");
        let a = &inputs[0];
        let b = &inputs[1];

        // Gradient w.r.t. first input: output_grad / b
        // Gradient w.r.t. second input: -a * output_grad / (b^2)
        let grad_a = output_grad.div(b);
        let b_squared = b.mul(b);
        let neg_a_grad = a.mul(output_grad).div(&b_squared).neg();

        vec![grad_a, neg_a_grad]
    }
}

/// Negation operation
pub struct NegOp;

impl<T: Dtype + std::ops::Neg<Output = T>> BackwardOp<T> for NegOp {
    #[allow(unused_variables)]
    fn backward(&self, inputs: &[&TensorRef<T>], output_grad: &TensorRef<T>) -> Vec<TensorRef<T>> {
        // Gradient w.r.t. input: -output_grad
        vec![output_grad.neg()]
    }
}

/// Sum operation
pub struct SumOp;

impl<T: Dtype> BackwardOp<T> for SumOp {
    fn backward(&self, inputs: &[&TensorRef<T>], output_grad: &TensorRef<T>) -> Vec<TensorRef<T>> {
        assert_eq!(inputs.len(), 1, "Sum requires 1 input");
        let input = &inputs[0];

        // Gradient is output_grad broadcasted to input shape
        vec![TensorRef::from_data(
            vec![output_grad.as_scalar(); input.numel()],
            input.shape().to_vec(),
        )]
    }
}

/// Exponential operation
pub struct ExpOp;

impl<T: Dtype + Float> BackwardOp<T> for ExpOp {
    fn backward(&self, inputs: &[&TensorRef<T>], output_grad: &TensorRef<T>) -> Vec<TensorRef<T>> {
        assert_eq!(inputs.len(), 1, "Exp requires 1 input");
        let input = &inputs[0];

        // Gradient w.r.t. input: exp(input) * output_grad
        let exp_input = TensorRef::from_data(
            input.data().iter().map(|&x| x.exp()).collect(),
            input.shape().to_vec(),
        );
        vec![exp_input.mul(output_grad)]
    }
}

/// Natural logarithm operation
pub struct LogOp;

impl<T: Dtype + Float> BackwardOp<T> for LogOp {
    fn backward(&self, inputs: &[&TensorRef<T>], output_grad: &TensorRef<T>) -> Vec<TensorRef<T>> {
        assert_eq!(inputs.len(), 1, "Log requires 1 input");
        let input = &inputs[0];

        // Gradient w.r.t. input: output_grad / input
        vec![output_grad.div(input)]
    }
}

/// Power operation
pub struct PowOp<T: Dtype + Float> {
    pub exponent: T,
}

impl<T: Dtype + Float> PowOp<T> {
    pub fn new(exponent: T) -> Self {
        Self { exponent }
    }
}

impl<T: Dtype + Float> BackwardOp<T> for PowOp<T> {
    fn backward(&self, inputs: &[&TensorRef<T>], output_grad: &TensorRef<T>) -> Vec<TensorRef<T>> {
        assert_eq!(inputs.len(), 1, "Pow requires 1 input");
        let input = &inputs[0];

        // Gradient w.r.t. input: exponent * input^(exponent-1) * output_grad
        let exp_minus_one = self.exponent - T::one();
        let input_pow = input
            .data()
            .iter()
            .map(|&x| x.powf(exp_minus_one))
            .collect::<Vec<_>>();
        let input_pow_tensor = TensorRef::from_data(input_pow, input.shape().to_vec());

        vec![input_pow_tensor
            .mul(&TensorRef::from_data(
                vec![self.exponent; input.numel()],
                input.shape().to_vec(),
            ))
            .mul(output_grad)]
    }
}

/// Sine operation
pub struct SinOp;

impl<T: Dtype + Float> BackwardOp<T> for SinOp {
    fn backward(&self, inputs: &[&TensorRef<T>], output_grad: &TensorRef<T>) -> Vec<TensorRef<T>> {
        assert_eq!(inputs.len(), 1, "Sin requires 1 input");
        let input = &inputs[0];

        // Gradient w.r.t. input: cos(input) * output_grad
        let cos_vals = input.data().iter().map(|&x| x.cos()).collect::<Vec<_>>();
        let cos_tensor = TensorRef::from_data(cos_vals, input.shape().to_vec());

        vec![cos_tensor.mul(output_grad)]
    }
}

/// Cosine operation
pub struct CosOp;

impl<T: Dtype + Float> BackwardOp<T> for CosOp {
    fn backward(&self, inputs: &[&TensorRef<T>], output_grad: &TensorRef<T>) -> Vec<TensorRef<T>> {
        assert_eq!(inputs.len(), 1, "Cos requires 1 input");
        let input = &inputs[0];

        // Gradient w.r.t. input: -sin(input) * output_grad
        let sin_vals = input.data().iter().map(|&x| x.sin()).collect::<Vec<_>>();
        let sin_tensor = TensorRef::from_data(sin_vals, input.shape().to_vec());

        vec![sin_tensor.neg().mul(output_grad)]
    }
}

/// ReLU activation
pub struct ReluOp;

impl<T: Dtype + PartialOrd + Zero> BackwardOp<T> for ReluOp {
    fn backward(&self, inputs: &[&TensorRef<T>], output_grad: &TensorRef<T>) -> Vec<TensorRef<T>> {
        assert_eq!(inputs.len(), 1, "ReLU requires 1 input");
        let input = &inputs[0];

        // Gradient w.r.t. input: 1 if input > 0, 0 otherwise
        let grad_data = input
            .data()
            .iter()
            .map(|&x| if x > T::zero() { T::one() } else { T::zero() })
            .collect::<Vec<_>>();
        let grad_tensor = TensorRef::from_data(grad_data, input.shape().to_vec());

        vec![grad_tensor.mul(output_grad)]
    }
}

/// Matrix multiplication operation
pub struct MatMulOp;

impl<T: Dtype> BackwardOp<T> for MatMulOp {
    fn backward(&self, inputs: &[&TensorRef<T>], output_grad: &TensorRef<T>) -> Vec<TensorRef<T>> {
        assert_eq!(inputs.len(), 2, "MatMul requires 2 inputs");
        let a = &inputs[0];
        let b = &inputs[1];

        // For C = A @ B, gradients are:
        // dA = output_grad @ B^T
        // dB = A^T @ output_grad
        let grad_a = output_grad.mul(&b.t());
        let grad_b = a.t().mul(output_grad);

        vec![grad_a, grad_b]
    }
}

/// Utility function to create an operation with backward function
pub fn create_operation<T: Dtype, F>(
    _name: impl Into<String>,
    backward_fn: F,
) -> Box<dyn BackwardOp<T>>
where
    F: Fn(&[&TensorRef<T>], &TensorRef<T>) -> Vec<TensorRef<T>> + 'static,
{
    struct CustomOp<F> {
        backward_fn: F,
    }

    impl<T: Dtype, F> BackwardOp<T> for CustomOp<F>
    where
        F: Fn(&[&TensorRef<T>], &TensorRef<T>) -> Vec<TensorRef<T>>,
    {
        fn backward(
            &self,
            inputs: &[&TensorRef<T>],
            output_grad: &TensorRef<T>,
        ) -> Vec<TensorRef<T>> {
            (self.backward_fn)(inputs, output_grad)
        }
    }

    Box::new(CustomOp { backward_fn })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_add_backward() {
        let op = AddOp;
        let a = TensorRef::scalar(2.0f32);
        let b = TensorRef::scalar(3.0f32);
        let output_grad = TensorRef::scalar(1.0f32);

        let grads = op.backward(&[&a, &b], &output_grad);
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].as_scalar(), 1.0);
        assert_eq!(grads[1].as_scalar(), 1.0);
    }

    #[test]
    fn test_mul_backward() {
        let op = MulOp;
        let a = TensorRef::scalar(2.0f32);
        let b = TensorRef::scalar(3.0f32);
        let output_grad = TensorRef::scalar(1.0f32);

        let grads = op.backward(&[&a, &b], &output_grad);
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].as_scalar(), 3.0); // b * output_grad
        assert_eq!(grads[1].as_scalar(), 2.0); // a * output_grad
    }

    #[test]
    fn test_exp_backward() {
        let op = ExpOp;
        let input = TensorRef::scalar(1.0f32);
        let output_grad = TensorRef::scalar(2.0f32);

        let grads = op.backward(&[&input], &output_grad);
        assert_eq!(grads.len(), 1);
        assert_relative_eq!(grads[0].as_scalar(), 5.4365635, epsilon = 1e-6); // e^1 * 2
    }
}
