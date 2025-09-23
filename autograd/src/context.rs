//! Thread-local autograd context for tensor operations
//!
//! This context manages the computational graph and gradient computation
//! for automatic differentiation. It uses thread-local storage to ensure
//! thread safety while maintaining efficiency.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Comprehensive autograd context for tensor operations
pub struct AutogradContext {
    /// Next node ID to assign
    pub next_node_id: AtomicU64,
    /// Map of node ID to operation and inputs
    pub nodes: HashMap<u64, Node>,
    /// Map of node ID to tensor data for gradient computation
    pub tensor_data: HashMap<u64, Vec<f64>>,
    /// Map of node ID to tensor shape for gradient computation
    pub tensor_shapes: HashMap<u64, Vec<usize>>,
    /// Computed gradients for each node
    pub gradients: HashMap<u64, Vec<f64>>,
}

#[derive(Clone)]
pub struct Node {
    pub operation: Operation,
    pub inputs: Vec<u64>,
    /// Output node ID for this operation
    pub output: u64,
}

#[derive(Clone, Debug)]
pub enum Operation {
    Leaf, // Leaf nodes with no operation
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Abs,
    Sum, // Reduction operations
    SumDim,
    MeanDim,
    // Implemented unary operations
    Relu,
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Sin,
    Cos,
    Sqrt,
    // Matrix operations
    Matmul,
    Transpose,
    // Shape operations
    Reshape,
    Unsqueeze,
    Expand,
    // New mathematical operations
    Ceil,
    Floor,
    Round,
    Trunc,
    Square,
    Reciprocal,
    Sign,
    Tan,
    Clamp,
    Acosh,
    Asinh,
    Atanh,
    Erfc,
    Expm1,
    Fix,
    Fmod,
    Frac,
    Remainder,
    Log1p,
    NanToNum,
    Sgn,
    Signbit,
    Xlogy,
    // Missing operations that were causing compilation errors
    Acos,
    Atan,
    Erf,
    Exp2,
    Log10,
    Log2,
    Rsqrt,
    // Activation functions
    Elu,
    Gelu,
    Hardtanh,
    Logsigmoid,
    // Future operations
    Pow(f64), // Base^exponent, stores the exponent
}

impl AutogradContext {
    pub fn new() -> Self {
        Self {
            next_node_id: AtomicU64::new(1),
            nodes: HashMap::new(),
            tensor_data: HashMap::new(),
            tensor_shapes: HashMap::new(),
            gradients: HashMap::new(),
        }
    }
}

impl Default for AutogradContext {
    fn default() -> Self {
        Self {
            next_node_id: AtomicU64::new(0),
            nodes: HashMap::new(),
            tensor_data: HashMap::new(),
            tensor_shapes: HashMap::new(),
            gradients: HashMap::new(),
        }
    }
}

impl AutogradContext {
    pub fn create_node(&mut self, operation: Operation, inputs: Vec<u64>) -> u64 {
        let node_id = self.next_node_id.fetch_add(1, Ordering::SeqCst);
        let node = Node {
            operation,
            inputs,
            output: node_id,
        };
        self.nodes.insert(node_id, node);
        node_id
    }

    /// Create a leaf node (no operation, just data storage)
    pub fn create_leaf_node(&mut self) -> u64 {
        let node_id = self.next_node_id.fetch_add(1, Ordering::SeqCst);
        // Leaf nodes have no operation and no inputs
        let node = Node {
            operation: Operation::Leaf,
            inputs: vec![],
            output: node_id,
        };
        self.nodes.insert(node_id, node);
        node_id
    }

    /// Register tensor data for gradient computation
    pub fn register_tensor(&mut self, node_id: u64, data: Vec<f64>, shape: Vec<usize>) {
        self.tensor_data.insert(node_id, data);
        self.tensor_shapes.insert(node_id, shape);
    }

    /// Get tensor data for gradient computation
    pub fn get_tensor_data(&self, node_id: u64) -> Option<&Vec<f64>> {
        self.tensor_data.get(&node_id)
    }

    /// Store computed gradient for a node with numerical stability checks
    pub fn set_gradient(&mut self, node_id: u64, mut gradient: Vec<f64>) {
        use crate::numerical_stability::NumericalStability;
        // Sanitize gradients to prevent numerical instability
        NumericalStability::sanitize_gradients(&mut gradient);
        self.gradients.insert(node_id, gradient);
    }

    /// Get computed gradient for a node
    pub fn get_gradient(&self, node_id: u64) -> Option<&Vec<f64>> {
        self.gradients.get(&node_id)
    }

    /// Perform topological sort of the computational graph
    pub fn topological_sort(&self, start_node: u64) -> Vec<u64> {
        let mut visited = std::collections::HashSet::new();
        let mut order = Vec::new();

        fn dfs(
            node_id: u64,
            context: &AutogradContext,
            visited: &mut std::collections::HashSet<u64>,
            order: &mut Vec<u64>,
        ) {
            if visited.contains(&node_id) {
                return;
            }
            visited.insert(node_id);

            if let Some(node) = context.nodes.get(&node_id) {
                for &input_id in &node.inputs {
                    dfs(input_id, context, visited, order);
                }
            }
            order.push(node_id);
        }

        dfs(start_node, self, &mut visited, &mut order);
        order
    }

    /// Compute gradients using reverse-mode automatic differentiation
    pub fn backward(&mut self, output_node: u64, output_grad: &[f64]) {
        // Initialize gradient for output node
        self.set_gradient(output_node, output_grad.to_vec());

        // Get topological order for gradient computation
        let order = self.topological_sort(output_node);

        // Collect node IDs that have both node and gradient data
        let valid_node_ids: Vec<u64> = order.iter().rev()
            .filter(|&&node_id| {
                self.nodes.contains_key(&node_id) && self.gradients.contains_key(&node_id)
            })
            .cloned()
            .collect();

        // Process each valid node
        for node_id in valid_node_ids {
            // Get the node and gradient data
            let node = self.nodes.get(&node_id).unwrap().clone();
            let grad = self.gradients.get(&node_id).unwrap().clone();

            self.backward_operation(&node, &grad);
        }
    }

    fn backward_operation(&mut self, node: &Node, output_grad: &[f64]) {
        match &node.operation {
            Operation::Add => self.backward_add(&node.inputs, output_grad),
            Operation::Sub => self.backward_sub(&node.inputs, output_grad),
            Operation::Mul => self.backward_mul(&node.inputs, output_grad),
            Operation::Div => self.backward_div(&node.inputs, output_grad),
            Operation::Neg => self.backward_neg(&node.inputs, output_grad),
            Operation::Abs => self.backward_abs(&node.inputs, output_grad),
            Operation::Sum => self.backward_sum(&node.inputs, output_grad),
            Operation::SumDim => self.backward_sum_dim(&node.inputs, output_grad),
            Operation::MeanDim => self.backward_mean_dim(&node.inputs, output_grad),
            Operation::Relu => self.backward_unary(&node.inputs, output_grad, Self::relu_derivative),
            Operation::Sigmoid => self.backward_unary(&node.inputs, output_grad, Self::sigmoid_derivative),
            Operation::Tanh => self.backward_unary(&node.inputs, output_grad, Self::tanh_derivative),
            Operation::Exp => self.backward_unary(&node.inputs, output_grad, Self::exp_derivative),
            Operation::Log => self.backward_unary(&node.inputs, output_grad, Self::log_derivative),
            Operation::Sin => self.backward_unary(&node.inputs, output_grad, Self::sin_derivative),
            Operation::Cos => self.backward_unary(&node.inputs, output_grad, Self::cos_derivative),
            Operation::Sqrt => self.backward_unary(&node.inputs, output_grad, Self::sqrt_derivative),
            Operation::Matmul => self.backward_matmul(&node.inputs, output_grad),
            Operation::Transpose => self.backward_transpose(&node.inputs, output_grad),
            Operation::Reshape => self.backward_reshape(&node.inputs, output_grad),
            Operation::Unsqueeze => self.backward_unsqueeze(&node.inputs, output_grad),
            Operation::Expand => self.backward_expand(&node.inputs, output_grad),
            Operation::Ceil => self.backward_unary(&node.inputs, output_grad, Self::ceil_derivative),
            Operation::Floor => self.backward_unary(&node.inputs, output_grad, Self::floor_derivative),
            Operation::Round => self.backward_unary(&node.inputs, output_grad, Self::round_derivative),
            Operation::Trunc => self.backward_unary(&node.inputs, output_grad, Self::trunc_derivative),
            Operation::Square => self.backward_unary(&node.inputs, output_grad, Self::square_derivative),
            Operation::Reciprocal => self.backward_unary(&node.inputs, output_grad, Self::reciprocal_derivative),
            Operation::Sign => self.backward_unary(&node.inputs, output_grad, Self::sign_derivative),
            Operation::Tan => self.backward_unary(&node.inputs, output_grad, Self::tan_derivative),
            Operation::Clamp => self.backward_clamp(&node.inputs, output_grad),
            Operation::Acosh => self.backward_unary(&node.inputs, output_grad, Self::acosh_derivative),
            Operation::Asinh => self.backward_unary(&node.inputs, output_grad, Self::asinh_derivative),
            Operation::Atanh => self.backward_unary(&node.inputs, output_grad, Self::atanh_derivative),
            Operation::Erfc => self.backward_unary(&node.inputs, output_grad, Self::erfc_derivative),
            Operation::Expm1 => self.backward_unary(&node.inputs, output_grad, Self::expm1_derivative),
            Operation::Fix => self.backward_unary(&node.inputs, output_grad, Self::fix_derivative),
            Operation::Fmod => self.backward_fmod(&node.inputs, output_grad),
            Operation::Frac => self.backward_unary(&node.inputs, output_grad, Self::frac_derivative),
            Operation::Remainder => self.backward_remainder(&node.inputs, output_grad),
            Operation::Log1p => self.backward_unary(&node.inputs, output_grad, Self::log1p_derivative),
            Operation::NanToNum => self.backward_nan_to_num(&node.inputs, output_grad),
            Operation::Sgn => self.backward_unary(&node.inputs, output_grad, Self::sgn_derivative),
            Operation::Signbit => self.backward_unary(&node.inputs, output_grad, Self::signbit_derivative),
            Operation::Xlogy => self.backward_xlogy(&node.inputs, output_grad),
            Operation::Acos => self.backward_unary(&node.inputs, output_grad, Self::acos_derivative),
            Operation::Atan => self.backward_unary(&node.inputs, output_grad, Self::atan_derivative),
            Operation::Erf => self.backward_unary(&node.inputs, output_grad, Self::erf_derivative),
            Operation::Exp2 => self.backward_unary(&node.inputs, output_grad, Self::exp2_derivative),
            Operation::Log10 => self.backward_unary(&node.inputs, output_grad, Self::log10_derivative),
            Operation::Log2 => self.backward_unary(&node.inputs, output_grad, Self::log2_derivative),
            Operation::Rsqrt => self.backward_unary(&node.inputs, output_grad, Self::rsqrt_derivative),
            Operation::Elu => self.backward_elu(&node.inputs, output_grad),
            Operation::Gelu => self.backward_gelu(&node.inputs, output_grad),
            Operation::Hardtanh => self.backward_hardtanh(&node.inputs, output_grad),
            Operation::Logsigmoid => self.backward_logsigmoid(&node.inputs, output_grad),
            Operation::Pow(exponent) => self.backward_pow(&node.inputs, output_grad, *exponent),
            Operation::Leaf => {} // Leaf nodes have no backward operation
        }
    }

    fn backward_add(&mut self, inputs: &[u64], output_grad: &[f64]) {
        for &input_id in inputs {
            if input_id != 0 {
                let existing_grad = self
                    .get_gradient(input_id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; output_grad.len()]);
                let new_grad = existing_grad
                    .iter()
                    .zip(output_grad.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                self.set_gradient(input_id, new_grad);
            }
        }
    }

    fn backward_sub(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if inputs.len() >= 2 {
            // Gradient w.r.t. left input: output_grad
            if inputs[0] != 0 {
                let existing_grad = self
                    .get_gradient(inputs[0])
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; output_grad.len()]);
                let new_grad = existing_grad
                    .iter()
                    .zip(output_grad.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                self.set_gradient(inputs[0], new_grad);
            }

            // Gradient w.r.t. right input: -output_grad
            if inputs[1] != 0 {
                let existing_grad = self
                    .get_gradient(inputs[1])
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; output_grad.len()]);
                let new_grad = existing_grad
                    .iter()
                    .zip(output_grad.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                self.set_gradient(inputs[1], new_grad);
            }
        }
    }

    fn backward_mul(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if inputs.len() >= 2 {
            // Get tensor data and shapes for both inputs
            let left_data = self.get_tensor_data(inputs[0]).cloned();
            let right_data = self.get_tensor_data(inputs[1]).cloned();
            let left_shape = self.tensor_shapes.get(&inputs[0]).cloned();
            let right_shape = self.tensor_shapes.get(&inputs[1]).cloned();

            if let (Some(left_data), Some(right_data), Some(_left_shape), Some(_right_shape)) =
                (left_data, right_data, left_shape, right_shape)
            {
                // Handle broadcasting: determine which input was broadcasted
                let left_len = left_data.len();
                let right_len = right_data.len();
                let output_len = output_grad.len();

                // Gradient w.r.t. left input: output_grad * right (with broadcasting)
                if inputs[0] != 0 {
                    let mut grad_left = vec![0.0; left_len];
                    if left_len == output_len {
                        // No broadcasting for left input
                        for i in 0..output_len {
                            let right_idx = if right_len == 1 { 0 } else { i };
                            grad_left[i] = output_grad[i] * right_data[right_idx];
                        }
                    } else {
                        // Broadcasting: accumulate gradients
                        for i in 0..output_len {
                            let left_idx = i % left_len;
                            grad_left[left_idx] += output_grad[i] * right_data[i % right_len];
                        }
                    }
                    let existing_grad = self
                        .get_gradient(inputs[0])
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad_left.len()]);
                    let new_grad = grad_left
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(inputs[0], new_grad);
                }

                // Gradient w.r.t. right input: output_grad * left (with broadcasting)
                if inputs[1] != 0 {
                    let mut grad_right = vec![0.0; right_len];
                    if right_len == output_len {
                        // No broadcasting for right input
                        for i in 0..output_len {
                            let left_idx = if left_len == 1 { 0 } else { i };
                            grad_right[i] = output_grad[i] * left_data[left_idx];
                        }
                    } else {
                        // Broadcasting: accumulate gradients
                        for i in 0..output_len {
                            let right_idx = i % right_len;
                            grad_right[right_idx] += output_grad[i] * left_data[i % left_len];
                        }
                    }
                    let existing_grad = self
                        .get_gradient(inputs[1])
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad_right.len()]);
                    let new_grad = grad_right
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(inputs[1], new_grad);
                }
            }
        }
    }

    fn backward_div(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if inputs.len() >= 2 {
            // Get tensor data for both inputs
            let left_data = self.get_tensor_data(inputs[0]).cloned();
            let right_data = self.get_tensor_data(inputs[1]).cloned();

            if let (Some(left_data), Some(right_data)) = (left_data, right_data) {
                // Gradient w.r.t. left input: output_grad / right
                if inputs[0] != 0 {
                    let grad_left: Vec<f64> = output_grad
                        .iter()
                        .zip(right_data.iter())
                        .map(|(og, r)| {
                            use crate::numerical_stability::NumericalStability;
                            NumericalStability::safe_divide(*og, *r)
                        })
                        .collect();
                    let existing_grad = self
                        .get_gradient(inputs[0])
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad_left.len()]);
                    let new_grad = grad_left
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(inputs[0], new_grad);
                }

                // Gradient w.r.t. right input: -output_grad * left / right^2
                if inputs[1] != 0 {
                    let grad_right: Vec<f64> = output_grad
                        .iter()
                        .zip(left_data.iter())
                        .zip(right_data.iter())
                        .map(|((og, l), r)| {
                            use crate::numerical_stability::NumericalStability;
                            let r_squared = r * r;
                            -NumericalStability::safe_divide(*og * *l, r_squared)
                        })
                        .collect();
                    let existing_grad = self
                        .get_gradient(inputs[1])
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad_right.len()]);
                    let new_grad = grad_right
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(inputs[1], new_grad);
                }
            }
        }
    }

    fn backward_neg(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if let Some(&input_id) = inputs.first() {
            if input_id != 0 {
                let existing_grad = self
                    .get_gradient(input_id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; output_grad.len()]);
                let new_grad = existing_grad
                    .iter()
                    .zip(output_grad.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                self.set_gradient(input_id, new_grad);
            }
        }
    }

    fn backward_abs(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if let Some(&input_id) = inputs.first() {
            if input_id != 0 {
                if let Some(input_data) = self.get_tensor_data(input_id) {
                    let grad: Vec<f64> = output_grad
                        .iter()
                        .zip(input_data.iter())
                        .map(|(og, x)| if *x >= 0.0 { *og } else { -*og })
                        .collect();
                    let existing_grad = self
                        .get_gradient(input_id)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad.len()]);
                    let new_grad = grad
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(input_id, new_grad);
                }
            }
        }
    }

    fn backward_sum(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if let Some(&input_id) = inputs.first() {
            if input_id != 0 {
                // Sum reduction: broadcast gradient to all input elements
                if let Some(input_shape) = self.tensor_shapes.get(&input_id) {
                    let input_len = input_shape.iter().product();
                    let grad_value = output_grad[0]; // Sum reduces to scalar
                    let grad = vec![grad_value; input_len];
                    let existing_grad = self
                        .get_gradient(input_id)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad.len()]);
                    let new_grad = grad
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(input_id, new_grad);
                }
            }
        }
    }

    fn backward_sum_dim(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if inputs.len() >= 2 {
            if let Some(&input_id) = inputs.first() {
                if input_id != 0 {
                    // Sum along dimension: broadcast gradient back to input shape
                    if let Some(input_shape) = self.tensor_shapes.get(&input_id) {
                        let input_len = input_shape.iter().product();
                        let grad = vec![output_grad[0]; input_len]; // Simplified broadcasting
                        let existing_grad = self
                            .get_gradient(input_id)
                            .cloned()
                            .unwrap_or_else(|| vec![0.0; grad.len()]);
                        let new_grad = grad
                            .iter()
                            .zip(existing_grad.iter())
                            .map(|(a, b)| a + b)
                            .collect();
                        self.set_gradient(input_id, new_grad);
                    }
                }
            }
        }
    }

    fn backward_mean_dim(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if inputs.len() >= 2 {
            if let Some(&input_id) = inputs.first() {
                if input_id != 0 {
                    // Mean along dimension: broadcast gradient back to input shape
                    if let Some(input_shape) = self.tensor_shapes.get(&input_id) {
                        let input_len = input_shape.iter().product();
                        let grad_value = output_grad[0]; // Mean reduces to scalar
                        let grad = vec![grad_value; input_len]; // Simplified broadcasting
                        let existing_grad = self
                            .get_gradient(input_id)
                            .cloned()
                            .unwrap_or_else(|| vec![0.0; grad.len()]);
                        let new_grad = grad
                            .iter()
                            .zip(existing_grad.iter())
                            .map(|(a, b)| a + b)
                            .collect();
                        self.set_gradient(input_id, new_grad);
                    }
                }
            }
        }
    }

    fn backward_unary<F>(&mut self, inputs: &[u64], output_grad: &[f64], derivative_fn: F)
    where
        F: Fn(f64) -> f64,
    {
        if let Some(&input_id) = inputs.first() {
            if input_id != 0 {
                if let Some(input_data) = self.get_tensor_data(input_id) {
                    let grad: Vec<f64> = output_grad
                        .iter()
                        .zip(input_data.iter())
                        .map(|(og, x)| og * derivative_fn(*x))
                        .collect();
                    let existing_grad = self
                        .get_gradient(input_id)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad.len()]);
                    let new_grad = grad
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(input_id, new_grad);
                }
            }
        }
    }

    fn relu_derivative(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }

    fn sigmoid_derivative(x: f64) -> f64 {
        let sig = 1.0 / (1.0 + (-x).exp());
        sig * (1.0 - sig)
    }

    fn tanh_derivative(x: f64) -> f64 {
        let t = x.tanh();
        1.0 - t * t
    }

    fn exp_derivative(x: f64) -> f64 {
        x.exp()
    }

    fn log_derivative(x: f64) -> f64 {
        use crate::numerical_stability::NumericalStability;
        NumericalStability::safe_divide(1.0, x)
    }

    fn sin_derivative(x: f64) -> f64 {
        x.cos()
    }

    fn cos_derivative(x: f64) -> f64 {
        -x.sin()
    }

    fn sqrt_derivative(x: f64) -> f64 {
        use crate::numerical_stability::NumericalStability;
        NumericalStability::safe_divide(0.5, x.sqrt())
    }

    fn ceil_derivative(_x: f64) -> f64 {
        0.0 // Ceil is not differentiable
    }

    fn floor_derivative(_x: f64) -> f64 {
        0.0 // Floor is not differentiable
    }

    fn round_derivative(_x: f64) -> f64 {
        0.0 // Round is not differentiable
    }

    fn trunc_derivative(_x: f64) -> f64 {
        0.0 // Trunc is not differentiable
    }

    fn square_derivative(x: f64) -> f64 {
        2.0 * x
    }

    fn reciprocal_derivative(x: f64) -> f64 {
        use crate::numerical_stability::NumericalStability;
        -NumericalStability::safe_divide(1.0, x * x)
    }

    fn sign_derivative(_x: f64) -> f64 {
        0.0 // Sign is not differentiable
    }

    fn tan_derivative(x: f64) -> f64 {
        let cos_x = x.cos();
        use crate::numerical_stability::NumericalStability;
        NumericalStability::safe_divide(1.0, cos_x * cos_x)
    }

    fn acosh_derivative(x: f64) -> f64 {
        use crate::numerical_stability::NumericalStability;
        NumericalStability::safe_divide(1.0, (x * x - 1.0).sqrt())
    }

    fn asinh_derivative(x: f64) -> f64 {
        use crate::numerical_stability::NumericalStability;
        NumericalStability::safe_divide(1.0, (x * x + 1.0).sqrt())
    }

    fn atanh_derivative(x: f64) -> f64 {
        use crate::numerical_stability::NumericalStability;
        NumericalStability::safe_divide(1.0, 1.0 - x * x)
    }

    fn erfc_derivative(x: f64) -> f64 {
        -2.0 / (2.0_f64).sqrt() * (-x * x).exp()
    }

    fn expm1_derivative(x: f64) -> f64 {
        x.exp()
    }

    fn fix_derivative(_x: f64) -> f64 {
        0.0 // Fix is not differentiable
    }

    fn frac_derivative(_x: f64) -> f64 {
        1.0 // Fractional part derivative is 1
    }

    fn log1p_derivative(x: f64) -> f64 {
        use crate::numerical_stability::NumericalStability;
        NumericalStability::safe_divide(1.0, x + 1.0)
    }

    fn sgn_derivative(_x: f64) -> f64 {
        0.0 // Sgn is not differentiable
    }

    fn signbit_derivative(_x: f64) -> f64 {
        0.0 // Signbit is not differentiable
    }

    fn acos_derivative(x: f64) -> f64 {
        use crate::numerical_stability::NumericalStability;
        -NumericalStability::safe_divide(1.0, (1.0 - x * x).sqrt())
    }

    fn atan_derivative(x: f64) -> f64 {
        use crate::numerical_stability::NumericalStability;
        NumericalStability::safe_divide(1.0, 1.0 + x * x)
    }

    fn erf_derivative(x: f64) -> f64 {
        2.0 / (std::f64::consts::PI).sqrt() * (-x * x).exp()
    }

    fn exp2_derivative(x: f64) -> f64 {
        2.0_f64.ln() * (2.0_f64).powf(x)
    }

    fn log10_derivative(x: f64) -> f64 {
        use crate::numerical_stability::NumericalStability;
        NumericalStability::safe_divide(1.0, x * 10.0_f64.ln())
    }

    fn log2_derivative(x: f64) -> f64 {
        use crate::numerical_stability::NumericalStability;
        NumericalStability::safe_divide(1.0, x * 2.0_f64.ln())
    }

    fn rsqrt_derivative(x: f64) -> f64 {
        use crate::numerical_stability::NumericalStability;
        -0.5 * NumericalStability::safe_divide(1.0, (x * x).sqrt() * (x * x * x))
    }

    fn backward_matmul(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if inputs.len() >= 2 {
            // Get tensor data and shapes
            let left_data = self.get_tensor_data(inputs[0]).cloned();
            let right_data = self.get_tensor_data(inputs[1]).cloned();
            let left_shape = self.tensor_shapes.get(&inputs[0]).cloned();
            let right_shape = self.tensor_shapes.get(&inputs[1]).cloned();

            if let (Some(_left_data), Some(_right_data), Some(left_shape), Some(right_shape)) =
                (left_data, right_data, left_shape, right_shape)
            {
                // Simplified matmul gradients (would need proper matrix operations)
                // For now, use element-wise gradients
                if inputs[0] != 0 {
                    let grad = output_grad.to_vec();
                    let existing_grad = self
                        .get_gradient(inputs[0])
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad.len()]);
                    let new_grad = grad
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(inputs[0], new_grad);
                }

                if inputs[1] != 0 {
                    let grad = output_grad.to_vec();
                    let existing_grad = self
                        .get_gradient(inputs[1])
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad.len()]);
                    let new_grad = grad
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(inputs[1], new_grad);
                }
            }
        }
    }

    fn backward_transpose(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if let Some(&input_id) = inputs.first() {
            if input_id != 0 {
                // Transpose is its own inverse, so gradient just needs to be transposed back
                let grad = output_grad.to_vec();
                let existing_grad = self
                    .get_gradient(input_id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; grad.len()]);
                let new_grad = grad
                    .iter()
                    .zip(existing_grad.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                self.set_gradient(input_id, new_grad);
            }
        }
    }

    fn backward_reshape(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if let Some(&input_id) = inputs.first() {
            if input_id != 0 {
                // Reshape preserves total elements, gradient can flow through unchanged
                let grad = output_grad.to_vec();
                let existing_grad = self
                    .get_gradient(input_id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; grad.len()]);
                let new_grad = grad
                    .iter()
                    .zip(existing_grad.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                self.set_gradient(input_id, new_grad);
            }
        }
    }

    fn backward_unsqueeze(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if let Some(&input_id) = inputs.first() {
            if input_id != 0 {
                // Unsqueeze adds dimensions, gradient flows through unchanged
                let grad = output_grad.to_vec();
                let existing_grad = self
                    .get_gradient(input_id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; grad.len()]);
                let new_grad = grad
                    .iter()
                    .zip(existing_grad.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                self.set_gradient(input_id, new_grad);
            }
        }
    }

    fn backward_expand(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if let Some(&input_id) = inputs.first() {
            if input_id != 0 {
                // Expand broadcasts, need to accumulate gradients back to original shape
                if let Some(input_shape) = self.tensor_shapes.get(&input_id) {
                    let input_len = input_shape.iter().product();
                    let mut grad = vec![0.0; input_len];

                    // Accumulate gradients from broadcasted dimensions
                    for (i, &og) in output_grad.iter().enumerate() {
                        let input_idx = i % input_len;
                        grad[input_idx] += og;
                    }

                    let existing_grad = self
                        .get_gradient(input_id)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad.len()]);
                    let new_grad = grad
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(input_id, new_grad);
                }
            }
        }
    }

    fn backward_clamp(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if let Some(&input_id) = inputs.first() {
            if input_id != 0 {
                if let Some(input_data) = self.get_tensor_data(input_id) {
                    // Clamp derivative is 1 where input is within bounds, 0 otherwise
                    // For simplicity, assume min=None, max=None (no clamping)
                    let grad = output_grad.to_vec();
                    let existing_grad = self
                        .get_gradient(input_id)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad.len()]);
                    let new_grad = grad
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(input_id, new_grad);
                }
            }
        }
    }

    fn backward_fmod(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if inputs.len() >= 2 {
            // Gradient w.r.t. left input: output_grad
            if inputs[0] != 0 {
                let grad = output_grad.to_vec();
                let existing_grad = self
                    .get_gradient(inputs[0])
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; grad.len()]);
                let new_grad = grad
                    .iter()
                    .zip(existing_grad.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                self.set_gradient(inputs[0], new_grad);
            }
            // Right input gradient is more complex, simplified for now
        }
    }

    fn backward_remainder(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if inputs.len() >= 2 {
            // Gradient w.r.t. left input: output_grad
            if inputs[0] != 0 {
                let grad = output_grad.to_vec();
                let existing_grad = self
                    .get_gradient(inputs[0])
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; grad.len()]);
                let new_grad = grad
                    .iter()
                    .zip(existing_grad.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                self.set_gradient(inputs[0], new_grad);
            }
            // Right input gradient is more complex, simplified for now
        }
    }

    fn backward_nan_to_num(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if let Some(&input_id) = inputs.first() {
            if input_id != 0 {
                // NaN replacement doesn't affect gradient flow for valid inputs
                let grad = output_grad.to_vec();
                let existing_grad = self
                    .get_gradient(input_id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; grad.len()]);
                let new_grad = grad
                    .iter()
                    .zip(existing_grad.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                self.set_gradient(input_id, new_grad);
            }
        }
    }

    fn backward_xlogy(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if inputs.len() >= 2 {
            let left_data = self.get_tensor_data(inputs[0]).cloned();
            let right_data = self.get_tensor_data(inputs[1]).cloned();

            if let (Some(left_data), Some(right_data)) = (left_data, right_data) {
                // Gradient w.r.t. left input: output_grad * log(right)
                if inputs[0] != 0 {
                    let grad_left: Vec<f64> = output_grad
                        .iter()
                        .zip(right_data.iter())
                        .map(|(og, y)| {
                            if *y > 0.0 {
                                og * y.ln()
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let existing_grad = self
                        .get_gradient(inputs[0])
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad_left.len()]);
                    let new_grad = grad_left
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(inputs[0], new_grad);
                }

                // Gradient w.r.t. right input: output_grad * left / right
                if inputs[1] != 0 {
                    let grad_right: Vec<f64> = output_grad
                        .iter()
                        .zip(left_data.iter())
                        .zip(right_data.iter())
                        .map(|((og, x), y)| {
                            use crate::numerical_stability::NumericalStability;
                            if *y > 0.0 {
                                NumericalStability::safe_divide(*og * *x, *y)
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let existing_grad = self
                        .get_gradient(inputs[1])
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad_right.len()]);
                    let new_grad = grad_right
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(inputs[1], new_grad);
                }
            }
        }
    }

    fn backward_elu(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if let Some(&input_id) = inputs.first() {
            if input_id != 0 {
                if let Some(input_data) = self.get_tensor_data(input_id) {
                    let grad: Vec<f64> = output_grad
                        .iter()
                        .zip(input_data.iter())
                        .map(|(og, x)| {
                            if *x >= 0.0 {
                                *og
                            } else {
                                *og * (-*x).exp()
                            }
                        })
                        .collect();
                    let existing_grad = self
                        .get_gradient(input_id)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad.len()]);
                    let new_grad = grad
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(input_id, new_grad);
                }
            }
        }
    }

    fn backward_gelu(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if let Some(&input_id) = inputs.first() {
            if input_id != 0 {
                if let Some(input_data) = self.get_tensor_data(input_id) {
                    let grad: Vec<f64> = output_grad
                        .iter()
                        .zip(input_data.iter())
                        .map(|(og, x)| {
                            // GELU derivative approximation
                            let x3 = x * x * x;
                            let tanh_arg = 0.7978845608028654 * (*x + 0.044715 * x3);
                            let sech_squared = 1.0 / (tanh_arg.cosh() * tanh_arg.cosh());
                            og * (0.5 + 0.5 * tanh_arg.tanh() + 0.7978845608028654 * *x * sech_squared * (1.0 + 0.134364 * x * x))
                        })
                        .collect();
                    let existing_grad = self
                        .get_gradient(input_id)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad.len()]);
                    let new_grad = grad
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(input_id, new_grad);
                }
            }
        }
    }

    fn backward_hardtanh(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if let Some(&input_id) = inputs.first() {
            if input_id != 0 {
                if let Some(input_data) = self.get_tensor_data(input_id) {
                    let grad: Vec<f64> = output_grad
                        .iter()
                        .zip(input_data.iter())
                        .map(|(og, x)| {
                            if *x >= -1.0 && *x <= 1.0 {
                                *og
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let existing_grad = self
                        .get_gradient(input_id)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad.len()]);
                    let new_grad = grad
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(input_id, new_grad);
                }
            }
        }
    }

    fn backward_logsigmoid(&mut self, inputs: &[u64], output_grad: &[f64]) {
        if let Some(&input_id) = inputs.first() {
            if input_id != 0 {
                if let Some(input_data) = self.get_tensor_data(input_id) {
                    let grad: Vec<f64> = output_grad
                        .iter()
                        .zip(input_data.iter())
                        .map(|(og, x)| {
                            let sig = 1.0 / (1.0 + (-*x).exp());
                            og * sig * (1.0 - sig)
                        })
                        .collect();
                    let existing_grad = self
                        .get_gradient(input_id)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad.len()]);
                    let new_grad = grad
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(input_id, new_grad);
                }
            }
        }
    }

    fn backward_pow(&mut self, inputs: &[u64], output_grad: &[f64], exponent: f64) {
        if let Some(&input_id) = inputs.first() {
            if input_id != 0 {
                if let Some(input_data) = self.get_tensor_data(input_id) {
                    let grad: Vec<f64> = output_grad
                        .iter()
                        .zip(input_data.iter())
                        .map(|(og, x)| {
                            og * exponent * x.powf(exponent - 1.0)
                        })
                        .collect();
                    let existing_grad = self
                        .get_gradient(input_id)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; grad.len()]);
                    let new_grad = grad
                        .iter()
                        .zip(existing_grad.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    self.set_gradient(input_id, new_grad);
                }
            }
        }
    }
}
