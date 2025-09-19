//! Mathematical operations for tensors
//!
//! This module contains element-wise mathematical operations and activation functions
//! for tensors, including automatic differentiation support.

use crate::{Tensor, Dtype, FloatDtype};
use crate::with_autograd_context;
use coeus_autograd::context::Operation;
use statrs::function::erf;

impl<T: Dtype + num_traits::FromPrimitive + num_traits::ToPrimitive> Tensor<T> {

    /// Compute the inverse cosine of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 0.0, -1.0], vec![3]);
    /// let acos_tensor = tensor.acos();
    /// // Result: [acos(1.0), acos(0.0), acos(-1.0)]
    /// ```
    pub fn acos(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.acos()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Acos, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the inverse tangent of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]);
    /// let atan_tensor = tensor.atan();
    /// // Result: [atan(0.0), atan(1.0), atan(-1.0)]
    /// ```
    pub fn atan(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.atan()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Atan, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the error function of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0], vec![1]);
    /// let erf_tensor = tensor.erf();
    /// // Result: [erf(0.0)]
    /// ```
    pub fn erf(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| {
            let x_f64 = Dtype::to_f64(x).unwrap_or(0.0);
            let erf_result = erf::erf(x_f64);
            T::from_f64(erf_result).unwrap_or(T::zero())
        }).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Erf, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute 2^x for each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![3]);
    /// let exp2_tensor = tensor.exp2();
    /// // Result: [2^0, 2^1, 2^2] = [1.0, 2.0, 4.0]
    /// ```
    pub fn exp2(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.exp2()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Exp2, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the base-10 logarithm of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 10.0, 100.0], vec![3]);
    /// let log10_tensor = tensor.log10();
    /// // Result: [log10(1), log10(10), log10(100)] = [0.0, 1.0, 2.0]
    /// ```
    pub fn log10(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.log10()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Log10, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the base-2 logarithm of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 4.0, 8.0], vec![4]);
    /// let log2_tensor = tensor.log2();
    /// // Result: [log2(1), log2(2), log2(4), log2(8)] = [0.0, 1.0, 2.0, 3.0]
    /// ```
    pub fn log2(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.log2()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Log2, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the reciprocal square root (1/sqrt(x)) of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 4.0, 9.0], vec![3]);
    /// let rsqrt_tensor = tensor.rsqrt();
    /// // Result: [1/sqrt(1), 1/sqrt(4), 1/sqrt(9)] = [1.0, 0.5, 1/3]
    /// ```
    pub fn rsqrt(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| T::one() / x.sqrt()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            if let Some(node_id) = self.node {
                with_autograd_context(|context| {
                    let inputs = vec![node_id];
                    let node_id = context.create_node(Operation::Rsqrt, inputs);

                    // Register tensor data for gradient computation
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node_id, data_f64, self.shape.clone());

                    result.node = Some(node_id);
                });
            }
        }

        result
    }


    /// Compute the complementary error function of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0], vec![1]);
    /// let erfc_tensor = tensor.erfc();
    /// // Result: [erfc(0.0)] = [1.0]
    /// ```
    pub fn erfc(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| {
            let x_f64 = Dtype::to_f64(x).unwrap_or(0.0);
            let erfc_result = erf::erfc(x_f64);
            T::from_f64(erfc_result).unwrap_or(T::one())
        }).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Erfc, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the sign bit of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]);
    /// let signbit_tensor = tensor.signbit();
    /// // Result: [1.0, 0.0, 0.0] (1.0 for negative, 0.0 for positive)
    /// ```
    pub fn signbit(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| {
            if x.is_sign_negative() {
                T::one()
            } else {
                T::zero()
            }
        }).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Signbit, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the inverse hyperbolic cosine of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    /// let acosh_tensor = tensor.acosh();
    /// // Result: [acosh(1.0), acosh(2.0)]
    /// ```
    pub fn acosh(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.acosh()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Acosh, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the inverse hyperbolic sine of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]);
    /// let asinh_tensor = tensor.asinh();
    /// // Result: [asinh(0.0), asinh(1.0), asinh(-1.0)]
    /// ```
    pub fn asinh(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.asinh()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Asinh, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the inverse hyperbolic tangent of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, 0.5, -0.5], vec![3]);
    /// let atanh_tensor = tensor.atanh();
    /// // Result: [atanh(0.0), atanh(0.5), atanh(-0.5)]
    /// ```
    pub fn atanh(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.atanh()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Atanh, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }


    /// Compute exp(x) - 1 for each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, 1.0], vec![2]);
    /// let expm1_tensor = tensor.expm1();
    /// // Result: [exp(0.0) - 1, exp(1.0) - 1] = [0.0, e - 1]
    /// ```
    pub fn expm1(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.exp_m1()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Expm1, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Truncate each element towards zero
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.7, -1.7, 1.3, -1.3], vec![4]);
    /// let fix_tensor = tensor.fix();
    /// // Result: [1.0, -1.0, 1.0, -1.0]
    /// ```
    pub fn fix(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| if *x >= T::zero() {
            x.floor()
        } else {
            x.ceil()
        }).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Fix, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the floating-point remainder of division
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![7.5, -7.5], vec![2]);
    /// let divisor = Tensor::from_vec(vec![3.0, -3.0], vec![2]);
    /// let fmod_tensor = tensor.fmod(&divisor).unwrap();
    /// // Result: [7.5 % 3.0, -7.5 % -3.0] = [1.5, -1.5]
    /// ```
    pub fn fmod(&self, divisor: &Tensor<T>) -> Result<Tensor<T>, crate::TensorError>
    where
        T: FloatDtype,
    {
        if self.shape() != divisor.shape() {
            return Err(crate::TensorError::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: divisor.shape().to_vec(),
            });
        }

        let result_data: Vec<T> = self.data.iter().zip(divisor.data.iter()).map(|(&x, &y)| {
            if y == T::zero() {
                T::nan()
            } else {
                let quotient = (x / y).floor();
                x - quotient * y
            }
        }).collect();

        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() || divisor.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let divisor_node = if let Some(node) = divisor.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = divisor
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, divisor.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Fmod, vec![self_node, divisor_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        Ok(result)
    }

    /// Compute the fractional part of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![3.7, -3.7, 3.0], vec![3]);
    /// let frac_tensor = tensor.frac();
    /// // Result: [0.7, -0.7, 0.0]
    /// ```
    pub fn frac(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| *x - x.trunc()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Frac, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the IEEE 754 remainder of division
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![7.5, -7.5], vec![2]);
    /// let divisor = Tensor::from_vec(vec![3.0, -3.0], vec![2]);
    /// let remainder_tensor = tensor.remainder(&divisor).unwrap();
    /// // Result: [7.5 % 3.0, -7.5 % -3.0] = [1.5, -1.5]
    /// ```
    pub fn remainder(&self, divisor: &Tensor<T>) -> Result<Tensor<T>, crate::TensorError>
    where
        T: FloatDtype,
    {
        if self.shape() != divisor.shape() {
            return Err(crate::TensorError::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: divisor.shape().to_vec(),
            });
        }

        let result_data: Vec<T> = self.data.iter().zip(divisor.data.iter()).map(|(&x, &y)| {
            if y == T::zero() {
                T::nan()
            } else {
                x % y
            }
        }).collect();

        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() || divisor.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let divisor_node = if let Some(node) = divisor.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = divisor
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, divisor.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Remainder, vec![self_node, divisor_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        Ok(result)
    }

    /// Compute log(1 + x) for each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, 1.0], vec![2]);
    /// let log1p_tensor = tensor.log1p();
    /// // Result: [log(1 + 0.0), log(1 + 1.0)] = [0.0, log(2.0)]
    /// ```
    pub fn log1p(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| (T::one() + *x).ln()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Log1p, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Replace NaN and infinite values with specified values
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![f32::NAN, f32::INFINITY, -f32::INFINITY, 1.0], vec![4]);
    /// let nan_to_num_tensor = tensor.nan_to_num(Some(0.0), Some(1.0), Some(-1.0));
    /// // Result: [0.0, 1.0, -1.0, 1.0]
    /// ```
    pub fn nan_to_num(&self, nan: Option<T>, posinf: Option<T>, neginf: Option<T>) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let nan_val = nan.unwrap_or(T::zero());
        let posinf_val = posinf.unwrap_or(T::from(1.0).unwrap());
        let neginf_val = neginf.unwrap_or(T::from(-1.0).unwrap());

        let result_data: Vec<T> = self.data.iter().map(|&x| {
            if x.is_nan() {
                nan_val
            } else if x.is_infinite() && x.is_sign_positive() {
                posinf_val
            } else if x.is_infinite() && x.is_sign_negative() {
                neginf_val
                } else {
                x
            }
        }).collect();

        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::NanToNum, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the sign function of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![-2.0, -0.5, 0.0, 0.5, 2.0], vec![5]);
    /// let sgn_tensor = tensor.sgn();
    /// // Result: [-1.0, -1.0, 0.0, 1.0, 1.0]
    /// ```
    pub fn sgn(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|&x| {
            if x > T::zero() {
                T::one()
            } else if x < T::zero() {
                -T::one()
                } else {
                T::zero()
            }
        }).collect();

        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Sgn, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }


    /// Compute x * log(y) with special handling for x = 0
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let x = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![3]);
    /// let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    /// let xlogy_tensor = x.xlogy(&y).unwrap();
    /// // Result: [0.0, 1.0 * log(2.0), 2.0 * log(3.0)]
    /// ```
    pub fn xlogy(&self, y: &Tensor<T>) -> Result<Tensor<T>, crate::TensorError>
    where
        T: FloatDtype,
    {
        if self.shape() != y.shape() {
            return Err(crate::TensorError::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: y.shape().to_vec(),
            });
        }

        let result_data: Vec<T> = self.data.iter().zip(y.data.iter()).map(|(&x, &y_val)| {
            if x == T::zero() {
                    T::zero()
            } else if y_val <= T::zero() {
                T::nan()
                } else {
                x * y_val.ln()
                }
        }).collect();

        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() || y.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let y_node = if let Some(node) = y.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = y
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, y.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Xlogy, vec![self_node, y_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        Ok(result)
    }
    }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acos() {
        let tensor = Tensor::from_vec(vec![1.0, 0.0, -1.0], vec![3]);
        let result = tensor.acos();
        assert!((result.data[0] - 0.0_f64).abs() < 1e-6); // acos(1) = 0
        assert!((result.data[1] - std::f64::consts::PI / 2.0).abs() < 1e-6); // acos(0) = π/2
        assert!((result.data[2] - std::f64::consts::PI).abs() < 1e-6); // acos(-1) = π
    }

    #[test]
    fn test_atan() {
        let tensor = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]);
        let result = tensor.atan();
        assert!((result.data[0] - 0.0_f64).abs() < 1e-6); // atan(0) = 0
        assert!((result.data[1] - std::f64::consts::PI / 4.0).abs() < 1e-6); // atan(1) = π/4
        assert!((result.data[2] - (-std::f64::consts::PI / 4.0)).abs() < 1e-6); // atan(-1) = -π/4
    }

    #[test]
    fn test_erf() {
        let tensor = Tensor::from_vec(vec![0.0], vec![1]);
        let result = tensor.erf();
        assert!((result.data[0] - 0.0_f64).abs() < 1e-6); // erf(0) = 0
    }

    #[test]
    fn test_exp2() {
        let tensor = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![3]);
        let result = tensor.exp2();
        assert!((result.data[0] - 1.0_f64).abs() < 1e-6); // 2^0 = 1
        assert!((result.data[1] - 2.0_f64).abs() < 1e-6); // 2^1 = 2
        assert!((result.data[2] - 4.0_f64).abs() < 1e-6); // 2^2 = 4
    }

    #[test]
    fn test_log10() {
        let tensor = Tensor::from_vec(vec![1.0, 10.0, 100.0], vec![3]);
        let result = tensor.log10();
        assert!((result.data[0] - 0.0_f64).abs() < 1e-6); // log10(1) = 0
        assert!((result.data[1] - 1.0_f64).abs() < 1e-6); // log10(10) = 1
        assert!((result.data[2] - 2.0_f64).abs() < 1e-6); // log10(100) = 2
    }

    #[test]
    fn test_log2() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 4.0, 8.0], vec![4]);
        let result = tensor.log2();
        assert!((result.data[0] - 0.0_f64).abs() < 1e-6); // log2(1) = 0
        assert!((result.data[1] - 1.0_f64).abs() < 1e-6); // log2(2) = 1
        assert!((result.data[2] - 2.0_f64).abs() < 1e-6); // log2(4) = 2
        assert!((result.data[3] - 3.0_f64).abs() < 1e-6); // log2(8) = 3
    }

    #[test]
    fn test_rsqrt() {
        let tensor = Tensor::from_vec(vec![1.0, 4.0, 9.0], vec![3]);
        let result = tensor.rsqrt();
        assert!((result.data[0] - 1.0_f64).abs() < 1e-6); // 1/sqrt(1) = 1
        assert!((result.data[1] - 0.5_f64).abs() < 1e-6); // 1/sqrt(4) = 0.5
        assert!((result.data[2] - (1.0_f64 / 3.0_f64)).abs() < 1e-6); // 1/sqrt(9) = 1/3
    }


    #[test]
    fn test_acosh() {
        let tensor = Tensor::from_vec(vec![1.0], vec![1]);
        let result = tensor.acosh();
        assert!((result.data[0] - 0.0_f64).abs() < 1e-6); // acosh(1) = 0
    }

    #[test]
    fn test_asinh() {
        let tensor = Tensor::from_vec(vec![0.0], vec![1]);
        let result = tensor.asinh();
        assert!((result.data[0] - 0.0_f64).abs() < 1e-6); // asinh(0) = 0
    }

    #[test]
    fn test_atanh() {
        let tensor = Tensor::from_vec(vec![0.0], vec![1]);
        let result = tensor.atanh();
        assert!((result.data[0] - 0.0_f64).abs() < 1e-6); // atanh(0) = 0
    }

    #[test]
    fn test_erfc() {
        let tensor = Tensor::from_vec(vec![0.0], vec![1]);
        let result = tensor.erfc();
        assert!((result.data[0] - 1.0_f64).abs() < 1e-6); // erfc(0) = 1 - erf(0) = 1
    }

    #[test]
    fn test_expm1() {
        let tensor = Tensor::from_vec(vec![0.0], vec![1]);
        let result = tensor.expm1();
        assert!((result.data[0] - 0.0_f64).abs() < 1e-6); // exp(0) - 1 = 0
    }

    #[test]
    fn test_fix() {
        let tensor = Tensor::from_vec(vec![1.7, -1.7, 1.3, -1.3], vec![4]);
        let result = tensor.fix();
        assert!((result.data[0] - 1.0_f64).abs() < 1e-6); // trunc(1.7) = 1
        assert!((result.data[1] - (-1.0_f64)).abs() < 1e-6); // trunc(-1.7) = -1
        assert!((result.data[2] - 1.0_f64).abs() < 1e-6); // trunc(1.3) = 1
        assert!((result.data[3] - (-1.0_f64)).abs() < 1e-6); // trunc(-1.3) = -1
    }

    #[test]
    fn test_fmod() {
        let tensor = Tensor::from_vec(vec![7.5, -7.5], vec![2]);
        let divisor = Tensor::from_vec(vec![3.0, -3.0], vec![2]);
        let result = tensor.fmod(&divisor).unwrap();
        assert!((result.data[0] - 1.5_f64).abs() < 1e-6); // 7.5 % 3.0 = 1.5
        assert!((result.data[1] - (-1.5_f64)).abs() < 1e-6); // -7.5 % -3.0 = -1.5
    }

    #[test]
    fn test_frac() {
        let tensor = Tensor::from_vec(vec![3.7, -3.7, 3.0], vec![3]);
        let result = tensor.frac();
        assert!((result.data[0] - 0.7_f64).abs() < 1e-6); // frac(3.7) = 0.7
        assert!((result.data[1] - (-0.7_f64)).abs() < 1e-6); // frac(-3.7) = -0.7
        assert!((result.data[2] - 0.0_f64).abs() < 1e-6); // frac(3.0) = 0.0
    }

    #[test]
    fn test_remainder() {
        let tensor = Tensor::from_vec(vec![7.5, -7.5], vec![2]);
        let divisor = Tensor::from_vec(vec![3.0, -3.0], vec![2]);
        let result = tensor.remainder(&divisor).unwrap();
        assert!((result.data[0] - 1.5_f64).abs() < 1e-6); // 7.5 % 3.0 = 1.5
        assert!((result.data[1] - (-1.5_f64)).abs() < 1e-6); // -7.5 % -3.0 = -1.5
    }

    #[test]
    fn test_log1p() {
        let tensor = Tensor::from_vec(vec![0.0], vec![1]);
        let result = tensor.log1p();
        assert!((result.data[0] - 0.0_f64).abs() < 1e-6); // log(1 + 0) = 0
    }

    #[test]
    fn test_nan_to_num() {
        let tensor = Tensor::from_vec(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0], vec![4]);
        let result = tensor.nan_to_num(Some(0.0), Some(1.0), Some(-1.0));
        assert!((result.data[0] - 0.0_f32).abs() < 1e-6); // NaN -> 0.0
        assert!((result.data[1] - 1.0_f32).abs() < 1e-6); // +inf -> 1.0
        assert!((result.data[2] - (-1.0_f32)).abs() < 1e-6); // -inf -> -1.0
        assert!((result.data[3] - 1.0_f32).abs() < 1e-6); // 1.0 -> 1.0
    }

    #[test]
    fn test_sgn() {
        let tensor = Tensor::from_vec(vec![-2.0, -0.5, 0.0, 0.5, 2.0], vec![5]);
        let result = tensor.sgn();
        assert!((result.data[0] - (-1.0_f64)).abs() < 1e-6); // sgn(-2) = -1
        assert!((result.data[1] - (-1.0_f64)).abs() < 1e-6); // sgn(-0.5) = -1
        assert!((result.data[2] - 0.0_f64).abs() < 1e-6); // sgn(0) = 0
        assert!((result.data[3] - 1.0_f64).abs() < 1e-6); // sgn(0.5) = 1
        assert!((result.data[4] - 1.0_f64).abs() < 1e-6); // sgn(2) = 1
    }

    #[test]
    fn test_signbit() {
        let tensor = Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]);
        let result = tensor.signbit();
        assert_eq!(result.data[0], 1.0); // signbit(-1) = 1 (negative)
        assert_eq!(result.data[1], 0.0); // signbit(0) = 0 (non-negative)
        assert_eq!(result.data[2], 0.0); // signbit(1) = 0 (non-negative)
    }

    #[test]
    fn test_xlogy() {
        let x = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let result = x.xlogy(&y).unwrap();
        assert!((result.data[0] - 0.0_f64).abs() < 1e-6); // 0 * log(1) = 0
        assert!((result.data[1] - 2.0_f64.ln()).abs() < 1e-6); // 1 * log(2)
        assert!((result.data[2] - (2.0 * 3.0_f64.ln())).abs() < 1e-6); // 2 * log(3)
    }
}