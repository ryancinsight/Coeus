//! Base optimizer functionality and generic wrapper
//!
//! This module provides shared functionality for all optimizers,
//! reducing code duplication across optimizer implementations.

use backend::CpuBackend;
use dtype::float::Float32;
use pyo3::prelude::*;
use std::collections::HashMap;
use storage::DenseStorage;

use crate::tensor::{PyTensor, TensorWrapper};

/// Type alias for the concrete tensor type used in optimizers
pub type CpuF32Tensor = tensor::Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

/// Helper to extract F32 tensors from PyTensors
///
/// This function validates that all parameters are float32 tensors
/// and extracts the underlying Rust tensors.
pub fn extract_f32_params(params: Vec<PyTensor>) -> PyResult<Vec<CpuF32Tensor>> {
    let mut result = Vec::with_capacity(params.len());
    for p in params {
        match p.inner {
            TensorWrapper::CpuDenseF32(t) => result.push(t),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Optimizer currently only supports float32 tensors",
                ))
            }
        }
    }
    Ok(result)
}

/// Convert Rust state dict to Python state dict
pub fn rust_state_to_py(state: HashMap<String, CpuF32Tensor>) -> HashMap<String, PyTensor> {
    state
        .into_iter()
        .map(|(k, v)| {
            (
                k,
                PyTensor {
                    inner: TensorWrapper::CpuDenseF32(v),
                },
            )
        })
        .collect()
}

/// Convert Python state dict to Rust state dict
pub fn py_state_to_rust(
    state_dict: HashMap<String, PyTensor>,
) -> PyResult<HashMap<String, CpuF32Tensor>> {
    let mut rust_state = HashMap::new();
    for (k, v) in state_dict {
        match v.inner {
            TensorWrapper::CpuDenseF32(t) => {
                rust_state.insert(k, t);
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "State dict must contain float32 tensors",
                ))
            }
        }
    }
    Ok(rust_state)
}

/// Macro to implement common optimizer methods
///
/// This macro reduces boilerplate by implementing the standard
/// optimizer interface for all optimizer types.
#[macro_export]
macro_rules! impl_optimizer_methods {
    ($optimizer_type:ty) => {
        impl $optimizer_type {
            pub fn step_impl(&mut self) -> PyResult<()> {
                optim::BaseOptimizer::step(&mut self.inner)
                    .map_err(|e| $crate::error::convert_error_with_context(e, "optimizer step"))?;
                Ok(())
            }

            pub fn zero_grad_impl(&mut self) {
                optim::BaseOptimizer::zero_grad(&mut self.inner);
            }

            pub fn add_param_group_impl(
                &mut self,
                params: Vec<$crate::tensor::PyTensor>,
            ) -> PyResult<()> {
                let rust_params = $crate::optim::base::extract_f32_params(params)?;
                optim::BaseOptimizer::add_param_group(&mut self.inner, rust_params);
                Ok(())
            }

            pub fn state_dict_impl(
                &self,
            ) -> PyResult<std::collections::HashMap<String, $crate::tensor::PyTensor>> {
                let state = optim::BaseOptimizer::state_dict(&self.inner);
                Ok($crate::optim::base::rust_state_to_py(state))
            }

            pub fn load_state_dict_impl(
                &mut self,
                state_dict: std::collections::HashMap<String, $crate::tensor::PyTensor>,
            ) -> PyResult<()> {
                let rust_state = $crate::optim::base::py_state_to_rust(state_dict)?;
                optim::BaseOptimizer::load_state_dict(&mut self.inner, rust_state).map_err(
                    |e| $crate::error::convert_error_with_context(e, "optimizer load_state_dict"),
                )?;
                Ok(())
            }

            pub fn get_lr_impl(&self) -> f64 {
                optim::BaseOptimizer::get_lr(&self.inner) as f64
            }

            pub fn set_lr_impl(&mut self, lr: f64) {
                optim::BaseOptimizer::set_lr(&mut self.inner, lr as f32);
            }
        }
    };
}
