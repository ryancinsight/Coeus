use crate::tensor::{PyTensor, TensorWrapper};
use optim::optimizers::Rprop;
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyResult};

#[pyclass(name = "Rprop", module = "coeus.optim", subclass)]
pub struct PyRprop {
    pub inner_f32: Option<Rprop<backend::CpuBackend<dtype::float::Float32>, storage::DenseStorage<dtype::float::Float32>, dtype::float::Float32>>,
    pub inner_f64: Option<Rprop<backend::CpuBackend<dtype::float::Float64>, storage::DenseStorage<dtype::float::Float64>, dtype::float::Float64>>,
}

#[pymethods]
impl PyRprop {
    #[new]
    #[pyo3(signature = (params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50.0)))]
    fn new(params: Vec<PyTensor>, lr: f64, etas: (f64, f64), step_sizes: (f64, f64)) -> PyResult<Self> {
        if params.is_empty() {
             return Ok(PyRprop { inner_f32: None, inner_f64: None });
        }
        
        match &params[0].inner {
            TensorWrapper::CpuDenseF32(_) => {
                let mut p_vec = Vec::new();
                for p in params {
                    if let TensorWrapper::CpuDenseF32(t) = p.inner {
                        p_vec.push(t);
                    }
                }
                let mut opt = Rprop::new(lr, etas, step_sizes);
                optim::BaseOptimizer::add_param_group(&mut opt, p_vec);
                Ok(PyRprop { inner_f32: Some(opt), inner_f64: None })
            }
            TensorWrapper::CpuDenseF64(_) => {
                let mut p_vec = Vec::new();
                for p in params {
                    if let TensorWrapper::CpuDenseF64(t) = p.inner {
                        p_vec.push(t);
                    }
                }
                let mut opt = Rprop::new(lr, etas, step_sizes);
                optim::BaseOptimizer::add_param_group(&mut opt, p_vec);
                Ok(PyRprop { inner_f32: None, inner_f64: Some(opt) })
            }
            _ => Err(pyo3::exceptions::PyTypeError::new_err("Rprop only supports CPU dense F32/F64")),
        }
    }

    fn step(&mut self) -> PyResult<()> {
        if let Some(opt) = &mut self.inner_f32 {
            optim::BaseOptimizer::step(opt).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        } else if let Some(opt) = &mut self.inner_f64 {
            optim::BaseOptimizer::step(opt).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    fn zero_grad(&mut self) -> PyResult<()> {
        if let Some(opt) = &mut self.inner_f32 {
            optim::BaseOptimizer::zero_grad(opt);
        } else if let Some(opt) = &mut self.inner_f64 {
            optim::BaseOptimizer::zero_grad(opt);
        }
        Ok(())
    }
}
