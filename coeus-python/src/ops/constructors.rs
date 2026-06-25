use crate::tensor::PyTensor;
use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (shape, requires_grad = false))]
pub fn zeros(shape: Vec<usize>, requires_grad: bool) -> PyTensor {
    let t = Tensor::<f64, MoiraiBackend>::zeros(shape);
    PyTensor {
        inner: Var::new(t, requires_grad),
    }
}

#[pyfunction]
#[pyo3(signature = (shape, requires_grad = false))]
pub fn ones(shape: Vec<usize>, requires_grad: bool) -> PyTensor {
    let t = Tensor::<f64, MoiraiBackend>::ones(shape);
    PyTensor {
        inner: Var::new(t, requires_grad),
    }
}

#[pyfunction]
#[pyo3(signature = (shape, value, requires_grad = false))]
pub fn full(shape: Vec<usize>, value: f64, requires_grad: bool) -> PyTensor {
    let backend = MoiraiBackend::new();
    let t = Tensor::<f64, MoiraiBackend>::full_on(shape, value, &backend);
    PyTensor {
        inner: Var::new(t, requires_grad),
    }
}

#[pyfunction]
#[pyo3(signature = (start, end, step = 1.0))]
pub fn arange(start: f64, end: f64, step: f64) -> PyTensor {
    let n = ((end - start) / step).ceil() as usize;
    let data: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
    let t = Tensor::from_slice(vec![n], &data);
    PyTensor {
        inner: Var::new(t, false),
    }
}

#[pyfunction]
pub fn linspace(start: f64, end: f64, steps: usize) -> PyTensor {
    let data: Vec<f64> = if steps <= 1 {
        vec![start]
    } else {
        (0..steps)
            .map(|i| start + (end - start) * i as f64 / (steps - 1) as f64)
            .collect()
    };
    let t = Tensor::from_slice(vec![steps], &data);
    PyTensor {
        inner: Var::new(t, false),
    }
}

#[pyfunction]
#[pyo3(signature = (shape, requires_grad = false))]
pub fn randn(shape: Vec<usize>, requires_grad: bool) -> PyTensor {
    use ::std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(12345);
    let zeros_t = Tensor::<f64, MoiraiBackend>::zeros(shape);
    let mut v = Var::new(zeros_t, requires_grad);
    coeus_nn::init::normal_with_seed(&mut v, 0.0, 1.0, seed);
    PyTensor { inner: v }
}

#[pyfunction]
#[pyo3(signature = (input, requires_grad = false))]
pub fn zeros_like(input: &PyTensor, requires_grad: bool) -> PyTensor {
    PyTensor {
        inner: Var::new(
            Tensor::<f64, MoiraiBackend>::zeros(input.inner.tensor.shape().to_vec()),
            requires_grad,
        ),
    }
}

#[pyfunction]
#[pyo3(signature = (input, requires_grad = false))]
pub fn ones_like(input: &PyTensor, requires_grad: bool) -> PyTensor {
    PyTensor {
        inner: Var::new(
            Tensor::<f64, MoiraiBackend>::ones(input.inner.tensor.shape().to_vec()),
            requires_grad,
        ),
    }
}

#[pyfunction]
#[pyo3(signature = (n, requires_grad = false))]
pub fn eye(n: usize, requires_grad: bool) -> PyTensor {
    let backend = MoiraiBackend::new();
    let t = Tensor::<f64, MoiraiBackend>::eye_on(n, &backend);
    PyTensor {
        inner: Var::new(t, requires_grad),
    }
}
