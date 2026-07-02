use crate::tensor::PyTensor;
use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn time_seed(fallback: u64, salt: u64) -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64 ^ salt)
        .unwrap_or(fallback)
}

fn next_xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

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
#[pyo3(signature = (start, end, steps, base = 10.0))]
pub fn logspace(start: f64, end: f64, steps: usize, base: f64) -> PyTensor {
    let data: Vec<f64> = if steps == 0 {
        vec![]
    } else if steps == 1 {
        vec![base.powf(start)]
    } else {
        (0..steps)
            .map(|i| {
                let exp = start + (end - start) * i as f64 / (steps - 1) as f64;
                base.powf(exp)
            })
            .collect()
    };
    let t = Tensor::from_slice(vec![steps], &data);
    PyTensor {
        inner: Var::new(t, false),
    }
}

#[pyfunction]
pub fn geomspace(start: f64, end: f64, steps: usize) -> PyResult<PyTensor> {
    if start == 0.0 || end == 0.0 {
        return Err(PyValueError::new_err(
            "geomspace requires non-zero start/end",
        ));
    }
    if start.signum() != end.signum() {
        return Err(PyValueError::new_err(
            "geomspace requires start/end to have the same sign",
        ));
    }
    let sign = start.signum();
    let start_abs = start.abs();
    let end_abs = end.abs();
    let ratio = if steps > 1 {
        (end_abs / start_abs).powf(1.0 / (steps - 1) as f64)
    } else {
        1.0
    };
    let data: Vec<f64> = if steps == 0 {
        vec![]
    } else if steps == 1 {
        vec![start]
    } else {
        (0..steps)
            .map(|i| sign * start_abs * ratio.powf(i as f64))
            .collect()
    };
    let t = Tensor::from_slice(vec![steps], &data);
    Ok(PyTensor {
        inner: Var::new(t, false),
    })
}

#[pyfunction]
#[pyo3(signature = (shape, requires_grad = false))]
pub fn randn(shape: Vec<usize>, requires_grad: bool) -> PyTensor {
    let seed = time_seed(12345, 0x2d35_8b72_a4c9_6e1d);
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

/// Uniform random tensor in `[0, 1)`.
///
/// Equivalent to `torch.rand(shape)`.
#[pyfunction]
#[pyo3(signature = (shape, requires_grad = false))]
pub fn rand(shape: Vec<usize>, requires_grad: bool) -> PyTensor {
    let seed = time_seed(54321, 0x9e37_79b9_7f4a_7c15);
    let zeros_t = Tensor::<f64, MoiraiBackend>::zeros(shape);
    let mut v = Var::new(zeros_t, requires_grad);
    coeus_nn::init::uniform_with_seed(&mut v, 0.0, 1.0, seed);
    PyTensor { inner: v }
}

/// Random integer tensor in `[low, high)` stored as f64.
///
/// Equivalent to `torch.randint(low, high, shape)`.
#[pyfunction]
#[pyo3(signature = (low, high, shape, requires_grad = false))]
pub fn randint(low: i64, high: i64, shape: Vec<usize>, requires_grad: bool) -> PyResult<PyTensor> {
    if high <= low {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "randint: high ({high}) must be greater than low ({low})"
        )));
    }
    let seed = time_seed(99991, 0x6c62_272e_07bb_0142);
    let numel: usize = shape.iter().product();
    let range = (high - low) as u64;
    let mut state = seed.wrapping_add(0x6c62272e07bb0142);
    let data: Vec<f64> = (0..numel)
        .map(|_| {
            let v = low + (next_xorshift64(&mut state) % range) as i64;
            v as f64
        })
        .collect();
    let t = Tensor::from_slice(shape, &data);
    Ok(PyTensor {
        inner: Var::new(t, requires_grad),
    })
}

/// Bernoulli random tensor: each element is 1.0 with probability `p`.
///
/// Equivalent to `torch.bernoulli(torch.full(shape, p))`.
#[pyfunction]
#[pyo3(signature = (shape, p = 0.5, requires_grad = false))]
pub fn bernoulli(shape: Vec<usize>, p: f64, requires_grad: bool) -> PyResult<PyTensor> {
    if !(0.0..=1.0).contains(&p) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "bernoulli: p must be in [0, 1], got {p}"
        )));
    }
    let seed = time_seed(77777, 0xdead_beef_cafe_1234);
    let numel: usize = shape.iter().product();
    let mut state = seed.wrapping_add(0xdeadbeefcafe1234);
    let data: Vec<f64> = (0..numel)
        .map(|_| {
            let frac = (next_xorshift64(&mut state) as f64) / (u64::MAX as f64);
            if frac < p {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let t = Tensor::from_slice(shape, &data);
    Ok(PyTensor {
        inner: Var::new(t, requires_grad),
    })
}
