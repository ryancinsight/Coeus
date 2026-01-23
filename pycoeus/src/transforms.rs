use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::{wrap_pyfunction, Bound, Py, PyResult, Python};

pub fn register(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyToTensor>()?;
    m.add_class::<PyNormalize>()?;
    m.add_class::<PyResize>()?;
    m.add_class::<PyRandomApply>()?;
    m.add_class::<PyCompose>()?;
    m.add_function(wrap_pyfunction!(to_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(resize, m)?)?;
    m.add_function(wrap_pyfunction!(random_apply, m)?)?;
    m.add_function(wrap_pyfunction!(compose, m)?)?;
    Ok(())
}

/// Python bindings for data transformations
///
/// Provides PyTorch-compatible data transformation utilities
/// for preprocessing machine learning data pipelines.

// Factory functions for easy transform creation
#[pyfunction]
#[pyo3(signature = ())]
pub fn to_tensor() -> PyToTensor {
    PyToTensor::new()
}

#[pyfunction]
#[pyo3(signature = (mean, std))]
pub fn normalize(mean: Vec<f32>, std: Vec<f32>) -> PyResult<PyNormalize> {
    PyNormalize::new(mean, std)
}

#[pyfunction]
#[pyo3(signature = (size))]
pub fn resize(size: (usize, usize)) -> PyResize {
    PyResize::new(size)
}

#[pyfunction]
#[pyo3(signature = (transforms, _p = 0.5))]
pub fn random_apply(transforms: Vec<Py<PyAny>>, _p: f32) -> PyResult<PyRandomApply> {
    PyRandomApply::new(transforms, _p)
}

#[pyfunction]
#[pyo3(signature = (transforms))]
pub fn compose(transforms: Vec<Py<PyAny>>) -> PyResult<PyCompose> {
    Ok(PyCompose::new(transforms))
}

/// ToTensor transform - converts data to tensors
#[pyclass(name = "ToTensor", module = "_coeus")]
pub struct PyToTensor {
    inner: vision::transforms::ToTensor,
}

#[pymethods]
impl PyToTensor {
    #[new]
    fn new() -> Self {
        Self {
            inner: vision::transforms::ToTensor::new(),
        }
    }

    /// Apply transform to f32 list
    fn __call__(&self, py: Python, data: Vec<f32>) -> PyResult<Py<PyAny>> {
        let result = self.inner.apply_f32(data).map_err(|e| {
            crate::error::convert_error(format!("Transform failed: {:?}", e))
        })?;

        // Convert tensor to Python list for now
        // TODO: Return proper tensor object when tensor bindings are ready
        let slice = result.as_slice();
        let vec: Vec<f32> = slice.iter().map(|x| x.get()).collect();
        let py_list = PyList::new(py, &vec)?;
        Ok(py_list.unbind().into())
    }

    fn __repr__(&self) -> String {
        "ToTensor()".to_string()
    }
}

/// Normalize transform - normalizes tensor data
#[pyclass(name = "Normalize", module = "_coeus")]
pub struct PyNormalize {
    inner: std::sync::Arc<vision::transforms::Normalize>,
}

#[pymethods]
impl PyNormalize {
    #[new]
    fn new(mean: Vec<f32>, std: Vec<f32>) -> PyResult<Self> {
        // Normalize::new panics on invalid input, so we need to validate manually
        if mean.len() != std.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "mean and std must have the same length",
            ));
        }
        for &s in &std {
            if s <= 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "standard deviation must be positive",
                ));
            }
        }

        // Since we validated inputs, this should succeed
        let inner = vision::transforms::Normalize::new(mean, std);
        Ok(Self {
            inner: std::sync::Arc::new(inner),
        })
    }

    /// Create single-channel normalize transform
    #[staticmethod]
    fn single_channel(mean: f32, std: f32) -> PyResult<Self> {
        let inner = vision::transforms::Normalize::single_channel(mean, std);
        Ok(Self {
            inner: std::sync::Arc::new(inner),
        })
    }

    /// Create ImageNet normalize transform
    #[staticmethod]
    fn imagenet() -> Self {
        let inner = vision::transforms::Normalize::imagenet();
        Self {
            inner: std::sync::Arc::new(inner),
        }
    }

    /// Create grayscale normalize transform
    #[staticmethod]
    fn grayscale() -> Self {
        let inner = vision::transforms::Normalize::grayscale();
        Self {
            inner: std::sync::Arc::new(inner),
        }
    }

    /// Apply transform to tensor data (placeholder for now)
    fn __call__(&self, _tensor: Py<PyAny>) -> PyResult<Py<PyAny>> {
        // TODO: Implement tensor input handling when tensor bindings are complete
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Normalize transform requires tensor input - implement when tensor bindings are complete"
        ))
    }

    /// Get mean values
    fn mean(&self) -> Vec<f32> {
        self.inner.mean().to_vec()
    }

    /// Get std values
    fn std(&self) -> Vec<f32> {
        self.inner.std().to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "Normalize(mean={:?}, std={:?})",
            self.inner.mean(),
            self.inner.std()
        )
    }
}

/// Resize transform - resizes tensor data to specified dimensions
#[pyclass(name = "Resize", module = "_coeus")]
pub struct PyResize {
    inner: std::sync::Arc<vision::transforms::Resize>,
}

#[pymethods]
impl PyResize {
    #[new]
    fn new(size: (usize, usize)) -> Self {
        Self {
            inner: std::sync::Arc::new(vision::transforms::Resize::new(size)),
        }
    }

    /// Create resize transform for ImageNet standard size
    #[staticmethod]
    fn imagenet() -> Self {
        Self {
            inner: std::sync::Arc::new(vision::transforms::Resize::imagenet()),
        }
    }

    /// Create resize transform for CIFAR-10 size
    #[staticmethod]
    fn cifar() -> Self {
        Self {
            inner: std::sync::Arc::new(vision::transforms::Resize::cifar()),
        }
    }

    /// Apply transform to tensor data (placeholder for now)
    fn __call__(&self, _tensor: Py<PyAny>) -> PyResult<Py<PyAny>> {
        // TODO: Implement tensor input handling when tensor bindings are complete
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Resize transform requires tensor input - implement when tensor bindings are complete",
        ))
    }

    /// Get target size
    fn size(&self) -> (usize, usize) {
        self.inner.size()
    }

    fn __repr__(&self) -> String {
        format!("Resize(size={:?})", self.inner.size())
    }
}

/// RandomApply transform - conditionally applies transforms with probability
#[pyclass(name = "RandomApply", module = "_coeus")]
pub struct PyRandomApply {
    transforms: Vec<Py<PyAny>>,
    probability: f32,
}

#[pymethods]
impl PyRandomApply {
    #[new]
    fn new(transforms: Vec<Py<PyAny>>, probability: f32) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Probability must be between 0.0 and 1.0",
            ));
        }

        Ok(Self {
            transforms,
            probability,
        })
    }

    /// Apply transform pipeline conditionally
    fn __call__(&self, py: Python, input: Vec<f32>) -> PyResult<Py<PyAny>> {
        // Generate random value to decide whether to apply transforms
        let random = py.import("random")?;
        let should_apply =
            random.call_method("random", (), None)?.extract::<f32>()? < self.probability;

        if !should_apply {
            // Return input unchanged
            let py_list = PyList::new(py, &input)?;
            return Ok(py_list.unbind().into());
        }

        // Apply all transforms in sequence
        let mut current = input;
        for transform in &self.transforms {
            let args = (current,);
            let result = transform.call1(py, args)?;
            current = result.extract::<Vec<f32>>(py)?;
        }

        // Return as Python list
        let py_list = PyList::new(py, &current)?;
        Ok(py_list.unbind().into())
    }

    /// Get probability
    fn p(&self) -> f32 {
        self.probability
    }

    /// Get number of transforms
    fn __len__(&self) -> usize {
        self.transforms.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "RandomApply(p={:.2}, {} transforms)",
            self.probability,
            self.transforms.len()
        )
    }
}

/// Compose transform - chains multiple transformations
#[pyclass(name = "Compose", module = "_coeus")]
pub struct PyCompose {
    transforms: Vec<Py<PyAny>>,
}

#[pymethods]
impl PyCompose {
    #[new]
    fn new(transforms: Vec<Py<PyAny>>) -> Self {
        Self { transforms }
    }

    /// Apply transform pipeline by calling each transform in sequence
    fn __call__(&self, py: Python, input: Vec<f32>) -> PyResult<Py<PyAny>> {
        let mut current = input;

        // Apply each transform in sequence
        for transform in &self.transforms {
            // Call the transform as a callable Python object
            let args = (current,);
            let result = transform.call1(py, args)?;
            current = result.extract::<Vec<f32>>(py)?;
        }

        // Return as Python list
        let py_list = PyList::new(py, &current)?;
        Ok(py_list.unbind().into())
    }

    /// Get number of transforms
    fn __len__(&self) -> usize {
        self.transforms.len()
    }

    fn __repr__(&self) -> String {
        format!("Compose({} transforms)", self.transforms.len())
    }
}
