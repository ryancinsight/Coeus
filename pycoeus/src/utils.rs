use pyo3::prelude::*;
use pyo3::pyclass;

/// Dataset base class (placeholder)
#[pyclass(name = "Dataset", module = "_coeus", subclass)]
pub struct PyDataset;

#[pymethods]
impl PyDataset {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Dataset not yet implemented in Coeus"
        ))
    }
}

/// DataLoader (placeholder)
#[pyclass(name = "DataLoader", module = "_coeus")]
pub struct PyDataLoader;

#[pymethods]
impl PyDataLoader {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "DataLoader not yet implemented in Coeus"
        ))
    }
}

/// DataLoader iterator (placeholder)
#[pyclass(name = "DataLoaderIter", module = "_coeus")]
pub struct PyDataLoaderIter;

#[pymethods]
impl PyDataLoaderIter {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "DataLoaderIter not yet implemented in Coeus"
        ))
    }
}

/// TensorDataset (placeholder)
#[pyclass(name = "TensorDataset", module = "_coeus")]
pub struct PyTensorDataset;

#[pymethods]
impl PyTensorDataset {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "TensorDataset not yet implemented in Coeus"
        ))
    }
}

/// ConcatDataset (placeholder)
#[pyclass(name = "ConcatDataset", module = "_coeus")]
pub struct PyConcatDataset;

#[pymethods]
impl PyConcatDataset {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "ConcatDataset not yet implemented in Coeus"
        ))
    }
}

/// Subset (placeholder)
#[pyclass(name = "Subset", module = "_coeus")]
pub struct PySubset;

#[pymethods]
impl PySubset {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Subset not yet implemented in Coeus"
        ))
    }
}

/// Transform base class (placeholder)
#[pyclass(name = "Transform", module = "_coeus", subclass)]
pub struct PyTransform;

#[pymethods]
impl PyTransform {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Transform not yet implemented in Coeus"
        ))
    }
}

/// Compose transform (placeholder)
#[pyclass(name = "Compose", module = "_coeus")]
pub struct PyCompose;

#[pymethods]
impl PyCompose {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Compose not yet implemented in Coeus"
        ))
    }
}

/// Normalize transform (placeholder)
#[pyclass(name = "Normalize", module = "_coeus")]
pub struct PyNormalize;

#[pymethods]
impl PyNormalize {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Normalize not yet implemented in Coeus"
        ))
    }
}

/// ToTensor transform (placeholder)
#[pyclass(name = "ToTensor", module = "_coeus")]
pub struct PyToTensor;

#[pymethods]
impl PyToTensor {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "ToTensor not yet implemented in Coeus"
        ))
    }
}

/// RandomHorizontalFlip transform (placeholder)
#[pyclass(name = "RandomHorizontalFlip", module = "_coeus")]
pub struct PyRandomHorizontalFlip;

#[pymethods]
impl PyRandomHorizontalFlip {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "RandomHorizontalFlip not yet implemented in Coeus"
        ))
    }
}

/// RandomVerticalFlip transform (placeholder)
#[pyclass(name = "RandomVerticalFlip", module = "_coeus")]
pub struct PyRandomVerticalFlip;

#[pymethods]
impl PyRandomVerticalFlip {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "RandomVerticalFlip not yet implemented in Coeus"
        ))
    }
}

/// ColorJitter transform (placeholder)
#[pyclass(name = "ColorJitter", module = "_coeus")]
pub struct PyColorJitter;

#[pymethods]
impl PyColorJitter {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "ColorJitter not yet implemented in Coeus"
        ))
    }
}
