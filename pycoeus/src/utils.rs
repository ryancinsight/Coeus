use crate::tensor::PyTensor;
use coeus_tensor::Tensor;
use coeus_utils::{
    data::{ConcatDataset, DataLoader, Dataset, Subset, TensorDataset},
    metrics::{
        accuracy, auc_roc, classification_report, confusion_matrix, mean_squared_error,
        top_k_accuracy,
    },
    transforms::{Identity, Transform},
};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyResult};

/// Dataset trait wrapper for Python
#[pyclass]
pub struct PyDataset {
    /// Internal dataset implementation
    dataset: Box<dyn Dataset<f32> + Send + Sync>,
}

impl Clone for PyDataset {
    fn clone(&self) -> Self {
        // Since Dataset trait objects can't be cloned, panic for now
        // This is a placeholder until proper dataset cloning is implemented
        panic!("PyDataset cloning not implemented")
    }
}

#[pymethods]
impl PyDataset {
    /// Get the length of the dataset
    fn __len__(&self) -> usize {
        self.dataset.len()
    }

    /// Get an item from the dataset
    fn __getitem__(&self, index: usize) -> PyResult<(PyTensor, PyTensor)> {
        let (data, target) = self.dataset.get(index);
        Ok((
            PyTensor::from_rust_tensor(data),
            PyTensor::from_rust_tensor(target),
        ))
    }
}

/// DataLoader wrapper for Python
#[pyclass]
#[derive(Clone)]
pub struct PyDataLoader {
    /// Internal data loader
    loader: DataLoader<coeus_utils::TensorDataset<f32>, f32>,
}

#[pymethods]
impl PyDataLoader {
    #[new]
    #[pyo3(signature = (_dataset, _batch_size=1, _shuffle=true, _num_workers=0))]
    fn new(
        _dataset: &PyDataset,
        _batch_size: usize,
        _shuffle: bool,
        _num_workers: usize,
    ) -> PyResult<Self> {
        // Note: The current implementation doesn't support parallel loading
        // due to Tensor not implementing Send + Sync
        // This is a placeholder implementation - full DataLoader support needs trait bound fixes
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "DataLoader construction from PyDataset not yet implemented",
        ))
    }

    /// Get an iterator over batches
    fn __iter__(&mut self) -> PyResult<PyDataLoaderIter> {
        Ok(PyDataLoaderIter {
            iter: self.loader.iter(),
        })
    }
}

/// Iterator for DataLoader
#[pyclass]
pub struct PyDataLoaderIter {
    iter: coeus_utils::DataLoaderIter<coeus_utils::TensorDataset<f32>, f32>,
}

#[pymethods]
impl PyDataLoaderIter {
    fn __next__(&mut self) -> PyResult<Option<(PyTensor, PyTensor)>> {
        match self.iter.next() {
            Some(batch) => Ok(Some((
                PyTensor::from_rust_tensor(batch.data),
                PyTensor::from_rust_tensor(batch.targets),
            ))),
            None => Ok(None),
        }
    }
}

/// TensorDataset wrapper
#[pyclass]
#[derive(Clone)]
pub struct PyTensorDataset {
    dataset: TensorDataset<f32>,
}

#[pymethods]
impl PyTensorDataset {
    #[new]
    fn new(data: Vec<PyTensor>, targets: Vec<PyTensor>) -> PyResult<Self> {
        let data_tensors: Vec<_> = data.into_iter().map(|t| t.tensor).collect();
        let target_tensors: Vec<_> = targets.into_iter().map(|t| t.tensor).collect();

        let dataset = TensorDataset::new(data_tensors, target_tensors);

        Ok(PyTensorDataset { dataset })
    }

    fn __len__(&self) -> usize {
        self.dataset.len()
    }

    fn __getitem__(&self, index: usize) -> PyResult<(PyTensor, PyTensor)> {
        let (data, target) = self.dataset.get(index);
        Ok((
            PyTensor::from_rust_tensor(data),
            PyTensor::from_rust_tensor(target),
        ))
    }
}

/// ConcatDataset wrapper
#[pyclass]
pub struct PyConcatDataset {
    dataset: ConcatDataset<f32>,
}

#[pymethods]
impl PyConcatDataset {
    #[new]
    fn new(_datasets: Vec<PyDataset>) -> PyResult<Self> {
        // Simplified implementation - full trait bound fixes needed for production
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "ConcatDataset construction not yet implemented",
        ))
    }

    fn __len__(&self) -> usize {
        self.dataset.len()
    }
}

/// Subset wrapper
#[pyclass]
#[derive(Clone)]
pub struct PySubset {
    dataset: Subset<coeus_utils::TensorDataset<f32>, f32>,
}

#[pymethods]
impl PySubset {
    #[new]
    fn new(_dataset: &PyDataset, _indices: Vec<usize>) -> PyResult<Self> {
        // Simplified implementation - full trait bound fixes needed for production
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Subset construction not yet implemented",
        ))
    }

    fn __len__(&self) -> usize {
        self.dataset.len()
    }
}

/// Transform base class
#[pyclass(subclass)]
#[allow(dead_code)]
pub struct PyTransform {
    transform: Box<dyn Transform<f32> + Send + Sync>,
}

#[pymethods]
impl PyTransform {
    #[new]
    fn new() -> Self {
        // Default identity transform
        PyTransform {
            transform: Box::new(Identity::new()),
        }
    }

    fn __call__(&self, _input: PyTensor) -> PyResult<PyTensor> {
        // Simplified implementation - full Transform trait support needed
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Transform apply not yet implemented",
        ))
    }
}

/// Compose transform
#[pyclass]
#[allow(dead_code)]
pub struct PyCompose {
    transforms: Vec<Box<dyn Transform<f32> + Send + Sync>>,
}

#[pymethods]
impl PyCompose {
    #[new]
    fn new(_transforms: Vec<PyObject>) -> PyResult<Self> {
        // For now, create an empty compose - full implementation would require
        // proper conversion from PyObject to specific transform types
        Ok(PyCompose {
            transforms: Vec::new(),
        })
    }
}

/// Normalize transform
#[pyclass]
#[allow(dead_code)]
pub struct PyNormalize {
    mean: Vec<f32>,
    std: Vec<f32>,
}

#[pymethods]
impl PyNormalize {
    #[new]
    #[pyo3(signature = (mean, std, _inplace=false))]
    fn new(mean: Vec<f32>, std: Vec<f32>, _inplace: bool) -> PyResult<Self> {
        Ok(PyNormalize { mean, std })
    }
}

/// ToTensor transform
#[pyclass]
#[derive(Clone)]
pub struct PyToTensor {}

#[pymethods]
impl PyToTensor {
    #[new]
    fn new() -> Self {
        PyToTensor {}
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        Ok(input.clone())
    }
}

/// Vision transforms
#[pyclass]
#[derive(Clone)]
pub struct PyRandomHorizontalFlip {}

#[pymethods]
impl PyRandomHorizontalFlip {
    #[new]
    #[pyo3(signature = (_p=0.5))]
    fn new(_p: f32) -> Self {
        PyRandomHorizontalFlip {}
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyRandomVerticalFlip {}

#[pymethods]
impl PyRandomVerticalFlip {
    #[new]
    #[pyo3(signature = (_p=0.5))]
    fn new(_p: f32) -> Self {
        PyRandomVerticalFlip {}
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyColorJitter {}

#[pymethods]
impl PyColorJitter {
    #[new]
    #[pyo3(signature = (_brightness=0.0, _contrast=0.0, _saturation=0.0, _hue=0.0))]
    fn new(_brightness: f32, _contrast: f32, _saturation: f32, _hue: f32) -> Self {
        PyColorJitter {}
    }
}

// Metrics functions

/// Accuracy metric
#[pyfunction]
pub fn py_accuracy(predictions: &PyTensor, targets: &PyTensor) -> PyResult<f32> {
    // Convert targets to i64 for metrics functions
    let targets_i64: Tensor<i64> = Tensor::from_vec(
        targets.tensor.data().iter().map(|&x| x as i64).collect(),
        targets.tensor.shape().to_vec(),
    );
    accuracy(&predictions.tensor, &targets_i64)
        .map(|result| result as f32)
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Accuracy calculation failed: {}", e))
        })
}

/// Top-k accuracy metric
#[pyfunction]
#[pyo3(signature = (predictions, targets, k=1))]
pub fn py_top_k_accuracy(predictions: &PyTensor, targets: &PyTensor, k: usize) -> PyResult<f32> {
    // Convert targets to i64 for metrics functions
    let targets_i64: Tensor<i64> = Tensor::from_vec(
        targets.tensor.data().iter().map(|&x| x as i64).collect(),
        targets.tensor.shape().to_vec(),
    );
    top_k_accuracy(&predictions.tensor, &targets_i64, k)
        .map(|result| result as f32)
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Top-k accuracy calculation failed: {}",
                e
            ))
        })
}

/// Confusion matrix
#[pyfunction]
pub fn py_confusion_matrix(
    predictions: &PyTensor,
    targets: &PyTensor,
    num_classes: usize,
) -> PyResult<Vec<Vec<i32>>> {
    // Convert predictions and targets to i64 for metrics functions
    let predictions_i64: Tensor<i64> = Tensor::from_vec(
        predictions
            .tensor
            .data()
            .iter()
            .map(|&x| x as i64)
            .collect(),
        predictions.tensor.shape().to_vec(),
    );
    let targets_i64: Tensor<i64> = Tensor::from_vec(
        targets.tensor.data().iter().map(|&x| x as i64).collect(),
        targets.tensor.shape().to_vec(),
    );
    confusion_matrix(&predictions_i64, &targets_i64, num_classes)
        .map(|tensor| {
            // Convert Tensor<i64> to Vec<Vec<i32>>
            let data = tensor.data();
            let shape = tensor.shape();
            let rows = shape[0];
            let cols = shape[1];
            let mut result = vec![vec![0i32; cols]; rows];
            for i in 0..rows {
                for j in 0..cols {
                    result[i][j] = data[i * cols + j] as i32;
                }
            }
            result
        })
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Confusion matrix calculation failed: {}",
                e
            ))
        })
}

/// Classification report
#[pyfunction]
pub fn py_classification_report(
    predictions: &PyTensor,
    targets: &PyTensor,
    num_classes: usize,
) -> PyResult<String> {
    // Convert predictions and targets to i64 for metrics functions
    let predictions_i64: Tensor<i64> = Tensor::from_vec(
        predictions
            .tensor
            .data()
            .iter()
            .map(|&x| x as i64)
            .collect(),
        predictions.tensor.shape().to_vec(),
    );
    let targets_i64: Tensor<i64> = Tensor::from_vec(
        targets.tensor.data().iter().map(|&x| x as i64).collect(),
        targets.tensor.shape().to_vec(),
    );
    classification_report(&predictions_i64, &targets_i64, num_classes)
        .map(|report| report.to_string())
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Classification report failed: {}",
                e
            ))
        })
}

/// Mean squared error
#[pyfunction]
pub fn py_mean_squared_error(predictions: &PyTensor, targets: &PyTensor) -> PyResult<f32> {
    // MSE works with f32 targets, no conversion needed
    mean_squared_error(&predictions.tensor, &targets.tensor)
        .map(|result| result as f32)
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("MSE calculation failed: {}", e))
        })
}

/// AUC-ROC score
#[pyfunction]
pub fn py_auc_roc(predictions: &PyTensor, targets: &PyTensor) -> PyResult<f32> {
    // Convert targets to i64 for AUC-ROC function
    let targets_i64: Tensor<i64> = Tensor::from_vec(
        targets.tensor.data().iter().map(|&x| x as i64).collect(),
        targets.tensor.shape().to_vec(),
    );
    auc_roc(&predictions.tensor, &targets_i64)
        .map(|result| result as f32)
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("AUC-ROC calculation failed: {}", e))
        })
}

// Legacy utility functions (kept for compatibility)

/// Set the number of threads for CPU operations
#[pyfunction]
pub fn set_num_threads(num_threads: usize) -> PyResult<()> {
    crate::tensor::PyTensor::set_num_threads(num_threads)
}

/// Get the current number of threads for CPU operations
#[pyfunction]
pub fn get_num_threads() -> PyResult<usize> {
    crate::tensor::PyTensor::get_num_threads()
}

/// Set the random seed for reproducible results
#[pyfunction]
pub fn manual_seed(seed: u64) -> PyResult<()> {
    crate::tensor::PyTensor::manual_seed(seed)
}

/// Check if CUDA is available
#[pyfunction]
pub fn cuda_is_available() -> PyResult<bool> {
    crate::tensor::PyTensor::cuda_is_available()
}
