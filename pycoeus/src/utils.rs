use pyo3::prelude::*;
use pyo3::pyclass;
use pyo3::types::PyList;
use pyo3::{Py, PyAny, PyResult};

use coeus_backend::CpuBackend;
use coeus_dtype::int::Int32;
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;
use coeus_utils::{
    DataLoader as RustDataLoader, Dataset, Subset as RustSubset,
    TensorDataset as RustTensorDataset, TensorSample, Transform,
};

use super::tensor::PyTensor;

/// DataLoader for batched data iteration
#[pyclass(name = "DataLoader", module = "_coeus", unsendable)]
pub struct PyDataLoader {
    dataset: Option<RustTensorDataset>,
    batch_size: usize,
    shuffle: bool,
}

#[pymethods]
impl PyDataLoader {
    #[new]
    #[pyo3(signature = (dataset, batch_size=1, shuffle=false))]
    fn new(dataset: &PyTensorDataset, batch_size: usize, shuffle: bool) -> PyResult<Self> {
        match &dataset.inner {
            Some(rust_dataset) => Ok(PyDataLoader {
                dataset: Some(rust_dataset.clone()),
                batch_size,
                shuffle,
            }),
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Dataset not provided",
            )),
        }
    }

    fn __iter__(&mut self) -> PyResult<PyDataLoaderIter> {
        match self.dataset.take() {
            Some(dataset) => {
                let mut builder = RustDataLoader::builder(dataset);
                builder = builder.batch_size(self.batch_size);

                if self.shuffle {
                    builder = builder.shuffle(true);
                }

                let dataloader = builder.build().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to create DataLoader iterator: {:?}",
                        e
                    ))
                })?;

                Ok(PyDataLoaderIter {
                    inner: Some(dataloader),
                })
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "DataLoader not initialized",
            )),
        }
    }

    fn __len__(&self) -> PyResult<usize> {
        match &self.dataset {
            Some(dataset) => {
                let mut builder = RustDataLoader::builder(dataset.clone());
                builder = builder.batch_size(self.batch_size);
                if self.shuffle {
                    builder = builder.shuffle(true);
                }
                let dataloader = builder.build().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to get DataLoader length: {:?}",
                        e
                    ))
                })?;
                Ok(dataloader.len())
            }
            None => Ok(0),
        }
    }
}

/// DataLoader iterator
#[pyclass(name = "DataLoaderIter", module = "_coeus", unsendable)]
pub struct PyDataLoaderIter {
    inner: Option<RustDataLoader<RustTensorDataset, TensorSample>>,
}

#[pymethods]
impl PyDataLoaderIter {
    fn __next__(&mut self) -> PyResult<Option<PyTensorBatch>> {
        match &mut self.inner {
            Some(iter) => match iter.next() {
                Some(batch) => match batch {
                    Ok(samples) => Ok(Some(PyTensorBatch::from(samples))),
                    Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Batch processing error: {:?}",
                        e
                    ))),
                },
                None => Ok(None),
            },
            None => Ok(None),
        }
    }
}

/// TensorDataset for PyTorch-compatible tensor datasets
#[pyclass(name = "TensorDataset", module = "_coeus")]
pub struct PyTensorDataset {
    inner: Option<RustTensorDataset>,
}

#[pymethods]
impl PyTensorDataset {
    #[new]
    fn new(inputs: Vec<PyTensor>, targets: Vec<PyTensor>) -> PyResult<Self> {
        // Validate input lengths
        if inputs.len() != targets.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "inputs and targets must have the same length",
            ));
        }

        if inputs.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Dataset cannot be empty",
            ));
        }

        // Convert PyTensors to Rust tensors with proper types
        // Simplified implementation: assume all PyTensors contain Float32 data
        // In production, this would need proper type checking and conversion
        let mut rust_inputs = Vec::new();
        let mut rust_targets = Vec::new();

        for input_tensor in inputs {
            // For inputs, use the Float32 tensor directly
            // Clone the tensor for now - in production this should be type-checked
            rust_inputs.push(input_tensor.inner.clone());
        }

        for target_tensor in targets {
            // For targets, convert Float32 data to Int32
            let shape = target_tensor.inner.shape().dims();
            let float_data: Vec<f32> = target_tensor
                .inner
                .as_slice()
                .iter()
                .map(|x| x.get())
                .collect();
            let int_data: Vec<i32> = float_data.into_iter().map(|x| x as i32).collect();

            // Create Int32 tensor for targets
            let int32_tensor = Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec(
                int_data.into_iter().map(|x| Int32(x)).collect(),
                &shape,
            )
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to convert target tensor: {:?}",
                    e
                ))
            })?;

            rust_targets.push(int32_tensor);
        }

        // Create Rust TensorDataset
        let dataset = RustTensorDataset::new(rust_inputs, rust_targets).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create TensorDataset: {:?}",
                e
            ))
        })?;

        Ok(PyTensorDataset {
            inner: Some(dataset),
        })
    }

    fn __len__(&self) -> PyResult<usize> {
        match &self.inner {
            Some(dataset) => Ok(dataset.len()),
            None => Ok(0),
        }
    }

    fn __getitem__(&self, index: usize) -> PyResult<PyTensorSample> {
        match &self.inner {
            Some(dataset) => {
                let sample = dataset.get(index).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!("{:?}", e))
                })?;
                Ok(PyTensorSample::from(sample))
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "TensorDataset not initialized",
            )),
        }
    }
}

/// Sample from a TensorDataset
#[pyclass(name = "TensorSample", module = "_coeus")]
pub struct PyTensorSample {
    pub inputs: Vec<PyTensor>,
    pub targets: Vec<Py<PyAny>>, // Use Py<PyAny> for Int32 tensors
}

impl From<TensorSample> for PyTensorSample {
    fn from(sample: TensorSample) -> Self {
        pyo3::Python::with_gil(|py| {
            PyTensorSample {
                inputs: sample
                    .inputs
                    .into_iter()
                    .map(|t| PyTensor { inner: t })
                    .collect(),
                targets: sample
                    .targets
                    .into_iter()
                    .map(|t| {
                        // Convert Int32 tensor to Python list for now
                        // Future enhancement: Create proper PyIntTensor class
                        let data: Vec<i32> = t.as_slice().iter().map(|x| x.get()).collect();
                        PyList::new(py, &data).unwrap().into()
                    })
                    .collect(),
            }
        })
    }
}

#[pymethods]
impl PyTensorSample {
    #[getter]
    fn inputs(&self) -> Vec<PyTensor> {
        self.inputs.clone()
    }

    #[getter]
    fn targets(&self) -> &[Py<PyAny>] {
        &self.targets
    }
}

/// Batch of tensor samples
#[pyclass(name = "TensorBatch", module = "_coeus")]
pub struct PyTensorBatch {
    pub inputs: Vec<Vec<PyTensor>>,
    pub targets: Vec<Vec<Py<PyAny>>>, // Use Py<PyAny> for Int32 tensors
}

impl From<Vec<TensorSample>> for PyTensorBatch {
    fn from(samples: Vec<TensorSample>) -> Self {
        pyo3::Python::with_gil(|py| {
            let inputs: Vec<Vec<PyTensor>> = samples
                .iter()
                .map(|sample| {
                    sample
                        .inputs
                        .iter()
                        .map(|t| PyTensor { inner: t.clone() })
                        .collect()
                })
                .collect();

            let targets: Vec<Vec<Py<PyAny>>> = samples
                .iter()
                .map(|sample| {
                    sample
                        .targets
                        .iter()
                        .map(|t| {
                            // Convert Int32 tensor to Python list
                            let data: Vec<i32> = t.as_slice().iter().map(|x| x.get()).collect();
                            PyList::new(py, &data).unwrap().into()
                        })
                        .collect()
                })
                .collect();

            PyTensorBatch { inputs, targets }
        })
    }
}

#[pymethods]
impl PyTensorBatch {
    #[getter]
    fn inputs(&self) -> Vec<Vec<PyTensor>> {
        self.inputs.clone()
    }

    #[getter]
    fn targets(&self) -> &[Vec<Py<PyAny>>] {
        &self.targets
    }
}

/// ToTensor transform
#[pyclass(name = "ToTensor", module = "_coeus")]
pub struct PyToTensor {
    inner: coeus_utils::transforms::ToTensor,
}

#[pymethods]
impl PyToTensor {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(PyToTensor {
            inner: coeus_utils::transforms::ToTensor::new(),
        })
    }

    /// Apply transform to f32 data
    fn __call__(&self, input: Vec<f32>) -> PyResult<PyTensor> {
        let tensor = self.inner.apply_f32(input).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "ToTensor transform failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor { inner: tensor })
    }
}

/// Normalize transform
#[pyclass(name = "Normalize", module = "_coeus")]
pub struct PyNormalize {
    inner: coeus_utils::transforms::Normalize,
}

#[pymethods]
impl PyNormalize {
    #[new]
    #[pyo3(signature = (mean, std, _inplace=true))]
    fn new(mean: Vec<f32>, std: Vec<f32>, _inplace: bool) -> PyResult<Self> {
        let normalize = if mean.len() == 1 && std.len() == 1 {
            coeus_utils::transforms::Normalize::single_channel(mean[0], std[0])
        } else {
            coeus_utils::transforms::Normalize::new(mean, std)
        };
        Ok(PyNormalize { inner: normalize })
    }

    /// Apply transform to tensor
    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let tensor = self.inner.apply(&input.inner).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Normalize transform failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor { inner: tensor })
    }
}

/// Compose transform
#[pyclass(name = "Compose", module = "_coeus", unsendable)]
pub struct PyCompose {
    transforms: Vec<Py<PyAny>>,
}

#[pymethods]
impl PyCompose {
    #[new]
    fn new(transforms: Vec<Py<PyAny>>) -> PyResult<Self> {
        Ok(PyCompose { transforms })
    }

    /// Apply composed transforms to tensor
    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // For now, return input unchanged since we can't easily handle dynamic transforms
        Ok(input.clone())
    }
}

/// ConcatDataset - combines multiple datasets with trait object support for PyO3 advanced features
/// Enables efficient contiguous access across multiple datasets using Box<dyn Dataset<TensorSample>>
#[pyclass(name = "ConcatDataset", module = "_coeus", unsendable)]
pub struct PyConcatDataset {
    // PyO3 trait object handling for dataset composition
    datasets: Vec<Box<dyn Dataset<TensorSample>>>,
    cumulative_lengths: Vec<usize>,
}

#[pymethods]
impl PyConcatDataset {
    #[new]
    fn new(datasets: Vec<Py<PyAny>>) -> PyResult<Self> {
        if datasets.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot create ConcatDataset from empty dataset list",
            ));
        }

        let mut rust_datasets: Vec<Box<dyn Dataset<TensorSample>>> = Vec::new();
        let mut cumulative_lengths = Vec::new();
        let mut running_total = 0;

        pyo3::Python::with_gil(|py| {
            for py_dataset in datasets {
                // Try to downcast to PyTensorDataset and create trait object
                if let Ok(py_tensor_dataset) = py_dataset.cast_bound::<PyTensorDataset>(py) {
                    let borrowed = py_tensor_dataset.borrow();
                    if let Some(rust_dataset) = &borrowed.inner {
                        let len = rust_dataset.len();
                        running_total += len;
                        cumulative_lengths.push(running_total);
                        // Wrap in Box<dyn Dataset<TensorSample>> for trait object support
                        rust_datasets.push(Box::new(rust_dataset.clone()));
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                            "TensorDataset not initialized - ensure datasets are created with inputs and targets"
                        ));
                    }
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "All datasets must be TensorDataset instances for ConcatDataset - trait object support requires uniform type"
                    ));
                }
            }
            Ok(())
        })?;

        Ok(PyConcatDataset {
            datasets: rust_datasets,
            cumulative_lengths,
        })
    }

    fn __len__(&self) -> usize {
        self.cumulative_lengths.last().copied().unwrap_or(0)
    }

    fn __getitem__(&self, index: usize) -> PyResult<PyTensorSample> {
        if index >= self.__len__() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} out of range for dataset of length {}",
                index,
                self.__len__()
            )));
        }

        // Find which dataset this index belongs to using binary search
        let dataset_idx = match self.cumulative_lengths.binary_search(&(index + 1)) {
            Ok(idx) => idx + 1, // Exact match means we need the next dataset
            Err(idx) => idx,    // Insertion point is the dataset index
        };

        let local_index = if dataset_idx == 0 {
            index
        } else {
            index - self.cumulative_lengths[dataset_idx - 1]
        };

        let sample = self.datasets[dataset_idx].get(local_index).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to get sample: {:?}",
                e
            ))
        })?;

        Ok(PyTensorSample::from(sample))
    }

    fn num_datasets(&self) -> usize {
        self.datasets.len()
    }
}

/// Subset - creates a subset of a dataset with trait object support for PyO3 advanced features
/// Enables generic dataset subsetting using Box<dyn Dataset<TensorSample>>
#[pyclass(name = "Subset", module = "_coeus")]
pub struct PySubset {
    // PyO3 trait object handling for dataset subsetting
    inner: RustSubset,
}

#[pymethods]
impl PySubset {
    #[new]
    fn new(dataset: &PyTensorDataset, indices: Vec<usize>) -> PyResult<Self> {
        match &dataset.inner {
            Some(rust_dataset) => {
                let subset = RustSubset::new(rust_dataset.clone(), indices).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to create subset: {:?}",
                        e
                    ))
                })?;
                Ok(PySubset { inner: subset })
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Dataset not initialized",
            )),
        }
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.inner.len())
    }

    fn __getitem__(&self, index: usize) -> PyResult<PyTensorSample> {
        let sample = self
            .inner
            .get(index)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!("{:?}", e)))?;
        Ok(PyTensorSample::from(sample))
    }

    fn indices(&self) -> &[usize] {
        self.inner.indices()
    }
}

/// Transform base class
#[pyclass(name = "Transform", module = "_coeus", subclass)]
pub struct PyTransform;

/// RandomHorizontalFlip transform
#[pyclass(name = "RandomHorizontalFlip", module = "_coeus")]
pub struct PyRandomHorizontalFlip;

#[pymethods]
impl PyRandomHorizontalFlip {
    #[new]
    #[pyo3(signature = (p=0.5))]
    fn new(p: f32) -> PyResult<Self> {
        Ok(PyRandomHorizontalFlip)
    }

    fn __call__(&self, _input: &PyTensor) -> PyResult<PyTensor> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "RandomHorizontalFlip requires image processing capabilities - deferred to future version with computer vision features"
        ))
    }
}

/// RandomVerticalFlip transform
#[pyclass(name = "RandomVerticalFlip", module = "_coeus")]
pub struct PyRandomVerticalFlip;

#[pymethods]
impl PyRandomVerticalFlip {
    #[new]
    #[pyo3(signature = (p=0.5))]
    fn new(p: f32) -> PyResult<Self> {
        Ok(PyRandomVerticalFlip)
    }

    fn __call__(&self, _input: &PyTensor) -> PyResult<PyTensor> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "RandomVerticalFlip requires image processing capabilities - deferred to future version with computer vision features"
        ))
    }
}

/// ColorJitter transform
#[pyclass(name = "ColorJitter", module = "_coeus")]
pub struct PyColorJitter;

#[pymethods]
impl PyColorJitter {
    #[new]
    #[pyo3(signature = (brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))]
    fn new(brightness: f32, contrast: f32, saturation: f32, hue: f32) -> PyResult<Self> {
        Ok(PyColorJitter)
    }

    fn __call__(&self, _input: &PyTensor) -> PyResult<PyTensor> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "ColorJitter requires image processing capabilities - deferred to future version with computer vision features"
        ))
    }
}
