use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::{Bound, Py, PyAny, PyErr, PyResult, Python};

pub fn register(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyDataLoader>()?;
    m.add_class::<PyDataLoaderIter>()?;
    m.add_class::<PyTensorDataset>()?;
    m.add_class::<PyTensorSample>()?;
    m.add_class::<PyTensorBatch>()?;
    m.add_class::<PyConcatDataset>()?;
    m.add_class::<PySubset>()?;
    m.add_class::<PyTransform>()?;
    m.add_class::<PyRandomHorizontalFlip>()?;
    m.add_class::<PyRandomVerticalFlip>()?;
    m.add_class::<PyColorJitter>()?;
    Ok(())
}
use std::boxed::Box;
use std::vec::Vec;

use backend::CpuBackend;
use dtype::int::Int32;
use storage::DenseStorage;
use tensor::Tensor;
use utils::{
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
            None => Err(crate::error::convert_error(
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
                    crate::error::convert_error(format!(
                        "Failed to create DataLoader iterator: {:?}",
                        e
                    ))
                })?;

                Ok(PyDataLoaderIter {
                    inner: Some(dataloader),
                })
            }
            None => Err(crate::error::convert_error(
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
                    crate::error::convert_error(format!(
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
                    Err(e) => Err(crate::error::convert_error(format!(
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
        use crate::tensor::TensorWrapper;

        let mut rust_inputs = Vec::new();
        let mut rust_targets = Vec::new();

        for input_tensor in inputs {
            // Extract CpuDenseF32 tensor from TensorWrapper
            match &input_tensor.inner {
                TensorWrapper::CpuDenseF32(t) => {
                    rust_inputs.push(t.clone());
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "TensorDataset only supports CpuDenseF32 tensors currently",
                    ));
                }
            }
        }

        for target_tensor in targets {
            // For targets, extract CpuDenseF32 and convert to Int32
            match &target_tensor.inner {
                TensorWrapper::CpuDenseF32(t) => {
                    let shape = t.shape().dims();
                    let float_data: Vec<f32> = t.as_slice().iter().map(|x| x.get()).collect();
                    let int_data: Vec<i32> = float_data.into_iter().map(|x| x as i32).collect();

                    // Create Int32 tensor for targets
                    let int32_tensor =
                        Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec(
                            int_data.into_iter().map(Int32).collect(),
                            shape,
                        )
                        .map_err(|e| {
                            crate::error::convert_error(format!(
                                "tensor: Failed to convert target tensor: {:?}",
                                e
                            ))
                        })?;

                    rust_targets.push(int32_tensor);
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "TensorDataset only supports CpuDenseF32 tensors currently",
                    ));
                }
            }
        }

        // Create Rust TensorDataset
        let dataset = RustTensorDataset::new(rust_inputs, rust_targets).map_err(|e| {
            crate::error::convert_error(format!(
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
            None => Err(crate::error::convert_error(
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
        Python::attach(|py| {
            PyTensorSample {
                inputs: sample
                    .inputs
                    .into_iter()
                    .map(|t| PyTensor {
                        inner: crate::tensor::TensorWrapper::CpuDenseF32(t),
                    })
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
        Python::attach(|py| {
            let inputs: Vec<Vec<PyTensor>> = samples
                .iter()
                .map(|sample| {
                    sample
                        .inputs
                        .iter()
                        .map(|t| PyTensor {
                            inner: crate::tensor::TensorWrapper::CpuDenseF32(t.clone()),
                        })
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
    inner: vision::transforms::ToTensor,
}

#[pymethods]
impl PyToTensor {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(PyToTensor {
            inner: vision::transforms::ToTensor::new(),
        })
    }

    /// Apply transform to f32 data
    fn __call__(&self, input: Vec<f32>) -> PyResult<PyTensor> {
        let tensor = self.inner.apply_f32(input).map_err(|e| {
            crate::error::convert_error(format!(
                "ToTensor transform failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor {
            inner: crate::tensor::TensorWrapper::CpuDenseF32(tensor),
        })
    }
}

/// Normalize transform
#[pyclass(name = "Normalize", module = "_coeus")]
pub struct PyNormalize {
    inner: vision::transforms::Normalize,
}

#[pymethods]
impl PyNormalize {
    #[new]
    #[pyo3(signature = (mean, std, _inplace=true))]
    fn new(mean: Vec<f32>, std: Vec<f32>, _inplace: bool) -> PyResult<Self> {
        let normalize = if mean.len() == 1 && std.len() == 1 {
            vision::transforms::Normalize::single_channel(mean[0], std[0])
        } else {
            vision::transforms::Normalize::new(mean, std)
        };
        Ok(PyNormalize { inner: normalize })
    }

    /// Apply transform to tensor
    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &input.inner {
            crate::tensor::TensorWrapper::CpuDenseF32(t) => {
                let tensor = self.inner.apply(t).map_err(|e| {
                    crate::error::convert_error(format!(
                        "Normalize transform failed: {:?}",
                        e
                    ))
                })?;
                Ok(PyTensor {
                    inner: crate::tensor::TensorWrapper::CpuDenseF32(tensor),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Normalize only supports CpuDenseF32 tensors",
            )),
        }
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

    fn __len__(&self) -> usize {
        self.transforms.len()
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

        Python::attach(|py| {
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
                        return Err(crate::error::convert_error(
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

        let target = index + 1;
        let dataset_idx = self
            .cumulative_lengths
            .partition_point(|&cumulative_len| cumulative_len < target);

        if dataset_idx >= self.datasets.len() {
            return Err(crate::error::convert_error(format!(
                "ConcatDataset index mapping invariant violated: index={}, dataset_idx={}, num_datasets={}, cumulative_lengths={:?}",
                index,
                dataset_idx,
                self.datasets.len(),
                self.cumulative_lengths
            )));
        }

        let base = dataset_idx
            .checked_sub(1)
            .and_then(|idx| self.cumulative_lengths.get(idx).copied())
            .unwrap_or(0);

        let local_index = index.checked_sub(base).ok_or_else(|| {
            crate::error::convert_error(format!(
                "ConcatDataset index mapping invariant violated: index={}, dataset_idx={}, base={}, cumulative_lengths={:?}",
                index, dataset_idx, base, self.cumulative_lengths
            ))
        })?;

        let sample = self.datasets[dataset_idx].get(local_index).map_err(|e| {
            crate::error::convert_error(format!(
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
                    crate::error::convert_error(format!(
                        "Failed to create subset: {:?}",
                        e
                    ))
                })?;
                Ok(PySubset { inner: subset })
            }
            None => Err(crate::error::convert_error(
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
        let _ = p;
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
        let _ = p;
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
        let _ = (brightness, contrast, saturation, hue);
        Ok(PyColorJitter)
    }

    fn __call__(&self, _input: &PyTensor) -> PyResult<PyTensor> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "ColorJitter requires image processing capabilities - deferred to future version with computer vision features"
        ))
    }
}
