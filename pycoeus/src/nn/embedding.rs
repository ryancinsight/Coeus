use crate::tensor::{to_py_err, PyTensor, TensorWrapper};
use pyo3::prelude::*;

use coeus_nn::core::module::Module;
use coeus_nn::modules::embedding::Embedding;

use backend::CpuBackend;
#[cfg(feature = "gpu")]
use backend::GpuBackend;
use dtype::float::{Float32, Float64};
use storage::DenseStorage;

#[derive(Clone)]
pub enum EmbeddingWrapper {
    CpuF32(Embedding<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(Embedding<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(Embedding<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[pyclass(name = "Embedding", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyEmbedding {
    pub inner: EmbeddingWrapper,
}

#[pymethods]
impl PyEmbedding {
    #[new]
    #[pyo3(signature = (num_embeddings, embedding_dim, padding_idx=None, dtype="float32", device="cpu"))]
    fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let dtype_str = dtype.unwrap_or("float32");
        let device_str = device.unwrap_or("cpu");

        let inner = match (device_str, dtype_str) {
            ("cpu", "float32") => {
                let embedding =
                    Embedding::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                        num_embeddings,
                        embedding_dim,
                        padding_idx,
                    )
                    .map_err(to_py_err)?;
                EmbeddingWrapper::CpuF32(embedding)
            }
            ("cpu", "float64") => {
                let embedding =
                    Embedding::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                        num_embeddings,
                        embedding_dim,
                        padding_idx,
                    )
                    .map_err(to_py_err)?;
                EmbeddingWrapper::CpuF64(embedding)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") => {
                let embedding =
                    Embedding::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                        num_embeddings,
                        embedding_dim,
                        padding_idx,
                    )
                    .map_err(to_py_err)?;
                EmbeddingWrapper::GpuF32(embedding)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported device/dtype combination: {}/{}",
                    device_str, dtype_str
                )))
            }
        };

        Ok(PyEmbedding { inner })
    }

    #[getter]
    fn num_embeddings(&self) -> usize {
        match &self.inner {
            EmbeddingWrapper::CpuF32(inner) => inner.num_embeddings,
            EmbeddingWrapper::CpuF64(inner) => inner.num_embeddings,
            #[cfg(feature = "gpu")]
            EmbeddingWrapper::GpuF32(inner) => inner.num_embeddings,
        }
    }

    #[getter]
    fn embedding_dim(&self) -> usize {
        match &self.inner {
            EmbeddingWrapper::CpuF32(inner) => inner.embedding_dim,
            EmbeddingWrapper::CpuF64(inner) => inner.embedding_dim,
            #[cfg(feature = "gpu")]
            EmbeddingWrapper::GpuF32(inner) => inner.embedding_dim,
        }
    }

    #[getter]
    fn weight(&self) -> PyResult<PyTensor> {
        match &self.inner {
            EmbeddingWrapper::CpuF32(inner) => Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(inner.weight.data().clone()),
            }),
            EmbeddingWrapper::CpuF64(inner) => Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(inner.weight.data().clone()),
            }),
            #[cfg(feature = "gpu")]
            EmbeddingWrapper::GpuF32(inner) => Ok(PyTensor {
                inner: TensorWrapper::GpuDenseF32(inner.weight.data().clone()),
            }),
        }
    }

    fn train(&mut self, mode: bool) {
        match &mut self.inner {
            EmbeddingWrapper::CpuF32(inner) => inner.train(mode),
            EmbeddingWrapper::CpuF64(inner) => inner.train(mode),
            #[cfg(feature = "gpu")]
            EmbeddingWrapper::GpuF32(inner) => inner.train(mode),
        }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (EmbeddingWrapper::CpuF32(s), TensorWrapper::CpuDenseF32(i)) => {
                let res = s.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (EmbeddingWrapper::CpuF64(s), TensorWrapper::CpuDenseF64(i)) => {
                let res = s.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (EmbeddingWrapper::GpuF32(s), TensorWrapper::GpuDenseF32(i)) => {
                let res = s.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Embedding forward: device/dtype mismatch between container and input",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        match &self.inner {
            EmbeddingWrapper::CpuF32(s) => s
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::CpuDenseF32(p.data().clone()),
                })
                .collect(),
            EmbeddingWrapper::CpuF64(s) => s
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::CpuDenseF64(p.data().clone()),
                })
                .collect(),
            #[cfg(feature = "gpu")]
            EmbeddingWrapper::GpuF32(s) => s
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::GpuDenseF32(p.data().clone()),
                })
                .collect(),
        }
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEmbedding>()?;

    // Add to module __dict__ for dir() visibility (PyTorch compatibility)
    let dict = m.dict();
    dict.set_item("Embedding", m.getattr("Embedding")?)?;

    Ok(())
}
