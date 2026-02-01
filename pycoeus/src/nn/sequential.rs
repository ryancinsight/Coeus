use crate::tensor::{to_py_err, PyTensor, TensorWrapper};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};

use backend::CpuBackend;
#[cfg(feature = "gpu")]
use backend::GpuBackend;
use dtype::float::{Float32, Float64};
use storage::DenseStorage;

use coeus_nn::containers::sequential::Sequential;
use coeus_nn::modules::activation::{GeLU, Hardtanh, LeakyReLU, Mish, ReLU, SiLU, Softplus, ELU};
use coeus_nn::modules::linear::Linear;

use coeus_nn::modules::regularization::dropout::Dropout;

use coeus_nn::core::module::Module;

// Import binding classes for type extraction
use crate::nn::activations::{
    PyELU, PyGeLU, PyHardtanh, PyLeakyReLU, PyMish, PyReLU, PySiLU, PySoftplus,
};
use crate::nn::conv::{Conv1DWrapper, Conv2DWrapper, Conv3DWrapper, PyConv1d, PyConv2d, PyConv3d};
use crate::nn::dropout::PyDropout;
use crate::nn::linear::{LinearWrapper, PyLinear};
use crate::nn::normalization::{
    BatchNorm1DWrapper, BatchNorm2DWrapper, BatchNorm3DWrapper, GroupNormWrapper, LayerNormWrapper,
    PyBatchNorm1d, PyBatchNorm2d, PyBatchNorm3d, PyGroupNorm, PyLayerNorm, PyRMSNorm,
    RMSNormWrapper,
};
use crate::nn::pooling::{MaxPool2dWrapper, PyMaxPool2d};

#[derive(Clone)]
pub enum SequentialWrapper {
    CpuF32(Sequential<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(Sequential<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(Sequential<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[pyclass(name = "Sequential", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PySequential {
    pub inner: SequentialWrapper,
}

#[pymethods]
impl PySequential {
    #[new]
    #[pyo3(signature = (*args, dtype="float32", device="cpu"))]
    fn new(args: &Bound<PyTuple>, dtype: Option<&str>, device: Option<&str>) -> PyResult<Self> {
        let dtype_str = dtype.unwrap_or("float32");
        let device_str = device.unwrap_or("cpu");

        let mut inner = match (device_str, dtype_str) {
            ("cpu", "float32") => SequentialWrapper::CpuF32(Sequential::new()),
            ("cpu", "float64") => SequentialWrapper::CpuF64(Sequential::new()),
            #[cfg(feature = "gpu")]
            ("cuda", "float32") => SequentialWrapper::GpuF32(Sequential::new()),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported device/dtype combination: {}/{}",
                    device_str, dtype_str
                )))
            }
        };

        for (i, arg) in args.iter().enumerate() {
            let name = i.to_string();

            match &mut inner {
                SequentialWrapper::CpuF32(seq) => {
                    if let Ok(m) = arg.extract::<PyLinear>() {
                        if let LinearWrapper::CpuF32(inner) = &m.inner {
                            seq.add_module(name, inner.clone());
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(cpu, f32)",
                            ));
                        }
                    } else if let Ok(m) = arg.extract::<PyConv2d>() {
                        if let Conv2DWrapper::CpuF32(inner) = &m.inner {
                            seq.add_module(name, inner.clone());
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(cpu, f32)",
                            ));
                        }
                    } else if let Ok(m) = arg.extract::<PyConv1d>() {
                        if let Conv1DWrapper::CpuF32(inner) = &m.inner {
                            seq.add_module(name, inner.clone());
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(cpu, f32)",
                            ));
                        }
                    } else if let Ok(m) = arg.extract::<PyConv3d>() {
                        if let Conv3DWrapper::CpuF32(inner) = &m.inner {
                            seq.add_module(name, inner.clone());
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(cpu, f32)",
                            ));
                        }
                    } else if let Ok(m) = arg.extract::<PyBatchNorm1d>() {
                        if let BatchNorm1DWrapper::CpuF32(inner) = &m.inner {
                            seq.add_module(name, inner.clone());
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(cpu, f32)",
                            ));
                        }
                    } else if let Ok(m) = arg.extract::<PyBatchNorm2d>() {
                        if let BatchNorm2DWrapper::CpuF32(inner) = &m.inner {
                            seq.add_module(name, inner.clone());
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(cpu, f32)",
                            ));
                        }
                    } else if let Ok(m) = arg.extract::<PyBatchNorm3d>() {
                        if let BatchNorm3DWrapper::CpuF32(inner) = &m.inner {
                            seq.add_module(name, inner.clone());
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(cpu, f32)",
                            ));
                        }
                    } else if let Ok(m) = arg.extract::<PyLayerNorm>() {
                        if let LayerNormWrapper::CpuF32(inner) = &m.inner {
                            seq.add_module(name, inner.clone());
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(cpu, f32)",
                            ));
                        }
                    } else if let Ok(m) = arg.extract::<PyGroupNorm>() {
                        if let GroupNormWrapper::CpuF32(inner) = &m.inner {
                            seq.add_module(name, inner.clone());
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(cpu, f32)",
                            ));
                        }
                    } else if let Ok(m) = arg.extract::<PyRMSNorm>() {
                        if let RMSNormWrapper::CpuF32(inner) = &m.inner {
                            seq.add_module(name, inner.clone());
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(cpu, f32)",
                            ));
                        }
                    } else if let Ok(_m) = arg.extract::<PyReLU>() {
                        seq.add_module(
                            name,
                            ReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(),
                        );
                    } else if let Ok(_m) = arg.extract::<PyGeLU>() {
                        seq.add_module(
                            name,
                            GeLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(),
                        );
                    } else if let Ok(_m) = arg.extract::<PySiLU>() {
                        seq.add_module(
                            name,
                            SiLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(),
                        );
                    } else if let Ok(m) = arg.extract::<PyLeakyReLU>() {
                        seq.add_module(
                            name,
                            LeakyReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                                Float32::new(m.negative_slope as f32),
                            ),
                        );
                    } else if let Ok(m) = arg.extract::<PyELU>() {
                        seq.add_module(
                            name,
                            ELU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                                Float32::new(m.alpha as f32),
                            ),
                        );
                    } else if let Ok(m) = arg.extract::<PySoftplus>() {
                        seq.add_module(
                            name,
                            Softplus::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                                Float32::new(m.beta as f32),
                                Float32::new(m.threshold as f32),
                            ),
                        );
                    } else if let Ok(m) = arg.extract::<PyHardtanh>() {
                        seq.add_module(
                            name,
                            Hardtanh::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                                Float32::new(m.min_val as f32),
                                Float32::new(m.max_val as f32),
                            ),
                        );
                    } else if let Ok(_m) = arg.extract::<PyMish>() {
                        seq.add_module(
                            name,
                            Mish::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(),
                        );
                    } else if let Ok(m) = arg.extract::<PyMaxPool2d>() {
                        if let MaxPool2dWrapper::CpuF32(inner) = &m.inner {
                            seq.add_module(name, inner.clone());
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(cpu, f32)",
                            ));
                        }
                    } else if let Ok(m) = arg.extract::<PyDropout>() {
                        seq.add_module(name, Dropout::new(m.inner.p));
                    } else if let Ok(m) = arg.extract::<PySequential>() {
                        if let SequentialWrapper::CpuF32(inner) = &m.inner {
                            seq.add_module(name, inner.clone());
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(cpu, f32)",
                            ));
                        }
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                            "Sequential: Argument {} is not a supported Module or type mismatch.",
                            i
                        )));
                    }
                }
                SequentialWrapper::CpuF64(seq) => {
                    if let Ok(m) = arg.extract::<PyLinear>() {
                        if let LinearWrapper::CpuF64(inner) = &m.inner {
                            seq.add_module(name, inner.clone());
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(cpu, f64)",
                            ));
                        }
                    } else if let Ok(_m) = arg.extract::<PyReLU>() {
                        seq.add_module(
                            name,
                            ReLU::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(),
                        );
                    } else if let Ok(_m) = arg.extract::<PyGeLU>() {
                        seq.add_module(
                            name,
                            GeLU::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(),
                        );
                    } else if let Ok(_m) = arg.extract::<PySiLU>() {
                        seq.add_module(
                            name,
                            SiLU::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(),
                        );
                    } else if let Ok(m) = arg.extract::<PyDropout>() {
                        seq.add_module(name, Dropout::new(m.inner.p));
                    } else if let Ok(m) = arg.extract::<PySequential>() {
                        if let SequentialWrapper::CpuF64(inner) = &m.inner {
                            seq.add_module(name, inner.clone());
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(cpu, f64)",
                            ));
                        }
                    } else if let Ok(m) = arg.extract::<PyMaxPool2d>() {
                        if let MaxPool2dWrapper::CpuF64(inner) = &m.inner {
                             seq.add_module(name, inner.clone());
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(cpu, f64)",
                            ));
                        }
                    }
                    // Add more mappings as needed for F64
                    else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Sequential: Argument {} is not a supported Module or type mismatch for f64.", i)));
                    }
                }
                #[cfg(feature = "gpu")]
                SequentialWrapper::GpuF32(seq) => {
                    if let Ok(m) = arg.extract::<PyLinear>() {
                        if let LinearWrapper::GpuF32(inner) = &m.inner {
                            seq.add_module(name, inner.clone());
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(gpu, f32)",
                            ));
                        }
                    } else if let Ok(_m) = arg.extract::<PyReLU>() {
                        seq.add_module(
                            name,
                            ReLU::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(),
                        );
                    } else if let Ok(m) = arg.extract::<PySequential>() {
                        if let SequentialWrapper::GpuF32(inner) = &m.inner {
                            seq.add_module(name, inner.clone());
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(gpu, f32)",
                            ));
                        }
                    } else if let Ok(m) = arg.extract::<PyMaxPool2d>() {
                         if let MaxPool2dWrapper::GpuF32(inner) = &m.inner {
                             seq.add_module(name, inner.clone());
                         } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Module variant mismatch for Sequential(gpu, f32)",
                            ));
                         }
                    }
                    // Add more mappings as needed for GPU F32
                    else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Sequential: Argument {} is not a supported Module or type mismatch for gpu.", i)));
                    }
                }
            }
        }
        Ok(PySequential { inner })
    }

    fn __len__(&self) -> usize {
        match &self.inner {
            SequentialWrapper::CpuF32(s) => s.len(),
            SequentialWrapper::CpuF64(s) => s.len(),
            #[cfg(feature = "gpu")]
            SequentialWrapper::GpuF32(s) => s.len(),
        }
    }

    #[allow(deprecated)]
    fn add_module(&mut self, name: String, module: Py<PyAny>) -> PyResult<()> {
        let _ = (name, module);
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Sequential.add_module is not yet implemented; use add_linear/add_conv2d/add_relu/etc",
        ))
    }

    #[pyo3(signature = (name, in_features, out_features))]
    fn add_linear(&mut self, name: &str, in_features: usize, out_features: usize) -> PyResult<()> {
        match &mut self.inner {
            SequentialWrapper::CpuF32(seq) => {
                let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    in_features,
                    out_features,
                )
                .map_err(to_py_err)?;
                seq.add_module(name.to_string(), linear);
                Ok(())
            }
            SequentialWrapper::CpuF64(seq) => {
                let linear = Linear::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                    in_features,
                    out_features,
                )
                .map_err(to_py_err)?;
                seq.add_module(name.to_string(), linear);
                Ok(())
            }
            #[cfg(feature = "gpu")]
            SequentialWrapper::GpuF32(seq) => {
                let linear = Linear::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    in_features,
                    out_features,
                )
                .map_err(to_py_err)?;
                seq.add_module(name.to_string(), linear);
                Ok(())
            }
        }
    }

    #[pyo3(signature = (name,))]
    fn add_relu(&mut self, name: &str) -> PyResult<()> {
        match &mut self.inner {
            SequentialWrapper::CpuF32(seq) => {
                seq.add_module(
                    name.to_string(),
                    ReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(),
                );
                Ok(())
            }
            SequentialWrapper::CpuF64(seq) => {
                seq.add_module(
                    name.to_string(),
                    ReLU::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(),
                );
                Ok(())
            }
            #[cfg(feature = "gpu")]
            SequentialWrapper::GpuF32(seq) => {
                seq.add_module(
                    name.to_string(),
                    ReLU::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(),
                );
                Ok(())
            }
        }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (SequentialWrapper::CpuF32(s), TensorWrapper::CpuDenseF32(i)) => {
                let res = s.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (SequentialWrapper::CpuF64(s), TensorWrapper::CpuDenseF64(i)) => {
                let res = s.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (SequentialWrapper::GpuF32(s), TensorWrapper::GpuDenseF32(i)) => {
                let res = s.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Sequential forward: device/dtype mismatch between container and input",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        match &self.inner {
            SequentialWrapper::CpuF32(s) => s
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::CpuDenseF32(p.data().clone()),
                })
                .collect(),
            SequentialWrapper::CpuF64(s) => s
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::CpuDenseF64(p.data().clone()),
                })
                .collect(),
            #[cfg(feature = "gpu")]
            SequentialWrapper::GpuF32(s) => s
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
    m.add_class::<PySequential>()?;

    // Add to module __dict__ for dir() visibility (PyTorch compatibility)
    let dict = m.dict();
    dict.set_item("Sequential", m.getattr("Sequential")?)?;

    Ok(())
}
