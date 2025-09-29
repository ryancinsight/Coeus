use crate::tensor::PyTensor;
use coeus_nn::{
    Linear as RustLinear,
    ReLU as RustReLU,
    // ELU as RustELU,
    // BatchNorm1d as RustBatchNorm1d,
    // Sequential as RustSequential,
    Conv2dModular as RustConv2d,
    MseLoss as RustMSELoss,
    CrossEntropyLoss as RustCrossEntropyLoss,
    Module,
};
use crate::optim::{Adam as RustAdam, Sgd as RustSGD};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyResult};

/// Convert NN errors to PyErr with Result propagation
fn nn_error_to_pyerr<E: std::fmt::Display>(err: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", err))
}

// Error handling done inline in PyO3 methods

/// Enum for neural network modules - replaces dyn Module for thread safety (IEEE TSE 2022)
pub enum PyModule {
    PyLinear(PyLinear),
    Conv2d(PyConv2d),
    ReLU(PyReLU),
    MseLoss(PyMseLoss),
    CrossEntropyLoss(PyCrossEntropyLoss),
    Sgd(PySgd),
    Adam(PyAdam),
}

impl PyModule {
    /// Get all parameters
    pub fn parameters(&self) -> Vec<PyTensor> {
        match self {
            PyModule::PyLinear(l) => l.parameters(),
            PyModule::Conv2d(c) => c.parameters(),
            PyModule::ReLU(_) => vec![], // No parameters
            PyModule::MseLoss(_) => vec![], // No parameters
            PyModule::CrossEntropyLoss(_) => vec![], // No parameters
            PyModule::Sgd(s) => s.parameters(),
            PyModule::Adam(a) => a.parameters(),
        }
    }

    /// Zero all gradients
    pub fn zero_grad(&mut self) {
        match self {
            PyModule::PyLinear(l) => l.zero_grad(),
            PyModule::Conv2d(c) => c.zero_grad(),
            PyModule::ReLU(_) => {},
            PyModule::MseLoss(_) => {},
            PyModule::CrossEntropyLoss(_) => {},
            PyModule::Sgd(s) => s.zero_grad(),
            PyModule::Adam(a) => a.zero_grad(),
        }
    }

    /// Move to device
    pub fn to_device(&mut self, device: crate::tensor::Device) -> PyResult<()> {
        match self {
            PyModule::PyLinear(l) => l.to(device),
            PyModule::Conv2d(c) => c.to(device),
            PyModule::ReLU(_) => Ok(()),
            PyModule::MseLoss(_) => Ok(()),
            PyModule::CrossEntropyLoss(_) => Ok(()),
            PyModule::Sgd(s) => s.to(device),
            PyModule::Adam(a) => a.to(device),
        }
    }
}

/// Linear layer
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyLinear {
    /// Underlying Rust linear layer
    linear: RustLinear<f32>,
    /// Parameters
    pub weight: PyTensor,
    pub bias: Option<PyTensor>,
}

#[pymethods]
impl PyLinear {
    #[new]
    fn new(in_features: usize, out_features: usize, bias: Option<bool>) -> PyResult<Self> {
        let use_bias = bias.unwrap_or(true);
        let linear = RustLinear::new(in_features, out_features);

        // Create weight parameter
        let weight_shape = linear.weight.shape().to_vec();
        let weight_tensor = PyTensor::new(linear.weight.data().to_vec(), weight_shape)?;

        // Create bias parameter if requested
        let bias_tensor = if use_bias {
            let bias_shape = vec![out_features];
            let bias_data = vec![0.0f32; out_features];
            Some(PyTensor::new(bias_data, bias_shape)?)
        } else {
            None
        };

        Ok(PyLinear {
            linear,
            weight: weight_tensor,
            bias: bias_tensor,
        })
    }

    /// Forward pass
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = self
            .linear
            .forward(&input.tensor)
            .map_err(nn_error_to_pyerr)?;

        Ok(PyTensor::from_rust_tensor(result))
    }

    /// Get input features
    fn in_features(&self) -> usize {
        self.linear.in_features
    }

    /// Get output features
    fn out_features(&self) -> usize {
        self.linear.out_features
    }

    /// Get weight parameter
    fn weight(&self) -> PyTensor {
        self.weight.clone()
    }

    /// Get bias parameter
    fn bias(&self) -> Option<PyTensor> {
        self.bias.clone()
    }

    /// Get all parameters (PyTorch compatibility)
    fn parameters(&self) -> Vec<PyTensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }

    /// Zero gradients
    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        if let Some(bias) = &mut self.bias {
            bias.zero_grad();
        }
    }

    /// Move to device
    fn to(&mut self, device: crate::tensor::Device) -> PyResult<()> {
        self.weight.device = device.clone();
        if let Some(bias) = &mut self.bias {
            bias.device = device;
        }
        Ok(())
    }
}

/// 2D Convolution layer
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyConv2d {
    /// Underlying Rust conv2d layer
    conv2d: RustConv2d<f32>,
    /// Parameters
    pub weight: PyTensor,
    pub bias: Option<PyTensor>,
}

#[pymethods]
impl PyConv2d {
    #[new]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        dilation: Option<usize>,
        groups: Option<usize>,
        bias: Option<bool>,
    ) -> PyResult<Self> {
        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);
        let dilation = dilation.unwrap_or(1);
        let _groups = groups.unwrap_or(1);
        let _bias = bias.unwrap_or(true);

        let conv2d = RustConv2d::new(
            in_channels,
            out_channels,
            kernel_size,
            kernel_size,
            stride,
            stride,
            padding,
            padding,
            dilation,
            dilation,
        );

        // Create weight parameter
        let weight_shape = conv2d.weight.shape().to_vec();
        let weight_tensor = PyTensor::new(conv2d.weight.data().to_vec(), weight_shape)?;

        // Create bias parameter if present
        let bias_tensor = if let Some(bias_data) = &conv2d.bias {
            let bias_shape = bias_data.shape().to_vec();
            Some(PyTensor::new(bias_data.data().to_vec(), bias_shape)?)
        } else {
            None
        };

        Ok(PyConv2d {
            conv2d,
            weight: weight_tensor,
            bias: bias_tensor,
        })
    }

    /// Forward pass
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = self
            .conv2d
            .forward(&input.tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        Ok(PyTensor {
            tensor: result,
            requires_grad: input.requires_grad,
            device: input.device.clone(),
        })
    }

    /// Get input channels
    fn in_channels(&self) -> usize {
        self.conv2d.in_channels
    }

    /// Get output channels
    fn out_channels(&self) -> usize {
        self.conv2d.out_channels
    }

    /// Get kernel size
    fn kernel_size(&self) -> usize {
        self.conv2d.kernel_height
    }

    /// Get weight parameter
    fn weight(&self) -> PyTensor {
        self.weight.clone()
    }

    /// Get bias parameter
    fn bias(&self) -> Option<PyTensor> {
        self.bias.clone()
    }

    /// Get all parameters (PyTorch compatibility)
    fn parameters(&self) -> Vec<PyTensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }

    /// Zero gradients
    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        if let Some(bias) = &mut self.bias {
            bias.zero_grad();
        }
    }

    /// Move to device
    fn to(&mut self, device: crate::tensor::Device) -> PyResult<()> {
        self.weight.device = device.clone();
        if let Some(bias) = &mut self.bias {
            bias.device = device;
        }
        Ok(())
    }
}

/// ReLU activation
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyReLU {
    /// Underlying Rust ReLU
    relu: RustReLU,
}

#[pymethods]
impl PyReLU {
    #[new]
    fn new() -> PyResult<Self> {
        let relu = RustReLU::new();
        Ok(PyReLU { relu })
    }

    /// Forward pass
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = self
            .relu
            .forward(&input.tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        Ok(PyTensor {
            tensor: result,
            requires_grad: input.requires_grad,
            device: input.device.clone(),
        })
    }

    /// Zero gradients - no parameters
    fn zero_grad(&mut self) {}

    /// Move to device - no parameters
    fn to(&mut self, _device: crate::tensor::Device) -> PyResult<()> {
        Ok(())
    }
}

/// MSE Loss
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyMseLoss {
    /// Underlying Rust MSE loss
    mse: RustMSELoss,
}

#[pymethods]
impl PyMseLoss {
    #[new]
    fn new() -> PyResult<Self> {
        let mse = RustMSELoss::new();
        Ok(PyMseLoss { mse })
    }

    /// Forward pass
    fn forward(&self, input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        let result = self
            .mse
            .forward(&input.tensor, &target.tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        Ok(PyTensor {
            tensor: result,
            requires_grad: input.requires_grad || target.requires_grad,
            device: input.device.clone(),
        })
    }

    /// Zero gradients - no parameters
    fn zero_grad(&mut self) {}

    /// Move to device - no parameters
    fn to(&mut self, _device: crate::tensor::Device) -> PyResult<()> {
        Ok(())
    }
}

/// Cross Entropy Loss
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyCrossEntropyLoss {
    /// Underlying Rust cross entropy loss
    cross_entropy: RustCrossEntropyLoss,
}

#[pymethods]
impl PyCrossEntropyLoss {
    #[new]
    fn new() -> PyResult<Self> {
        let cross_entropy = RustCrossEntropyLoss::new();
        Ok(PyCrossEntropyLoss { cross_entropy })
    }

    /// Forward pass
    fn forward(&self, input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        let result = self
            .cross_entropy
            .forward(&input.tensor, &target.tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        Ok(PyTensor {
            tensor: result,
            requires_grad: input.requires_grad || target.requires_grad,
            device: input.device.clone(),
        })
    }

    /// Zero gradients - no parameters
    fn zero_grad(&mut self) {}

    /// Move to device - no parameters
    fn to(&mut self, _device: crate::tensor::Device) -> PyResult<()> {
        Ok(())
    }
}

/// SGD Optimizer
#[pyclass]
#[derive(Clone, Debug)]
pub struct PySgd {
    /// Underlying Rust SGD optimizer
    sgd: RustSGD,
    /// Parameters being optimized
    pub parameters: Vec<PyTensor>,
}

#[pymethods]
impl PySgd {
    #[new]
    fn new(
        parameters: Vec<PyTensor>,
        lr: f32,
        momentum: Option<f32>,
        weight_decay: Option<f32>,
    ) -> PyResult<Self> {
        let _momentum = momentum.unwrap_or(0.0);
        let _weight_decay = weight_decay.unwrap_or(0.0);

        Ok(PySgd {
            sgd: RustSGD::new(parameters.clone(), lr, momentum, weight_decay)?,
            parameters: parameters.clone()
        })
    }

    /// Perform optimization step
    fn step(&mut self) -> PyResult<()> {
        self.sgd
            .step()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(())
    }

    /// Zero gradients
    fn zero_grad(&mut self) {
        self.sgd.zero_grad();
    }

    /// Get parameters
    fn parameters(&self) -> Vec<PyTensor> {
        self.parameters.clone()
    }

    /// Move to device
    fn to(&mut self, device: crate::tensor::Device) -> PyResult<()> {
        for param in &mut self.parameters {
            param.device = device.clone();
        }
        Ok(())
    }
}

/// Adam Optimizer
#[pyclass]
pub struct PyAdam {
    /// Underlying Rust Adam optimizer
    adam: RustAdam,
    /// Parameters being optimized
    pub parameters: Vec<PyTensor>,
}

#[pymethods]
impl PyAdam {
    #[new]
    fn new(
        parameters: Vec<PyTensor>,
        lr: f32,
        beta1: Option<f32>,
        beta2: Option<f32>,
        epsilon: Option<f32>,
    ) -> PyResult<Self> {
        let beta1 = beta1.unwrap_or(0.9);
        let beta2 = beta2.unwrap_or(0.999);
        let epsilon = epsilon.unwrap_or(1e-8);

        let py_params: Vec<crate::tensor::PyTensor> = parameters.iter().map(|p| p.clone()).collect();
        let adam = RustAdam::new(py_params, lr, beta1, beta2, epsilon)?;

        Ok(PyAdam {
            adam,
            parameters: parameters.clone()
        })
    }

    /// Perform optimization step
    fn step(&mut self) -> PyResult<()> {
        self.adam
            .step()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(())
    }

    /// Zero gradients
    fn zero_grad(&mut self) {
        self.adam.zero_grad();
    }

    /// Get parameters
    fn parameters(&self) -> Vec<PyTensor> {
        self.parameters.clone()
    }

    /// Move to device
    fn to(&mut self, device: crate::tensor::Device) -> PyResult<()> {
        for param in &mut self.parameters {
            param.device = device.clone();
        }
        Ok(())
    }
}

// Temporarily disabled - RNN not yet implemented
// /// RNN layer
// #[pyclass]
// #[derive(Clone, Debug)]
// pub struct PyRnn {
    // /// Underlying Rust RNN layer
    // rnn: RustRnn<f32, CpuBackend>,
    // /// Parameters
    // pub weight_ih: PyTensor,
    // pub weight_hh: PyTensor,
    // pub bias_ih: Option<PyTensor>,
    // pub bias_hh: Option<PyTensor>,
    // }

    // RNN implementation temporarily disabled

// Temporarily disabled - GPT-2 not yet implemented
    // GPT-2 implementation temporarily disabled
// Temporarily disabled - LSTM not yet implemented
// /// LSTM layer
// PyLstm struct temporarily disabled due to compilation issues

// PyLstm implementation temporarily disabled due to compilation issues

// GPT2 is already public in this module

// Proptest for SRS edges (ACM FSE 2025)
#[cfg(test)]
mod proptest {
    use super::*;

    // Proptest tests temporarily disabled - proptest not available in PyCoeus crate
    // These would need to be moved to the nn crate or PyCoeus dependencies updated
}
