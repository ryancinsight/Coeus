use crate::tensor::PyTensor;
use coeus_nn::{
    Conv2d as RustConv2d, CrossEntropyLoss as RustCrossEntropyLoss, GPTConfig, Gru as RustGru,
    Linear as RustLinear, Lstm as RustLstm, Module, MseLoss as RustMSELoss, ReLU as RustReLU,
    Rnn as RustRnn, GPT2 as RustGPT2,
};
use coeus_optim::{Adam as RustAdam, Optimizer, Sgd as RustSGD};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyResult};
use std::collections::HashMap;

/// Convert NN errors to PyErr
fn nn_error_to_pyerr<E: std::fmt::Display>(err: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", err))
}

/// Base neural network module
#[pyclass]
#[derive(Clone)]
pub struct NNModule {
    /// Parameter storage
    pub parameters: HashMap<String, PyTensor>,
}

#[pymethods]
impl NNModule {
    #[new]
    fn new() -> Self {
        NNModule {
            parameters: HashMap::new(),
        }
    }

    /// Get all parameters
    fn parameters(&self) -> HashMap<String, PyTensor> {
        self.parameters
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Register a parameter
    fn register_parameter(&mut self, name: String, param: PyTensor) {
        self.parameters.insert(name, param);
    }

    /// Zero all gradients
    fn zero_grad(&mut self) {
        for param in self.parameters.values_mut() {
            param.zero_grad();
        }
    }

    /// Move module to specified device
    fn to(&mut self, device: crate::tensor::Device) -> PyResult<()> {
        // Transfer all parameters to the specified device
        for param in self.parameters.values_mut() {
            param.device = device.clone();
        }
        Ok(())
    }
}

/// Linear layer
#[pyclass]
pub struct Linear {
    /// Underlying Rust linear layer
    linear: RustLinear<f32>,
    /// Parameters
    pub weight: PyTensor,
    pub bias: Option<PyTensor>,
}

#[pymethods]
impl Linear {
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=None))]
    fn new(in_features: usize, out_features: usize, bias: Option<bool>) -> PyResult<Self> {
        let _bias = bias.unwrap_or(true);
        let linear = RustLinear::new_with_bias(in_features, out_features, _bias);

        // Create weight parameter
        let weight_shape = linear.weight.shape().to_vec();
        let weight_tensor = PyTensor::new(linear.weight.data().to_vec(), weight_shape)?;

        // Create bias parameter if present
        let bias_tensor = if let Some(bias_data) = &linear.bias {
            let bias_shape = bias_data.shape().to_vec();
            Some(PyTensor::new(bias_data.data().to_vec(), bias_shape)?)
        } else {
            None
        };

        Ok(Linear {
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

        Ok(PyTensor {
            tensor: result,
            requires_grad: input.requires_grad,
            device: input.device.clone(),
        })
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
}

/// 2D Convolution layer
#[pyclass]
pub struct Conv2d {
    /// Underlying Rust conv2d layer
    conv2d: RustConv2d<f32>,
    /// Parameters
    pub weight: PyTensor,
    pub bias: Option<PyTensor>,
}

#[pymethods]
impl Conv2d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=None, padding=None, bias=None))]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        bias: Option<bool>,
    ) -> PyResult<Self> {
        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);
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
            1,
            1,
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

        Ok(Conv2d {
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
}

/// ReLU activation
#[pyclass]
pub struct ReLU {
    /// Underlying Rust ReLU
    relu: RustReLU,
}

#[pymethods]
impl ReLU {
    #[new]
    fn new() -> PyResult<Self> {
        let relu = RustReLU::new();
        Ok(ReLU { relu })
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
}

/// MSE Loss
#[pyclass]
pub struct MseLoss {
    /// Underlying Rust MSE loss
    mse: RustMSELoss,
}

#[pymethods]
impl MseLoss {
    #[new]
    fn new() -> PyResult<Self> {
        let mse = RustMSELoss::new();
        Ok(MseLoss { mse })
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
}

/// Cross Entropy Loss
#[pyclass]
pub struct CrossEntropyLoss {
    /// Underlying Rust cross entropy loss
    cross_entropy: RustCrossEntropyLoss,
}

#[pymethods]
impl CrossEntropyLoss {
    #[new]
    fn new() -> PyResult<Self> {
        let cross_entropy = RustCrossEntropyLoss::new();
        Ok(CrossEntropyLoss { cross_entropy })
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
}

/// SGD Optimizer
#[pyclass]
pub struct Sgd {
    /// Underlying Rust SGD optimizer
    sgd: RustSGD<f32>,
    /// Parameters being optimized
    pub parameters: Vec<PyTensor>,
}

#[pymethods]
impl Sgd {
    #[new]
    #[pyo3(signature = (parameters, lr, momentum=None, weight_decay=None))]
    fn new(
        parameters: Vec<PyTensor>,
        lr: f32,
        momentum: Option<f32>,
        weight_decay: Option<f32>,
    ) -> PyResult<Self> {
        let momentum = momentum.unwrap_or(0.0);
        let weight_decay = weight_decay.unwrap_or(0.0);

        let rust_params: Vec<_> = parameters.iter().map(|p| p.tensor.clone()).collect();
        let sgd = RustSGD::with_options(
            rust_params,
            lr.into(),
            momentum.into(),
            weight_decay.into(),
            false,
        );

        Ok(Sgd { sgd, parameters })
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
}

/// Adam Optimizer
#[pyclass]
pub struct Adam {
    /// Underlying Rust Adam optimizer
    adam: RustAdam<f32>,
    /// Parameters being optimized
    pub parameters: Vec<PyTensor>,
}

#[pymethods]
impl Adam {
    #[new]
    #[pyo3(signature = (parameters, lr, beta1=None, beta2=None, epsilon=None))]
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

        let rust_params: Vec<_> = parameters.iter().map(|p| p.tensor.clone()).collect();
        let adam = RustAdam::with_options(
            rust_params,
            lr.into(),
            beta1.into(),
            beta2.into(),
            epsilon.into(),
            false, // amsgrad
        );

        Ok(Adam { adam, parameters })
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
}

/// RNN layer
#[pyclass]
pub struct Rnn {
    /// Underlying Rust RNN layer
    rnn: RustRnn<f32>,
    /// Parameters
    pub weight_ih: PyTensor,
    pub weight_hh: PyTensor,
    pub bias_ih: Option<PyTensor>,
    pub bias_hh: Option<PyTensor>,
}

#[pymethods]
impl Rnn {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers=None, nonlinearity=None, bias=None, batch_first=None, dropout=None, bidirectional=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: Option<usize>,
        nonlinearity: Option<String>,
        bias: Option<bool>,
        batch_first: Option<bool>,
        dropout: Option<f32>,
        bidirectional: Option<bool>,
    ) -> PyResult<Self> {
        let num_layers = num_layers.unwrap_or(1);
        let nonlinearity = nonlinearity.unwrap_or_else(|| "tanh".to_string());
        let _bias = bias.unwrap_or(true);
        let batch_first = batch_first.unwrap_or(false); // Default to sequence-first for PyTorch compatibility
        let dropout = dropout.unwrap_or(0.0);
        let bidirectional = bidirectional.unwrap_or(false);

        // Validate parameters
        if num_layers != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Multi-layer RNN not yet implemented",
            ));
        }
        if nonlinearity != "tanh" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Only tanh nonlinearity is currently supported",
            ));
        }
        if batch_first {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "batch_first=True not yet implemented",
            ));
        }
        if dropout != 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Dropout not yet implemented",
            ));
        }
        if bidirectional {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Bidirectional RNN not yet implemented",
            ));
        }

        let rnn = RustRnn::new(input_size, hidden_size);

        // Create weight parameters
        let weight_ih_shape = rnn.weight_ih.shape().to_vec();
        let weight_ih_tensor = PyTensor::new(rnn.weight_ih.data().to_vec(), weight_ih_shape)?;

        let weight_hh_shape = rnn.weight_hh.shape().to_vec();
        let weight_hh_tensor = PyTensor::new(rnn.weight_hh.data().to_vec(), weight_hh_shape)?;

        // Create bias parameters if present
        let bias_ih_tensor = if let Some(bias_ih_data) = &rnn.bias_ih {
            let bias_ih_shape = bias_ih_data.shape().to_vec();
            Some(PyTensor::new(bias_ih_data.data().to_vec(), bias_ih_shape)?)
        } else {
            None
        };

        let bias_hh_tensor = if let Some(bias_hh_data) = &rnn.bias_hh {
            let bias_hh_shape = bias_hh_data.shape().to_vec();
            Some(PyTensor::new(bias_hh_data.data().to_vec(), bias_hh_shape)?)
        } else {
            None
        };

        Ok(Rnn {
            rnn,
            weight_ih: weight_ih_tensor,
            weight_hh: weight_hh_tensor,
            bias_ih: bias_ih_tensor,
            bias_hh: bias_hh_tensor,
        })
    }

    /// Forward pass
    #[pyo3(signature = (input, hx=None))]
    fn forward(&self, input: &PyTensor, hx: Option<&PyTensor>) -> PyResult<(PyTensor, PyTensor)> {
        let result = if let Some(hx_tensor) = hx {
            self.rnn.forward(&input.tensor, Some(&hx_tensor.tensor))
        } else {
            self.rnn.forward(&input.tensor, None)
        }
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        let (output, h_n) = result;

        Ok((
            PyTensor {
                tensor: output,
                requires_grad: input.requires_grad || hx.as_ref().is_some_and(|h| h.requires_grad),
                device: input.device.clone(),
            },
            PyTensor {
                tensor: h_n,
                requires_grad: input.requires_grad || hx.as_ref().is_some_and(|h| h.requires_grad),
                device: input.device.clone(),
            },
        ))
    }

    /// Get input size
    #[getter]
    fn input_size(&self) -> usize {
        self.rnn.input_size
    }

    /// Get hidden size
    #[getter]
    fn hidden_size(&self) -> usize {
        self.rnn.hidden_size
    }

    /// Get number of layers
    #[getter]
    fn num_layers(&self) -> usize {
        1 // Currently only single layer supported
    }

    /// Get nonlinearity
    #[getter]
    fn nonlinearity(&self) -> String {
        "tanh".to_string()
    }

    /// Get bias flag
    #[getter]
    fn bias(&self) -> bool {
        self.bias_ih.is_some() && self.bias_hh.is_some()
    }

    /// Get batch_first flag
    #[getter]
    fn batch_first(&self) -> bool {
        false // Currently only sequence-first supported
    }

    /// Get dropout
    #[getter]
    fn dropout(&self) -> f32 {
        0.0
    }

    /// Get bidirectional flag
    #[getter]
    fn bidirectional(&self) -> bool {
        false
    }

    /// Get weight_ih parameter
    #[getter]
    fn weight_ih(&self) -> PyTensor {
        self.weight_ih.clone()
    }

    /// Get weight_hh parameter
    #[getter]
    fn weight_hh(&self) -> PyTensor {
        self.weight_hh.clone()
    }

    /// Get bias_ih parameter
    #[getter]
    fn bias_ih(&self) -> Option<PyTensor> {
        self.bias_ih.clone()
    }

    /// Get bias_hh parameter
    #[getter]
    fn bias_hh(&self) -> Option<PyTensor> {
        self.bias_hh.clone()
    }
}

/// GPT-2 model
#[pyclass]
pub struct GPT2 {
    /// Underlying Rust GPT-2 model
    gpt2: RustGPT2<f32>,
    /// Parameters
    pub wte: PyTensor,
    pub wpe: PyTensor,
}

#[pymethods]
impl GPT2 {
    #[new]
    #[pyo3(signature = (vocab_size, n_embd=None, n_head=None, n_layer=None, block_size=None, dropout=None))]
    fn new(
        vocab_size: usize,
        n_embd: Option<usize>,
        n_head: Option<usize>,
        n_layer: Option<usize>,
        block_size: Option<usize>,
        dropout: Option<f64>,
    ) -> PyResult<Self> {
        let config = GPTConfig {
            attn_config: coeus_nn::attention::AttentionConfig {
                n_embd: n_embd.unwrap_or(768),
                n_head: n_head.unwrap_or(12),
                block_size: block_size.unwrap_or(1024),
                dropout: dropout.unwrap_or(0.1),
                causal: true,
            },
            vocab_size,
            n_layer: n_layer.unwrap_or(12),
            dropout: dropout.unwrap_or(0.1),
        };

        let gpt2 = RustGPT2::new(config);

        // Create token embeddings parameter
        let wte_shape = gpt2.wte.weight.shape().to_vec();
        let wte_tensor = PyTensor::new(gpt2.wte.weight.data().to_vec(), wte_shape)?;

        // Create position embeddings parameter
        let wpe_shape = gpt2.wpe.weight.shape().to_vec();
        let wpe_tensor = PyTensor::new(gpt2.wpe.weight.data().to_vec(), wpe_shape)?;

        Ok(GPT2 {
            gpt2,
            wte: wte_tensor,
            wpe: wpe_tensor,
        })
    }

    /// Forward pass for language modeling
    #[pyo3(signature = (input, targets=None))]
    fn forward_lm(&self, input: &PyTensor, targets: Option<&PyTensor>) -> PyResult<PyTensor> {
        let result = if let Some(targets_tensor) = targets {
            self.gpt2
                .forward_lm(&input.tensor, Some(&targets_tensor.tensor))
        } else {
            self.gpt2.forward_lm(&input.tensor, None)
        }
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        Ok(PyTensor {
            tensor: result,
            requires_grad: input.requires_grad || targets.as_ref().is_some_and(|t| t.requires_grad),
            device: input.device.clone(),
        })
    }

    /// Generate text autoregressively
    #[pyo3(signature = (input, max_new_tokens=None, temperature=None))]
    fn generate(
        &self,
        input: &PyTensor,
        max_new_tokens: Option<usize>,
        temperature: Option<f64>,
    ) -> PyResult<PyTensor> {
        let max_new_tokens = max_new_tokens.unwrap_or(50);
        let temperature = temperature.unwrap_or(1.0);

        let result = self
            .gpt2
            .generate(&input.tensor, max_new_tokens, temperature)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        Ok(PyTensor {
            tensor: result,
            requires_grad: input.requires_grad,
            device: input.device.clone(),
        })
    }

    /// Get vocabulary size
    #[getter]
    fn vocab_size(&self) -> usize {
        self.gpt2.wte.vocab_size
    }

    /// Get embedding dimension
    #[getter]
    fn n_embd(&self) -> usize {
        self.gpt2.wte.embedding_dim
    }

    /// Get number of attention heads
    #[getter]
    fn n_head(&self) -> usize {
        self.gpt2.h[0].attn.config.n_head
    }

    /// Get number of layers
    #[getter]
    fn n_layer(&self) -> usize {
        self.gpt2.h.len()
    }

    /// Get block size (maximum sequence length)
    #[getter]
    fn block_size(&self) -> usize {
        self.gpt2.h[0].attn.config.block_size
    }

    /// Get token embeddings weight
    #[getter]
    fn wte(&self) -> PyTensor {
        self.wte.clone()
    }

    /// Get position embeddings weight
    #[getter]
    fn wpe(&self) -> PyTensor {
        self.wpe.clone()
    }
}

/// LSTM layer
#[pyclass]
pub struct Lstm {
    /// Underlying Rust LSTM layer
    lstm: RustLstm<f32>,
    /// Parameters
    pub weight_ih: PyTensor,
    pub weight_hh: PyTensor,
    pub bias_ih: Option<PyTensor>,
    pub bias_hh: Option<PyTensor>,
}

#[pymethods]
impl Lstm {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers=None, bias=None, batch_first=None, dropout=None, bidirectional=None, proj_size=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: Option<usize>,
        bias: Option<bool>,
        batch_first: Option<bool>,
        dropout: Option<f32>,
        bidirectional: Option<bool>,
        proj_size: Option<usize>,
    ) -> PyResult<Self> {
        let num_layers = num_layers.unwrap_or(1);
        let _bias = bias.unwrap_or(true);
        let batch_first = batch_first.unwrap_or(false);
        let dropout = dropout.unwrap_or(0.0);
        let bidirectional = bidirectional.unwrap_or(false);

        // Validate parameters
        if num_layers != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Multi-layer LSTM not yet implemented",
            ));
        }
        if batch_first {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "batch_first=True not yet implemented",
            ));
        }
        if dropout != 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Dropout not yet implemented",
            ));
        }
        if bidirectional {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Bidirectional LSTM not yet implemented",
            ));
        }
        if proj_size.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Projection not yet implemented",
            ));
        }

        let lstm = RustLstm::new(input_size, hidden_size);

        // Create combined weight parameters (concatenate gate-specific weights)
        // LSTM weights are organized as: [input_gate, forget_gate, cell_gate, output_gate]
        use coeus_tensor::ops::reduction::cat;

        let weight_ih_combined = cat(
            &[
                &lstm.weight_ih_i,
                &lstm.weight_ih_f,
                &lstm.weight_ih_g,
                &lstm.weight_ih_o,
            ],
            0,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        let weight_ih_shape = weight_ih_combined.shape().to_vec();
        let weight_ih_tensor = PyTensor::new(weight_ih_combined.data().to_vec(), weight_ih_shape)?;

        let weight_hh_combined = cat(
            &[
                &lstm.weight_hh_i,
                &lstm.weight_hh_f,
                &lstm.weight_hh_g,
                &lstm.weight_hh_o,
            ],
            0,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        let weight_hh_shape = weight_hh_combined.shape().to_vec();
        let weight_hh_tensor = PyTensor::new(weight_hh_combined.data().to_vec(), weight_hh_shape)?;

        // Create combined bias parameters if present
        let bias_ih_tensor = if lstm.bias_ih_i.is_some() || lstm.bias_ih_g.is_some() {
            let bias_ih_combined = cat(
                &[
                    lstm.bias_ih_i.as_ref().unwrap(),
                    lstm.bias_ih_g.as_ref().unwrap(),
                ],
                0,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            let bias_ih_shape = bias_ih_combined.shape().to_vec();
            Some(PyTensor::new(
                bias_ih_combined.data().to_vec(),
                bias_ih_shape,
            )?)
        } else {
            None
        };

        let bias_hh_tensor = if lstm.bias_hh_i.is_some() || lstm.bias_hh_o.is_some() {
            let bias_hh_combined = cat(
                &[
                    lstm.bias_hh_i.as_ref().unwrap(),
                    lstm.bias_hh_o.as_ref().unwrap(),
                ],
                0,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            let bias_hh_shape = bias_hh_combined.shape().to_vec();
            Some(PyTensor::new(
                bias_hh_combined.data().to_vec(),
                bias_hh_shape,
            )?)
        } else {
            None
        };

        Ok(Lstm {
            lstm,
            weight_ih: weight_ih_tensor,
            weight_hh: weight_hh_tensor,
            bias_ih: bias_ih_tensor,
            bias_hh: bias_hh_tensor,
        })
    }

    /// Forward pass
    #[pyo3(signature = (input, hx=None))]
    fn forward(
        &self,
        input: &PyTensor,
        hx: Option<(PyTensor, PyTensor)>,
    ) -> PyResult<(PyTensor, (PyTensor, PyTensor))> {
        let (h0, c0) = if let Some((hx_h, hx_c)) = hx {
            (Some(hx_h.tensor), Some(hx_c.tensor))
        } else {
            (None, None)
        };

        let result = self
            .lstm
            .forward(&input.tensor, h0.as_ref(), c0.as_ref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        let (output, (h_n, c_n)) = result;

        Ok((
            PyTensor {
                tensor: output,
                requires_grad: input.requires_grad
                    || h0.as_ref().is_some_and(|h| h.requires_grad())
                    || c0.as_ref().is_some_and(|c| c.requires_grad()),
                device: input.device.clone(),
            },
            (
                PyTensor {
                    tensor: h_n,
                    requires_grad: input.requires_grad
                        || h0.as_ref().is_some_and(|h| h.requires_grad())
                        || c0.as_ref().is_some_and(|c| c.requires_grad()),
                    device: input.device.clone(),
                },
                PyTensor {
                    tensor: c_n,
                    requires_grad: input.requires_grad
                        || h0.as_ref().is_some_and(|h| h.requires_grad())
                        || c0.as_ref().is_some_and(|c| c.requires_grad()),
                    device: input.device.clone(),
                },
            ),
        ))
    }

    /// Get input size
    #[getter]
    fn input_size(&self) -> usize {
        self.lstm.input_size
    }

    /// Get hidden size
    #[getter]
    fn hidden_size(&self) -> usize {
        self.lstm.hidden_size
    }

    /// Get number of layers
    #[getter]
    fn num_layers(&self) -> usize {
        1 // Currently only single layer supported
    }

    /// Get bias flag
    #[getter]
    fn bias(&self) -> bool {
        self.bias_ih.is_some() && self.bias_hh.is_some()
    }

    /// Get batch_first flag
    #[getter]
    fn batch_first(&self) -> bool {
        false // Currently only sequence-first supported
    }

    /// Get dropout
    #[getter]
    fn dropout(&self) -> f32 {
        0.0
    }

    /// Get bidirectional flag
    #[getter]
    fn bidirectional(&self) -> bool {
        false
    }

    /// Get weight_ih parameter
    #[getter]
    fn weight_ih(&self) -> PyTensor {
        self.weight_ih.clone()
    }

    /// Get weight_hh parameter
    #[getter]
    fn weight_hh(&self) -> PyTensor {
        self.weight_hh.clone()
    }

    /// Get bias_ih parameter
    #[getter]
    fn bias_ih(&self) -> Option<PyTensor> {
        self.bias_ih.clone()
    }

    /// Get bias_hh parameter
    #[getter]
    fn bias_hh(&self) -> Option<PyTensor> {
        self.bias_hh.clone()
    }
}

/// GRU layer
#[pyclass]
pub struct Gru {
    /// Underlying Rust GRU layer
    gru: RustGru<f32>,
    /// Parameters
    pub weight_ih: PyTensor,
    pub weight_hh: PyTensor,
    pub bias_ih: Option<PyTensor>,
    pub bias_hh: Option<PyTensor>,
}

#[pymethods]
impl Gru {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers=None, bias=None, batch_first=None, dropout=None, bidirectional=None))]
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: Option<usize>,
        bias: Option<bool>,
        batch_first: Option<bool>,
        dropout: Option<f32>,
        bidirectional: Option<bool>,
    ) -> PyResult<Self> {
        let num_layers = num_layers.unwrap_or(1);
        let _bias = bias.unwrap_or(true);
        let batch_first = batch_first.unwrap_or(false);
        let dropout = dropout.unwrap_or(0.0);
        let bidirectional = bidirectional.unwrap_or(false);

        // Validate parameters
        if num_layers != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Multi-layer GRU not yet implemented",
            ));
        }
        if batch_first {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "batch_first=True not yet implemented",
            ));
        }
        if dropout != 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Dropout not yet implemented",
            ));
        }
        if bidirectional {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Bidirectional GRU not yet implemented",
            ));
        }

        let gru = RustGru::new(input_size, hidden_size);

        // Create combined weight parameters (concatenate gate-specific weights)
        // GRU weights are organized as: [reset_gate, update_gate, new_gate]
        use coeus_tensor::ops::reduction::cat;

        let weight_ih_combined = cat(&[&gru.weight_ih_r, &gru.weight_ih_z, &gru.weight_ih_n], 0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        let weight_ih_shape = weight_ih_combined.shape().to_vec();
        let weight_ih_tensor = PyTensor::new(weight_ih_combined.data().to_vec(), weight_ih_shape)?;

        let weight_hh_combined = cat(&[&gru.weight_hh_r, &gru.weight_hh_z, &gru.weight_hh_n], 0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        let weight_hh_shape = weight_hh_combined.shape().to_vec();
        let weight_hh_tensor = PyTensor::new(weight_hh_combined.data().to_vec(), weight_hh_shape)?;

        // Create combined bias parameters if present
        let bias_ih_tensor = if gru.bias_ih_r.is_some() || gru.bias_ih_n.is_some() {
            let bias_ih_combined = cat(
                &[
                    gru.bias_ih_r.as_ref().unwrap(),
                    gru.bias_ih_n.as_ref().unwrap(),
                ],
                0,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            let bias_ih_shape = bias_ih_combined.shape().to_vec();
            Some(PyTensor::new(
                bias_ih_combined.data().to_vec(),
                bias_ih_shape,
            )?)
        } else {
            None
        };

        let bias_hh_tensor = if gru.bias_hh_r.is_some() || gru.bias_hh_n.is_some() {
            let bias_hh_combined = cat(
                &[
                    gru.bias_hh_r.as_ref().unwrap(),
                    gru.bias_hh_n.as_ref().unwrap(),
                ],
                0,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            let bias_hh_shape = bias_hh_combined.shape().to_vec();
            Some(PyTensor::new(
                bias_hh_combined.data().to_vec(),
                bias_hh_shape,
            )?)
        } else {
            None
        };

        Ok(Gru {
            gru,
            weight_ih: weight_ih_tensor,
            weight_hh: weight_hh_tensor,
            bias_ih: bias_ih_tensor,
            bias_hh: bias_hh_tensor,
        })
    }

    /// Forward pass
    #[pyo3(signature = (input, hx=None))]
    fn forward(&self, input: &PyTensor, hx: Option<&PyTensor>) -> PyResult<(PyTensor, PyTensor)> {
        let result = if let Some(hx_tensor) = hx {
            self.gru.forward(&input.tensor, Some(&hx_tensor.tensor))
        } else {
            self.gru.forward(&input.tensor, None)
        }
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        let (output, h_n) = result;

        Ok((
            PyTensor {
                tensor: output,
                requires_grad: input.requires_grad || hx.as_ref().is_some_and(|h| h.requires_grad),
                device: input.device.clone(),
            },
            PyTensor {
                tensor: h_n,
                requires_grad: input.requires_grad || hx.as_ref().is_some_and(|h| h.requires_grad),
                device: input.device.clone(),
            },
        ))
    }

    /// Get input size
    #[getter]
    fn input_size(&self) -> usize {
        self.gru.input_size
    }

    /// Get hidden size
    #[getter]
    fn hidden_size(&self) -> usize {
        self.gru.hidden_size
    }

    /// Get number of layers
    #[getter]
    fn num_layers(&self) -> usize {
        1 // Currently only single layer supported
    }

    /// Get bias flag
    #[getter]
    fn bias(&self) -> bool {
        self.bias_ih.is_some() && self.bias_hh.is_some()
    }

    /// Get batch_first flag
    #[getter]
    fn batch_first(&self) -> bool {
        false // Currently only sequence-first supported
    }

    /// Get dropout
    #[getter]
    fn dropout(&self) -> f32 {
        0.0
    }

    /// Get bidirectional flag
    #[getter]
    fn bidirectional(&self) -> bool {
        false
    }

    /// Get weight_ih parameter
    #[getter]
    fn weight_ih(&self) -> PyTensor {
        self.weight_ih.clone()
    }

    /// Get weight_hh parameter
    #[getter]
    fn weight_hh(&self) -> PyTensor {
        self.weight_hh.clone()
    }

    /// Get bias_ih parameter
    #[getter]
    fn bias_ih(&self) -> Option<PyTensor> {
        self.bias_ih.clone()
    }

    /// Get bias_hh parameter
    #[getter]
    fn bias_hh(&self) -> Option<PyTensor> {
        self.bias_hh.clone()
    }
}

// GPT2 is already public in this module
