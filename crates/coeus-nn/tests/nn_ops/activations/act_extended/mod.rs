//! G-037 extended activation value-semantic contracts.

use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::{
    celu, hardshrink, hardsigmoid, hardswish, hardtanh, log_sigmoid, prelu, softshrink, softsign,
    tanhshrink, threshold, Hardsigmoid, Hardswish, Module, PReLU, Softsign,
};
use coeus_optim::{Optimizer, SGD};
use coeus_tensor::Tensor;

#[path = "module_smoke.rs"]
mod module_smoke;
#[path = "parameterized.rs"]
mod parameterized;
#[path = "piecewise.rs"]
mod piecewise;
#[path = "smooth.rs"]
mod smooth;
mod support;
