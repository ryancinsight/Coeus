use crate::tensor::PyTensor;
use pyo3::prelude::*;

#[pyfunction]
pub fn uniform_(tensor: &mut PyTensor, a: f64, b: f64) {
    coeus_nn::init::uniform(&mut tensor.inner, a, b);
}

#[pyfunction]
pub fn normal_(tensor: &mut PyTensor, mean: f64, std_dev: f64) {
    coeus_nn::init::normal(&mut tensor.inner, mean, std_dev);
}

#[pyfunction]
pub fn constant_(tensor: &mut PyTensor, val: f64) {
    coeus_nn::init::constant(&mut tensor.inner, val);
}

#[pyfunction]
pub fn zeros_(tensor: &mut PyTensor) {
    coeus_nn::init::zeros(&mut tensor.inner);
}

#[pyfunction]
pub fn ones_(tensor: &mut PyTensor) {
    coeus_nn::init::ones(&mut tensor.inner);
}

#[pyfunction]
pub fn xavier_uniform_(tensor: &mut PyTensor, fan_in: usize, fan_out: usize) {
    coeus_nn::init::xavier_uniform(&mut tensor.inner, fan_in, fan_out);
}

#[pyfunction]
pub fn xavier_normal_(tensor: &mut PyTensor, fan_in: usize, fan_out: usize) {
    coeus_nn::init::xavier_normal(&mut tensor.inner, fan_in, fan_out);
}

#[pyfunction]
pub fn kaiming_uniform_(tensor: &mut PyTensor, fan_in: usize) {
    coeus_nn::init::kaiming_uniform(&mut tensor.inner, fan_in);
}

#[pyfunction]
pub fn kaiming_normal_(tensor: &mut PyTensor, fan_in: usize) {
    coeus_nn::init::kaiming_normal(&mut tensor.inner, fan_in);
}
