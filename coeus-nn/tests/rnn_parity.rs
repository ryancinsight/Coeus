//! Differential parity for `GRUCell` and `LSTMCell`.
//!
//! `Linear::new` initializes weights to all-ones and bias to zeros, so these
//! cells are fully deterministic at construction time — no random seed needed.
//!
//! Zero-input analytical oracle (x=0, h=0, c=0, all-ones weights, zero bias):
//!
//! GRU:
//!   ih = 0·W_ih.T = 0,  hh = 0·W_hh.T = 0
//!   r = sigmoid(0) = 0.5,  z = sigmoid(0) = 0.5,  n = tanh(0+0.5·0) = 0
//!   h_new = (1-0.5)·0 + 0.5·0 = 0  ← exact
//!
//! LSTM:
//!   gates = 0·W_ih.T + 0·W_hh.T = 0
//!   i=sigmoid(0)=0.5, f=sigmoid(0)=0.5, g=tanh(0)=0, o=sigmoid(0)=0.5
//!   c_new = 0.5·0 + 0.5·0 = 0  ← exact
//!   h_new = 0.5·tanh(0) = 0   ← exact
//!
//! Both SequentialBackend and MoiraiBackend must return bitwise-identical results.

use coeus_autograd::Var;
use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_nn::{GRUCell, LSTMCell, Module};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

fn zeros_var<B: BackendOps<f64> + Default>(shape: &[usize], backend: &B) -> Var<f64, B>
where
    B::DeviceBuffer<f64>: coeus_core::CpuAddressableStorageMut<f64>,
{
    Var::new(Tensor::zeros_on(shape.to_vec(), backend), false)
}

fn check_gru_cell<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>:
        coeus_core::CpuAddressableStorage<f64> + coeus_core::CpuAddressableStorageMut<f64>,
{
    // input_size=2, hidden_size=2 — small but non-trivial.
    let cell = GRUCell::<f64, B>::new(2, 2);

    // x=zeros, h=zeros → h_new = zeros exactly (proven above).
    let x = zeros_var(&[1, 2], backend);
    let h = zeros_var(&[1, 2], backend);
    let h_new = cell.step(&x, &h);
    assert_eq!(h_new.tensor.shape(), &[1, 2], "GRU output shape");
    assert_eq!(
        h_new.tensor.as_slice(),
        &[0.0_f64, 0.0],
        "GRU zero-input → zero h_new"
    );

    // Module::forward (h_0 = zeros internally) must equal step with h=zeros.
    let h_new2 = Module::<f64, B>::forward(&cell, &x);
    assert_eq!(
        h_new2.tensor.as_slice(),
        h_new.tensor.as_slice(),
        "GRU Module::forward == step with h=zeros"
    );
}

fn check_lstm_cell<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>:
        coeus_core::CpuAddressableStorage<f64> + coeus_core::CpuAddressableStorageMut<f64>,
{
    // input_size=2, hidden_size=2.
    let cell = LSTMCell::<f64, B>::new(2, 2);

    // x=zeros, h=zeros, c=zeros → h_new=zeros, c_new=zeros exactly.
    let x = zeros_var(&[1, 2], backend);
    let h = zeros_var(&[1, 2], backend);
    let c = zeros_var(&[1, 2], backend);
    let (h_new, c_new) = cell.step(&x, &h, &c);
    assert_eq!(h_new.tensor.shape(), &[1, 2], "LSTM h_new shape");
    assert_eq!(c_new.tensor.shape(), &[1, 2], "LSTM c_new shape");
    assert_eq!(
        h_new.tensor.as_slice(),
        &[0.0_f64, 0.0],
        "LSTM zero-input → zero h_new"
    );
    assert_eq!(
        c_new.tensor.as_slice(),
        &[0.0_f64, 0.0],
        "LSTM zero-input → zero c_new"
    );

    // Module::forward must equal step with h=zeros, c=zeros.
    let h_new2 = Module::<f64, B>::forward(&cell, &x);
    assert_eq!(
        h_new2.tensor.as_slice(),
        h_new.tensor.as_slice(),
        "LSTM Module::forward == step with h=zeros,c=zeros"
    );
}

fn check_all<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>:
        coeus_core::CpuAddressableStorage<f64> + coeus_core::CpuAddressableStorageMut<f64>,
{
    check_gru_cell(backend);
    check_lstm_cell(backend);
}

#[test]
fn sequential_rnn_match_reference() {
    check_all(&SequentialBackend);
}

#[test]
fn moirai_rnn_match_reference() {
    check_all(&MoiraiBackend);
}
