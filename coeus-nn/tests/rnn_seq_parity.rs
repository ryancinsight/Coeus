//! Differential parity for the sequence-level `Lstm` and `Gru` modules.
//!
//! Analytical oracle — zeros input, zeros initial state, any weight matrix:
//!
//! **LSTM** at each timestep `t`:
//! ```text
//! gates = W_ih · 0 + b_ih + W_hh · h_{t-1} + b_hh   where h_0 = 0
//!       = 0 + 0 + 0 + 0 = 0
//! i = f = o = σ(0) = 0.5     g = tanh(0) = 0
//! c_t = f·c_{t-1} + i·g = 0  h_t = o·tanh(c_t) = 0
//! ```
//! ∴ `Lstm::forward_seq(zeros)` → output = zeros, h_n = zeros, c_n = zeros.
//! This holds for any seq_len ≥ 1 because the zeros state propagates exactly.
//!
//! **GRU** at each timestep `t`:
//! ```text
//! r = z = σ(0) = 0.5     n = tanh(0 + 0.5·0) = 0
//! h_t = (1−z)·n + z·h_{t-1} = 0
//! ```
//! ∴ `Gru::forward_seq(zeros)` → output = zeros, h_n = zeros.
//!
//! **`Module::forward` consistency**: `Module::forward(x) == forward_seq(x).0` by
//! construction (forward_seq is the canonical path).
//!
//! All assertions use `assert_eq!`.
//! SequentialBackend and MoiraiBackend must produce bitwise-identical results.

use coeus_autograd::Var;
use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_nn::{Gru, Lstm, Module};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

fn zeros_var<B: BackendOps<f64> + Default>(shape: &[usize], backend: &B) -> Var<f64, B>
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Var::new(Tensor::zeros_on(shape.to_vec(), backend), false)
}

fn check_lstm<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // Lstm::new uses Linear::new (ones weights, zeros bias).
    // With zeros input and zeros initial state:
    //   gates = ones·0 + 0 + ones·0 + 0 = 0
    //   i=f=o=0.5, g=0 → c_new=0, h_new=0 for every timestep.
    //
    // Shape: [batch=1, seq_len=3, input_size=2] → [1, 3, hidden=2]
    let lstm = Lstm::<f64, B>::new(2, 2);
    let inp = zeros_var(&[1, 3, 2], backend);

    let (out, (h_n, c_n)) = lstm.forward_seq(&inp);

    assert_eq!(out.tensor.shape(), &[1, 3, 2], "Lstm output shape");
    assert_eq!(
        out.tensor.as_slice(),
        &[0.0_f64; 6],
        "Lstm zeros input → zeros output"
    );
    assert_eq!(
        h_n.tensor.as_slice(),
        &[0.0_f64; 2],
        "Lstm zeros input → zeros h_n"
    );
    assert_eq!(
        c_n.tensor.as_slice(),
        &[0.0_f64; 2],
        "Lstm zeros input → zeros c_n"
    );

    // Module::forward must agree with forward_seq output slice.
    let module_out = Module::<f64, B>::forward(&lstm, &inp);
    assert_eq!(
        module_out.tensor.as_slice(),
        out.tensor.as_slice(),
        "Lstm Module::forward == forward_seq output"
    );

    // Single-timestep: seq_len=1 gives [1,1,2] with zeros entry.
    let inp1 = zeros_var(&[1, 1, 2], backend);
    let (out1, _) = lstm.forward_seq(&inp1);
    assert_eq!(out1.tensor.shape(), &[1, 1, 2], "Lstm seq=1 shape");
    assert_eq!(out1.tensor.as_slice(), &[0.0_f64; 2], "Lstm seq=1 zeros");
}

fn check_gru<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // Gru::new uses Linear::new (ones weights, zeros bias).
    // With zeros input and zeros initial state:
    //   ih = ones·0 + 0 = 0, hh = ones·0 + 0 = 0
    //   r = z = σ(0) = 0.5   n = tanh(0 + 0.5·0) = 0
    //   h_new = (1−0.5)·0 + 0.5·0 = 0 for every timestep.
    //
    // Shape: [batch=1, seq_len=3, input_size=2] → [1, 3, hidden=2]
    let gru = Gru::<f64, B>::new(2, 2);
    let inp = zeros_var(&[1, 3, 2], backend);

    let (out, h_n) = gru.forward_seq(&inp);

    assert_eq!(out.tensor.shape(), &[1, 3, 2], "Gru output shape");
    assert_eq!(
        out.tensor.as_slice(),
        &[0.0_f64; 6],
        "Gru zeros input → zeros output"
    );
    assert_eq!(
        h_n.tensor.as_slice(),
        &[0.0_f64; 2],
        "Gru zeros input → zeros h_n"
    );

    // Module::forward must agree with forward_seq output slice.
    let module_out = Module::<f64, B>::forward(&gru, &inp);
    assert_eq!(
        module_out.tensor.as_slice(),
        out.tensor.as_slice(),
        "Gru Module::forward == forward_seq output"
    );

    // Batch dimension: batch=2, each sequence element is zeros.
    let inp2 = zeros_var(&[2, 4, 2], backend);
    let (out2, h_n2) = gru.forward_seq(&inp2);
    assert_eq!(out2.tensor.shape(), &[2, 4, 2], "Gru batch=2 shape");
    assert_eq!(out2.tensor.as_slice(), &[0.0_f64; 16], "Gru batch=2 zeros");
    assert_eq!(
        h_n2.tensor.as_slice(),
        &[0.0_f64; 4],
        "Gru batch=2 h_n zeros"
    );
}

fn check_all<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    check_lstm(backend);
    check_gru(backend);
}

#[test]
fn sequential_rnn_seq_match_reference() {
    check_all(&SequentialBackend);
}

#[test]
fn moirai_rnn_seq_match_reference() {
    check_all(&MoiraiBackend);
}
