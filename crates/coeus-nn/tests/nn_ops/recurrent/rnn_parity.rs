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
use coeus_nn::{GRUCell, LSTMCell, Module, ModuleError, RNNCell, RnnNonlinearity};
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
    let h_new = cell.step(&x, &h).expect("valid GRUCell input");
    assert_eq!(h_new.tensor.shape(), &[1, 2], "GRU output shape");
    assert_eq!(
        h_new.tensor.as_slice(),
        &[0.0_f64, 0.0],
        "GRU zero-input → zero h_new"
    );

    // Module::forward (h_0 = zeros internally) must equal step with h=zeros.
    let h_new2 = Module::<f64, B>::forward(&cell, &x).expect("valid GRUCell input");
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
    let (h_new, c_new) = cell.step(&x, &h, &c).expect("valid LSTMCell input");
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
    let h_new2 = Module::<f64, B>::forward(&cell, &x).expect("valid LSTMCell input");
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
    check_rnn_cell(backend);
}

fn check_rnn_cell<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>:
        coeus_core::CpuAddressableStorage<f64> + coeus_core::CpuAddressableStorageMut<f64>,
{
    // Linear::new → all-ones weights, zero bias. RNNCell step:
    //   h_new = f(x @ W_ih.T + h @ W_hh.T) with W all-ones.
    let cell = RNNCell::<f64, B>::new(2, 2, RnnNonlinearity::Tanh);

    // x=0, h=0 → pre=0 → tanh(0)=0 exactly.
    let x0 = zeros_var(&[1, 2], backend);
    let h0 = zeros_var(&[1, 2], backend);
    let h_z = cell.step(&x0, &h0).expect("valid RNNCell input");
    assert_eq!(h_z.tensor.shape(), &[1, 2], "RNN h_new shape");
    assert_eq!(
        h_z.tensor.as_slice(),
        &[0.0_f64, 0.0],
        "RNN zero-input → zero h_new"
    );

    // x=[1,2], h=0, all-ones W_ih → pre = [1+2, 1+2] = [3,3]; h_new = tanh(3).
    let x = Var::new(Tensor::from_slice_on([1, 2], &[1.0, 2.0], backend), false);
    let h_new = cell.step(&x, &h0).expect("valid RNNCell input");
    let t3 = 3.0_f64.tanh();
    let got = h_new.tensor.as_slice();
    assert!(
        (got[0] - t3).abs() <= 1e-12 && (got[1] - t3).abs() <= 1e-12,
        "RNN tanh: got {got:?}, expected [{t3}, {t3}]"
    );

    // Relu nonlinearity: pre=[3,3] → relu(3)=3 exactly.
    let cell_relu = RNNCell::<f64, B>::new(2, 2, RnnNonlinearity::Relu);
    let h_relu = cell_relu.step(&x, &h0).expect("valid RNNCell input");
    assert_eq!(
        h_relu.tensor.as_slice(),
        &[3.0_f64, 3.0],
        "RNN relu: pre=3 → relu(3)=3"
    );

    // Module::forward (h_0 = zeros) == step with h=zeros.
    let h_mod = Module::<f64, B>::forward(&cell, &x).expect("valid RNNCell input");
    assert_eq!(
        h_mod.tensor.as_slice(),
        h_new.tensor.as_slice(),
        "RNN Module::forward == step with h=zeros"
    );
}

#[test]
fn sequential_rnn_match_reference() {
    check_all(&SequentialBackend);
}

#[test]
fn moirai_rnn_match_reference() {
    check_all(&MoiraiBackend);
}

fn expect_invalid_rank(
    result: Result<Var<f64, SequentialBackend>, ModuleError<coeus_core::BackendError>>,
    module: &'static str,
    actual: usize,
) {
    match result {
        Err(ModuleError::InvalidRank {
            module: actual_module,
            expected: "2",
            actual: actual_rank,
        }) => {
            assert_eq!(actual_module, module);
            assert_eq!(actual_rank, actual);
        }
        Err(other) => panic!("expected {module} InvalidRank, got {other:?}"),
        Ok(_) => panic!("expected {module} to reject rank {actual}"),
    }
}

fn expect_shape_mismatch(
    result: Result<Var<f64, SequentialBackend>, ModuleError<coeus_core::BackendError>>,
    module: &'static str,
    parameter: &'static str,
    expected: &[usize],
    actual: &[usize],
) {
    match result {
        Err(ModuleError::ShapeMismatch {
            module: actual_module,
            parameter: actual_parameter,
            expected: actual_expected,
            actual: actual_shape,
        }) => {
            assert_eq!(actual_module, module);
            assert_eq!(actual_parameter, parameter);
            assert_eq!(actual_expected, expected);
            assert_eq!(actual_shape, actual);
        }
        Err(other) => panic!("expected {module} ShapeMismatch, got {other:?}"),
        Ok(_) => panic!("expected {module} to reject shape {actual:?}"),
    }
}

#[test]
fn recurrent_cells_reject_invalid_input_contracts() {
    let backend = SequentialBackend;
    let rank_one = zeros_var(&[2], &backend);
    let wrong_features = zeros_var(&[1, 3], &backend);

    let rnn = RNNCell::<f64, SequentialBackend>::new(2, 2, RnnNonlinearity::Tanh);
    expect_invalid_rank(rnn.forward(&rank_one), "RNNCell", 1);
    expect_shape_mismatch(
        rnn.forward(&wrong_features),
        "RNNCell",
        "input",
        &[1, 2],
        &[1, 3],
    );

    let gru = GRUCell::<f64, SequentialBackend>::new(2, 2);
    expect_invalid_rank(gru.forward(&rank_one), "GRUCell", 1);
    expect_shape_mismatch(
        gru.forward(&wrong_features),
        "GRUCell",
        "input",
        &[1, 2],
        &[1, 3],
    );

    let lstm = LSTMCell::<f64, SequentialBackend>::new(2, 2);
    expect_invalid_rank(lstm.forward(&rank_one), "LSTMCell", 1);
    expect_shape_mismatch(
        lstm.forward(&wrong_features),
        "LSTMCell",
        "input",
        &[1, 2],
        &[1, 3],
    );
}

#[test]
fn recurrent_steps_reject_incompatible_state_shapes() {
    let backend = SequentialBackend;
    let input = zeros_var(&[1, 2], &backend);
    let wrong_state = zeros_var(&[2, 2], &backend);
    let valid_state = zeros_var(&[1, 2], &backend);

    let rnn = RNNCell::<f64, SequentialBackend>::new(2, 2, RnnNonlinearity::Tanh);
    expect_shape_mismatch(
        rnn.step(&input, &wrong_state),
        "RNNCell",
        "hidden state",
        &[1, 2],
        &[2, 2],
    );

    let gru = GRUCell::<f64, SequentialBackend>::new(2, 2);
    expect_shape_mismatch(
        gru.step(&input, &wrong_state),
        "GRUCell",
        "hidden state",
        &[1, 2],
        &[2, 2],
    );

    let lstm = LSTMCell::<f64, SequentialBackend>::new(2, 2);
    match lstm.step(&input, &valid_state, &wrong_state) {
        Err(ModuleError::ShapeMismatch {
            module: "LSTMCell",
            parameter: "cell state",
            expected,
            actual,
        }) => {
            assert_eq!(expected, vec![1, 2]);
            assert_eq!(actual, vec![2, 2]);
        }
        Err(other) => panic!("expected LSTMCell cell-state ShapeMismatch, got {other:?}"),
        Ok(_) => panic!("expected LSTMCell to reject incompatible cell state"),
    }
}
