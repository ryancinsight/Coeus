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
use coeus_nn::{Bidirectional, Gru, Lstm, Module, ModuleError, Rnn, RnnNonlinearity};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

fn zeros_var<B: BackendOps<f64> + coeus_ops::RandomInitOps<f64> + Default>(
    shape: &[usize],
    backend: &B,
) -> Var<f64, B>
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Var::new(Tensor::zeros_on(shape.to_vec(), backend), false)
}

fn check_lstm<B: BackendOps<f64> + coeus_ops::RandomInitOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // Lstm::new uses Linear::new (ones weights, zeros bias).
    // With zeros input and zeros initial state:
    //   gates = ones·0 + 0 + ones·0 + 0 = 0
    //   i=f=o=0.5, g=0 → c_new=0, h_new=0 for every timestep.
    //
    // Shape: [batch=1, seq_len=3, input_size=2] → [1, 3, hidden=2]
    let lstm =
        Lstm::<f64, B>::new(2, 2).expect("invariant: the fixture's layer dimensions are non-zero");
    let inp = zeros_var(&[1, 3, 2], backend);

    let (out, (h_n, c_n)) = lstm.forward_seq(&inp).expect("valid Lstm sequence input");

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
    let module_out = Module::<f64, B>::forward(&lstm, &inp).expect("valid Lstm sequence input");
    assert_eq!(
        module_out.tensor.as_slice(),
        out.tensor.as_slice(),
        "Lstm Module::forward == forward_seq output"
    );

    // Single-timestep: seq_len=1 gives [1,1,2] with zeros entry.
    let inp1 = zeros_var(&[1, 1, 2], backend);
    let (out1, _) = lstm.forward_seq(&inp1).expect("valid Lstm sequence input");
    assert_eq!(out1.tensor.shape(), &[1, 1, 2], "Lstm seq=1 shape");
    assert_eq!(out1.tensor.as_slice(), &[0.0_f64; 2], "Lstm seq=1 zeros");
}

fn check_gru<B: BackendOps<f64> + coeus_ops::RandomInitOps<f64> + Default>(backend: &B)
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
    let gru =
        Gru::<f64, B>::new(2, 2).expect("invariant: the fixture's layer dimensions are non-zero");
    let inp = zeros_var(&[1, 3, 2], backend);

    let (out, h_n) = gru.forward_seq(&inp).expect("valid Gru sequence input");

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
    let module_out = Module::<f64, B>::forward(&gru, &inp).expect("valid Gru sequence input");
    assert_eq!(
        module_out.tensor.as_slice(),
        out.tensor.as_slice(),
        "Gru Module::forward == forward_seq output"
    );

    // Batch dimension: batch=2, each sequence element is zeros.
    let inp2 = zeros_var(&[2, 4, 2], backend);
    let (out2, h_n2) = gru.forward_seq(&inp2).expect("valid Gru sequence input");
    assert_eq!(out2.tensor.shape(), &[2, 4, 2], "Gru batch=2 shape");
    assert_eq!(out2.tensor.as_slice(), &[0.0_f64; 16], "Gru batch=2 zeros");
    assert_eq!(
        h_n2.tensor.as_slice(),
        &[0.0_f64; 4],
        "Gru batch=2 h_n zeros"
    );
}

fn check_all<B: BackendOps<f64> + coeus_ops::RandomInitOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    check_lstm(backend);
    check_gru(backend);
    check_bidirectional(backend);
}

fn check_bidirectional<B: BackendOps<f64> + coeus_ops::RandomInitOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // Bidirectional<Rnn>: forward over x, backward over reversed x, concat on hidden.
    let fwd = Rnn::<f64, B>::new(2, 3, RnnNonlinearity::Tanh)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let bwd = Rnn::<f64, B>::new(2, 3, RnnNonlinearity::Tanh)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let bi = Bidirectional::new(fwd.clone(), bwd.clone());

    // Zero input → zeros, shape [batch, seq, 2*hidden].
    let xz = Var::new(Tensor::zeros_on([1, 2, 2], backend), false);
    let oz = Module::<f64, B>::forward(&bi, &xz).expect("valid Bidirectional RNN input");
    assert_eq!(oz.tensor.shape(), &[1, 2, 6], "bidirectional output shape");
    assert!(
        oz.tensor.as_slice().iter().all(|&v| v == 0.0),
        "bidirectional zero-input → zeros"
    );

    // Non-zero: the concatenation must place forward output in [0:H] and the
    // re-reversed backward output in [H:2H] at every timestep.
    let x = Var::new(
        Tensor::from_slice_on([1, 2, 2], &[1.0, 2.0, 3.0, 4.0], backend),
        false,
    );
    let o = Module::<f64, B>::forward(&bi, &x).expect("valid Bidirectional RNN input");
    assert_eq!(o.tensor.shape(), &[1, 2, 6]);
    let o_s = o.tensor.as_slice();

    let f_out = Module::<f64, B>::forward(&fwd, &x).expect("valid forward RNN input");
    let reversed = Module::<f64, B>::forward(&bwd, &coeus_autograd::flip(&x, 1))
        .expect("valid backward RNN input");
    let b_out = coeus_autograd::flip(&reversed, 1);
    let f_s = f_out.tensor.as_slice(); // [1,2,3]
    let b_s = b_out.tensor.as_slice(); // [1,2,3]
                                       // Layout per (batch, seq): [forward(3), backward(3)].
    assert_eq!(&o_s[0..3], &f_s[0..3], "t0 forward half");
    assert_eq!(&o_s[3..6], &b_s[0..3], "t0 backward half");
    assert_eq!(&o_s[6..9], &f_s[3..6], "t1 forward half");
    assert_eq!(&o_s[9..12], &b_s[3..6], "t1 backward half");
}

#[test]
fn sequential_rnn_seq_match_reference() {
    check_all(&SequentialBackend);
}

#[test]
fn moirai_rnn_seq_match_reference() {
    check_all(&MoiraiBackend);
}

// ── Bidirectional tests ──

#[test]
fn bidirectional_lstm_doubles_hidden_dim() {
    // A Bidirectional<Lstm> with hidden=4 should produce [batch, seq, 8] output.
    use coeus_nn::{Bidirectional, Lstm, Module};
    let fwd = Lstm::<f32, SequentialBackend>::new(2, 4)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let bwd = Lstm::<f32, SequentialBackend>::new(2, 4)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let bi = Bidirectional::new(fwd, bwd);

    let x = Var::new(
        coeus_tensor::Tensor::<f32, SequentialBackend>::zeros(vec![2, 3, 2]),
        false,
    );
    let y = bi.forward(&x).expect("valid Bidirectional RNN input");
    assert_eq!(
        y.tensor.shape(),
        &[2, 3, 8],
        "bidirectional hidden dim should be 2*hidden"
    );
}

#[test]
fn bidirectional_gru_zeros_output_for_zeros_input() {
    // Zeros input + zeros state → zeros output for any weight init.
    use coeus_nn::{Bidirectional, Gru, Module};
    let fwd = Gru::<f32, SequentialBackend>::new(2, 4)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let bwd = Gru::<f32, SequentialBackend>::new(2, 4)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    let bi = Bidirectional::new(fwd, bwd);

    let x = Var::new(
        coeus_tensor::Tensor::<f32, SequentialBackend>::zeros(vec![1, 5, 2]),
        false,
    );
    let y = bi.forward(&x).expect("valid Bidirectional RNN input");
    assert_eq!(y.tensor.shape(), &[1, 5, 8]);
    // Zeros input → zeros output (analytically, same as unidirectional).
    for &v in y.tensor.as_slice() {
        assert!((v).abs() < 1e-6, "expected zero output, got {v}");
    }
}

fn expect_sequence_invalid_rank(
    result: Result<Var<f64, SequentialBackend>, ModuleError<coeus_core::BackendError>>,
    module: &'static str,
) {
    match result {
        Err(ModuleError::InvalidRank {
            module: actual_module,
            expected: "3",
            actual: 2,
        }) => assert_eq!(actual_module, module),
        Err(other) => panic!("expected {module} InvalidRank, got {other:?}"),
        Ok(_) => panic!("expected {module} to reject rank-two input"),
    }
}

fn expect_sequence_feature_mismatch(
    result: Result<Var<f64, SequentialBackend>, ModuleError<coeus_core::BackendError>>,
    module: &'static str,
) {
    match result {
        Err(ModuleError::ShapeMismatch {
            module: actual_module,
            parameter: "input",
            expected,
            actual,
        }) => {
            assert_eq!(actual_module, module);
            assert_eq!(expected, vec![1, 2, 2]);
            assert_eq!(actual, vec![1, 2, 3]);
        }
        Err(other) => panic!("expected {module} ShapeMismatch, got {other:?}"),
        Ok(_) => panic!("expected {module} to reject the trailing feature size"),
    }
}

fn expect_empty_sequence(
    result: Result<Var<f64, SequentialBackend>, ModuleError<coeus_core::BackendError>>,
    module: &'static str,
) {
    match result {
        Err(ModuleError::InsufficientElements {
            module: actual_module,
            minimum: 1,
            actual: 0,
        }) => assert_eq!(actual_module, module),
        Err(other) => panic!("expected {module} InsufficientElements, got {other:?}"),
        Ok(_) => panic!("expected {module} to reject an empty sequence"),
    }
}

#[test]
fn recurrent_sequences_reject_invalid_input_contracts() {
    let backend = SequentialBackend;
    let rank_two = zeros_var(&[1, 2], &backend);
    let wrong_features = zeros_var(&[1, 2, 3], &backend);
    let empty = zeros_var(&[1, 0, 2], &backend);

    let rnn = Rnn::<f64, SequentialBackend>::new(2, 2, RnnNonlinearity::Tanh)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    expect_sequence_invalid_rank(rnn.forward(&rank_two), "Rnn");
    expect_sequence_feature_mismatch(rnn.forward(&wrong_features), "Rnn");
    expect_empty_sequence(rnn.forward(&empty), "Rnn");

    let gru = Gru::<f64, SequentialBackend>::new(2, 2)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    expect_sequence_invalid_rank(gru.forward(&rank_two), "Gru");
    expect_sequence_feature_mismatch(gru.forward(&wrong_features), "Gru");
    expect_empty_sequence(gru.forward(&empty), "Gru");

    let lstm = Lstm::<f64, SequentialBackend>::new(2, 2)
        .expect("invariant: the fixture's layer dimensions are non-zero");
    expect_sequence_invalid_rank(lstm.forward(&rank_two), "Lstm");
    expect_sequence_feature_mismatch(lstm.forward(&wrong_features), "Lstm");
    expect_empty_sequence(lstm.forward(&empty), "Lstm");
}

#[derive(Clone)]
struct ReshapeSequence {
    shape: Vec<usize>,
}

impl Module<f64, SequentialBackend> for ReshapeSequence {
    fn parameters(&self) -> Vec<Var<f64, SequentialBackend>> {
        Vec::new()
    }

    fn forward(
        &self,
        input: &Var<f64, SequentialBackend>,
    ) -> Result<Var<f64, SequentialBackend>, ModuleError<coeus_core::BackendError>> {
        Ok(coeus_autograd::reshape(input, self.shape.clone()))
    }
}

#[test]
fn bidirectional_rejects_child_sequence_shape_changes() {
    let bidirectional = Bidirectional::new(
        ReshapeSequence {
            shape: vec![1, 2, 2],
        },
        ReshapeSequence {
            shape: vec![1, 1, 4],
        },
    );
    let input = zeros_var(&[1, 2, 2], &SequentialBackend);

    match bidirectional.forward(&input) {
        Err(ModuleError::ShapeMismatch {
            module: "Bidirectional",
            parameter: "backward output",
            expected,
            actual,
        }) => {
            assert_eq!(expected, vec![1, 2, 4]);
            assert_eq!(actual, vec![1, 1, 4]);
        }
        Err(other) => panic!("expected Bidirectional ShapeMismatch, got {other:?}"),
        Ok(_) => panic!("expected Bidirectional to reject the child output shape"),
    }
}
