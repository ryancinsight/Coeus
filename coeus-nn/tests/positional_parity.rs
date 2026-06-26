//! Differential parity for `SinusoidalEncoding` and `RotaryEmbedding`.
//!
//! Analytical oracles:
//!
//! SinusoidalEncoding at pos=0: angle = 0 / denom = 0 for all (i, denom).
//!   sin(0) = 0, cos(0) = 1 (IEEE-754 exact).
//!   The pos=0 row is [0, 1, 0, 1, ..., 0, 1] (even→sin=0, odd→cos=1).
//!   forward(zeros_input, seq_len=1) = zeros + table[0] = [0,1,0,1,...].
//!
//! RotaryEmbedding at pos=0: angle = 0 * theta = 0 for all (i, theta).
//!   cos(0) = 1, sin(0) = 0 (IEEE-754 exact).
//!   x' = x * cos + rotate_half(x) * sin = x * 1 + anything * 0 = x.
//!   forward(x, seq_len=1) = x (identity).
//!
//! All assertions use `assert_eq!`.
//! SequentialBackend and MoiraiBackend must return bitwise-identical results.

use coeus_autograd::Var;
use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_nn::{Module, RotaryEmbedding, SinusoidalEncoding};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

fn zeros_var<B: BackendOps<f64> + Default>(shape: &[usize], backend: &B) -> Var<f64, B>
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Var::new(Tensor::zeros_on(shape.to_vec(), backend), false)
}

fn v<B: BackendOps<f64> + Default>(shape: &[usize], vals: &[f64], backend: &B) -> Var<f64, B>
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Var::new(Tensor::from_slice_on(shape.to_vec(), vals, backend), false)
}

fn check_sinusoidal<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // d_model=4, max_len=2. Pos-0 row: [sin(0),cos(0),sin(0),cos(0)] = [0,1,0,1].
    let pe = SinusoidalEncoding::<f64, B>::new(2, 4);

    // input [1,1,4] (batch=1, seq_len=1): forward adds table[0] = [0,1,0,1].
    let inp = zeros_var(&[1, 1, 4], backend);
    let out = Module::<f64, B>::forward(&pe, &inp);
    assert_eq!(
        out.tensor.shape(),
        &[1, 1, 4],
        "SinusoidalEncoding output shape"
    );
    assert_eq!(
        out.tensor.as_slice(),
        &[0.0_f64, 1.0, 0.0, 1.0],
        "SinusoidalEncoding zeros+pos0=[0,1,0,1]"
    );

    // Non-zero input [1,1,4] = [[[ 1,2,3,4 ]]]: adds [0,1,0,1] → [[[ 1,3,3,5 ]]].
    let inp2 = v(&[1, 1, 4], &[1.0, 2.0, 3.0, 4.0], backend);
    let out2 = Module::<f64, B>::forward(&pe, &inp2);
    assert_eq!(
        out2.tensor.as_slice(),
        &[1.0_f64, 3.0, 3.0, 5.0],
        "SinusoidalEncoding [1,2,3,4]+pos0=[1,3,3,5]"
    );
}

fn check_rope<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // d_head=4, max_len=4. At pos=0: angle=0 for all i → cos=1, sin=0 → identity.
    // Input shape [batch=1, seq_len=1, heads=1, d_head=4] = [[[[1,2,3,4]]]].
    let rope = RotaryEmbedding::<f64, B>::new(4, 4, 10000.0);
    let inp = v(&[1, 1, 1, 4], &[1.0, 2.0, 3.0, 4.0], backend);
    let out = rope.forward(&inp);
    assert_eq!(
        out.tensor.shape(),
        &[1, 1, 1, 4],
        "RotaryEmbedding output shape"
    );
    assert_eq!(
        out.tensor.as_slice(),
        inp.tensor.as_slice(),
        "RotaryEmbedding at pos=0 is identity"
    );

    // Zero input: output is zero regardless of rotation.
    let inp_zero = zeros_var(&[1, 1, 1, 4], backend);
    let out_zero = rope.forward(&inp_zero);
    assert_eq!(
        out_zero.tensor.as_slice(),
        &[0.0_f64; 4],
        "RotaryEmbedding zeros → zeros"
    );
}

fn check_all<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    check_sinusoidal(backend);
    check_rope(backend);
}

#[test]
fn sequential_positional_match_reference() {
    check_all(&SequentialBackend);
}

#[test]
fn moirai_positional_match_reference() {
    check_all(&MoiraiBackend);
}
