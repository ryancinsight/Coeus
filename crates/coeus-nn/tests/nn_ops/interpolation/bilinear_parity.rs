//! Differential parity for the `Bilinear` module.
//!
//! Analytical oracle — weight = ones, bias = zeros:
//! ```text
//! out[n, k] = Σ_i Σ_j  x1[n,i] · W[k,i,j] · x2[n,j]
//!           = (Σ_i x1[n,i]) · (Σ_j x2[n,j])      (W = ones everywhere)
//! ```
//! With x1 = [1, 2], x2 = [3, 4], out_features = 2:
//!   `out[0, k] = (1+2) · (3+4) = 3 · 7 = 21`  for k = 0, 1
//!
//! With x1 = x2 = [1, 2] (Module::forward uses x1 = x2 = input):
//!   `out[0, k] = 3 · 3 = 9`
//!
//! All values are exact integers representable in f64.
//! SequentialBackend and MoiraiBackend must produce bitwise-identical results.

use coeus_autograd::Var;
use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_nn::{bilinear as bilinear_fn, Bilinear, Module};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

/// Construct a `Bilinear` with weight = ones and bias = zeros directly.
/// Bypasses the random xavier init in `Bilinear::new` for a deterministic oracle.
fn ones_bilinear<B: BackendOps<f64> + Default>(
    in1: usize,
    in2: usize,
    out: usize,
    backend: &B,
) -> Bilinear<f64, B>
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Bilinear {
        weight: Var::new(Tensor::ones_on([out, in1, in2], backend).expect("construct tensor"), false).expect("construct variable"),
        bias: Some(Var::new(Tensor::zeros_on([out], backend).expect("construct tensor"), false).expect("construct variable")),
        in1_features: in1,
        in2_features: in2,
        out_features: out,
    }
}

fn v<B: BackendOps<f64> + Default>(shape: &[usize], vals: &[f64], backend: &B) -> Var<f64, B>
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Var::new(Tensor::from_slice_on(shape.to_vec(), vals, backend).expect("construct tensor"), false).expect("construct variable")
}

fn check_bilinear<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // W=ones [2,2,2], b=zeros [2]: out[n,k] = (sum x1[n]) * (sum x2[n]).
    let bil = ones_bilinear::<B>(2, 2, 2, backend);

    // x1=[1,2] (sum=3), x2=[3,4] (sum=7): out[0,k] = 21 for both k.
    let x1 = v(&[1, 2], &[1.0, 2.0], backend);
    let x2 = v(&[1, 2], &[3.0, 4.0], backend);
    let out = bil.bilinear_forward(&x1, &x2).expect("run operation");
    assert_eq!(out.tensor.shape(), &[1, 2], "Bilinear output shape");
    assert_eq!(
        out.tensor.as_slice(),
        &[21.0_f64, 21.0],
        "Bilinear([1,2],[3,4]) = [21,21]"
    );
    let out_fn = bilinear_fn(&x1, &x2, &bil.weight, bil.bias.as_ref()).expect("run operation");
    assert_eq!(
        out_fn.tensor.as_slice(),
        out.tensor.as_slice(),
        "functional bilinear matches module bilinear_forward"
    );

    // x1=x2=[1,2] (sum=3): out[0,k] = 9 for both k.
    let x_same = v(&[1, 2], &[1.0, 2.0], backend);
    let out_same = bil.bilinear_forward(&x_same, &x_same).expect("run operation");
    assert_eq!(
        out_same.tensor.as_slice(),
        &[9.0_f64, 9.0],
        "Bilinear([1,2],[1,2]) = [9,9]"
    );

    // Module::forward(x) delegates to bilinear_forward(x, x): must equal [9,9].
    let out_module = Module::<f64, B>::forward(&bil, &x_same).expect("run forward");
    assert_eq!(
        out_module.tensor.as_slice(),
        out_same.tensor.as_slice(),
        "Bilinear Module::forward == bilinear_forward(x,x)"
    );

    // Zero input: out = 0 exactly.
    let xz = v(&[1, 2], &[0.0, 0.0], backend);
    let out_zero = bil.bilinear_forward(&xz, &xz).expect("run operation");
    assert_eq!(
        out_zero.tensor.as_slice(),
        &[0.0_f64, 0.0],
        "Bilinear zeros → zeros"
    );

    // No-bias variant: same arithmetic, just no b term (already 0 here).
    let bil_no_bias = Bilinear {
        weight: Var::new(Tensor::ones_on([1_usize, 2, 2], backend).expect("construct tensor"), false).expect("construct variable"),
        bias: None,
        in1_features: 2,
        in2_features: 2,
        out_features: 1,
    };
    let out_nb = bil_no_bias.bilinear_forward(&x1, &x2).expect("run operation");
    assert_eq!(
        out_nb.tensor.as_slice(),
        &[21.0_f64],
        "Bilinear no-bias([1,2],[3,4]) = [21]"
    );

    // Per-output weights verify the [out, in1, in2] indexing contract:
    // W[0] is identity, W[1] swaps x2 coordinates, bias=[0.5,-0.5].
    // x1=[2,3], x2=[4,5] -> [2*4 + 3*5 + 0.5, 2*5 + 3*4 - 0.5].
    let indexed_bilinear = Bilinear {
        weight: Var::new(
            Tensor::from_slice_on(
                vec![2_usize, 2, 2],
                &[1.0_f64, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                backend,
            ).expect("construct tensor"),
            false,
        ).expect("construct variable"),
        bias: Some(Var::new(
            Tensor::from_slice_on(vec![2_usize], &[0.5_f64, -0.5], backend).expect("construct tensor"),
            false,
        ).expect("construct variable")),
        in1_features: 2,
        in2_features: 2,
        out_features: 2,
    };
    let indexed_x1 = v(&[1, 2], &[2.0, 3.0], backend);
    let indexed_x2 = v(&[1, 2], &[4.0, 5.0], backend);
    let indexed_out = indexed_bilinear.bilinear_forward(&indexed_x1, &indexed_x2).expect("run operation");
    assert_eq!(
        indexed_out.tensor.as_slice(),
        &[23.5_f64, 21.5],
        "Bilinear per-output weight indexing"
    );

    // Batch dimension: 2 samples.
    // x1=[1,2; 0,1] (sums=[3,1]), x2=[3,4; 2,0] (sums=[7,2])
    // out[0,k]=21, out[1,k]=2
    let x1b = v(&[2, 2], &[1.0, 2.0, 0.0, 1.0], backend);
    let x2b = v(&[2, 2], &[3.0, 4.0, 2.0, 0.0], backend);
    let outb = bil.bilinear_forward(&x1b, &x2b).expect("run operation");
    assert_eq!(outb.tensor.shape(), &[2, 2], "Bilinear batch shape");
    assert_eq!(
        outb.tensor.as_slice(),
        &[21.0_f64, 21.0, 2.0, 2.0],
        "Bilinear batch=[21,21,2,2]"
    );
}

#[test]
fn sequential_bilinear_match_reference() {
    check_bilinear(&SequentialBackend);
}

#[test]
fn moirai_bilinear_match_reference() {
    check_bilinear(&MoiraiBackend);
}
