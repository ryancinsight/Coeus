// ── Pool1d parity tests ──
//
// Verifies MaxPool1d and AvgPool1d forward outputs against hand-computed values.

use coeus_autograd::Var;
use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_nn::{AvgPool1d, MaxPool1d, Module, ModuleError};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

fn seq_var(shape: impl Into<coeus_core::Shape>, data: &[f32]) -> Var<f32, SequentialBackend> {
    Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(shape, data),
        false,
    )
}

#[test]
fn maxpool1d_no_pad() {
    // Input [1, 1, 6]: [1, 3, 2, 4, 1, 5], k=2, s=2, p=0
    // Windows: [1,3]=3, [2,4]=4, [1,5]=5  → output [3, 4, 5]
    let pool = MaxPool1d::<f32, SequentialBackend>::new(2);
    let x = seq_var([1, 1, 6], &[1.0, 3.0, 2.0, 4.0, 1.0, 5.0]);
    let y = pool.forward(&x).expect("valid MaxPool1d input");
    assert_eq!(y.tensor.shape(), &[1, 1, 3]);
    let s = y.tensor.as_slice();
    assert!((s[0] - 3.0).abs() < 1e-6, "window 0");
    assert!((s[1] - 4.0).abs() < 1e-6, "window 1");
    assert!((s[2] - 5.0).abs() < 1e-6, "window 2");
}

#[test]
fn maxpool1d_stride1() {
    // Input [1, 1, 4]: [1, 4, 2, 3], k=2, s=1
    // Windows: [1,4]=4, [4,2]=4, [2,3]=3  → output [4, 4, 3]
    let pool = MaxPool1d::<f32, SequentialBackend>::with_params(2, 1, 0, 1);
    let x = seq_var([1, 1, 4], &[1.0, 4.0, 2.0, 3.0]);
    let y = pool.forward(&x).expect("valid MaxPool1d input");
    assert_eq!(y.tensor.shape(), &[1, 1, 3]);
    let s = y.tensor.as_slice();
    assert!((s[0] - 4.0).abs() < 1e-6);
    assert!((s[1] - 4.0).abs() < 1e-6);
    assert!((s[2] - 3.0).abs() < 1e-6);
}

#[test]
fn maxpool1d_multi_channel() {
    // Input [1, 2, 4], k=2, s=2
    // Ch0: [1,3]=3, [2,4]=4  → [3, 4]
    // Ch1: [5,7]=7, [6,8]=8  → [7, 8]
    let pool = MaxPool1d::<f32, SequentialBackend>::new(2);
    let x = seq_var([1, 2, 4], &[1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]);
    let y = pool.forward(&x).expect("valid MaxPool1d input");
    assert_eq!(y.tensor.shape(), &[1, 2, 2]);
    let s = y.tensor.as_slice();
    assert!((s[0] - 3.0).abs() < 1e-6, "ch0 w0");
    assert!((s[1] - 4.0).abs() < 1e-6, "ch0 w1");
    assert!((s[2] - 7.0).abs() < 1e-6, "ch1 w0");
    assert!((s[3] - 8.0).abs() < 1e-6, "ch1 w1");
}

#[test]
fn avgpool1d_no_pad() {
    // Input [1, 1, 4]: [1, 3, 2, 4], k=2, s=2
    // Windows: (1+3)/2=2.0, (2+4)/2=3.0
    let pool = AvgPool1d::<f32, SequentialBackend>::new(2);
    let x = seq_var([1, 1, 4], &[1.0, 3.0, 2.0, 4.0]);
    let y = pool.forward(&x).expect("valid AvgPool1d input");
    assert_eq!(y.tensor.shape(), &[1, 1, 2]);
    let s = y.tensor.as_slice();
    assert!((s[0] - 2.0).abs() < 1e-6, "window 0");
    assert!((s[1] - 3.0).abs() < 1e-6, "window 1");
}

#[test]
fn avgpool1d_stride1() {
    // Input [1, 1, 4]: [2, 4, 6, 8], k=3, s=1
    // Windows: (2+4+6)/3=4.0, (4+6+8)/3=6.0
    let pool = AvgPool1d::<f32, SequentialBackend>::with_params(3, 1, 0, 1);
    let x = seq_var([1, 1, 4], &[2.0, 4.0, 6.0, 8.0]);
    let y = pool.forward(&x).expect("valid AvgPool1d input");
    assert_eq!(y.tensor.shape(), &[1, 1, 2]);
    let s = y.tensor.as_slice();
    assert!((s[0] - 4.0).abs() < 1e-6, "window 0");
    assert!((s[1] - 6.0).abs() < 1e-6, "window 1");
}

#[test]
fn avgpool1d_batch() {
    // Input [2, 1, 4], k=2, s=2: batch of two independent rows
    let pool = AvgPool1d::<f32, SequentialBackend>::new(2);
    let x = seq_var([2, 1, 4], &[1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
    let y = pool.forward(&x).expect("valid AvgPool1d input");
    assert_eq!(y.tensor.shape(), &[2, 1, 2]);
    let s = y.tensor.as_slice();
    // batch0: 2.0, 6.0
    assert!((s[0] - 2.0).abs() < 1e-6);
    assert!((s[1] - 6.0).abs() < 1e-6);
    // batch1: 3.0, 7.0
    assert!((s[2] - 3.0).abs() < 1e-6);
    assert!((s[3] - 7.0).abs() < 1e-6);
}

fn provider_var<B: BackendOps<f64> + Default>(
    shape: &[usize],
    data: &[f64],
    backend: &B,
) -> Var<f64, B>
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Var::new(Tensor::from_slice_on(shape.to_vec(), data, backend), false)
}

fn assert_pool1d_provider_contract<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // [N=1, C=2, L=8], kernel=2, stride=2. Every average is integral, so the
    // exact outputs also rule out provider-specific reduction-order drift.
    let input = provider_var(
        &[1, 2, 8],
        &[
            1.0, 3.0, 2.0, 4.0, 1.0, 5.0, 0.0, 6.0, 7.0, 9.0, 8.0, 10.0, 7.0, 11.0, 6.0, 12.0,
        ],
        backend,
    );

    let max_pool = MaxPool1d::<f64, B>::with_params(2, 2, 0, 1);
    let max_output = Module::<f64, B>::forward(&max_pool, &input).expect("valid MaxPool1d input");
    assert_eq!(max_output.tensor.shape(), &[1, 2, 4], "MaxPool1d shape");
    assert_eq!(
        max_output.tensor.as_slice(),
        &[3.0, 4.0, 5.0, 6.0, 9.0, 10.0, 11.0, 12.0],
        "MaxPool1d value oracle"
    );

    let average_pool = AvgPool1d::<f64, B>::with_params(2, 2, 0, 1);
    let average_output =
        Module::<f64, B>::forward(&average_pool, &input).expect("valid AvgPool1d input");
    assert_eq!(average_output.tensor.shape(), &[1, 2, 4], "AvgPool1d shape");
    assert_eq!(
        average_output.tensor.as_slice(),
        &[2.0, 3.0, 3.0, 3.0, 8.0, 9.0, 9.0, 9.0],
        "AvgPool1d value oracle"
    );
}

#[test]
fn sequential_pool1d_matches_analytical_contract() {
    assert_pool1d_provider_contract(&SequentialBackend);
}

#[test]
fn moirai_pool1d_matches_analytical_contract() {
    assert_pool1d_provider_contract(&MoiraiBackend);
}

#[test]
fn pool1d_rejects_zero_window_configuration_and_wrong_rank() {
    let input = Var::new(Tensor::<f32, SequentialBackend>::ones([1, 1, 4]), false);
    for error in [
        MaxPool1d::<f32, SequentialBackend>::new(0)
            .forward(&input)
            .err()
            .expect("zero MaxPool1d kernel must be rejected"),
        AvgPool1d::<f32, SequentialBackend>::with_params(2, 0, 0, 1)
            .forward(&input)
            .err()
            .expect("zero AvgPool1d stride must be rejected"),
    ] {
        match error {
            ModuleError::ShapeMismatch {
                parameter, actual, ..
            } => {
                assert_eq!(parameter, "pooling window");
                assert!(actual.contains(&0));
            }
            other => panic!("expected typed Pool1d configuration error, got {other:?}"),
        }
    }

    let wrong_rank = Var::new(Tensor::<f32, SequentialBackend>::ones([1, 4]), false);
    let error = MaxPool1d::<f32, SequentialBackend>::new(2)
        .forward(&wrong_rank)
        .err()
        .expect("rank-two MaxPool1d input must be rejected");
    match error {
        ModuleError::InvalidRank {
            module,
            expected,
            actual,
        } => {
            assert_eq!(module, "MaxPool1d");
            assert_eq!(expected, "3");
            assert_eq!(actual, 2);
        }
        other => panic!("expected typed MaxPool1d rank error, got {other:?}"),
    }
}
