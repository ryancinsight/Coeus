//! Differential verification of the CPU unary `BackendOps` path.
//!
//! `SequentialBackend` and `MoiraiBackend` delegate unary CPU kernels through
//! `coeus-leto::elementwise_unary_into`, which maps each lane with
//! `Scalar::eval_unary`. The reference below calls that scalar operation
//! directly, so exact equality is the correct oracle: every output lane is one
//! invocation of the same native-precision scalar contract.

use coeus_core::{
    ComputeBackend, CpuAddressableStorageMut, CpuUnaryDispatch, CpuUnaryOp, Layout, MoiraiBackend,
    Scalar, SequentialBackend, Shape,
};
use coeus_ops::{BackendOps, CpuBackend};

const LEAKY_SLOPE: f64 = 0.125;

const OPS: &[CpuUnaryOp] = &[
    CpuUnaryOp::Relu,
    CpuUnaryOp::ReluGrad,
    CpuUnaryOp::Sigmoid,
    CpuUnaryOp::SigmoidGrad,
    CpuUnaryOp::Tanh,
    CpuUnaryOp::TanhGrad,
    CpuUnaryOp::Gelu,
    CpuUnaryOp::GeluGrad,
    CpuUnaryOp::Sin,
    CpuUnaryOp::Cos,
    CpuUnaryOp::Exp,
    CpuUnaryOp::Log,
    CpuUnaryOp::Neg,
    CpuUnaryOp::Abs,
    CpuUnaryOp::Sqrt,
    CpuUnaryOp::Silu,
    CpuUnaryOp::SiluGrad,
    CpuUnaryOp::Mish,
    CpuUnaryOp::MishGrad,
    CpuUnaryOp::Elu,
    CpuUnaryOp::EluGrad,
    CpuUnaryOp::Softplus,
    CpuUnaryOp::SoftplusGrad,
    CpuUnaryOp::GeluTanh,
    CpuUnaryOp::GeluTanhGrad,
    CpuUnaryOp::LeakyRelu(LEAKY_SLOPE.to_bits()),
    CpuUnaryOp::LeakyReluGrad(LEAKY_SLOPE.to_bits()),
];

fn device_unary<T, B>(backend: &B, op: CpuUnaryOp, input: &[T]) -> Vec<T>
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let layout = Layout::new(Shape::from(vec![input.len()]));
    let mut input_buffer = ComputeBackend::allocate::<T>(backend, input.len());
    let mut output_buffer = ComputeBackend::allocate::<T>(backend, input.len());

    backend.copy_to_device(input, &mut input_buffer);
    backend.elementwise_unary(op, &input_buffer, &layout, &mut output_buffer, &layout);

    let mut output = vec![T::zero(); input.len()];
    backend.copy_to_host(&output_buffer, &mut output);
    output
}

fn inputs_for<T: Scalar>(op: CpuUnaryOp) -> Vec<T> {
    let values: &[f64] = match op {
        CpuUnaryOp::Log | CpuUnaryOp::Sqrt => &[0.25, 1.0, 4.0, 16.0],
        _ => &[-2.0, -0.5, 0.0, 0.5, 2.0],
    };
    values.iter().map(|&value| T::from_f64(value)).collect()
}

fn check_unary<T, B>(backend: &B)
where
    T: Scalar + leto_ops::Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    for &op in OPS {
        let input = inputs_for::<T>(op);
        let got = device_unary(backend, op, &input);
        let expected: Vec<T> = input
            .iter()
            .map(|&value| <T as CpuUnaryDispatch>::eval_unary(op, value))
            .collect();

        for (index, (&actual, &reference)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_f64().to_bits(),
                reference.to_f64().to_bits(),
                "{op:?} mismatch at index {index}"
            );
        }
    }
}

#[test]
fn sequential_unary_matches_scalar_reference() {
    let backend = SequentialBackend;
    check_unary::<f32, _>(&backend);
    check_unary::<f64, _>(&backend);
}

#[test]
fn moirai_unary_matches_scalar_reference() {
    let backend = MoiraiBackend;
    check_unary::<f32, _>(&backend);
    check_unary::<f64, _>(&backend);
}
