//! Differential verification of neural-network initializers.
//!
//! CPU `coeus_nn::init::{uniform_with_seed, normal_with_seed}` dispatch through
//! destination-writing `coeus-leto` operations. The oracle is direct Leto
//! output for the same shape, scalar type, and seed.

use coeus_autograd::Var;
use coeus_core::{
    ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, Scalar,
    SequentialBackend,
};
use coeus_tensor::Tensor;

use coeus_nn::init::InitializationError;

fn assert_values<T: Scalar>(got: &[T], expected: &[T], context: &str) {
    assert_eq!(got.len(), expected.len(), "{context}: length mismatch");
    for (index, (&actual, &reference)) in got.iter().zip(expected).enumerate() {
        assert_eq!(actual, reference, "{context}: mismatch at index {index}");
    }
}

fn check_backend<T, B>(backend: &B)
where
    T: coeus_core::Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + coeus_ops::RandomInitOps<T> + ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = [2usize, 3];
    let mut weight = Var::new(Tensor::<T, B>::zeros_on(shape, backend), true);

    coeus_nn::init::uniform_with_seed(&mut weight, -2.0, 5.0, 42)
        .expect("valid seeded uniform initializer fixture");
    let expected_uniform = coeus_leto::uniform_values(
        &shape,
        <T as coeus_core::Scalar>::from_f64(-2.0),
        <T as coeus_core::Scalar>::from_f64(5.0),
        42,
    )
    .expect("valid direct Leto uniform fixture");
    assert_values(
        weight.tensor.as_slice(),
        &expected_uniform,
        "uniform initializer",
    );

    coeus_nn::init::normal_with_seed(&mut weight, 1.0, 2.0, 11)
        .expect("valid seeded normal initializer fixture");
    let expected_normal = coeus_leto::normal_values(
        &shape,
        <T as coeus_core::Scalar>::from_f64(1.0),
        <T as coeus_core::Scalar>::from_f64(2.0),
        11,
    )
    .expect("valid direct Leto normal fixture");
    assert_values(
        weight.tensor.as_slice(),
        &expected_normal,
        "normal initializer",
    );
}

// kaiming_uniform and kaiming_normal delegate to uniform/normal with analytically
// computed bounds. Verify by reproducing those bounds and comparing against direct
// coeus-leto dispatch.
//
// kaiming_uniform bound: limit = sqrt(6 / fan_in)
// kaiming_normal std_dev: sigma = sqrt(2 / fan_in)
fn check_kaiming<T, B>(backend: &B)
where
    T: coeus_core::Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + coeus_ops::RandomInitOps<T> + ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = [3usize, 4];
    let fan_in = 4usize;
    let seed = 37u64;

    let mut weight = Var::new(Tensor::<T, B>::zeros_on(shape, backend), true);
    coeus_nn::init::kaiming_uniform_with_seed(&mut weight, fan_in, seed)
        .expect("valid seeded Kaiming uniform fixture");

    let fan = <T as Scalar>::from_usize(fan_in);
    let limit = (<T as Scalar>::from_f64(6.0) / fan).sqrt_val();
    let expected_uniform = coeus_leto::uniform_values(&shape, T::zero() - limit, limit, seed)
        .expect("valid direct Leto Kaiming uniform fixture");
    assert_values(
        weight.tensor.as_slice(),
        &expected_uniform,
        "kaiming_uniform",
    );

    coeus_nn::init::kaiming_normal_with_seed(&mut weight, fan_in, seed)
        .expect("valid seeded Kaiming normal fixture");
    let std_dev = (<T as Scalar>::from_f64(2.0) / fan).sqrt_val();
    let expected_normal =
        coeus_leto::normal_values(&shape, <T as Scalar>::from_f64(0.0), std_dev, seed)
            .expect("valid direct Leto Kaiming normal fixture");
    assert_values(weight.tensor.as_slice(), &expected_normal, "kaiming_normal");
}

#[test]
fn sequential_initializers_match_leto_dispatch() {
    let backend = SequentialBackend;
    check_backend::<f32, _>(&backend);
    check_backend::<f64, _>(&backend);
}

#[test]
fn moirai_initializers_match_leto_dispatch() {
    let backend = MoiraiBackend;
    check_backend::<f32, _>(&backend);
    check_backend::<f64, _>(&backend);
}

// xavier_uniform bound: limit = sqrt(6 / (fan_in + fan_out))
// xavier_normal std_dev: sigma = sqrt(2 / (fan_in + fan_out))
fn check_xavier<T, B>(backend: &B)
where
    T: coeus_core::Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + coeus_ops::RandomInitOps<T> + ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = [3usize, 4];
    let fan_in = 3usize;
    let fan_out = 4usize;
    let seed = 53u64;

    let mut weight = Var::new(Tensor::<T, B>::zeros_on(shape, backend), true);
    coeus_nn::init::xavier_uniform_with_seed(&mut weight, fan_in, fan_out, seed)
        .expect("valid seeded Xavier uniform fixture");

    let fan = <T as Scalar>::from_usize(fan_in + fan_out);
    let limit = (<T as Scalar>::from_f64(6.0) / fan).sqrt_val();
    let expected_uniform = coeus_leto::uniform_values(&shape, T::zero() - limit, limit, seed)
        .expect("valid direct Leto Xavier uniform fixture");
    assert_values(
        weight.tensor.as_slice(),
        &expected_uniform,
        "xavier_uniform",
    );

    coeus_nn::init::xavier_normal_with_seed(&mut weight, fan_in, fan_out, seed)
        .expect("valid seeded Xavier normal fixture");
    let std_dev = (<T as Scalar>::from_f64(2.0) / fan).sqrt_val();
    let expected_normal =
        coeus_leto::normal_values(&shape, <T as Scalar>::from_f64(0.0), std_dev, seed)
            .expect("valid direct Leto Xavier normal fixture");
    assert_values(weight.tensor.as_slice(), &expected_normal, "xavier_normal");
}

#[test]
fn sequential_kaiming_matches_leto_dispatch() {
    let backend = SequentialBackend;
    check_kaiming::<f32, _>(&backend);
    check_kaiming::<f64, _>(&backend);
}

#[test]
fn moirai_kaiming_matches_leto_dispatch() {
    let backend = MoiraiBackend;
    check_kaiming::<f32, _>(&backend);
    check_kaiming::<f64, _>(&backend);
}

#[test]
fn sequential_xavier_matches_leto_dispatch() {
    let backend = SequentialBackend;
    check_xavier::<f32, _>(&backend);
    check_xavier::<f64, _>(&backend);
}

#[test]
fn moirai_xavier_matches_leto_dispatch() {
    let backend = MoiraiBackend;
    check_xavier::<f32, _>(&backend);
    check_xavier::<f64, _>(&backend);
}

fn assert_unchanged(
    weight: &Var<f32, SequentialBackend>,
    expected_shape: &[usize],
    expected_values: &[f32],
) {
    assert_eq!(weight.tensor.shape(), expected_shape);
    assert_eq!(weight.tensor.as_slice(), expected_values);
}

#[test]
fn initializer_rejects_unsupported_ranks_without_mutation() {
    for shape in [vec![], vec![1, 1, 1, 1, 1, 1, 1]] {
        let rank = shape.len();
        let mut weight = Var::new(Tensor::<f32, SequentialBackend>::ones(shape.clone()), true);
        let original = weight.tensor.as_slice().to_vec();

        let error = coeus_nn::init::uniform_with_seed(&mut weight, -1.0, 1.0, 7)
            .expect_err("rank outside 1..=6 must be rejected");

        match error {
            InitializationError::InvalidRank {
                actual,
                minimum,
                maximum,
            } => {
                assert_eq!(actual, rank);
                assert_eq!(minimum, 1);
                assert_eq!(maximum, 6);
            }
            other => panic!("expected InvalidRank, got {other:?}"),
        }
        assert_unchanged(&weight, &shape, &original);
    }
}

#[test]
fn initializer_rejects_invalid_distribution_parameters_without_mutation() {
    let cases = [
        (f64::NAN, 1.0, "low", f64::NAN),
        (-1.0, f64::INFINITY, "high", f64::INFINITY),
        (f64::MAX, f64::MAX, "low", f64::MAX),
    ];

    for (low, high, expected_parameter, expected_value) in cases {
        let mut weight = Var::new(
            Tensor::<f32, SequentialBackend>::from_slice([2], &[3.0, 4.0]),
            true,
        );
        let error = coeus_nn::init::uniform_with_seed(&mut weight, low, high, 7)
            .expect_err("non-finite uniform parameter must be rejected");
        match error {
            InitializationError::NonFiniteParameter { parameter, value } => {
                assert_eq!(parameter, expected_parameter);
                if expected_value.is_nan() {
                    assert!(value.is_nan());
                } else {
                    assert_eq!(value, expected_value);
                }
            }
            other => panic!("expected NonFiniteParameter, got {other:?}"),
        }
        assert_unchanged(&weight, &[2], &[3.0, 4.0]);
    }

    let mut weight = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice([2], &[3.0, 4.0]),
        true,
    );
    let error = coeus_nn::init::uniform_with_seed(&mut weight, 2.0, -2.0, 7)
        .expect_err("reversed uniform bounds must be rejected");
    match error {
        InitializationError::InvalidUniformBounds { low, high } => {
            assert_eq!(low, 2.0);
            assert_eq!(high, -2.0);
        }
        other => panic!("expected InvalidUniformBounds, got {other:?}"),
    }
    assert_unchanged(&weight, &[2], &[3.0, 4.0]);

    for (mean, std_dev, expected_parameter, expected_value) in [
        (f64::NAN, 1.0, "mean", f64::NAN),
        (0.0, f64::INFINITY, "std_dev", f64::INFINITY),
    ] {
        let error = coeus_nn::init::normal_with_seed(&mut weight, mean, std_dev, 7)
            .expect_err("non-finite normal parameter must be rejected");
        match error {
            InitializationError::NonFiniteParameter { parameter, value } => {
                assert_eq!(parameter, expected_parameter);
                if expected_value.is_nan() {
                    assert!(value.is_nan());
                } else {
                    assert_eq!(value, expected_value);
                }
            }
            other => panic!("expected NonFiniteParameter, got {other:?}"),
        }
        assert_unchanged(&weight, &[2], &[3.0, 4.0]);
    }

    let error = coeus_nn::init::normal_with_seed(&mut weight, 0.0, -0.5, 7)
        .expect_err("negative normal standard deviation must be rejected");
    match error {
        InitializationError::NegativeStandardDeviation { value } => assert_eq!(value, -0.5),
        other => panic!("expected NegativeStandardDeviation, got {other:?}"),
    }
    assert_unchanged(&weight, &[2], &[3.0, 4.0]);
}

#[test]
fn initializer_rejects_invalid_fans_without_mutation() {
    let mut weight = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice([2], &[3.0, 4.0]),
        true,
    );

    let error = coeus_nn::init::kaiming_uniform_with_seed(&mut weight, 0, 7)
        .expect_err("zero Kaiming fan must be rejected");
    match error {
        InitializationError::InvalidFan { parameter, value } => {
            assert_eq!(parameter, "fan_in");
            assert_eq!(value, 0);
        }
        other => panic!("expected InvalidFan, got {other:?}"),
    }
    assert_unchanged(&weight, &[2], &[3.0, 4.0]);

    let error = coeus_nn::init::xavier_uniform_with_seed(&mut weight, 1, 0, 7)
        .expect_err("zero Xavier output fan must be rejected");
    match error {
        InitializationError::InvalidFan { parameter, value } => {
            assert_eq!(parameter, "fan_out");
            assert_eq!(value, 0);
        }
        other => panic!("expected InvalidFan, got {other:?}"),
    }
    assert_unchanged(&weight, &[2], &[3.0, 4.0]);

    let error = coeus_nn::init::xavier_normal_with_seed(&mut weight, usize::MAX, 1, 7)
        .expect_err("overflowing Xavier fan sum must be rejected");
    match error {
        InitializationError::FanArithmeticOverflow { fan_in, fan_out } => {
            assert_eq!(fan_in, usize::MAX);
            assert_eq!(fan_out, 1);
        }
        other => panic!("expected FanArithmeticOverflow, got {other:?}"),
    }
    assert_unchanged(&weight, &[2], &[3.0, 4.0]);
}

#[test]
fn default_initializers_match_explicit_seed_42() {
    type Initializer = fn(&mut Var<f32, SequentialBackend>);
    let initializers: [(Initializer, Initializer); 6] = [
        (
            |weight| coeus_nn::init::uniform(weight, -1.0, 1.0).expect("valid uniform fixture"),
            |weight| {
                coeus_nn::init::uniform_with_seed(weight, -1.0, 1.0, 42)
                    .expect("valid seeded uniform fixture")
            },
        ),
        (
            |weight| coeus_nn::init::normal(weight, 0.5, 1.5).expect("valid normal fixture"),
            |weight| {
                coeus_nn::init::normal_with_seed(weight, 0.5, 1.5, 42)
                    .expect("valid seeded normal fixture")
            },
        ),
        (
            |weight| {
                coeus_nn::init::xavier_uniform(weight, 2, 3).expect("valid Xavier uniform fixture")
            },
            |weight| {
                coeus_nn::init::xavier_uniform_with_seed(weight, 2, 3, 42)
                    .expect("valid seeded Xavier uniform fixture")
            },
        ),
        (
            |weight| {
                coeus_nn::init::xavier_normal(weight, 2, 3).expect("valid Xavier normal fixture")
            },
            |weight| {
                coeus_nn::init::xavier_normal_with_seed(weight, 2, 3, 42)
                    .expect("valid seeded Xavier normal fixture")
            },
        ),
        (
            |weight| {
                coeus_nn::init::kaiming_uniform(weight, 2).expect("valid Kaiming uniform fixture")
            },
            |weight| {
                coeus_nn::init::kaiming_uniform_with_seed(weight, 2, 42)
                    .expect("valid seeded Kaiming uniform fixture")
            },
        ),
        (
            |weight| {
                coeus_nn::init::kaiming_normal(weight, 2).expect("valid Kaiming normal fixture")
            },
            |weight| {
                coeus_nn::init::kaiming_normal_with_seed(weight, 2, 42)
                    .expect("valid seeded Kaiming normal fixture")
            },
        ),
    ];

    for (initialize_default, initialize_seeded) in initializers {
        let mut default = Var::new(Tensor::<f32, SequentialBackend>::zeros([2, 3]), true);
        let mut seeded = Var::new(Tensor::<f32, SequentialBackend>::zeros([2, 3]), true);
        initialize_default(&mut default);
        initialize_seeded(&mut seeded);
        assert_eq!(default.tensor.as_slice(), seeded.tensor.as_slice());
    }
}
