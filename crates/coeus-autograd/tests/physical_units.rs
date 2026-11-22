#![cfg(feature = "physical-units")]

use aequitas::{
    systems::si::{dimensions as d, quantities as q, units as u},
    unit::{prefixes::Milli, Prefixed},
    Quantity,
};
use coeus_autograd::physical::{self, value_and_gradient, Coordinates, Error};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use tracing::info_span;
use tracing_subscriber::fmt::format::FmtSpan;

type Data<T = f64> = Tensor<T, MoiraiBackend>;
type ReciprocalPressure =
    <d::Dimensionless as aequitas::dimension::DivideDimension<d::Pressure>>::Output;

fn tracing() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(FmtSpan::FULL)
            .finish(),
    )
}

fn field<D>(xs: &[f64]) -> Quantity<Data, D> {
    Quantity::from_base(Data::from_slice([xs.len()], xs).expect("valid shape"))
}

fn close(actual: f64, expected: f64, rounds: i32) {
    let n_u = f64::from(rounds) * f64::EPSILON / 2.0;
    assert!(
        (actual - expected).abs() <= expected.abs() * n_u / (1.0 - n_u),
        "{actual} vs {expected}"
    );
}

#[test]
fn work_and_coordinate_scale() {
    let _tracing = tracing();
    let mm = Coordinates::<d::Length>::linear::<u::Millimeter>();
    let joule = Coordinates::<d::Energy>::linear::<u::Joule>();
    let millijoule = Coordinates::<d::Energy>::linear::<Prefixed<Milli, u::Joule>>();
    let x = info_span!("arrange").in_scope(|| {
        mm.normalize(Data::from_slice([1], &[2.0]).expect("scalar"))
            .expect("mm conversion")
    });
    let (energy, grad) = info_span!("act").in_scope(|| {
        value_and_gradient(x, |scope, length| {
            let force = scope.constant(field::<d::Force>(&[1.0]))?;
            physical::mul(&force, &length)
        })
        .expect("work gradient")
    });
    info_span!("assert").in_scope(|| {
        let energy: q::Energy<Data> = energy;
        assert_eq!(energy.as_base().as_slice(), &[0.002]);
        let grad_j = grad.in_coordinates(&joule, &mm).expect("J/mm");
        assert_eq!(grad_j.shape(), &[1]);
        assert_eq!(grad_j.as_slice(), &[0.001]);
        assert_eq!(
            grad.in_coordinates(&millijoule, &mm)
                .expect("mJ/mm")
                .as_slice(),
            &[1.0]
        );
        let force: q::Force<Data> = grad.into_base();
        assert_eq!(force.as_base().as_slice(), &[1.0]);
        assert_eq!(force.as_base().shape(), &[1]);
    });
}

#[test]
fn pressure_ramp_and_dimensionless_response() {
    let _tracing = tracing();
    let (time, pressure) = info_span!("arrange")
        .in_scope(|| (field::<d::Time>(&[2.0]), field::<d::Pressure>(&[108_000.0])));
    let ((ramp_value, ramp_gradient), (response, response_gradient)) =
        info_span!("act").in_scope(|| {
            let ramp = value_and_gradient(time, |scope, t| {
                let initial = scope.constant(field::<d::Pressure>(&[100_000.0]))?;
                let rate = scope.constant(field::<d::PressureRate>(&[4000.0]))?;
                physical::add(&initial, &physical::mul(&rate, &t)?)
            })
            .expect("pressure ramp");
            let response = value_and_gradient(pressure, |scope, p| {
                let reference = scope.constant(field::<d::Pressure>(&[100_000.0]))?;
                physical::log(&physical::div(&p, &reference)?)
            })
            .expect("log-pressure response");
            (ramp, response)
        });
    info_span!("assert").in_scope(|| {
        let value: q::Pressure<Data> = ramp_value;
        let rate: q::PressureRate<Data> = ramp_gradient.into_base();
        let response: q::Dimensionless<Data> = response;
        let derivative: Quantity<Data, ReciprocalPressure> = response_gradient.into_base();
        assert_eq!(value.as_base().as_slice(), &[108_000.0]);
        assert_eq!(rate.as_base().as_slice(), &[4000.0]);
        // A logarithm, evaluated near 1, amplifies ratio rounding by at most 1/ratio.
        let ratio = 1.08_f64;
        assert!((response.as_base().as_slice()[0] - ratio.ln()).abs() <= f64::EPSILON);
        close(derivative.as_base().as_slice()[0], 1.0 / 108_000.0, 6);
    });
}

#[test]
fn affine_point_interval_and_heat() {
    let _tracing = tracing();
    let ft = Coordinates::<d::ThermodynamicTemperature>::affine::<u::DegreeFahrenheit>();
    let fd = Coordinates::<d::TemperatureDifference>::linear::<u::DegreeFahrenheit>();
    let kelvin = Coordinates::<d::ThermodynamicTemperature>::linear::<u::Kelvin>();
    let joule = Coordinates::<d::Energy>::linear::<u::Joule>();
    let (warm, cold, interval) = info_span!("arrange").in_scope(|| {
        (
            ft.normalize(Data::from_slice([1], &[68.0]).expect("warm reading"))
                .expect("warm SI"),
            ft.normalize(Data::from_slice([1], &[32.0]).expect("cold reading"))
                .expect("cold SI"),
            fd.normalize(Data::from_slice([1], &[36.0]).expect("rise reading"))
                .expect("rise SI"),
        )
    });
    let (point, q) = info_span!("act").in_scope(|| {
        let point = value_and_gradient(warm.clone(), |_, x| Ok(x)).expect("identity");
        let heat = value_and_gradient(warm, |scope, hot| {
            let cold = scope.constant(cold)?;
            let rise = physical::sub(&hot, &cold)?;
            let capacity = scope.constant(field::<d::HeatCapacity>(&[500.0]))?;
            physical::mul(&capacity, &rise)
        })
        .expect("heat gradient");
        (point, heat)
    });
    info_span!("assert").in_scope(|| {
        let rise: q::TemperatureDifference<Data> = interval;
        assert_eq!(rise.as_base().as_slice(), &[20.0]);
        close(point.0.as_base().as_slice()[0], 293.15, 5);
        assert_eq!(
            point
                .1
                .in_coordinates(&kelvin, &ft)
                .expect("K/°F")
                .as_slice(),
            &[5.0 / 9.0]
        );
        let fahrenheit_per_kelvin = point.1.in_coordinates(&ft, &kelvin).expect("°F/K");
        // One rounding for the stored 5/9 coefficient and one for its reciprocal.
        close(fahrenheit_per_kelvin.as_slice()[0], 9.0 / 5.0, 2);
        let identity: q::Dimensionless<Data> = point.1.into_base();
        assert_eq!(identity.as_base().as_slice(), &[1.0]);
        assert_eq!(q.0.as_base().as_slice(), &[10_000.0]);
        close(
            q.1.in_coordinates(&joule, &ft).expect("J/°F").as_slice()[0],
            2500.0 / 9.0,
            4,
        );
        let capacity: q::HeatCapacity<Data> = q.1.into_base();
        assert_eq!(capacity.as_base().as_slice(), &[500.0]);
    });
}

#[test]
fn vector_gradient_and_fresh_objectives() {
    let _tracing = tracing();
    let x = info_span!("arrange").in_scope(|| field::<d::Length>(&[0.002, -0.003]));
    let ((energy, first), (norm, second)) = info_span!("act").in_scope(|| {
        let work = value_and_gradient(x.clone(), |scope, displacement| {
            let force = scope.constant(field::<d::Force>(&[1.0, 2.0]))?;
            physical::sum(&physical::mul(&force, &displacement)?)
        })
        .expect("vector work");
        let normalized = value_and_gradient(x, |scope, displacement| {
            let scale = scope.constant(field::<d::Length>(&[0.001]))?;
            let ratio = physical::div(&displacement, &scale)?;
            physical::sum(&physical::mul(&ratio, &ratio)?)
        })
        .expect("squared normalized displacement");
        (work, normalized)
    });
    info_span!("assert").in_scope(|| {
        let force: q::Force<Data> = first.into_base();
        let length_inverse: Quantity<
            Data,
            <d::Dimensionless as aequitas::dimension::DivideDimension<d::Length>>::Output,
        > = second.into_base();
        close(energy.as_base().as_slice()[0], -0.004, 4);
        assert_eq!(force.as_base().as_slice(), &[1.0, 2.0]);
        assert_eq!(norm.as_base().as_slice(), &[13.0]);
        assert_eq!(length_inverse.as_base().as_slice(), &[4000.0, -6000.0]);
    });
}

#[test]
fn rejects_non_scalar_and_suppressed_recording() {
    let _tracing = tracing();
    let x = info_span!("arrange").in_scope(|| field::<d::Length>(&[0.002, 0.004]));
    let (vector, no_grad, disconnected) = info_span!("act").in_scope(|| {
        let vector = value_and_gradient(x.clone(), |_, x| Ok(x));
        let no_grad = value_and_gradient(x.clone(), |_, x| {
            let _guard = coeus_autograd::no_grad_guard();
            physical::mul(&x, &x)
        });
        let disconnected =
            value_and_gradient(x, |scope, _| scope.constant(field::<d::Time>(&[7.0])))
                .expect("independent objective");
        (vector, no_grad, disconnected)
    });
    info_span!("assert").in_scope(|| {
        assert!(matches!(
            vector,
            Err(Error::NonScalarObjective { elements: 2 })
        ));
        assert!(matches!(no_grad, Err(Error::RecordingDisabled)));
        assert_eq!(disconnected.0.as_base().as_slice(), &[7.0]);
        assert_eq!(disconnected.1.into_base().as_base().as_slice(), &[0.0, 0.0]);
    });
}

#[test]
fn absolute_normalization_preserves_binary32() {
    let _tracing = tracing();
    let fahrenheit = Coordinates::<d::ThermodynamicTemperature>::affine::<u::DegreeFahrenheit>();
    let kelvin = Coordinates::<d::ThermodynamicTemperature>::linear::<u::Kelvin>();
    let readings = info_span!("arrange").in_scope(|| {
        Tensor::<f32, MoiraiBackend>::from_slice([1], &[68.0_f32]).expect("point reading")
    });
    let (value, grad) = info_span!("act").in_scope(|| {
        let point = fahrenheit.normalize(readings).expect("point normalization");
        value_and_gradient(point, |_, x| Ok(x)).expect("identity")
    });
    info_span!("assert").in_scope(|| {
        let expected =
            q::ThermodynamicTemperature::from_affine_unit::<u::DegreeFahrenheit>(68.0_f32);
        assert_eq!(value.as_base().as_slice(), &[*expected.as_base()]);
        assert_eq!(
            grad.in_coordinates(&kelvin, &fahrenheit)
                .expect("K/°F")
                .as_slice(),
            &[5.0_f32 / 9.0]
        );
    });
}

#[test]
fn rank_zero_and_bad_shapes_are_errors() {
    let _tracing = tracing();
    let (rank_zero, length) = info_span!("arrange").in_scope(|| {
        (
            Data::from_slice([], &[2.0]).expect("rank-zero fixture"),
            field::<d::Length>(&[0.002, 0.004]),
        )
    });
    let (normalization, derivative, broadcast) = info_span!("act").in_scope(|| {
        let normalization =
            Coordinates::<d::Length>::linear::<u::Millimeter>().normalize(rank_zero.clone());
        let derivative =
            value_and_gradient(Quantity::<_, d::Length>::from_base(rank_zero), |_, x| Ok(x));
        let broadcast = value_and_gradient(length, |scope, x| {
            let other = scope.constant(field::<d::Length>(&[1.0, 2.0, 3.0]))?;
            physical::add(&x, &other)
        });
        (normalization, derivative, broadcast)
    });
    info_span!("assert").in_scope(|| {
        assert!(matches!(normalization, Err(Error::RankZero)));
        assert!(matches!(derivative, Err(Error::RankZero)));
        assert!(matches!(
            broadcast,
            Err(Error::Backend(
                coeus_core::BackendError::IncompatibleBroadcast { .. }
            ))
        ));
    });
}

#[test]
fn reciprocal_pressure_keeps_volume_in_the_denominator() {
    let _tracing = tracing();
    let volume = info_span!("arrange").in_scope(|| {
        Coordinates::<d::Volume>::linear::<u::Liter>()
            .normalize(Data::from_slice([1], &[2.0]).expect("two litres"))
            .expect("volume conversion")
    });
    let (pressure, derivative) = info_span!("act").in_scope(|| {
        value_and_gradient(volume, |scope, v| {
            let pv = scope.constant(field::<d::Energy>(&[200.0]))?;
            physical::div(&pv, &v)
        })
        .expect("isothermal vessel")
    });
    info_span!("assert").in_scope(|| {
        let pressure: q::Pressure<Data> = pressure;
        type PressurePerVolume =
            <d::Pressure as aequitas::dimension::DivideDimension<d::Volume>>::Output;
        let slope: Quantity<Data, PressurePerVolume> = derivative.into_base();
        close(pressure.as_base().as_slice()[0], 100_000.0, 4);
        close(slope.as_base().as_slice()[0], -50_000_000.0, 8);
    });
}
