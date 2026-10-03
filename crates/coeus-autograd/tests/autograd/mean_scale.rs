//! The mean gradient scale `1/n` is the element-format value nearest the
//! exact reciprocal, for every shipped float and at the counts where
//! rounding `n` first goes wrong: just above `2^p` (269 for `Bf16`, 2079 for
//! `F16`), at the largest finite `F16` count and the first count past it
//! (65519, 65520), and at a count past `2^20`.

use coeus_autograd::Var;
use coeus_core::{FloatElement, Scalar, SequentialBackend};
use coeus_tensor::Tensor;
use eunomia::{Bf16, F16};

const COUNTS: [usize; 6] = [3, 269, 2079, 65_519, 65_520, (1 << 20) + 1];

/// A float format whose bit-pattern neighbours bracket each value.
trait Grid: Scalar + FloatElement + leto_ops::Scalar {
    fn neighbours(self) -> (Self, Self);
}

impl Grid for f64 {
    fn neighbours(self) -> (Self, Self) {
        (
            Self::from_bits(self.to_bits() - 1),
            Self::from_bits(self.to_bits() + 1),
        )
    }
}

impl Grid for f32 {
    fn neighbours(self) -> (Self, Self) {
        (
            Self::from_bits(self.to_bits() - 1),
            Self::from_bits(self.to_bits() + 1),
        )
    }
}

impl Grid for F16 {
    fn neighbours(self) -> (Self, Self) {
        (
            Self::from_bits(self.to_bits() - 1),
            Self::from_bits(self.to_bits() + 1),
        )
    }
}

impl Grid for Bf16 {
    fn neighbours(self) -> (Self, Self) {
        (
            Self::from_bits(self.to_bits() - 1),
            Self::from_bits(self.to_bits() + 1),
        )
    }
}

/// `value = m * 2^e` for a finite positive `f64`, `m` its integer significand.
fn significand(value: f64) -> (u128, i32) {
    let bits = value.to_bits();
    let field = i32::try_from(bits >> 52).expect("invariant: a positive f64 has no sign bit");
    let fraction = u128::from(bits & ((1_u64 << 52) - 1));
    if field == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1 << 52), field - 1075)
    }
}

/// `|a * n - 1| * 2^-scale` in exact integer arithmetic.
fn scaled_distance(a: f64, n: usize, scale: i32) -> u128 {
    let (m, e) = significand(a);
    let one = 1_u128 << u32::try_from(-scale).expect("invariant: 1/n is below 1");
    let shift = u32::try_from(e - scale).expect("invariant: scale is the smallest exponent");
    let count = u128::try_from(n).expect("invariant: usize fits u128");
    ((m * count) << shift).abs_diff(one)
}

/// `scale` is strictly nearer to `1/n` than both of its neighbours.
fn assert_nearest_reciprocal<T: Grid>(scale: T, n: usize, context: &str) {
    let (below, above) = scale.neighbours();
    let values = [scale, below, above].map(<T as Scalar>::to_f64);
    assert!(
        values[0] > 0.0,
        "{context}: count {n} scale underflowed to zero"
    );
    let exponent = values
        .iter()
        .map(|&v| significand(v).1)
        .min()
        .expect("invariant: three values");
    let [nearest, low, high] = values.map(|v| scaled_distance(v, n, exponent));
    assert!(
        nearest < low && nearest < high,
        "{context}: count {n} scale {} is not the nearest reciprocal",
        values[0]
    );
}

fn mean_gradient_scale<T: Grid>() {
    for n in COUNTS {
        let ones = Tensor::<T, SequentialBackend>::full([n], T::ONE);
        let x = Var::new(ones.clone(), true);
        coeus_autograd::mean(&x)
            .backward()
            .expect("invariant: mean backward completes");
        let gradient = x.grad().expect("invariant: x requires grad");
        let scale = gradient.as_slice()[0];
        assert!(
            gradient.as_slice().iter().all(|&g| g == scale),
            "uniform mean gradient"
        );
        assert_nearest_reciprocal(scale, n, "mean");

        let column = Var::new(ones.reshape([n, 1]), true);
        coeus_autograd::mean_axis(&column, 0)
            .backward()
            .expect("invariant: mean_axis backward completes");
        let axis_gradient = column.grad().expect("invariant: column requires grad");
        assert_eq!(
            axis_gradient.as_slice()[0],
            scale,
            "mean_axis scale at count {n}"
        );
    }
}

#[test]
fn mean_gradient_scale_is_the_nearest_reciprocal() {
    mean_gradient_scale::<f64>();
    mean_gradient_scale::<f32>();
    mean_gradient_scale::<F16>();
    mean_gradient_scale::<Bf16>();
}
