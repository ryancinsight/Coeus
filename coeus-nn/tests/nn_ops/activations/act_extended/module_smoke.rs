use super::piecewise::{hardsigmoid_expected, hardswish_expected, softsign_expected};
use super::support::assert_close_slice;
use super::{Hardsigmoid, Hardswish, Module, MoiraiBackend, Softsign, Tensor, Var};

// ── Module-level forward smoke tests (no parameters) ────────────────────

#[test]
fn hardsigmoid_module_forward() {
    let m = Hardsigmoid;
    let data = vec![-4.0_f64, -2.0, 0.0, 2.0, 4.0];
    let expected: Vec<f64> = data.iter().map(|&x| hardsigmoid_expected(x)).collect();
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = m.forward(&input);
    assert_close_slice(
        "hardsigmoid_module_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
}

#[test]
fn hardswish_module_forward() {
    let m = Hardswish;
    let data = vec![-4.0_f64, -2.0, 0.0, 2.0, 4.0];
    let expected: Vec<f64> = data.iter().map(|&x| hardswish_expected(x)).collect();
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = m.forward(&input);
    assert_close_slice(
        "hardswish_module_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
}

#[test]
fn softsign_module_forward() {
    let m = Softsign;
    let data = vec![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let expected: Vec<f64> = data.iter().map(|&x| softsign_expected(x)).collect();
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = m.forward(&input);
    assert_close_slice(
        "softsign_module_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
}
