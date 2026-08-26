//! `Linear::new` must produce units that can train apart from each other.
//!
//! Every weight used to be 1.0. A layer's units then computed the same value
//! from the same input, took the same gradient, and applied the same update --
//! identical at step zero and identical forever, so a `Linear(3, 4)` had the
//! expressive capacity of a `Linear(3, 1)` no matter how long it trained. The
//! measurement that established it: the weight gradient of a `Linear(3, 4)`
//! came back as `[2.0, -0.75, 1.25]` repeated four times, one row per unit.
//!
//! See ADR 0067. These tests fail if that initialisation ever returns.

use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::linear::Linear;
use coeus_nn::module::Module;
use coeus_tensor::Tensor;

const IN: usize = 3;
const OUT: usize = 4;

fn batch(backend: &MoiraiBackend) -> Var<f32, MoiraiBackend> {
    Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice_on(
            vec![2, IN],
            &[0.5, -1.0, 2.0, 1.5, 0.25, -0.75],
            backend,
        ),
        false,
    )
}

/// The units must not all hold the same weights.
#[test]
fn units_start_distinct() {
    let layer = Linear::<f32, MoiraiBackend>::new(IN, OUT, true).expect("in_features is non-zero");
    let weights = layer.weight.tensor.as_slice().to_vec();

    let rows: Vec<&[f32]> = weights.chunks(IN).collect();
    assert_eq!(rows.len(), OUT, "one row per unit");
    assert!(
        rows.iter().any(|row| *row != rows[0]),
        "every unit holds identical weights, so the layer has the capacity of \
         one unit however wide it is: {weights:?}"
    );
}

/// The units must not all compute the same thing.
///
/// This is the property that mattered. `y[n,j] = sum_i x[n,i] * W[j,i] + b[j]`,
/// so identical rows of `W` make every output column identical, and the layer
/// carries one unit's worth of information however wide it is.
#[test]
fn units_compute_distinct_outputs() {
    let backend = MoiraiBackend;
    let layer = Linear::<f32, MoiraiBackend>::new(IN, OUT, true).expect("in_features is non-zero");

    let output = layer.forward(&batch(&backend)).expect("valid batch");
    let values = output.tensor.as_slice().to_vec();

    // `[2, OUT]`: one row per batch element, one column per unit.
    for (n, row) in values.chunks(OUT).enumerate() {
        assert!(
            row.iter().any(|v| *v != row[0]),
            "every unit produced the same value for batch element {n}, so the              layer has the capacity of one unit: {row:?}"
        );
    }
}

/// The units must not all receive the same gradient.
///
/// The loss has to depend on the weights for this to say anything. Under
/// `L = sum(y)` the gradient is `dL/dW[j,i] = sum_n x[n,i]` for every unit `j`
/// -- identical across units whatever `W` holds, which is correct arithmetic
/// and not a symmetry at all. `L = sum(y * y)` gives `2 * y[n,j] * x[n,i]`,
/// which does depend on the unit's own weights.
#[test]
fn units_take_distinct_gradients_under_a_weight_dependent_loss() {
    let backend = MoiraiBackend;
    let layer = Linear::<f32, MoiraiBackend>::new(IN, OUT, true).expect("in_features is non-zero");

    let output = layer.forward(&batch(&backend)).expect("valid batch");
    let squared = coeus_autograd::mul(&output, &output);
    coeus_autograd::sum(&squared).backward().expect("backward");

    let grad = layer
        .weight
        .grad()
        .expect("a tracked weight has a gradient after backward")
        .as_slice()
        .to_vec();

    let rows: Vec<&[f32]> = grad.chunks(IN).collect();
    assert!(
        rows.iter().any(|row| *row != rows[0]),
        "every unit took the same gradient under a weight-dependent loss, so          they cannot diverge under training: {grad:?}"
    );
}

/// One seed must give one layer, and different seeds different layers.
#[test]
fn the_draw_is_reproducible_from_its_seed() {
    let of = |seed: u64| {
        Linear::<f32, MoiraiBackend>::with_seed(IN, OUT, true, seed)
            .expect("in_features is non-zero")
            .weight
            .tensor
            .as_slice()
            .to_vec()
    };

    assert_eq!(of(7), of(7), "one seed gave two different layers");
    assert_ne!(of(7), of(8), "two seeds gave the same layer");
}

/// `new` must be reproducible too, so a network is replayable by default.
#[test]
fn new_is_reproducible_without_a_seed() {
    let of = || {
        Linear::<f32, MoiraiBackend>::new(IN, OUT, true)
            .expect("in_features is non-zero")
            .weight
            .tensor
            .as_slice()
            .to_vec()
    };
    assert_eq!(of(), of(), "`new` is not reproducible run to run");
}

/// A zero `in_features` has no Kaiming bound, and must be reported.
#[test]
fn a_zero_input_width_is_rejected() {
    assert!(
        Linear::<f32, MoiraiBackend>::new(0, OUT, true).is_err(),
        "a zero fan_in has no initialisation bound and must not build a layer"
    );
}
