//! Reference forward and backward values for normalization layers.
//!
//! A normalization layer maps each index group `S` to
//! `xhat_i = (x_i - mean_S) / sqrt(var_S + eps)` with the biased (population)
//! variance, then applies a per-channel affine `weight * xhat + bias`.
//! BatchNorm takes `S` as one channel across the batch, GroupNorm as one
//! sample's channel group, InstanceNorm as one sample's channel. With upstream
//! gradient `g` and `m = |S|` the input gradient is
//! `dx_i = (1 / sqrt(var_S + eps)) * (g_i - mean_S(g) - xhat_i * mean_S(g * xhat))`
//! (Ioffe and Szegedy, "Batch Normalization", 2015, section 3), the weight
//! gradient is `sum(g * xhat)` and the bias gradient is `sum(g)` per channel.
//! The oracle is two-pass (mean first, then centered squares), so it carries
//! none of the cancellation a single-pass `E[x^2] - mean^2` form would add.

use coeus_autograd::Var;
use coeus_tensor::Tensor;

/// Absolute tolerance for `f64` normalization outputs and gradients on the
/// fixtures in this harness.
///
/// A `p`-term sum rounds to at most `(p - 1) * eps / 2` relative error and the
/// mean, centered square, reciprocal square root and scale each add one
/// rounding of `eps / 2`, with `eps = f64::EPSILON`; groups here hold `p <= 32`
/// terms, so the layer and the oracle each stay under `(32 + 4) * eps / 2 =
/// 18 * eps`. Cancellation in the layer's variance amplifies that by at most
/// `1 + mean^2 / var`; the fixtures keep `|mean| <= sqrt(var)`, so the factor
/// is at most 2 and the output differs by at most `2 * 2 * 18 * eps = 72 * eps`
/// relative to values of magnitude at most 4. `256 * eps` covers the bound.
pub(super) const NORMALIZATION_TOLERANCE: f64 = 256.0 * f64::EPSILON;

/// Converts a fixture length to `f64`; fixture lengths are far below `2^24`.
fn count_f64(length: usize) -> f64 {
    f64::from(u32::try_from(length).expect("invariant: fixture length fits u32"))
}

/// Deterministic fixture values in `[-4.5, 5.5]` that do not repeat within a
/// 32-element group, so no group is constant and no layout mix-up is hidden.
pub(super) fn fixture_values(length: u32) -> Vec<f64> {
    (0..length)
        .map(|index| f64::from((index * 7 + 3) % 11) - 4.5)
        .collect()
}

/// Deterministic positive upstream gradient weights in `[0.5, 1.5]`.
pub(super) fn fixture_upstream(length: u32) -> Vec<f64> {
    (0..length)
        .map(|index| 0.5 + 0.25 * f64::from((index * 3 + 1) % 5))
        .collect()
}

/// Index groups of one channel across the batch for `[N, C, spatial...]` data.
pub(super) fn channel_groups(batch: usize, channels: usize, spatial: usize) -> Vec<Vec<usize>> {
    (0..channels)
        .map(|channel| {
            (0..batch)
                .flat_map(|sample| {
                    let start = (sample * channels + channel) * spatial;
                    start..start + spatial
                })
                .collect()
        })
        .collect()
}

/// One normalization scenario: data, upstream gradient, statistic groups and
/// the per-channel index sets that carry the affine parameters.
pub(super) struct Scenario {
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
    pub upstream: Vec<f64>,
    pub statistic_groups: Vec<Vec<usize>>,
    pub channels: Vec<Vec<usize>>,
    pub eps: f64,
}

fn assert_close(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= NORMALIZATION_TOLERANCE,
            "{context}[{index}]: got {got}, expected {want}, tolerance {NORMALIZATION_TOLERANCE}"
        );
    }
}

/// Runs `forward` on the scenario input under the loss `sum(output * upstream)`
/// and asserts the output, input gradient, weight gradient and bias gradient
/// against the reference. The layer must hold its default `weight = 1`,
/// `bias = 0` so its output is `xhat`.
pub(super) fn assert_matches_reference(
    scenario: &Scenario,
    forward: impl FnOnce(&Var<f64>) -> Var<f64>,
    weight: &Var<f64>,
    bias: &Var<f64>,
    context: &str,
) {
    let mut xhat = vec![0.0; scenario.values.len()];
    let mut input_grad = vec![0.0; scenario.values.len()];
    for group in &scenario.statistic_groups {
        let count = count_f64(group.len());
        let mean = group.iter().map(|&i| scenario.values[i]).sum::<f64>() / count;
        let variance = group
            .iter()
            .map(|&i| (scenario.values[i] - mean) * (scenario.values[i] - mean))
            .sum::<f64>()
            / count;
        let inv_std = 1.0 / (variance + scenario.eps).sqrt();
        for &i in group {
            xhat[i] = (scenario.values[i] - mean) * inv_std;
        }
        let mean_upstream = group.iter().map(|&i| scenario.upstream[i]).sum::<f64>() / count;
        let mean_projected = group
            .iter()
            .map(|&i| scenario.upstream[i] * xhat[i])
            .sum::<f64>()
            / count;
        for &i in group {
            input_grad[i] =
                inv_std * (scenario.upstream[i] - mean_upstream - xhat[i] * mean_projected);
        }
    }
    let weight_grad: Vec<f64> = scenario
        .channels
        .iter()
        .map(|channel| {
            channel
                .iter()
                .map(|&i| scenario.upstream[i] * xhat[i])
                .sum()
        })
        .collect();
    let bias_grad: Vec<f64> = scenario
        .channels
        .iter()
        .map(|channel| channel.iter().map(|&i| scenario.upstream[i]).sum())
        .collect();

    let input = Var::new(
        Tensor::from_slice(scenario.shape.clone(), &scenario.values)
            .expect("invariant: scenario shape matches its values"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = forward(&input);
    assert_eq!(output.tensor.shape(), scenario.shape.as_slice());
    assert_close(
        output.tensor.as_slice(),
        &xhat,
        &format!("{context} output"),
    );

    let upstream = Var::new(
        Tensor::from_slice(scenario.shape.clone(), &scenario.upstream)
            .expect("invariant: scenario shape matches its upstream gradient"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let weighted =
        coeus_autograd::mul(&output, &upstream).expect("invariant: equal-shaped operands multiply");
    let loss = coeus_autograd::sum(&weighted).expect("invariant: test operation succeeds");
    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");

    assert_close(
        input
            .grad()
            .expect("invariant: input tracks gradients")
            .as_slice(),
        &input_grad,
        &format!("{context} input gradient"),
    );
    assert_close(
        weight
            .grad()
            .expect("invariant: weight tracks gradients")
            .as_slice(),
        &weight_grad,
        &format!("{context} weight gradient"),
    );
    assert_close(
        bias.grad()
            .expect("invariant: bias tracks gradients")
            .as_slice(),
        &bias_grad,
        &format!("{context} bias gradient"),
    );
}
