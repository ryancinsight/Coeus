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
use coeus_core::FloatElement;
use coeus_tensor::Tensor;

/// Absolute tolerances for `f64` normalization assertions, derived per
/// quantity at unit roundoff `u = eps / 2` (`eps = f64::EPSILON`). Sums use
/// sequential bounds, covering every backend reduction order. Exact fixture
/// facts: `p <= 12`, `|x| <= 5.5`, `sum|x| <= 34`, `|mean| <= 1.75`,
/// `8.48 <= var <= 10.41`, `|xhat| <= 1.69`, `|g| <= 1.5`, `sum|g| <= 12`,
/// `sum|g*xhat| <= 11.3`. Mean `<= 33u`; centered values `<= 38.5u`, whose
/// shared shift cancels to first order in the two-pass variance (`<= 156u`);
/// outputs agree within `37.4u`, input gradients within `63u`, bias sums
/// within `264u`, weight sums within `980u` across both sides.
pub(super) const NORMALIZATION_OUTPUT_TOLERANCE: f64 = 64.0 * f64::EPSILON;
pub(super) const NORMALIZATION_GRADIENT_TOLERANCE: f64 = 256.0 * f64::EPSILON;
pub(super) const NORMALIZATION_WEIGHT_TOLERANCE: f64 = 1024.0 * f64::EPSILON;
/// Deterministic fixture values in `[-4.5, 5.5]`, repeating every 11 indices.
/// Every group is non-constant and the channel means are pairwise distinct,
/// so no layout mix-up is hidden.
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

fn assert_close(actual: &[f64], expected: &[f64], context: &str, tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= tolerance,
            "{context}[{index}]: got {got}, expected {want}, tolerance {tolerance}"
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
        let count = f64::from_count(group.len());
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
        NORMALIZATION_OUTPUT_TOLERANCE,
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
        NORMALIZATION_GRADIENT_TOLERANCE,
    );
    assert_close(
        weight
            .grad()
            .expect("invariant: weight tracks gradients")
            .as_slice(),
        &weight_grad,
        &format!("{context} weight gradient"),
        NORMALIZATION_WEIGHT_TOLERANCE,
    );
    assert_close(
        bias.grad()
            .expect("invariant: bias tracks gradients")
            .as_slice(),
        &bias_grad,
        &format!("{context} bias gradient"),
        NORMALIZATION_GRADIENT_TOLERANCE,
    );
}
