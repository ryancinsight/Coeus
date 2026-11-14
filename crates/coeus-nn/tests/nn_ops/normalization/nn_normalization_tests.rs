use super::value_oracle::{
    assert_matches_reference, channel_groups, contiguous_groups, fixture_upstream, fixture_values,
    Scenario,
};
use coeus_core::MoiraiBackend;
use coeus_nn::{GroupNorm, InstanceNorm1d, InstanceNorm2d, Module};

const EPS: f64 = 1e-5;

#[test]
fn test_group_norm() {
    // [batch=2, channels=6, spatial=4] in 3 groups of 2 channels: each group
    // spans 8 consecutive elements of one sample.
    let (batch, channels, spatial) = (2, 6, 4);
    let gn = GroupNorm::<f64, MoiraiBackend, 3>::new(channels, EPS)
        .expect("invariant: test operation succeeds");

    let params = gn.parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].tensor.shape(), &[6]); // weight
    assert_eq!(params[1].tensor.shape(), &[6]); // bias

    let total = u32::try_from(batch * channels * spatial).expect("invariant: fixture fits u32");
    assert_matches_reference(
        &Scenario {
            shape: vec![batch, channels, spatial],
            values: fixture_values(total),
            upstream: fixture_upstream(total),
            statistic_groups: contiguous_groups(batch * channels * spatial, 2 * spatial),
            channels: channel_groups(batch, channels, spatial),
            eps: EPS,
        },
        |input| gn.forward(input).expect("valid GroupNorm input"),
        &gn.weight,
        &gn.bias,
        "GroupNorm",
    );
}

#[test]
fn test_instance_norm_1d() {
    // [batch=2, channels=4, spatial=5]: one statistic per sample and channel.
    let (batch, channels, spatial) = (2, 4, 5);
    let in1d = InstanceNorm1d::<f64, MoiraiBackend>::new(channels, EPS)
        .expect("invariant: test operation succeeds");

    let params = in1d.parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].tensor.shape(), &[4]);
    assert_eq!(params[1].tensor.shape(), &[4]);

    let total = u32::try_from(batch * channels * spatial).expect("invariant: fixture fits u32");
    assert_matches_reference(
        &Scenario {
            shape: vec![batch, channels, spatial],
            values: fixture_values(total),
            upstream: fixture_upstream(total),
            statistic_groups: contiguous_groups(batch * channels * spatial, spatial),
            channels: channel_groups(batch, channels, spatial),
            eps: EPS,
        },
        |input| in1d.forward(input).expect("valid InstanceNorm1d input"),
        &in1d.weight,
        &in1d.bias,
        "InstanceNorm1d",
    );
}

#[test]
fn test_instance_norm_2d() {
    // [batch=2, channels=3, height=4, width=4]: one statistic per sample and channel.
    let (batch, channels, height, width) = (2, 3, 4, 4);
    let in2d = InstanceNorm2d::<f64, MoiraiBackend>::new(channels, EPS)
        .expect("invariant: test operation succeeds");

    assert_eq!(in2d.parameters().len(), 2);

    let spatial = height * width;
    let total = u32::try_from(batch * channels * spatial).expect("invariant: fixture fits u32");
    assert_matches_reference(
        &Scenario {
            shape: vec![batch, channels, height, width],
            values: fixture_values(total),
            upstream: fixture_upstream(total),
            statistic_groups: contiguous_groups(batch * channels * spatial, spatial),
            channels: channel_groups(batch, channels, spatial),
            eps: EPS,
        },
        |input| in2d.forward(input).expect("valid InstanceNorm2d input"),
        &in2d.weight,
        &in2d.bias,
        "InstanceNorm2d",
    );
}
