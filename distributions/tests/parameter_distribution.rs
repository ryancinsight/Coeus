use coeus_distributions::{
    FiniteF64, NonNegativeFiniteF64, ParameterDistribution, PositiveFiniteF64,
};
use proptest::prelude::*;

fn non_finite_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
        (1u64..(1u64 << 52)).prop_map(|mantissa| f64::from_bits(0x7ff0_0000_0000_0000 | mantissa)),
        (1u64..(1u64 << 52)).prop_map(|mantissa| f64::from_bits(0xfff0_0000_0000_0000 | mantissa)),
    ]
}

proptest! {
    #[test]
    fn continuous_rejects_inverted_ranges(min in -1.0e6f64..1.0e6, width in 1.0f64..1.0e6) {
        let max = min - width.abs();
        prop_assert!(ParameterDistribution::continuous(min, max).is_err());
    }

    #[test]
    fn categorical_rejects_len_mismatch(a in 0usize..10, b in 0usize..10) {
        prop_assume!(a != b);
        let categories: Vec<String> = (0..a).map(|i| format!("c{i}")).collect();
        let weights: Vec<f64> = (0..b).map(|_| 1.0).collect();
        prop_assert!(ParameterDistribution::categorical(categories, weights).is_err());
    }

    #[test]
    fn categorical_rejects_negative_weights(n in 1usize..10, neg_idx in 0usize..10) {
        let categories: Vec<String> = (0..n).map(|i| format!("c{i}")).collect();
        let mut weights: Vec<f64> = vec![1.0; n];
        let i = neg_idx % n;
        weights[i] = -1.0;
        prop_assert!(ParameterDistribution::categorical(categories, weights).is_err());
    }

    #[test]
    fn finite_f64_rejects_non_finite(x in non_finite_f64()) {
        prop_assert!(FiniteF64::new("x", x).is_err());
    }

    #[test]
    fn positive_finite_f64_rejects_non_finite(x in non_finite_f64()) {
        prop_assert!(PositiveFiniteF64::new("x", x).is_err());
    }

    #[test]
    fn positive_finite_f64_rejects_non_positive(x in -1.0e12f64..=0.0) {
        prop_assert!(PositiveFiniteF64::new("x", x).is_err());
    }

    #[test]
    fn non_negative_finite_f64_rejects_non_finite(x in non_finite_f64()) {
        prop_assert!(NonNegativeFiniteF64::new("x", x).is_err());
    }

    #[test]
    fn non_negative_finite_f64_rejects_negative(x in -1.0e12f64..0.0) {
        prop_assume!(x < 0.0);
        prop_assert!(NonNegativeFiniteF64::new("x", x).is_err());
    }

    #[test]
    fn continuous_json_roundtrip(min in -1.0e6f64..1.0e6, width in 0.0f64..1.0e6) {
        let max = min + width.abs();
        let dist = ParameterDistribution::continuous(min, max).unwrap();
        let json = serde_json::to_string(&dist).unwrap();
        let decoded: ParameterDistribution = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(dist, decoded);
    }

    #[test]
    fn categorical_json_roundtrip(n in 1usize..10, weights in prop::collection::vec(1.0f64..1.0e6, 1usize..10)) {
        let n = n.min(weights.len());
        let categories: Vec<String> = (0..n).map(|i| format!("c{i}")).collect();
        let weights = weights.into_iter().take(n).collect::<Vec<_>>();
        let dist = ParameterDistribution::categorical(categories, weights).unwrap();
        let json = serde_json::to_string(&dist).unwrap();
        let decoded: ParameterDistribution = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(dist, decoded);
    }
}

#[test]
fn serde_rejects_invalid_normal_std() {
    let json = r#"{"Normal":{"mean":0.0,"std":0.0}}"#;
    let res: Result<ParameterDistribution, _> = serde_json::from_str(json);
    assert!(res.is_err());
}

#[test]
fn categorical_rejects_zero_total_weight() {
    let categories = vec!["a".to_string(), "b".to_string()];
    let weights = vec![0.0, 0.0];
    assert!(ParameterDistribution::categorical(categories, weights).is_err());
}

#[test]
fn sampling_discrete_always_from_support() {
    let dist = match ParameterDistribution::discrete(vec![
        serde_json::Value::String("a".to_string()),
        serde_json::Value::String("b".to_string()),
    ]) {
        Ok(d) => d,
        Err(e) => panic!("{e}"),
    };

    let mut rng = rand_pcg::Pcg64Mcg::new(123);
    for _ in 0..1000 {
        let v = match dist.sample_json(&mut rng) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        assert!(
            v == serde_json::Value::String("a".to_string())
                || v == serde_json::Value::String("b".to_string())
        );
    }
}
