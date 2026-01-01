use super::*;

#[test]
fn test_clip_config_default_invariants() {
    let config = ClipConfig::default();
    assert!(config.temperature > 0.0);
    assert!(config.embed_dim > 0);
    assert!(config.projection_dim > 0);
    assert!(config.max_grad_norm.is_some());
}
