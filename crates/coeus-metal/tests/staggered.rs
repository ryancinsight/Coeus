// The staggered oracle needs only `assert_values`/`upload` from the shared
// optimizer helpers; allow the rest of the included module to stay unused.
#[allow(dead_code)]
#[path = "../../coeus-ops/tests/ops/ownership/optimizer.rs"]
mod optimizer;
#[path = "../../coeus-ops/tests/ops/ownership/staggered.rs"]
mod staggered_oracle;

use coeus_accelerator::HephaestusBackend;
use coeus_metal::MetalProvider;
use staggered_oracle::staggered_preserves_clones;

type Backend = HephaestusBackend<MetalProvider>;

#[test]
fn staggered_preserves_metal_output_clones() {
    if !require_device() {
        return;
    }
    staggered_preserves_clones(&Backend::new(), 1.0_f32);
}

fn require_device() -> bool {
    let available = hephaestus_metal::MetalDevice::try_default().is_ok();
    if !available {
        assert_ne!(
            std::env::var("HEPHAESTUS_METAL_REQUIRE_DEVICE").as_deref(),
            Ok("1"),
            "Metal CI requires an acquired device"
        );
    }
    available
}
