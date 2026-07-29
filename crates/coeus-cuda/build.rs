use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let uses_msvc = env::var("CARGO_CFG_TARGET_ENV").is_ok_and(|value| value == "msvc");
    if uses_msvc && env::var_os("CARGO_FEATURE_CUDA").is_some() {
        // CUDA 13.3's driver import library requests LIBCMT even though Rust's
        // MSVC target uses the dynamic CRT. Select the Rust runtime explicitly
        // so the final binary has one allocator and CRT contract.
        println!("cargo:rustc-link-arg=/NODEFAULTLIB:LIBCMT");
    }
}
