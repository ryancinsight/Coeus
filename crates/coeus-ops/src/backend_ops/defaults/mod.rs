// ── Default BackendOps method bodies ──
//
// Large host-fallback default implementations extracted from the trait
// definition so trait_def.rs stays under 500 lines.
//
// Each function takes `backend: &impl BackendOps<T>` and matches the trait
// method signature.  Trait default methods delegate to these free functions
// so backends that override the method are unaffected.

pub mod matmul;
pub mod reductions;
