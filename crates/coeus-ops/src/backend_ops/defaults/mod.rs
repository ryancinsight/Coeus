// ── Default BackendOps method bodies ──
//
// Large default implementations extracted from the trait
// definition so trait_def.rs stays under 500 lines.
//
// Each function takes `backend: &impl BackendOps<T>` and matches the trait
// method signature. Provider-preserving compositions remain available to
// accelerator backends; CPU-only defaults carry an explicit `CpuBackend`
// bound, so no operation silently crosses a host boundary.

pub mod conv_transpose;
pub mod matmul;
pub mod reductions;
