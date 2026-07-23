//! Hierarchical Python tensor-operation binding contracts.

#[path = "operations/algebra.rs"]
mod algebra;
#[path = "operations/autograd.rs"]
mod autograd;
#[path = "operations/constructors.rs"]
mod constructors;
#[path = "operations/dtype.rs"]
mod dtype;
#[path = "operations/elementwise.rs"]
mod elementwise;
#[path = "operations/indexing.rs"]
mod indexing;
#[path = "operations/layout.rs"]
mod layout;
#[path = "operations/nn/mod.rs"]
mod nn;
#[path = "operations/optim.rs"]
mod optim;
#[path = "operations/reductions.rs"]
mod reductions;
#[path = "operations/statistics.rs"]
mod statistics;
#[path = "operations/support.rs"]
mod support;
