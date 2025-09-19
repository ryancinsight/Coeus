/// Modular tests for automatic differentiation
/// This file has been modularized to improve maintainability and reduce file size from 3191 lines to ~20 lines
///
// Import test modules
#[path = "autograd/fundamental_tests.rs"]
mod fundamental_tests;
#[path = "autograd/activation_tests.rs"]
mod activation_tests;
#[path = "autograd/gradient_flow_tests.rs"]
mod gradient_flow_tests;
#[path = "autograd/math_utils.rs"]
mod math_utils;
#[path = "autograd/transpose_tests.rs"]
mod transpose_tests;

// Test modules are organized separately for maintainability