//! Iterator implementations for tensors
//!
//! This module provides various iterator types for traversing tensor elements,
//! including parallel iterators and functional programming constructs.
//!
//! ## Available Iterators
//!
//! - **Standard Iterator**: `tensor.iter()` - immutable element access
//! - **Mutable Iterator**: `tensor.iter_mut()` - mutable element access
//! - **Parallel Iterator**: `tensor.par_iter()` - parallel element access (with `parallel` feature)
//! - **Parallel Mutable Iterator**: `tensor.par_iter_mut()` - parallel mutable access
//! - **Functional Operations**: `map`, `fold`, `filter`, `any`, `all`, `find`
//!
//! ## Iterator Patterns
//!
/// ### Element-wise Mapping
/// ```rust
/// use coeus_tensor::Tensor;
///
/// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
/// let squared = tensor.map(|x| x * x);
/// // Result: [1.0, 4.0, 9.0]
/// ```
///
/// ### Folding/Reducing
/// ```rust
/// use coeus_tensor::Tensor;
///
/// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
/// let sum = tensor.fold(0.0, |acc, x| acc + x);
/// // Result: 10.0
/// ```
///
/// ### Filtering
/// ```rust
/// use coeus_tensor::Tensor;
///
/// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
/// let even_indices = tensor.filter_indices(|x| *x % 2.0 == 0.0);
/// // Result: [1, 3] (indices of even values)
/// ```
///
/// ## Parallel Iteration
///
/// With the `parallel` feature enabled, tensors support parallel iteration:
///
/// ```rust
/// use coeus_tensor::Tensor;
/// use rayon::iter::ParallelIterator;
///
/// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
/// let sum: f64 = tensor.par_iter().sum();
/// ```
///
/// ## References
///
/// - [Rust Iterator Trait](https://doc.rust-lang.org/std/iter/trait.Iterator.html)
/// - [Rayon Parallel Iterators](https://docs.rs/rayon/latest/rayon/)
pub mod impls;
