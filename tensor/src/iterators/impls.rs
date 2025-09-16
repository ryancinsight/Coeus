//! Iterator implementations for tensor operations
//!
//! This module contains the implementation of various iterator patterns
//! for tensor element access and functional programming.

use crate::Tensor;

/// Iterator implementation for tensors
///
/// Provides standard iterator functionality over tensor elements.
impl<T: crate::Dtype> Tensor<T> {
    /// Create an iterator over the tensor elements
    ///
    /// # Returns
    /// An iterator that yields immutable references to tensor elements
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    /// let sum: f64 = tensor.iter().sum();
    /// assert_eq!(sum, 6.0);
    /// ```
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data().iter()
    }

    /// Create a mutable iterator over the tensor elements
    ///
    /// # Returns
    /// An iterator that yields mutable references to tensor elements
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    /// for elem in tensor.iter_mut() {
    ///     *elem *= 2.0;
    /// }
    /// assert_eq!(tensor.data(), &[2.0, 4.0, 6.0]);
    /// ```
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data_mut().iter_mut()
    }

    /// Create a parallel iterator over the tensor elements (requires rayon)
    ///
    /// # Returns
    /// A parallel iterator that yields immutable references to tensor elements
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    /// use rayon::iter::ParallelIterator;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    /// let sum: f64 = tensor.par_iter().sum();
    /// assert_eq!(sum, 10.0);
    /// ```
    pub fn par_iter(&self) -> rayon::slice::Iter<'_, T> {
        use rayon::prelude::*;
        self.data().par_iter()
    }

    /// Create a parallel mutable iterator over the tensor elements
    ///
    /// # Returns
    /// A parallel iterator that yields mutable references to tensor elements
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    /// use rayon::iter::ParallelIterator;
    ///
    /// let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    /// tensor.par_iter_mut().for_each(|elem| *elem *= 2.0);
    /// assert_eq!(tensor.data(), &[2.0, 4.0, 6.0, 8.0]);
    /// ```
    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<'_, T> {
        use rayon::prelude::*;
        self.data_mut().par_iter_mut()
    }

    /// Apply a function to each element and collect into a new tensor
    ///
    /// # Arguments
    /// * `f` - Function to apply to each element
    ///
    /// # Returns
    /// A new tensor with the transformed elements
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    /// let squared = tensor.map(|x| x * x);
    /// assert_eq!(squared.data(), &[1.0, 4.0, 9.0]);
    /// ```
    pub fn map<F, U>(&self, f: F) -> Tensor<U>
    where
        F: Fn(&T) -> U,
        U: crate::Dtype,
    {
        let result_data: Vec<U> = self.data().iter().map(f).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Apply a function to each element in place
    ///
    /// # Arguments
    /// * `f` - Function to apply to each element (takes a mutable reference)
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    /// tensor.map_inplace(|x| *x * 2.0);
    /// assert_eq!(tensor.data(), &[2.0, 4.0, 6.0]);
    /// ```
    pub fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&T) -> T,
    {
        for elem in self.data_mut().iter_mut() {
            *elem = f(elem);
        }
    }

    /// Filter elements based on a predicate and return indices
    ///
    /// # Arguments
    /// * `predicate` - Function that returns true for elements to include
    ///
    /// # Returns
    /// Vector of indices where the predicate returns true
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    /// let even_indices = tensor.filter_indices(|x| *x % 2.0 == 0.0);
    /// assert_eq!(even_indices, vec![1, 3]); // indices of 2.0 and 4.0
    /// ```
    pub fn filter_indices<F>(&self, predicate: F) -> Vec<usize>
    where
        F: Fn(&T) -> bool,
    {
        self.data()
            .iter()
            .enumerate()
            .filter(|(_, elem)| predicate(elem))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Fold elements using a function
    ///
    /// # Arguments
    /// * `init` - Initial value for the accumulator
    /// * `f` - Function to combine accumulator with each element
    ///
    /// # Returns
    /// Final accumulator value
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    /// let sum = tensor.fold(0.0, |acc, x| acc + x);
    /// assert_eq!(sum, 10.0);
    /// ```
    pub fn fold<U, F>(&self, init: U, f: F) -> U
    where
        F: Fn(U, &T) -> U,
    {
        self.data().iter().fold(init, f)
    }

    /// Check if any element satisfies a predicate
    ///
    /// # Arguments
    /// * `predicate` - Function to test each element
    ///
    /// # Returns
    /// True if any element satisfies the predicate, false otherwise
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    /// assert!(tensor.any(|x| *x > 2.5));
    /// assert!(!tensor.any(|x| *x > 5.0));
    /// ```
    pub fn any<F>(&self, predicate: F) -> bool
    where
        F: Fn(&T) -> bool,
    {
        self.data().iter().any(predicate)
    }

    /// Check if all elements satisfy a predicate
    ///
    /// # Arguments
    /// * `predicate` - Function to test each element
    ///
    /// # Returns
    /// True if all elements satisfy the predicate, false otherwise
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![2.0, 4.0, 6.0], vec![3]);
    /// assert!(tensor.all(|x| *x % 2.0 == 0.0));
    /// assert!(!tensor.all(|x| *x > 3.0));
    /// ```
    pub fn all<F>(&self, predicate: F) -> bool
    where
        F: Fn(&T) -> bool,
    {
        self.data().iter().all(predicate)
    }

    /// Find the first element that satisfies a predicate
    ///
    /// # Arguments
    /// * `predicate` - Function to test each element
    ///
    /// # Returns
    /// Option containing the index and reference to the first matching element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    /// let result = tensor.find(|x| *x > 2.5);
    /// assert_eq!(result, Some((2, &3.0)));
    /// ```
    pub fn find<F>(&self, predicate: F) -> Option<(usize, &T)>
    where
        F: Fn(&T) -> bool,
    {
        self.data()
            .iter()
            .enumerate()
            .find(|(_, elem)| predicate(elem))
    }

    /// Create a tensor from an iterator
    ///
    /// # Arguments
    /// * `iter` - Iterator over elements
    /// * `shape` - Shape of the resulting tensor
    ///
    /// # Returns
    /// A new tensor containing the elements from the iterator
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::from_iter(data, vec![2, 2]);
    /// assert_eq!(tensor.shape(), &[2, 2]);
    /// ```
    pub fn from_iter<I>(iter: I, shape: Vec<usize>) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let data: Vec<T> = iter.into_iter().collect();
        Tensor::from_vec(data, shape)
    }

    /// Chain multiple tensors together
    ///
    /// # Arguments
    /// * `other` - Another tensor to chain with
    ///
    /// # Returns
    /// An iterator that yields elements from both tensors
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    /// let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
    /// let chained: Vec<f64> = a.chain(&b).cloned().collect();
    /// assert_eq!(chained, vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn chain<'a>(&'a self, other: &'a Tensor<T>) -> impl Iterator<Item = &'a T> + 'a {
        self.data.iter().chain(other.data.iter())
    }

    /// Zip two tensors together
    ///
    /// # Arguments
    /// * `other` - Another tensor to zip with
    ///
    /// # Returns
    /// An iterator that yields pairs of elements from both tensors
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    /// let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
    /// let zipped: Vec<(f64, f64)> = a.zip(&b).map(|(x, y)| (*x, *y)).collect();
    /// assert_eq!(zipped, vec![(1.0, 3.0), (2.0, 4.0)]);
    /// ```
    pub fn zip<'a>(&'a self, other: &'a Tensor<T>) -> impl Iterator<Item = (&'a T, &'a T)> + 'a {
        self.data.iter().zip(other.data.iter())
    }

    /// Create windows of elements
    ///
    /// # Arguments
    /// * `size` - Size of each window
    ///
    /// # Returns
    /// An iterator over windows of the tensor data
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    /// let windows: Vec<&[f64]> = tensor.windows(2).collect();
    /// assert_eq!(windows[0], &[1.0, 2.0]);
    /// assert_eq!(windows[1], &[2.0, 3.0]);
    /// ```
    pub fn windows(&self, size: usize) -> impl Iterator<Item = &[T]> {
        self.data.windows(size)
    }

    /// Take the first n elements
    ///
    /// # Arguments
    /// * `n` - Number of elements to take
    ///
    /// # Returns
    /// An iterator over the first n elements
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    /// let first_two: Vec<f64> = tensor.take(2).cloned().collect();
    /// assert_eq!(first_two, vec![1.0, 2.0]);
    /// ```
    pub fn take(&self, n: usize) -> impl Iterator<Item = &T> {
        self.data.iter().take(n)
    }

    /// Skip the first n elements
    ///
    /// # Arguments
    /// * `n` - Number of elements to skip
    ///
    /// # Returns
    /// An iterator over the remaining elements
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    /// let last_two: Vec<f64> = tensor.skip(2).cloned().collect();
    /// assert_eq!(last_two, vec![3.0, 4.0]);
    /// ```
    pub fn skip(&self, n: usize) -> impl Iterator<Item = &T> {
        self.data.iter().skip(n)
    }

    /// Get elements at specific indices
    ///
    /// # Arguments
    /// * `indices` - Vector of indices to select
    ///
    /// # Returns
    /// Vector of references to elements at the specified indices
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    /// let selected = tensor.select(&[0, 2]);
    /// let values: Vec<f64> = selected.iter().map(|&x| *x).collect();
    /// assert_eq!(values, vec![1.0, 3.0]);
    /// ```
    pub fn select(&self, indices: &[usize]) -> Vec<&T> {
        indices.iter().map(|&idx| &self.data[idx]).collect()
    }

    /// Create a new tensor with elements that satisfy a predicate
    ///
    /// # Arguments
    /// * `predicate` - Function to test each element
    ///
    /// # Returns
    /// A new tensor containing only elements that satisfy the predicate
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    /// let even = tensor.filter(|x| *x % 2.0 == 0.0);
    /// assert_eq!(even.data(), &[2.0, 4.0]);
    /// assert_eq!(even.shape(), &[2]);
    /// ```
    pub fn filter<F>(&self, predicate: F) -> Tensor<T>
    where
        F: Fn(&T) -> bool,
        T: Clone,
    {
        let filtered_data: Vec<T> = self
            .data
            .iter()
            .filter(|elem| predicate(elem))
            .cloned()
            .collect();
        let filtered_len = filtered_data.len();
        Tensor::from_vec(filtered_data, vec![filtered_len])
    }
}

// IntoIterator implementations

impl<T: crate::Dtype> IntoIterator for Tensor<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T: crate::Dtype + std::ops::Neg<Output = T> + num_traits::FromPrimitive> IntoIterator
    for &'a Tensor<T>
{
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let sum: f64 = tensor.iter().sum();
        assert_eq!(sum, 6.0);
    }

    #[test]
    fn test_iter_mut() {
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        for elem in tensor.iter_mut() {
            *elem *= 2.0;
        }
        assert_eq!(tensor.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_map() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let squared = tensor.map(|x| x * x);
        assert_eq!(squared.data(), &[1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_map_inplace() {
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        tensor.map_inplace(|x| *x * 2.0);
        assert_eq!(tensor.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_filter_indices() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let even_indices = tensor.filter_indices(|x| *x % 2.0 == 0.0);
        assert_eq!(even_indices, vec![1, 3]);
    }

    #[test]
    fn test_fold() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let sum = tensor.fold(0.0, |acc, x| acc + x);
        assert_eq!(sum, 10.0);
    }

    #[test]
    fn test_any_all() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert!(tensor.any(|x| *x > 2.5));
        assert!(!tensor.any(|x| *x > 5.0));

        assert!(tensor.all(|x| *x > 0.0));
        assert!(!tensor.all(|x| *x > 1.5));
    }

    #[test]
    fn test_find() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let result = tensor.find(|x| *x > 2.5);
        assert_eq!(result, Some((2, &3.0)));

        let not_found = tensor.find(|x| *x > 10.0);
        assert_eq!(not_found, None);
    }

    #[test]
    fn test_from_iter() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_iter(data, vec![2, 2]);
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_chain() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let chained: Vec<f64> = a.chain(&b).cloned().collect();
        assert_eq!(chained, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_zip() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let zipped: Vec<(f64, f64)> = a.zip(&b).map(|(x, y)| (*x, *y)).collect();
        assert_eq!(zipped, vec![(1.0, 3.0), (2.0, 4.0)]);
    }

    #[test]
    fn test_windows() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let windows: Vec<&[f64]> = tensor.windows(2).collect();
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0], &[1.0, 2.0]);
        assert_eq!(windows[1], &[2.0, 3.0]);
        assert_eq!(windows[2], &[3.0, 4.0]);
    }

    #[test]
    fn test_take_skip() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

        let first_two: Vec<f64> = tensor.take(2).cloned().collect();
        assert_eq!(first_two, vec![1.0, 2.0]);

        let last_two: Vec<f64> = tensor.skip(2).cloned().collect();
        assert_eq!(last_two, vec![3.0, 4.0]);
    }

    #[test]
    fn test_select() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let selected = tensor.select(&[0, 2]);
        let values: Vec<f64> = selected.iter().map(|&x| *x).collect();
        assert_eq!(values, vec![1.0, 3.0]);
    }

    #[test]
    fn test_filter() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let even = tensor.filter(|x| *x % 2.0 == 0.0);
        assert_eq!(even.data(), &[2.0, 4.0]);
        assert_eq!(even.shape(), &[2]);
    }

    #[test]
    fn test_into_iterator() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let sum: f64 = tensor.into_iter().sum();
        assert_eq!(sum, 6.0);
    }

    #[test]
    fn test_into_iterator_ref() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let sum: f64 = (&tensor).into_iter().sum::<f64>();
        assert_eq!(sum, 6.0);
    }
}
