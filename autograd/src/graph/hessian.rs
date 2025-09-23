//! Hessian matrix iterator for second-order automatic differentiation
//!
//! This module contains the HessianIter struct for traversing Hessian matrix elements.

use coeus_dtype::Dtype;

/// Iterator for traversing Hessian matrix elements
pub struct HessianIter<T: Dtype> {
    /// Flattened Hessian matrix data
    data: Vec<T>,
    /// Current position in iteration
    position: usize,
    /// Matrix dimension (n x n matrix has size n)
    size: usize,
}

impl<T: Dtype> HessianIter<T> {
    /// Create a new Hessian iterator
    pub fn new(data: Vec<T>, size: usize) -> Self {
        assert_eq!(
            data.len(),
            size * size,
            "Data length must equal size squared"
        );
        Self {
            data,
            position: 0,
            size,
        }
    }

    /// Create iterator from nested vector representation
    pub fn from_nested(matrix: Vec<Vec<T>>) -> Self {
        let size = matrix.len();
        let mut data = Vec::with_capacity(size * size);

        for row in matrix {
            assert_eq!(row.len(), size, "Matrix must be square");
            data.extend(row);
        }

        Self::new(data, size)
    }

    /// Get the matrix size (n for n x n matrix)
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get element at position (i, j)
    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        if i < self.size && j < self.size {
            self.data.get(i * self.size + j)
        } else {
            None
        }
    }

    /// Convert back to nested vector representation
    pub fn to_nested(&self) -> Vec<Vec<T>>
    where
        T: Clone,
    {
        let mut result = Vec::with_capacity(self.size);
        for i in 0..self.size {
            let start = i * self.size;
            let end = start + self.size;
            result.push(self.data[start..end].to_vec());
        }
        result
    }
}

impl<T: Dtype> Iterator for HessianIter<T> {
    type Item = (usize, usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.data.len() {
            let i = self.position / self.size;
            let j = self.position % self.size;
            let value = self.data[self.position];
            self.position += 1;
            Some((i, j, value))
        } else {
            None
        }
    }
}

impl<T: Dtype> ExactSizeIterator for HessianIter<T> {
    fn len(&self) -> usize {
        self.data.len() - self.position
    }
}
