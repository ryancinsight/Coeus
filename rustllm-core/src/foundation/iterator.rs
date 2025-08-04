//! Advanced iterator combinators for zero-copy token processing.
//!
//! This module provides custom iterator adapters and combinators specifically
//! designed for efficient token processing with minimal allocations.

use core::iter::{Iterator, FusedIterator};
use core::marker::PhantomData;

/// Iterator adapter for sliding windows over tokens.
#[derive(Debug, Clone)]
pub struct Windows<I, T> {
    iter: I,
    window: Vec<T>,
    size: usize,
}

impl<I, T> Windows<I, T>
where
    I: Iterator<Item = T>,
    T: Clone,
{
    /// Creates a new sliding window iterator.
    pub fn new(mut iter: I, size: usize) -> Self {
        assert!(size > 0, "Window size must be greater than 0");
        
        let mut window = Vec::with_capacity(size);
        
        // Fill initial window
        for _ in 0..size {
            if let Some(item) = iter.next() {
                window.push(item);
            } else {
                break;
            }
        }
        
        Self { iter, window, size }
    }
}

impl<I, T> Iterator for Windows<I, T>
where
    I: Iterator<Item = T>,
    T: Clone,
{
    type Item = Vec<T>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.window.len() < self.size {
            return None;
        }
        
        let result = self.window.clone();
        
        if let Some(next_item) = self.iter.next() {
            self.window.remove(0);
            self.window.push(next_item);
        } else {
            self.window.clear();
        }
        
        Some(result)
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        
        if self.window.len() < self.size {
            (0, Some(0))
        } else {
            (lower, upper.map(|u| u + 1))
        }
    }
}

impl<I, T> FusedIterator for Windows<I, T>
where
    I: Iterator<Item = T> + FusedIterator,
    T: Clone,
{}

/// Iterator adapter for chunking tokens into batches.
#[derive(Debug)]
pub struct Chunks<I, T> {
    iter: I,
    size: usize,
    _marker: PhantomData<T>,
}

impl<I, T> Chunks<I, T>
where
    I: Iterator<Item = T>,
{
    /// Creates a new chunking iterator.
    pub fn new(iter: I, size: usize) -> Self {
        assert!(size > 0, "Chunk size must be greater than 0");
        
        Self {
            iter,
            size,
            _marker: PhantomData,
        }
    }
}

impl<I, T> Iterator for Chunks<I, T>
where
    I: Iterator<Item = T>,
{
    type Item = Vec<T>;
    
    fn next(&mut self) -> Option<Self::Item> {
        let mut chunk = Vec::with_capacity(self.size);
        
        for _ in 0..self.size {
            match self.iter.next() {
                Some(item) => chunk.push(item),
                None => break,
            }
        }
        
        if chunk.is_empty() {
            None
        } else {
            Some(chunk)
        }
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        
        let lower_chunks = (lower + self.size - 1) / self.size;
        let upper_chunks = upper.map(|u| (u + self.size - 1) / self.size);
        
        (lower_chunks, upper_chunks)
    }
}

impl<I, T> FusedIterator for Chunks<I, T>
where
    I: Iterator<Item = T> + FusedIterator,
{}

/// Iterator adapter for striding through tokens.
#[derive(Debug, Clone)]
pub struct Stride<I> {
    iter: I,
    step: usize,
}

impl<I> Stride<I>
where
    I: Iterator,
{
    /// Creates a new striding iterator.
    pub fn new(iter: I, step: usize) -> Self {
        assert!(step > 0, "Stride step must be greater than 0");
        
        Self { iter, step }
    }
}

impl<I> Iterator for Stride<I>
where
    I: Iterator,
{
    type Item = I::Item;
    
    fn next(&mut self) -> Option<Self::Item> {
        let item = self.iter.next()?;
        
        // Skip step-1 items
        for _ in 1..self.step {
            self.iter.next();
        }
        
        Some(item)
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        
        let lower_stride = (lower + self.step - 1) / self.step;
        let upper_stride = upper.map(|u| (u + self.step - 1) / self.step);
        
        (lower_stride, upper_stride)
    }
}

impl<I> FusedIterator for Stride<I>
where
    I: Iterator + FusedIterator,
{}

/// Iterator adapter for parallel-friendly token processing.
#[derive(Debug)]
pub struct ParChunks<I, T> {
    iter: I,
    chunk_size: usize,
    _marker: PhantomData<T>,
}

impl<I, T> ParChunks<I, T>
where
    I: Iterator<Item = T>,
{
    /// Creates a new parallel chunks iterator.
    pub fn new(iter: I, chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "Chunk size must be greater than 0");
        
        Self {
            iter,
            chunk_size,
            _marker: PhantomData,
        }
    }
}

impl<I, T> Iterator for ParChunks<I, T>
where
    I: Iterator<Item = T>,
    T: Send,
{
    type Item = Vec<T>;
    
    fn next(&mut self) -> Option<Self::Item> {
        let mut chunk = Vec::with_capacity(self.chunk_size);
        
        for _ in 0..self.chunk_size {
            match self.iter.next() {
                Some(item) => chunk.push(item),
                None => break,
            }
        }
        
        if chunk.is_empty() {
            None
        } else {
            Some(chunk)
        }
    }
}

/// Extension trait for iterators with advanced combinators.
pub trait IteratorExt: Iterator {
    /// Creates a sliding window iterator.
    fn windows(self, size: usize) -> Windows<Self, Self::Item>
    where
        Self: Sized,
        Self::Item: Clone,
    {
        Windows::new(self, size)
    }
    
    /// Creates a chunking iterator.
    fn chunks(self, size: usize) -> Chunks<Self, Self::Item>
    where
        Self: Sized,
    {
        Chunks::new(self, size)
    }
    
    /// Creates a striding iterator.
    fn stride(self, step: usize) -> Stride<Self>
    where
        Self: Sized,
    {
        Stride::new(self, step)
    }
    
    /// Creates a parallel chunks iterator.
    fn par_chunks(self, size: usize) -> ParChunks<Self, Self::Item>
    where
        Self: Sized,
        Self::Item: Send,
    {
        ParChunks::new(self, size)
    }
    
    /// Collects into a Vec with a size hint for capacity.
    fn collect_vec_with_capacity(self) -> Vec<Self::Item>
    where
        Self: Sized,
    {
        let (lower, upper) = self.size_hint();
        let capacity = upper.unwrap_or(lower);
        let mut vec = Vec::with_capacity(capacity);
        vec.extend(self);
        vec
    }
}

impl<I: Iterator> IteratorExt for I {}

/// Zero-copy string iterator for token processing.
#[derive(Debug, Clone)]
pub struct StrTokens<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> StrTokens<'a> {
    /// Creates a new string tokens iterator.
    pub fn new(text: &'a str) -> Self {
        Self { text, pos: 0 }
    }
}

impl<'a> Iterator for StrTokens<'a> {
    type Item = &'a str;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.text.len() {
            return None;
        }
        
        let start = self.pos;
        
        // Simple whitespace tokenization for demonstration
        while self.pos < self.text.len() && !self.text.as_bytes()[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        
        if start == self.pos {
            // Skip whitespace
            self.pos += 1;
            self.next()
        } else {
            Some(&self.text[start..self.pos])
        }
    }
}

impl<'a> FusedIterator for StrTokens<'a> {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_windows() {
        let tokens = vec!["a", "b", "c", "d", "e"];
        let windows: Vec<_> = tokens.into_iter().windows(3).collect();
        
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0], vec!["a", "b", "c"]);
        assert_eq!(windows[1], vec!["b", "c", "d"]);
        assert_eq!(windows[2], vec!["c", "d", "e"]);
    }
    
    #[test]
    fn test_chunks() {
        let tokens = vec![1, 2, 3, 4, 5, 6, 7];
        let chunks: Vec<_> = tokens.into_iter().chunks(3).collect();
        
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], vec![1, 2, 3]);
        assert_eq!(chunks[1], vec![4, 5, 6]);
        assert_eq!(chunks[2], vec![7]);
    }
    
    #[test]
    fn test_stride() {
        let tokens = vec![1, 2, 3, 4, 5, 6];
        let strided: Vec<_> = tokens.into_iter().stride(2).collect();
        
        assert_eq!(strided, vec![1, 3, 5]);
    }
    
    #[test]
    fn test_str_tokens() {
        let text = "hello world rust";
        let tokens: Vec<_> = StrTokens::new(text).collect();
        
        assert_eq!(tokens, vec!["hello", "world", "rust"]);
    }
}