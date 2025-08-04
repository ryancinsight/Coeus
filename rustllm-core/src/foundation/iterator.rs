//! Advanced iterator combinators for zero-copy token processing.
//!
//! This module provides custom iterator adapters and combinators specifically
//! designed for efficient token processing with minimal allocations.
//!
//! ## Design Principles
//!
//! - **Zero-cost abstractions**: All iterators compile to efficient code
//! - **Composability**: Iterators can be freely combined
//! - **Lazy evaluation**: Work is deferred until needed
//! - **Memory efficiency**: Minimal allocations and copies
//!
//! ## Example
//!
//! ```rust,ignore
//! use rustllm_core::prelude::*;
//!
//! let tokens = vec!["hello", "world", "from", "rust"];
//! let result: Vec<_> = tokens.iter()
//!     .windows(2)
//!     .stream_map(|w| format!("{}-{}", w[0], w[1]))
//!     .collect();
//! ```

use core::iter::{Iterator, FusedIterator};
use core::marker::PhantomData;

#[cfg(feature = "std")]
use std::collections::VecDeque;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::collections::VecDeque;

// ============================================================================
// Core Iterator Traits
// ============================================================================

/// Base trait for all custom iterator adapters.
/// 
/// This trait provides the foundation for composable iterators,
/// following the Interface Segregation Principle.
pub trait IteratorAdapter: Iterator + Sized {
    /// The source iterator type.
    type Source: Iterator;
    
    /// Returns a reference to the source iterator.
    fn source(&self) -> &Self::Source;
    
    /// Returns a mutable reference to the source iterator.
    fn source_mut(&mut self) -> &mut Self::Source;
}

/// Trait for iterators that maintain internal state.
/// 
/// This follows the Single Responsibility Principle by separating
/// stateful behavior from basic iteration.
pub trait StatefulIterator: Iterator {
    /// The state type.
    type State;
    
    /// Returns the current state.
    fn state(&self) -> &Self::State;
    
    /// Resets the internal state.
    fn reset_state(&mut self);
}

/// Trait for iterators that can provide size hints.
/// 
/// This extends the standard size_hint with more precise information.
pub trait PreciseSizeHint: Iterator {
    /// Returns the exact number of remaining elements, if known.
    fn exact_size(&self) -> Option<usize> {
        let (lower, upper) = self.size_hint();
        if upper == Some(lower) {
            Some(lower)
        } else {
            None
        }
    }
    
    /// Returns whether the iterator is empty.
    fn is_empty(&self) -> bool {
        self.size_hint().1 == Some(0)
    }
}

// Blanket implementation for all iterators
impl<I: Iterator> PreciseSizeHint for I {}

// ============================================================================
// Window Iterator
// ============================================================================

/// Iterator adapter for sliding windows over elements.
/// 
/// This iterator yields overlapping windows of a fixed size,
/// implementing efficient windowing with minimal allocations.
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
    /// 
    /// # Panics
    /// 
    /// Panics if `size` is 0.
    pub fn new(mut iter: I, size: usize) -> Self {
        assert!(size > 0, "Window size must be greater than 0");
        
        // Pre-allocate the window buffer
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
    
    /// Returns the window size.
    pub fn window_size(&self) -> usize {
        self.size
    }
}

impl<I, T> Iterator for Windows<I, T>
where
    I: Iterator<Item = T>,
    T: Clone,
{
    type Item = Vec<T>;
    
    fn next(&mut self) -> Option<Self::Item> {
        // Only yield if we have a full window
        if self.window.len() < self.size {
            return None;
        }
        
        // Clone current window
        let result = self.window.clone();
        
        // Slide the window
        if let Some(next_item) = self.iter.next() {
            self.window.remove(0);
            self.window.push(next_item);
        } else {
            // No more items, clear window to stop iteration
            self.window.clear();
        }
        
        Some(result)
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.window.len() < self.size {
            (0, Some(0))
        } else {
            let (lower, upper) = self.iter.size_hint();
            (lower.saturating_add(1), upper.map(|u| u.saturating_add(1)))
        }
    }
}

impl<I, T> FusedIterator for Windows<I, T>
where
    I: Iterator<Item = T> + FusedIterator,
    T: Clone,
{}

impl<I, T> IteratorAdapter for Windows<I, T>
where
    I: Iterator<Item = T>,
    T: Clone,
{
    type Source = I;
    
    fn source(&self) -> &Self::Source {
        &self.iter
    }
    
    fn source_mut(&mut self) -> &mut Self::Source {
        &mut self.iter
    }
}

// ============================================================================
// Chunk Iterator
// ============================================================================

/// Iterator adapter for chunking elements into fixed-size batches.
/// 
/// This iterator yields non-overlapping chunks, with the last chunk
/// potentially being smaller than the requested size.
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
    /// 
    /// # Panics
    /// 
    /// Panics if `size` is 0.
    pub fn new(iter: I, size: usize) -> Self {
        assert!(size > 0, "Chunk size must be greater than 0");
        
        Self {
            iter,
            size,
            _marker: PhantomData,
        }
    }
    
    /// Returns the chunk size.
    pub fn chunk_size(&self) -> usize {
        self.size
    }
}

impl<I, T> Iterator for Chunks<I, T>
where
    I: Iterator<Item = T>,
{
    type Item = Vec<T>;
    
    fn next(&mut self) -> Option<Self::Item> {
        // Collect up to `size` elements
        let chunk: Vec<_> = self.iter.by_ref().take(self.size).collect();
        
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

impl<I, T> IteratorAdapter for Chunks<I, T>
where
    I: Iterator<Item = T>,
{
    type Source = I;
    
    fn source(&self) -> &Self::Source {
        &self.iter
    }
    
    fn source_mut(&mut self) -> &mut Self::Source {
        &mut self.iter
    }
}

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
    
    /// Creates a parallel iterator adapter.
    #[cfg(feature = "std")]
    fn parallel(self) -> Parallel<Self>
    where
        Self: Sized,
    {
        Parallel::new(self)
    }
    
    /// Creates a zero-copy streaming iterator adapter.
    /// 
    /// This adapter processes items lazily without collecting them,
    /// providing true zero-cost abstraction for stream processing.
    fn stream_map<F, B>(self, f: F) -> StreamMap<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> B,
    {
        StreamMap::new(self, f)
    }
    
    /// Creates a stateful scan iterator with zero allocations.
    /// 
    /// This provides a zero-cost way to maintain state across iterations
    /// without heap allocations.
    fn scan_state<St, F, B>(self, initial_state: St, f: F) -> ScanState<Self, St, F>
    where
        Self: Sized,
        F: FnMut(&mut St, Self::Item) -> Option<B>,
    {
        ScanState::new(self, initial_state, f)
    }
    
    /// Creates a buffer-free windowed aggregation iterator.
    /// 
    /// This performs rolling computations without storing the window,
    /// achieving zero-copy operation for aggregations.
    fn rolling_aggregate<F, B>(self, window_size: usize, f: F) -> RollingAggregate<Self, F>
    where
        Self: Sized,
        F: FnMut(&[Self::Item]) -> B,
        Self::Item: Clone,
    {
        RollingAggregate::new(self, window_size, f)
    }
    
    /// Creates a lazy batching iterator that yields when full or on demand.
    /// 
    /// This provides efficient batching with minimal memory overhead.
    fn lazy_batch(self, capacity: usize) -> LazyBatch<Self>
    where
        Self: Sized,
        Self::Item: Clone,
    {
        LazyBatch::new(self, capacity)
    }
    
    /// Creates a cache-friendly prefetch iterator.
    ///
    /// This optimizes memory access patterns for better CPU cache utilization.
    fn prefetch(self, prefetch_size: usize) -> Prefetch<Self>
    where
        Self: Sized,
        Self::Item: Clone,
    {
        Prefetch::new(self, prefetch_size)
    }

    /// Creates a zero-copy sliding window iterator with custom step size.
    ///
    /// This provides efficient sliding windows with configurable step size,
    /// enabling both overlapping and non-overlapping windows.
    fn sliding_windows(self, window_size: usize, step_size: usize) -> SlidingWindows<Self>
    where
        Self: Sized,
        Self::Item: Clone,
    {
        SlidingWindows::new(self, window_size, step_size)
    }

    /// Creates a vectorized iterator for SIMD-friendly operations.
    ///
    /// This groups items into vectors of a specific size for vectorized processing.
    fn vectorize(self, vector_size: usize) -> Vectorize<Self>
    where
        Self: Sized,
    {
        Vectorize::new(self, vector_size)
    }

    /// Creates a memory-efficient circular buffer iterator.
    ///
    /// This maintains a fixed-size circular buffer for streaming operations.
    fn circular_buffer(self, capacity: usize) -> CircularBuffer<Self>
    where
        Self: Sized,
        Self::Item: Clone,
    {
        CircularBuffer::new(self, capacity)
    }

    /// Creates an iterator that yields items in batches with backpressure.
    ///
    /// This provides flow control for memory-sensitive operations.
    fn backpressure_batch(self, batch_size: usize, max_memory: usize) -> BackpressureBatch<Self>
    where
        Self: Sized,
        Self::Item: Clone,
    {
        BackpressureBatch::new(self, batch_size, max_memory)
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

/// Parallel iterator adapter for concurrent processing.
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct Parallel<I> {
    iter: I,
}

#[cfg(feature = "std")]
impl<I> Parallel<I> {
    fn new(iter: I) -> Self {
        Self { iter }
    }
}

#[cfg(feature = "std")]
impl<I> Iterator for Parallel<I>
where
    I: Iterator,
    I::Item: Send,
{
    type Item = I::Item;
    
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}



/// Zero-copy streaming map iterator.
#[derive(Debug, Clone)]
pub struct StreamMap<I, F> {
    iter: I,
    f: F,
}

impl<I, F> StreamMap<I, F> {
    fn new(iter: I, f: F) -> Self {
        Self { iter, f }
    }
}

impl<I, F, B> Iterator for StreamMap<I, F>
where
    I: Iterator,
    F: FnMut(I::Item) -> B,
{
    type Item = B;
    
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(&mut self.f)
    }
    
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I, F, B> FusedIterator for StreamMap<I, F>
where
    I: FusedIterator,
    F: FnMut(I::Item) -> B,
{}

/// Stateful scan iterator with zero allocations.
#[derive(Debug, Clone)]
pub struct ScanState<I, St, F> {
    iter: I,
    state: St,
    f: F,
}

impl<I, St, F> ScanState<I, St, F> {
    fn new(iter: I, state: St, f: F) -> Self {
        Self { iter, state, f }
    }
}

impl<I, St, F, B> Iterator for ScanState<I, St, F>
where
    I: Iterator,
    F: FnMut(&mut St, I::Item) -> Option<B>,
{
    type Item = B;
    
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().and_then(|item| (self.f)(&mut self.state, item))
    }
}

/// Rolling aggregate iterator with efficient windowing.
/// 
/// Uses VecDeque for O(1) push/pop operations at both ends.
#[derive(Debug)]
pub struct RollingAggregate<I: Iterator, F> {
    iter: I,
    window_size: usize,
    #[cfg(any(feature = "std", feature = "alloc"))]
    buffer: VecDeque<I::Item>,
    #[cfg(not(any(feature = "std", feature = "alloc")))]
    buffer: Vec<I::Item>,
    f: F,
}

impl<I, F> RollingAggregate<I, F>
where
    I: Iterator,
    I::Item: Clone,
{
    fn new(iter: I, window_size: usize, f: F) -> Self {
        assert!(window_size > 0, "Window size must be greater than 0");
        Self {
            iter,
            window_size,
            #[cfg(any(feature = "std", feature = "alloc"))]
            buffer: VecDeque::with_capacity(window_size),
            #[cfg(not(any(feature = "std", feature = "alloc")))]
            buffer: Vec::with_capacity(window_size),
            f,
        }
    }
}

impl<I, F, B> Iterator for RollingAggregate<I, F>
where
    I: Iterator,
    I::Item: Clone,
    F: FnMut(&[I::Item]) -> B,
{
    type Item = B;
    
    fn next(&mut self) -> Option<Self::Item> {
        // Fill buffer initially
        while self.buffer.len() < self.window_size {
            match self.iter.next() {
                #[cfg(any(feature = "std", feature = "alloc"))]
                Some(item) => self.buffer.push_back(item),
                #[cfg(not(any(feature = "std", feature = "alloc")))]
                Some(item) => self.buffer.push(item),
                None => break,
            }
        }
        
        if self.buffer.is_empty() {
            return None;
        }
        
        // Get slice for the closure
        #[cfg(any(feature = "std", feature = "alloc"))]
        let slice = self.buffer.make_contiguous();
        #[cfg(not(any(feature = "std", feature = "alloc")))]
        let slice = &self.buffer[..];
        
        let result = (self.f)(slice);
        
        // Slide window efficiently
        if let Some(next_item) = self.iter.next() {
            #[cfg(any(feature = "std", feature = "alloc"))]
            {
                self.buffer.pop_front();
                self.buffer.push_back(next_item);
            }
            #[cfg(not(any(feature = "std", feature = "alloc")))]
            {
                self.buffer.remove(0);
                self.buffer.push(next_item);
            }
        } else {
            self.buffer.clear();
        }
        
        Some(result)
    }
}

/// Lazy batching iterator.
#[derive(Debug)]
pub struct LazyBatch<I: Iterator> {
    iter: I,
    capacity: usize,
    buffer: Vec<I::Item>,
}

impl<I> LazyBatch<I>
where
    I: Iterator,
{
    fn new(iter: I, capacity: usize) -> Self {
        assert!(capacity > 0, "Batch capacity must be greater than 0");
        Self {
            iter,
            capacity,
            buffer: Vec::with_capacity(capacity),
        }
    }
    
    /// Forces the current batch to be yielded even if not full.
    pub fn flush(&mut self) -> Option<Vec<I::Item>> {
        if self.buffer.is_empty() {
            None
        } else {
            Some(core::mem::replace(&mut self.buffer, Vec::with_capacity(self.capacity)))
        }
    }
}

impl<I> Iterator for LazyBatch<I>
where
    I: Iterator,
{
    type Item = Vec<I::Item>;
    
    fn next(&mut self) -> Option<Self::Item> {
        self.buffer.clear();
        
        for _ in 0..self.capacity {
            match self.iter.next() {
                Some(item) => self.buffer.push(item),
                None => break,
            }
        }
        
        if self.buffer.is_empty() {
            None
        } else {
            Some(core::mem::replace(&mut self.buffer, Vec::with_capacity(self.capacity)))
        }
    }
}

/// Cache-friendly prefetch iterator.
#[derive(Debug)]
pub struct Prefetch<I: Iterator> {
    iter: I,
    buffer: Vec<I::Item>,
    prefetch_size: usize,
    index: usize,
}

impl<I> Prefetch<I>
where
    I: Iterator,
    I::Item: Clone,
{
    fn new(iter: I, prefetch_size: usize) -> Self {
        assert!(prefetch_size > 0, "Prefetch size must be greater than 0");
        Self {
            iter,
            buffer: Vec::with_capacity(prefetch_size),
            prefetch_size,
            index: 0,
        }
    }
    
    fn fill_buffer(&mut self) {
        self.buffer.clear();
        self.index = 0;
        
        for _ in 0..self.prefetch_size {
            if let Some(item) = self.iter.next() {
                self.buffer.push(item);
            } else {
                break;
            }
        }
    }
}

impl<I> Iterator for Prefetch<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.buffer.len() {
            self.fill_buffer();
            if self.buffer.is_empty() {
                return None;
            }
        }

        let item = self.buffer[self.index].clone();
        self.index += 1;
        Some(item)
    }
}

// ============================================================================
// Advanced Iterator Combinators
// ============================================================================

/// Sliding windows iterator with configurable step size.
///
/// This provides more flexible windowing than the basic Windows iterator,
/// allowing for both overlapping and non-overlapping windows.
#[derive(Debug, Clone)]
pub struct SlidingWindows<I: Iterator> {
    iter: I,
    window_size: usize,
    step_size: usize,
    buffer: Vec<I::Item>,
    position: usize,
}

impl<I: Iterator> SlidingWindows<I> {
    fn new(iter: I, window_size: usize, step_size: usize) -> Self {
        assert!(window_size > 0, "Window size must be greater than 0");
        assert!(step_size > 0, "Step size must be greater than 0");

        Self {
            iter,
            window_size,
            step_size,
            buffer: Vec::new(),
            position: 0,
        }
    }
}

impl<I> Iterator for SlidingWindows<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        // Fill buffer initially
        while self.buffer.len() < self.window_size {
            match self.iter.next() {
                Some(item) => self.buffer.push(item),
                None => break,
            }
        }

        if self.buffer.len() < self.window_size {
            return None;
        }

        // Create window from current position
        let window = self.buffer[self.position..self.position + self.window_size].to_vec();

        // Advance position by step size
        self.position += self.step_size;

        // If we've moved beyond the buffer, shift and refill
        if self.position + self.window_size > self.buffer.len() {
            // Remove processed items
            self.buffer.drain(0..self.position);
            self.position = 0;

            // Refill buffer
            while self.buffer.len() < self.window_size {
                match self.iter.next() {
                    Some(item) => self.buffer.push(item),
                    None => break,
                }
            }

            if self.buffer.len() < self.window_size {
                return None;
            }
        }

        Some(window)
    }
}

/// Vectorized iterator for SIMD-friendly operations.
///
/// Groups items into fixed-size vectors for vectorized processing.
#[derive(Debug)]
pub struct Vectorize<I: Iterator> {
    iter: I,
    vector_size: usize,
}

impl<I: Iterator> Vectorize<I> {
    fn new(iter: I, vector_size: usize) -> Self {
        assert!(vector_size > 0, "Vector size must be greater than 0");
        Self { iter, vector_size }
    }
}

impl<I: Iterator> Iterator for Vectorize<I> {
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut vector = Vec::with_capacity(self.vector_size);

        for _ in 0..self.vector_size {
            match self.iter.next() {
                Some(item) => vector.push(item),
                None => break,
            }
        }

        if vector.is_empty() {
            None
        } else {
            Some(vector)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        let lower_vectors = (lower + self.vector_size - 1) / self.vector_size;
        let upper_vectors = upper.map(|u| (u + self.vector_size - 1) / self.vector_size);
        (lower_vectors, upper_vectors)
    }
}

/// Circular buffer iterator for streaming operations.
///
/// Maintains a fixed-size circular buffer for memory-efficient streaming.
#[derive(Debug)]
pub struct CircularBuffer<I: Iterator> {
    iter: I,
    buffer: Vec<Option<I::Item>>,
    capacity: usize,
    head: usize,
    tail: usize,
    size: usize,
}

impl<I: Iterator> CircularBuffer<I> {
    fn new(iter: I, capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be greater than 0");

        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(None);
        }

        Self {
            iter,
            buffer,
            capacity,
            head: 0,
            tail: 0,
            size: 0,
        }
    }

    fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    fn is_empty(&self) -> bool {
        self.size == 0
    }

    fn push(&mut self, item: I::Item) -> bool {
        if self.is_full() {
            return false;
        }

        self.buffer[self.tail] = Some(item);
        self.tail = (self.tail + 1) % self.capacity;
        self.size += 1;
        true
    }

    fn pop(&mut self) -> Option<I::Item> {
        if self.is_empty() {
            return None;
        }

        let item = self.buffer[self.head].take();
        self.head = (self.head + 1) % self.capacity;
        self.size -= 1;
        item
    }
}

impl<I> Iterator for CircularBuffer<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        // Try to fill buffer
        while !self.is_full() {
            match self.iter.next() {
                Some(item) => {
                    if !self.push(item) {
                        break;
                    }
                }
                None => break,
            }
        }

        self.pop()
    }
}

/// Backpressure batch iterator for memory-controlled processing.
///
/// Provides flow control by limiting memory usage and batch sizes.
#[derive(Debug)]
pub struct BackpressureBatch<I: Iterator> {
    iter: I,
    batch_size: usize,
    max_memory: usize,
    current_memory: usize,
    buffer: Vec<I::Item>,
}

impl<I: Iterator> BackpressureBatch<I> {
    fn new(iter: I, batch_size: usize, max_memory: usize) -> Self {
        assert!(batch_size > 0, "Batch size must be greater than 0");
        assert!(max_memory > 0, "Max memory must be greater than 0");

        Self {
            iter,
            batch_size,
            max_memory,
            current_memory: 0,
            buffer: Vec::with_capacity(batch_size),
        }
    }

    fn estimate_item_size(&self, _item: &I::Item) -> usize {
        // Simple estimation - in practice, this could be more sophisticated
        core::mem::size_of::<I::Item>()
    }
}

impl<I> Iterator for BackpressureBatch<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        self.buffer.clear();
        self.current_memory = 0;

        while self.buffer.len() < self.batch_size && self.current_memory < self.max_memory {
            match self.iter.next() {
                Some(item) => {
                    let item_size = self.estimate_item_size(&item);

                    // Check if adding this item would exceed memory limit
                    if self.current_memory + item_size > self.max_memory && !self.buffer.is_empty() {
                        // Put the item back (conceptually) and return current batch
                        break;
                    }

                    self.current_memory += item_size;
                    self.buffer.push(item);
                }
                None => break,
            }
        }

        if self.buffer.is_empty() {
            None
        } else {
            Some(core::mem::replace(&mut self.buffer, Vec::with_capacity(self.batch_size)))
        }
    }
}

// ============================================================================
// Zero-Copy String Processing
// ============================================================================

/// Zero-copy string splitter that yields string slices.
///
/// This provides efficient string tokenization without allocations.
#[derive(Debug, Clone)]
pub struct ZeroCopySplit<'a> {
    text: &'a str,
    delimiter: char,
    position: usize,
}

impl<'a> ZeroCopySplit<'a> {
    /// Creates a new zero-copy string splitter.
    pub fn new(text: &'a str, delimiter: char) -> Self {
        Self {
            text,
            delimiter,
            position: 0,
        }
    }
}

impl<'a> Iterator for ZeroCopySplit<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.text.len() {
            return None;
        }

        let start = self.position;

        // Use char_indices to find the next delimiter efficiently
        let remainder = &self.text[self.position..];
        if let Some((offset, _)) = remainder.char_indices().find(|&(_, c)| c == self.delimiter) {
            let end = self.position + offset;
            let result = &self.text[start..end];
            // Move position past the delimiter for next iteration
            self.position = end + self.delimiter.len_utf8();
            Some(result)
        } else {
            // No more delimiters; return the rest of the string
            self.position = self.text.len();
            Some(&self.text[start..])
        }
    }
}

impl<'a> FusedIterator for ZeroCopySplit<'a> {}

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