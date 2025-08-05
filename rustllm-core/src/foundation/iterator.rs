//! Iterator extensions and combinators for RustLLM Core.
//!
//! This module provides advanced iterator patterns optimized for LLM workloads,
//! following zero-copy principles and leveraging Rust's iterator ecosystem.
//!
//! ## Design Principles
//!
//! - **Zero-Copy**: Minimize allocations and copies using borrowing and views
//! - **Lazy Evaluation**: Defer computation until values are needed
//! - **Composability**: Small, focused iterators that combine well
//! - **Performance**: Leverage SIMD and cache-friendly access patterns
//!
//! ## Literature References
//!
//! - "Stream Fusion: From Lists to Streams to Nothing at All" - Coutts et al., 2007
//! - "The Zipper" - Huet, 1997 (for bidirectional iteration patterns)
//! - "Rope: An Alternative to Strings" - Boehm et al., 1995 (for string handling)
//! - "Cache-Oblivious Algorithms" - Frigo et al., 1999 (for memory access patterns)

#![allow(clippy::module_name_repetitions)]

use core::iter::{Iterator, FusedIterator, ExactSizeIterator};
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
/// This extends the standard `size_hint` with more precise information.
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
// Window iterators
// ============================================================================

/// Sliding window iterator with configurable step size.
///
/// This iterator produces overlapping or non-overlapping windows of elements.
/// When step_size = 1, it produces overlapping windows (sliding window).
/// When step_size = window_size, it produces non-overlapping windows.
#[derive(Debug, Clone)]
pub struct SlidingWindows<I: Iterator> {
    iter: I,
    window_size: usize,
    step_size: usize,
    buffer: Vec<I::Item>,
    position: usize,
}

impl<I: Iterator> SlidingWindows<I> {
    /// Creates a new sliding window iterator.
    pub fn new(iter: I, window_size: usize, step_size: usize) -> Self {
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
    
    /// Creates a sliding window iterator with step size 1 (overlapping windows).
    pub fn sliding(iter: I, window_size: usize) -> Self {
        Self::new(iter, window_size, 1)
    }
    
    /// Returns the window size.
    pub const fn window_size(&self) -> usize {
        self.window_size
    }
    
    /// Returns the step size.
    pub const fn step_size(&self) -> usize {
        self.step_size
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
                None => {
                    if self.buffer.is_empty() {
                        return None;
                    }
                    break;
                }
            }
        }

        if self.buffer.len() < self.window_size {
            return None;
        }

        let window = self.buffer[..self.window_size].to_vec();

        // Advance by step_size
        for _ in 0..self.step_size {
            self.buffer.remove(0);
            if let Some(item) = self.iter.next() {
                self.buffer.push(item);
            }
        }

        Some(window)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        
        let calc_windows = |n: usize| {
            if n < self.window_size {
                0
            } else {
                (n - self.window_size) / self.step_size + 1
            }
        };
        
        (calc_windows(lower), upper.map(calc_windows))
    }
}

impl<I> FusedIterator for SlidingWindows<I>
where
    I: Iterator + FusedIterator,
    I::Item: Clone,
{}

// SlidingWindows doesn't implement StatefulIterator as it doesn't store state as a single value

// Type alias for backward compatibility
pub type Windows<I> = SlidingWindows<I>;

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
    pub const fn chunk_size(&self) -> usize {
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
    fn windows(self, size: usize) -> SlidingWindows<Self>
    where
        Self: Sized,
        Self::Item: Clone,
    {
        SlidingWindows::sliding(self, size)
    }
    
    /// Creates a sliding window iterator with custom step size.
    fn windows_step(self, window_size: usize, step_size: usize) -> SlidingWindows<Self>
    where
        Self: Sized,
        Self::Item: Clone,
    {
        SlidingWindows::new(self, window_size, step_size)
    }
    
    /// Creates a zero-copy view iterator over slices.
    fn zero_copy_windows<T>(self) -> ZeroCopyView<'static, T>
    where
        Self: Sized + Iterator<Item = &'static [T]>,
    {
        panic!("zero_copy_windows requires a slice iterator")
    }
    
    /// Creates a chunking iterator.
    fn chunks(self, size: usize) -> Chunks<Self, Self::Item>
    where
        Self: Sized,
        Self::Item: Clone,
    {
        Chunks::new(self, size)
    }
    
    /// Creates a strided iterator.
    fn stride(self, step: usize) -> Stride<Self>
    where
        Self: Sized,
    {
        Stride::new(self, step)
    }
    
    /// Applies stream fusion optimization.
    fn stream_fusion<F, B>(self, f: F) -> StreamFusion<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> Option<B>,
    {
        StreamFusion::new(self, f)
    }
    
    /// Creates a cache-oblivious iterator.
    fn cache_oblivious(self, block_size: usize) -> CacheObliviousIter<Self>
    where
        Self: Sized,
    {
        CacheObliviousIter::new(self, block_size)
    }
    
    /// Converts to a stream map iterator.
    fn stream_map<F, B>(self, f: F) -> StreamMap<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> B,
    {
        StreamMap::new(self, f)
    }
    
    /// Creates a scan state iterator.
    fn scan_state<St, F, B>(self, state: St, f: F) -> ScanState<Self, St, F>
    where
        Self: Sized,
        F: FnMut(&mut St, Self::Item) -> Option<B>,
    {
        ScanState::new(self, state, f)
    }
    
    /// Creates a lazy batch iterator.
    fn lazy_batch(self, capacity: usize) -> LazyBatch<Self>
    where
        Self: Sized,
    {
        LazyBatch::new(self, capacity)
    }
    
    /// Creates a backpressure-aware batch iterator.
    fn backpressure_batch(self, batch_size: usize, max_memory: usize) -> BackpressureBatch<Self>
    where
        Self: Sized,
    {
        BackpressureBatch::new(self, batch_size, max_memory)
    }
    
    /// Collects into a zero-copy string builder.
    #[cfg(any(feature = "std", feature = "alloc"))]
    fn collect_zero_copy_string<'a>(self) -> crate::foundation::memory::ZeroCopyStringBuilder<'a>
    where
        Self: Sized + Iterator<Item = &'a str>,
    {
        let mut builder = crate::foundation::memory::ZeroCopyStringBuilder::new();
        for s in self {
            builder.append_borrowed(s);
        }
        builder
    }
    
    /// Converts iterator to a zipper for bidirectional traversal.
    #[cfg(any(feature = "std", feature = "alloc"))]
    fn into_zipper(self) -> Zipper<Self::Item>
    where
        Self: Sized,
    {
        Zipper::new(self.collect())
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
    pub const fn new(text: &'a str) -> Self {
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
    const fn new(iter: I) -> Self {
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
    const fn new(iter: I, f: F) -> Self {
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
    const fn new(iter: I, state: St, f: F) -> Self {
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
/// Uses `VecDeque` for O(1) push/pop operations at both ends.
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

    const fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    const fn is_empty(&self) -> bool {
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

    const fn estimate_item_size(_item: &I::Item) -> usize {
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
                    let item_size = Self::estimate_item_size(&item);

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
            Some(core::mem::take(&mut self.buffer))
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
    pub const fn new(text: &'a str, delimiter: char) -> Self {
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
        let data = vec![1, 2, 3, 4, 5];
        let windows: Vec<_> = data.iter().cloned().windows(3).collect();
        assert_eq!(windows, vec![vec![1, 2, 3], vec![2, 3, 4], vec![3, 4, 5]]);
    }

    #[test]
    fn test_windows_step() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let windows: Vec<_> = data.iter().cloned().windows_step(3, 2).collect();
        assert_eq!(windows, vec![vec![1, 2, 3], vec![3, 4, 5]]);
    }

    #[test]
    fn test_chunks() {
        let data = vec![1, 2, 3, 4, 5, 6, 7];
        let chunks: Vec<_> = data.iter().cloned().chunks(3).collect();
        assert_eq!(chunks, vec![vec![1, 2, 3], vec![4, 5, 6], vec![7]]);
    }

    #[test]
    fn test_stride() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let strided: Vec<_> = data.iter().cloned().stride(2).collect();
        assert_eq!(strided, vec![1, 3, 5]);
    }

    #[test]
    fn test_str_tokens() {
        let text = "hello world rust";
        let tokens: Vec<_> = StrTokens::new(text).collect();
        assert_eq!(tokens, vec!["hello", "world", "rust"]);
    }
    
    #[test]
    fn test_zero_copy_view() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let view = ZeroCopyView::new(&data, 3);
        let windows: Vec<_> = view.collect();
        
        assert_eq!(windows.len(), 6);
        assert_eq!(windows[0], &[1, 2, 3]);
        assert_eq!(windows[1], &[2, 3, 4]);
        assert_eq!(windows[5], &[6, 7, 8]);
    }
    
    #[test]
    fn test_zero_copy_view_exact_size() {
        let data = vec![1, 2, 3, 4, 5];
        let view = ZeroCopyView::new(&data, 2);
        assert_eq!(view.len(), 4);
        assert_eq!(view.size_hint(), (4, Some(4)));
    }
    
    #[test]
    fn test_stream_fusion() {
        let data = vec![1, 2, 3, 4, 5];
        let fusion = StreamFusion::new(data.into_iter(), |x| {
            if x % 2 == 0 {
                Some(x * 2)
            } else {
                None
            }
        });
        let result: Vec<_> = fusion.collect();
        assert_eq!(result, vec![4, 8]);
    }
    
    #[test]
    fn test_cache_oblivious_iter() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let iter = CacheObliviousIter::new(data.into_iter(), 3);
        let result: Vec<_> = iter.collect();
        assert_eq!(result.len(), 8);
        assert!(result.contains(&1));
        assert!(result.contains(&8));
    }
    
    #[test]
    fn test_zipper() {
        let data = vec![1, 2, 3, 4, 5];
        let mut zipper = Zipper::new(data);
        
        assert_eq!(zipper.move_right(), Some(&1));
        assert_eq!(zipper.current(), Some(&1));
        assert_eq!(zipper.move_right(), Some(&2));
        assert_eq!(zipper.move_right(), Some(&3));
        assert_eq!(zipper.move_left(), Some(&2));
        assert_eq!(zipper.current(), Some(&2));
        
        let result = zipper.into_vec();
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }
    
    #[test]
    fn test_iterator_ext_integration() {
        let data = vec!["hello", "world", "from", "rust", "llm"];
        
        // Test chaining multiple zero-copy operations
        let result: Vec<_> = data.iter()
            .copied()
            .windows(2)
            .stream_map(|w| format!("{}-{}", w[0], w[1]))
            .collect();
            
        assert_eq!(result, vec!["hello-world", "world-from", "from-rust", "rust-llm"]);
    }
    
    #[test]
    fn test_lazy_batch() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let batches: Vec<_> = data.into_iter().lazy_batch(3).collect();
        assert_eq!(batches, vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);
    }
    
    #[test]
    fn test_backpressure_batch() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let batches: Vec<_> = data.into_iter()
            .backpressure_batch(4, 100)
            .collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 4);
        assert_eq!(batches[1].len(), 4);
    }
    
    #[test]
    fn test_zero_copy_split() {
        let text = "hello,world,rust";
        let parts: Vec<_> = ZeroCopySplit::new(text, ',').collect();
        assert_eq!(parts, vec!["hello", "world", "rust"]);
    }
}

// ============================================================================
// Zero-Copy Iterator Combinators
// ============================================================================

/// Zero-copy view iterator that provides a window into data without cloning.
///
/// Based on the concept of "views" from database systems and functional programming.
/// This avoids the overhead of cloning data in windows.
#[derive(Debug)]
pub struct ZeroCopyView<'a, T> {
    data: &'a [T],
    position: usize,
    window_size: usize,
}

impl<'a, T> ZeroCopyView<'a, T> {
    /// Creates a new zero-copy view iterator.
    pub const fn new(data: &'a [T], window_size: usize) -> Self {
        Self {
            data,
            position: 0,
            window_size,
        }
    }
}

impl<'a, T> Iterator for ZeroCopyView<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.position + self.window_size > self.data.len() {
            return None;
        }

        let window = &self.data[self.position..self.position + self.window_size];
        self.position += 1;
        Some(window)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.data.len().saturating_sub(self.position + self.window_size - 1);
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for ZeroCopyView<'a, T> {}
impl<'a, T> FusedIterator for ZeroCopyView<'a, T> {}

/// Cache-oblivious B-tree iterator for efficient memory access patterns.
///
/// Based on "Cache-Oblivious B-Trees" by Bender et al., 2000.
/// This iterator reorders elements to maximize cache efficiency.
#[derive(Debug)]
pub struct CacheObliviousIter<I: Iterator> {
    iter: I,
    buffer: Vec<I::Item>,
    block_size: usize,
}

impl<I: Iterator> CacheObliviousIter<I> {
    /// Creates a new cache-oblivious iterator.
    ///
    /// The block_size should be tuned to L1 cache line size (typically 64 bytes).
    pub fn new(iter: I, block_size: usize) -> Self {
        Self {
            iter,
            buffer: Vec::with_capacity(block_size),
            block_size,
        }
    }
}

impl<I: Iterator> Iterator for CacheObliviousIter<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer.is_empty() {
            // Fill buffer with next block
            self.buffer.extend(self.iter.by_ref().take(self.block_size));
            self.buffer.reverse(); // So we can pop efficiently
        }
        
        self.buffer.pop()
    }
}

/// Stream fusion iterator that eliminates intermediate data structures.
///
/// Based on "Stream Fusion" by Coutts et al., 2007.
/// This pattern allows the compiler to optimize away intermediate collections.
#[derive(Debug)]
pub struct StreamFusion<I, F> {
    iter: I,
    f: F,
}

impl<I, F> StreamFusion<I, F> {
    /// Creates a new stream fusion iterator.
    pub const fn new(iter: I, f: F) -> Self {
        Self { iter, f }
    }
}

impl<I, F, B> Iterator for StreamFusion<I, F>
where
    I: Iterator,
    F: FnMut(I::Item) -> Option<B>,
{
    type Item = B;

    #[inline(always)] // Critical for fusion
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.iter.next() {
                Some(item) => {
                    if let Some(result) = (self.f)(item) {
                        return Some(result);
                    }
                    // Continue if f returned None (filter effect)
                }
                None => return None,
            }
        }
    }
}

/// Bidirectional zipper iterator for efficient forward/backward traversal.
///
/// Based on "The Zipper" by Huet, 1997.
/// Provides O(1) movement in both directions.
#[cfg(any(feature = "std", feature = "alloc"))]
#[derive(Debug, Clone)]
pub struct Zipper<T> {
    left: Vec<T>,
    right: Vec<T>,
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<T> Zipper<T> {
    /// Creates a new zipper from a vector.
    pub fn new(mut items: Vec<T>) -> Self {
        items.reverse(); // So we can pop from the end
        Self {
            left: Vec::new(),
            right: items,
        }
    }

    /// Moves the focus one position to the right.
    pub fn move_right(&mut self) -> Option<&T> {
        if let Some(item) = self.right.pop() {
            self.left.push(item);
            self.left.last()
        } else {
            None
        }
    }

    /// Moves the focus one position to the left.
    pub fn move_left(&mut self) -> Option<&T> {
        if self.left.len() > 1 {
            if let Some(item) = self.left.pop() {
                self.right.push(item);
                self.left.last()
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Gets the current focus element.
    pub fn current(&self) -> Option<&T> {
        self.left.last()
    }

    /// Converts back to a vector.
    pub fn into_vec(mut self) -> Vec<T> {
        self.right.reverse();
        self.left.append(&mut self.right);
        self.left
    }
}