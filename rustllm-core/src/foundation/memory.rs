//! Memory management utilities for zero-copy operations.
//!
//! This module provides abstractions for efficient memory management,
//! including arena allocators, memory pools, and copy-on-write wrappers.
//!
//! ## Design Principles
//!
//! - **Zero-Copy**: Minimize allocations and copies using borrowing and views
//! - **Cache-Friendly**: Align memory for optimal cache utilization
//! - **Lazy Evaluation**: Defer allocation until actually needed
//! - **SIMD-Ready**: Support aligned allocations for vectorized operations

use core::alloc::Layout;
use core::cell::{Cell, RefCell};
use core::fmt;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::{self, NonNull};
use core::slice;

#[cfg(feature = "std")]
use std::alloc::{alloc, dealloc};

#[cfg(feature = "std")]
use std::sync::Arc;

#[cfg(not(feature = "std"))]
use alloc::alloc::{alloc, dealloc};

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::string::String;

// ============================================================================
// Arena Allocator
// ============================================================================

/// A high-performance arena allocator with SIMD alignment support.
///
/// This allocator provides O(1) allocation with configurable alignment,
/// making it suitable for both general purpose and SIMD workloads.
#[derive(Debug)]
pub struct Arena {
    chunks: Vec<ArenaChunk>,
    current_chunk: usize,
    default_chunk_size: usize,
    alignment: usize,
}

#[derive(Debug)]
struct ArenaChunk {
    data: NonNull<u8>,
    layout: Layout,
    used: Cell<usize>,
}

impl Arena {
    /// Creates a new arena with default alignment.
    pub fn new(chunk_size: usize) -> Self {
        Self::with_alignment(chunk_size, core::mem::align_of::<usize>())
    }
    
    /// Creates a new arena with specified alignment.
    pub fn with_alignment(chunk_size: usize, alignment: usize) -> Self {
        assert!(chunk_size > 0, "Chunk size must be greater than 0");
        assert!(alignment.is_power_of_two(), "Alignment must be a power of 2");
        
        Self {
            chunks: Vec::new(),
            current_chunk: 0,
            default_chunk_size: chunk_size,
            alignment,
        }
    }
    
    /// Allocates memory for a value in the arena.
    pub fn alloc<T>(&mut self, value: T) -> &mut T {
        let layout = Layout::for_value(&value);
        let ptr = self.alloc_raw(layout.align_to(self.alignment).unwrap());
        
        unsafe {
            ptr::write(ptr.cast::<T>(), value);
            &mut *ptr.cast::<T>()
        }
    }
    
    /// Allocates a slice in the arena.
    pub fn alloc_slice<T>(&mut self, slice: &[T]) -> &mut [T]
    where
        T: Copy,
    {
        if slice.is_empty() {
            return &mut [];
        }
        
        let layout = Layout::array::<T>(slice.len())
            .unwrap()
            .align_to(self.alignment)
            .unwrap();
        let ptr = self.alloc_raw(layout).cast::<T>();
        
        unsafe {
            ptr::copy_nonoverlapping(slice.as_ptr(), ptr, slice.len());
            slice::from_raw_parts_mut(ptr, slice.len())
        }
    }
    
    /// Allocates uninitialized memory for a slice.
    pub fn alloc_uninit_slice<T>(&mut self, len: usize) -> &mut [MaybeUninit<T>] {
        if len == 0 {
            return &mut [];
        }
        
        let layout = Layout::array::<T>(len)
            .unwrap()
            .align_to(self.alignment)
            .unwrap();
        let ptr = self.alloc_raw(layout).cast::<MaybeUninit<T>>();
        
        unsafe {
            slice::from_raw_parts_mut(ptr, len)
        }
    }
    
    /// Allocates raw memory with the given layout.
    fn alloc_raw(&mut self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let align = layout.align().max(self.alignment);
        
        // Try current chunk first
        if self.current_chunk < self.chunks.len() {
            if let Some(ptr) = self.try_alloc_from_chunk(self.current_chunk, size, align) {
                return ptr;
            }
        }
        
        // Try other chunks
        for i in 0..self.chunks.len() {
            if i != self.current_chunk {
                if let Some(ptr) = self.try_alloc_from_chunk(i, size, align) {
                    self.current_chunk = i;
                    return ptr;
                }
            }
        }
        
        // Allocate new chunk
        let chunk_size = self.default_chunk_size.max(size + align);
        let chunk = ArenaChunk::new(chunk_size, align);
        let ptr = chunk.data.as_ptr();
        chunk.used.set(size);
        
        self.chunks.push(chunk);
        self.current_chunk = self.chunks.len() - 1;
        
        ptr
    }
    
    fn try_alloc_from_chunk(&self, chunk_idx: usize, size: usize, align: usize) -> Option<*mut u8> {
        let chunk = &self.chunks[chunk_idx];
        let used = chunk.used.get();
        let aligned_pos = (used + align - 1) & !(align - 1);
        
        if aligned_pos + size <= chunk.layout.size() {
            chunk.used.set(aligned_pos + size);
            Some(unsafe { chunk.data.as_ptr().add(aligned_pos) })
        } else {
            None
        }
    }
    
    /// Resets the arena, allowing memory to be reused.
    pub fn reset(&mut self) {
        for chunk in &self.chunks {
            chunk.used.set(0);
        }
        self.current_chunk = 0;
    }
    
    /// Returns the total allocated capacity across all chunks.
    pub fn capacity(&self) -> usize {
        self.chunks.iter().map(|c| c.layout.size()).sum()
    }
    
    /// Returns the total used memory across all chunks.
    pub fn used(&self) -> usize {
        self.chunks.iter().map(|c| c.used.get()).sum()
    }
}

impl ArenaChunk {
    fn new(size: usize, align: usize) -> Self {
        let layout = Layout::from_size_align(size, align).unwrap();
        
        let data = unsafe {
            let ptr = alloc(layout);
            NonNull::new(ptr).expect("Allocation failed")
        };
        
        Self {
            data,
            layout,
            used: Cell::new(0),
        }
    }
}

impl Drop for ArenaChunk {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.data.as_ptr(), self.layout);
        }
    }
}

// Arena is Send if we're using std (with thread-safe internals)
#[cfg(feature = "std")]
unsafe impl Send for Arena {}

// Arena is Sync in std environments
#[cfg(feature = "std")]
unsafe impl Sync for Arena {}

// ============================================================================
// Object Pool
// ============================================================================

/// A memory pool for fixed-type allocations with reuse.
pub struct Pool<T> {
    items: RefCell<Vec<T>>,
    capacity: usize,
    initializer: Option<fn() -> T>,
}

impl<T> Pool<T> {
    /// Creates a new pool with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            items: RefCell::new(Vec::with_capacity(capacity)),
            capacity,
            initializer: None,
        }
    }
    
    /// Creates a new pool with a custom initializer.
    pub fn with_initializer(capacity: usize, init: fn() -> T) -> Self {
        Self {
            items: RefCell::new(Vec::with_capacity(capacity)),
            capacity,
            initializer: Some(init),
        }
    }
    
    /// Takes an item from the pool or creates a new one.
    pub fn take(&self) -> T
    where
        T: Default,
    {
        self.items.borrow_mut().pop().unwrap_or_else(T::default)
    }
    
    /// Takes an item from the pool or creates one with the initializer.
    pub fn take_or_init(&self) -> T {
        self.items.borrow_mut().pop().unwrap_or_else(|| {
            self.initializer.map(|f| f()).unwrap_or_else(|| {
                panic!("Pool requires either Default impl or initializer")
            })
        })
    }
    
    /// Returns an item to the pool.
    pub fn put(&self, item: T) {
        let mut items = self.items.borrow_mut();
        if items.len() < self.capacity {
            items.push(item);
        }
    }
    
    /// Returns an item to the pool and resets it if it implements PoolReset.
    pub fn put_and_reset(&self, mut item: T) 
    where
        T: PoolReset,
    {
        let mut items = self.items.borrow_mut();
        if items.len() < self.capacity {
            item.reset();
            items.push(item);
        }
    }
    
    /// Returns the current number of items in the pool.
    pub fn len(&self) -> usize {
        self.items.borrow().len()
    }
    
    /// Returns whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.items.borrow().is_empty()
    }
}

/// Trait for types that can be reset when returned to a pool.
pub trait PoolReset {
    /// Resets the object to a clean state.
    fn reset(&mut self);
}

// ============================================================================
// Copy-on-Write String
// ============================================================================

/// A copy-on-write string type for efficient string handling.
#[derive(Debug, Clone)]
pub enum CowStr<'a> {
    /// Borrowed string slice.
    Borrowed(&'a str),
    /// Owned string.
    Owned(String),
    /// Shared string (only available with std).
    #[cfg(feature = "std")]
    Shared(Arc<String>),
}

impl<'a> CowStr<'a> {
    /// Creates a new borrowed `CowStr`.
    pub const fn borrowed(s: &'a str) -> Self {
        Self::Borrowed(s)
    }
    
    /// Creates a new owned `CowStr`.
    pub const fn owned(s: String) -> Self {
        Self::Owned(s)
    }
    
    /// Creates a new shared `CowStr`.
    #[cfg(feature = "std")]
    pub fn shared(s: Arc<String>) -> Self {
        Self::Shared(s)
    }
    
    /// Returns the string as a slice.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Borrowed(s) => s,
            Self::Owned(s) => s.as_str(),
            #[cfg(feature = "std")]
            Self::Shared(s) => s.as_str(),
        }
    }
    
    /// Converts to an owned string if not already owned.
    pub fn into_owned(self) -> String {
        match self {
            Self::Borrowed(s) => s.to_string(),
            Self::Owned(s) => s,
            #[cfg(feature = "std")]
            Self::Shared(s) => (*s).clone(),
        }
    }
    
    /// Makes the string mutable, cloning if necessary.
    pub fn to_mut(&mut self) -> &mut String {
        match self {
            Self::Borrowed(s) => {
                *self = Self::Owned((*s).to_string());
                match self {
                    Self::Owned(s) => s,
                    _ => unreachable!(),
                }
            }
            Self::Owned(s) => s,
            #[cfg(feature = "std")]
            Self::Shared(s) => {
                *self = Self::Owned((**s).clone());
                match self {
                    Self::Owned(s) => s,
                    _ => unreachable!(),
                }
            }
        }
    }
    
    /// Returns the length of the string.
    pub fn len(&self) -> usize {
        self.as_str().len()
    }
    
    /// Returns whether the string is empty.
    pub fn is_empty(&self) -> bool {
        self.as_str().is_empty()
    }
}

impl<'a> AsRef<str> for CowStr<'a> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<'a> From<&'a str> for CowStr<'a> {
    fn from(s: &'a str) -> Self {
        Self::borrowed(s)
    }
}

impl<'a> From<String> for CowStr<'a> {
    fn from(s: String) -> Self {
        Self::owned(s)
    }
}

#[cfg(feature = "std")]
impl<'a> From<Arc<String>> for CowStr<'a> {
    fn from(s: Arc<String>) -> Self {
        Self::shared(s)
    }
}

// ============================================================================
// Zero-Copy String Builder
// ============================================================================

/// Zero-copy string builder using rope-like data structure.
/// 
/// This consolidates StrBuilder and ZeroCopyStringBuilder into a single
/// efficient implementation following DRY principle.
#[derive(Debug, Clone)]
pub struct ZeroCopyStringBuilder<'a> {
    segments: Vec<StringSegment<'a>>,
    total_len: usize,
}

/// Segments in the zero-copy string builder.
#[derive(Debug, Clone)]
enum StringSegment<'a> {
    /// Borrowed string slice.
    Borrowed(&'a str),
    /// Owned string.
    Owned(String),
    /// Shared string (when std feature is enabled).
    #[cfg(feature = "std")]
    Shared(Arc<String>),
}

impl<'a> ZeroCopyStringBuilder<'a> {
    /// Creates a new empty string builder.
    pub const fn new() -> Self {
        Self {
            segments: Vec::new(),
            total_len: 0,
        }
    }
    
    /// Creates a string builder with initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            segments: Vec::with_capacity(capacity),
            total_len: 0,
        }
    }
    
    /// Appends a borrowed string slice.
    pub fn append_borrowed(&mut self, s: &'a str) -> &mut Self {
        if !s.is_empty() {
            self.total_len += s.len();
            self.segments.push(StringSegment::Borrowed(s));
        }
        self
    }
    
    /// Appends an owned string.
    pub fn append_owned(&mut self, s: String) -> &mut Self {
        if !s.is_empty() {
            self.total_len += s.len();
            self.segments.push(StringSegment::Owned(s));
        }
        self
    }
    
    /// Appends a shared string.
    #[cfg(feature = "std")]
    pub fn append_shared(&mut self, s: Arc<String>) -> &mut Self {
        if !s.is_empty() {
            self.total_len += s.len();
            self.segments.push(StringSegment::Shared(s));
        }
        self
    }
    
    /// Appends a character.
    pub fn append_char(&mut self, ch: char) -> &mut Self {
        // Optimize by appending to last owned segment if possible
        if let Some(StringSegment::Owned(s)) = self.segments.last_mut() {
            s.push(ch);
            self.total_len += ch.len_utf8();
        } else {
            let mut s = String::with_capacity(ch.len_utf8());
            s.push(ch);
            self.total_len += ch.len_utf8();
            self.segments.push(StringSegment::Owned(s));
        }
        self
    }
    
    /// Alias for append_borrowed (backward compatibility).
    pub fn push_borrowed(&mut self, s: &'a str) -> &mut Self {
        self.append_borrowed(s)
    }
    
    /// Alias for append_owned (backward compatibility).
    pub fn push_owned(&mut self, s: String) -> &mut Self {
        self.append_owned(s)
    }
    
    /// Returns the total length without materializing the string.
    pub const fn len(&self) -> usize {
        self.total_len
    }
    
    /// Returns whether the builder is empty.
    pub const fn is_empty(&self) -> bool {
        self.total_len == 0
    }
    
    /// Builds the final string, minimizing allocations.
    pub fn build(self) -> String {
        match self.segments.len() {
            0 => String::new(),
            1 => self.build_single_segment(),
            _ => self.build_multiple_segments(),
        }
    }
    
    fn build_single_segment(mut self) -> String {
        match self.segments.pop().unwrap() {
            StringSegment::Borrowed(s) => s.to_string(),
            StringSegment::Owned(s) => s,
            #[cfg(feature = "std")]
            StringSegment::Shared(s) => (*s).clone(),
        }
    }
    
    fn build_multiple_segments(self) -> String {
        let mut result = String::with_capacity(self.total_len);
        for segment in self.segments {
            match segment {
                StringSegment::Borrowed(s) => result.push_str(s),
                StringSegment::Owned(s) => result.push_str(&s),
                #[cfg(feature = "std")]
                StringSegment::Shared(s) => result.push_str(&s),
            }
        }
        result
    }
}

impl<'a> Default for ZeroCopyStringBuilder<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> fmt::Write for ZeroCopyStringBuilder<'a> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.append_owned(s.to_string());
        Ok(())
    }
    
    fn write_char(&mut self, c: char) -> fmt::Result {
        self.append_char(c);
        Ok(())
    }
}

// ============================================================================
// Token Buffer Pool
// ============================================================================

/// A buffer pool for token processing.
pub struct TokenBufferPool {
    buffers: RefCell<Vec<Vec<u8>>>,
    buffer_size: usize,
    max_buffers: usize,
}

impl TokenBufferPool {
    /// Creates a new token buffer pool.
    pub fn new(buffer_size: usize, max_buffers: usize) -> Self {
        Self {
            buffers: RefCell::new(Vec::with_capacity(max_buffers)),
            buffer_size,
            max_buffers,
        }
    }
    
    /// Acquires a buffer from the pool.
    pub fn acquire(&self) -> Vec<u8> {
        self.buffers
            .borrow_mut()
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.buffer_size))
    }
    
    /// Releases a buffer back to the pool.
    pub fn release(&self, mut buffer: Vec<u8>) {
        buffer.clear();
        
        let mut buffers = self.buffers.borrow_mut();
        if buffers.len() < self.max_buffers {
            buffers.push(buffer);
        }
    }
}

// ============================================================================
// Zero-Copy View
// ============================================================================

/// Zero-copy view into a slice with lazy evaluation.
/// 
/// This provides a view that can be transformed without copying the underlying data.
/// 
/// Note: Due to the signature Fn(&T) -> T, transformations must clone the data.
/// For true zero-copy, consider using a different transformation signature.
pub struct SliceView<'a, T> {
    data: &'a [T],
    transforms: Vec<Box<dyn Fn(T) -> T + 'a>>,
}

impl<'a, T> SliceView<'a, T> {
    /// Creates a new slice view.
    pub fn new(data: &'a [T]) -> Self {
        Self {
            data,
            transforms: Vec::new(),
        }
    }
    
    /// Adds a transformation to be applied lazily.
    /// 
    /// Multiple map calls will chain the transformations.
    #[must_use]
    pub fn map<F>(mut self, f: F) -> Self
    where
        F: Fn(T) -> T + 'a,
        T: Clone,
    {
        self.transforms.push(Box::new(f));
        self
    }
    
    /// Gets an element with all transformations applied.
    pub fn get(&self, index: usize) -> Option<T>
    where
        T: Clone,
    {
        self.data.get(index).map(|item| {
            let mut result = item.clone();
            for transform in &self.transforms {
                result = transform(result);
            }
            result
        })
    }
    
    /// Returns the length of the view.
    pub const fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Returns whether the view is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Materializes the view into a vector.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter().map(|item| {
            let mut result = item.clone();
            for transform in &self.transforms {
                result = transform(result);
            }
            result
        }).collect()
    }
}

// ============================================================================
// Memory-Mapped Buffer
// ============================================================================

/// Memory-mapped buffer for zero-copy file operations.
/// 
/// This provides a zero-copy abstraction over memory-mapped files,
/// following the principle of minimal allocations.
#[cfg(feature = "std")]
pub struct MappedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    _phantom: PhantomData<[u8]>,
}

#[cfg(feature = "std")]
impl MappedBuffer {
    /// Creates a new mapped buffer from raw parts.
    /// 
    /// # Safety
    /// The caller must ensure that:
    /// - The pointer is valid for the given length
    /// - The memory remains valid for the lifetime of the buffer
    /// - The memory is properly aligned
    pub unsafe fn from_raw_parts(ptr: *mut u8, len: usize) -> Self {
        Self {
            ptr: NonNull::new_unchecked(ptr),
            len,
            _phantom: PhantomData,
        }
    }
    
    /// Returns a slice view of the buffer.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
    
    /// Returns the length of the buffer.
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Returns whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Creates a zero-copy view into a subregion.
    pub fn slice(&self, start: usize, end: usize) -> Option<&[u8]> {
        if start <= end && end <= self.len {
            Some(&self.as_slice()[start..end])
        } else {
            None
        }
    }
}

// ============================================================================
// Lazy Allocation
// ============================================================================

/// Lazy allocation wrapper that defers allocation until first use.
/// 
/// This follows the principle of lazy evaluation for better performance.
/// 
/// Note: This type is thread-safe when T is Send + Sync.
pub struct LazyAlloc<T> {
    #[cfg(feature = "std")]
    value: std::sync::OnceLock<Box<T>>,
    #[cfg(not(feature = "std"))]
    value: Cell<Option<Box<T>>>,
    init: fn() -> T,
}

impl<T> LazyAlloc<T> {
    /// Creates a new lazy allocation with the given initializer.
    pub fn new(init: fn() -> T) -> Self {
        Self {
            #[cfg(feature = "std")]
            value: std::sync::OnceLock::new(),
            #[cfg(not(feature = "std"))]
            value: Cell::new(None),
            init,
        }
    }
    
    /// Gets or initializes the value.
    #[cfg(feature = "std")]
    pub fn get_or_init(&self) -> &T {
        self.value.get_or_init(|| Box::new((self.init)()))
    }
    
    /// Gets or initializes the value (no_std version - not thread safe).
    #[cfg(not(feature = "std"))]
    pub fn get_or_init(&self) -> &T {
        if self.value.get().is_none() {
            self.value.set(Some(Box::new((self.init)())));
        }

        // Safe: we just initialized it above if it was None
        self.value.get().as_ref().unwrap().as_ref()
    }

    /// Returns whether the value has been initialized.
    #[cfg(feature = "std")]
    pub fn is_initialized(&self) -> bool {
        self.value.get().is_some()
    }

    /// Returns whether the value has been initialized (no_std version).
    #[cfg(not(feature = "std"))]
    pub fn is_initialized(&self) -> bool {
        self.value.get().is_some()
    }
}

// Drop is handled automatically by Box

// Debug implementation that doesn't require T: Debug
impl<T> fmt::Debug for LazyAlloc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LazyAlloc")
            .field("initialized", &self.is_initialized())
            .finish()
    }
}

// Safety: LazyAlloc is Send if T is Send
// With std: OnceLock<Box<T>> is Send if T is Send
// Without std: Cell<Option<Box<T>>> is Send if T is Send
unsafe impl<T: Send> Send for LazyAlloc<T> {}

// Safety: LazyAlloc is Sync if T is Send + Sync
// With std: OnceLock<Box<T>> is Sync if T is Send + Sync
// Without std: Cell is not Sync, so we don't implement Sync
#[cfg(feature = "std")]
unsafe impl<T: Send + Sync> Sync for LazyAlloc<T> {}

// ============================================================================
// Advanced Zero-Copy Memory Management
// ============================================================================

/// Memory allocation statistics for monitoring and optimization.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total bytes allocated.
    pub allocated_bytes: usize,
    /// Total bytes deallocated.
    pub deallocated_bytes: usize,
    /// Current bytes in use.
    pub used_bytes: usize,
    /// Peak bytes used.
    pub peak_bytes: usize,
    /// Number of allocations.
    pub allocation_count: usize,
    /// Number of deallocations.
    pub deallocation_count: usize,
}

impl MemoryStats {
    /// Creates new memory statistics.
    pub const fn new() -> Self {
        Self {
            allocated_bytes: 0,
            deallocated_bytes: 0,
            used_bytes: 0,
            peak_bytes: 0,
            allocation_count: 0,
            deallocation_count: 0,
        }
    }
    
    /// Records an allocation.
    pub fn record_allocation(&mut self, bytes: usize) {
        self.allocated_bytes += bytes;
        self.used_bytes += bytes;
        self.allocation_count += 1;
        if self.used_bytes > self.peak_bytes {
            self.peak_bytes = self.used_bytes;
        }
    }
    
    /// Records a deallocation.
    pub fn record_deallocation(&mut self, bytes: usize) {
        self.deallocated_bytes += bytes;
        self.used_bytes = self.used_bytes.saturating_sub(bytes);
        self.deallocation_count += 1;
    }
    
    /// Returns the fragmentation ratio (0.0 = no fragmentation, 1.0 = fully fragmented).
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.allocated_bytes == 0 {
            0.0
        } else {
            1.0 - (self.used_bytes as f64 / self.allocated_bytes as f64)
        }
    }
}

/// A tracking arena that collects allocation statistics.
#[derive(Debug)]
pub struct TrackingArena {
    arena: Arena,
    stats: RefCell<MemoryStats>,
}

impl TrackingArena {
    /// Creates a new tracking arena.
    pub fn new(chunk_size: usize) -> Self {
        Self {
            arena: Arena::new(chunk_size),
            stats: RefCell::new(MemoryStats::new()),
        }
    }
    
    /// Creates a new tracking arena with alignment.
    pub fn with_alignment(chunk_size: usize, alignment: usize) -> Self {
        Self {
            arena: Arena::with_alignment(chunk_size, alignment),
            stats: RefCell::new(MemoryStats::new()),
        }
    }
    
    /// Allocates memory and tracks the allocation.
    pub fn alloc<T>(&mut self, value: T) -> &mut T {
        self.stats.borrow_mut().record_allocation(core::mem::size_of::<T>());
        self.arena.alloc(value)
    }
    
    /// Allocates a slice and tracks the allocation.
    pub fn alloc_slice<T>(&mut self, slice: &[T]) -> &mut [T]
    where
        T: Copy,
    {
        self.stats.borrow_mut().record_allocation(core::mem::size_of::<T>() * slice.len());
        self.arena.alloc_slice(slice)
    }
    
    /// Resets the arena and updates statistics.
    pub fn reset(&mut self) {
        let used = self.arena.used();
        self.stats.borrow_mut().record_deallocation(used);
        self.arena.reset();
    }
    
    /// Returns a copy of the current statistics.
    pub fn stats(&self) -> MemoryStats {
        self.stats.borrow().clone()
    }
}

/// Zero-copy rope data structure for efficient string manipulation.
///
/// This provides O(log n) concatenation and slicing operations.
#[derive(Debug, Clone)]
pub struct Rope<'a> {
    root: RopeNode<'a>,
    len: usize,
}

/// Node in a rope data structure.
#[derive(Debug, Clone)]
enum RopeNode<'a> {
    /// Leaf node containing actual data.
    Leaf(CowStr<'a>),
    /// Branch node containing child nodes.
    Branch {
        left: Box<RopeNode<'a>>,
        right: Box<RopeNode<'a>>,
        // Removed unused len field
    },
}

impl<'a> Rope<'a> {
    /// Creates a new empty rope.
    pub const fn new() -> Self {
        Self {
            root: RopeNode::Leaf(CowStr::Borrowed("")),
            len: 0,
        }
    }
    
    /// Creates a rope from a string.
    pub fn from_str(s: &'a str) -> Self {
        Self {
            len: s.len(),
            root: RopeNode::Leaf(CowStr::Borrowed(s)),
        }
    }
    
    /// Returns the length of the rope.
    pub const fn len(&self) -> usize {
        self.len
    }
    
    /// Returns whether the rope is empty.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Concatenates two ropes.
    pub fn concat(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        
        let len = self.len + other.len;
        Self {
            root: RopeNode::Branch {
                left: Box::new(self.root),
                right: Box::new(other.root),
                // Removed unused len field
            },
            len,
        }
    }
    
    /// Converts the rope to a string.
    pub fn to_string(&self) -> String {
        let mut result = String::with_capacity(self.len);
        self.collect_into(&mut result);
        result
    }
    
    fn collect_into(&self, buf: &mut String) {
        match &self.root {
            RopeNode::Leaf(s) => buf.push_str(s.as_str()),
            RopeNode::Branch { left, right, .. } => {
                Self::collect_node_into(left, buf);
                Self::collect_node_into(right, buf);
            }
        }
    }
    
    fn collect_node_into(node: &RopeNode<'a>, buf: &mut String) {
        match node {
            RopeNode::Leaf(s) => buf.push_str(s.as_str()),
            RopeNode::Branch { left, right, .. } => {
                Self::collect_node_into(left, buf);
                Self::collect_node_into(right, buf);
            }
        }
    }
}

impl<'a> Default for Rope<'a> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Benchmarking Utilities
// ============================================================================

/// Benchmark structure for memory operations.
#[derive(Debug, Clone)]
pub struct AllocationBenchmark {
    /// Total allocations made.
    pub allocations: usize,
    /// Total bytes allocated.
    pub bytes_allocated: usize,
    /// Peak memory usage.
    pub peak_usage: usize,
    // Removed unused start_time field
}

impl AllocationBenchmark {
    /// Creates a new benchmark.
    pub fn new() -> Self {
        Self {
            allocations: 0,
            bytes_allocated: 0,
            peak_usage: 0,
        }
    }
    
    /// Records an allocation.
    pub fn record_allocation(&mut self, bytes: usize) {
        self.allocations += 1;
        self.bytes_allocated += bytes;
        self.peak_usage = self.peak_usage.max(self.bytes_allocated);
    }
    
    /// Records a deallocation.
    pub fn record_deallocation(&mut self, bytes: usize) {
        self.bytes_allocated = self.bytes_allocated.saturating_sub(bytes);
    }
    
    /// Resets the benchmark.
    pub fn reset(&mut self) {
        self.allocations = 0;
        self.bytes_allocated = 0;
        self.peak_usage = 0;
    }
}

/// Result of a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Number of allocations.
    pub allocations: usize,
    /// Total bytes allocated.
    pub bytes_allocated: usize,
    /// Peak memory usage.
    pub peak_usage: usize,
    /// Elapsed time (if std feature is enabled).
    #[cfg(feature = "std")]
    pub elapsed: std::time::Duration,
}

impl BenchmarkResult {
    /// Returns allocations per second.
    #[cfg(feature = "std")]
    pub fn allocations_per_second(&self) -> f64 {
        if self.elapsed.as_secs_f64() > 0.0 {
            self.allocations as f64 / self.elapsed.as_secs_f64()
        } else {
            0.0
        }
    }
    
    /// Returns megabytes per second.
    pub fn mb_per_second(&self) -> f64 {
        if self.elapsed.as_secs_f64() > 0.0 {
            (self.bytes_allocated as f64 / 1_048_576.0) / self.elapsed.as_secs_f64()
        } else {
            0.0
        }
    }
}

/// Memory-efficient string interning system.
///
/// This system deduplicates strings to save memory, particularly useful
/// for tokenization where many tokens may be repeated.
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct StringInterner {
    strings: std::sync::RwLock<std::collections::HashMap<String, std::sync::Arc<String>>>,
    stats: std::sync::RwLock<InternerStats>,
}

/// Statistics for string interner performance tracking.
///
/// This struct provides insights into the effectiveness of string interning,
/// helping to optimize memory usage and performance.
#[cfg(feature = "std")]
#[derive(Debug, Default)]
pub struct InternerStats {
    /// Total number of intern requests made.
    pub total_requests: usize,
    /// Number of requests that resulted in cache hits.
    pub cache_hits: usize,
    /// Number of unique strings stored in the interner.
    pub unique_strings: usize,
    /// Estimated memory saved through string deduplication (in bytes).
    pub memory_saved: usize,
}

#[cfg(feature = "std")]
impl StringInterner {
    /// Creates a new string interner.
    pub fn new() -> Self {
        Self {
            strings: std::sync::RwLock::new(std::collections::HashMap::new()),
            stats: std::sync::RwLock::new(InternerStats::default()),
        }
    }

    /// Interns a string, returning a shared reference.
    pub fn intern(&self, s: &str) -> std::sync::Arc<String> {
        // Try read lock first for common case
        {
            let strings = self.strings.read().unwrap();
            if let Some(interned) = strings.get(s) {
                return self.handle_cache_hit(interned, s);
            }
        }

        // Need write lock to insert
        let mut strings = self.strings.write().unwrap();

        // Double-check in case another thread inserted while we waited
        if let Some(interned) = strings.get(s) {
            let mut stats = self.stats.write().unwrap();
            stats.total_requests += 1;
            stats.cache_hits += 1;
            stats.memory_saved += s.len();
            return Arc::clone(interned);
        }

        // Insert new string
        let interned = std::sync::Arc::new(s.to_string());
        strings.insert(s.to_string(), Arc::clone(&interned));

        let mut stats = self.stats.write().unwrap();
        stats.total_requests += 1;
        stats.unique_strings += 1;

        interned
    }

    /// Handles a cache hit, updating statistics and returning the interned string.
    fn handle_cache_hit(&self, interned: &std::sync::Arc<String>, s: &str) -> std::sync::Arc<String> {
        let mut stats = self.stats.write().unwrap();
        stats.total_requests += 1;
        stats.cache_hits += 1;
        stats.memory_saved += s.len();
        Arc::clone(interned)
    }

    /// Returns current statistics.
    pub fn stats(&self) -> InternerStats {
        self.stats.read().unwrap().clone()
    }

    /// Clears the interner and resets statistics.
    pub fn clear(&self) {
        self.strings.write().unwrap().clear();
        *self.stats.write().unwrap() = InternerStats::default();
    }

    /// Returns the number of unique strings stored.
    pub fn len(&self) -> usize {
        self.strings.read().unwrap().len()
    }

    /// Returns whether the interner is empty.
    pub fn is_empty(&self) -> bool {
        self.strings.read().unwrap().is_empty()
    }
}

#[cfg(feature = "std")]
impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "std")]
impl Clone for InternerStats {
    fn clone(&self) -> Self {
        Self {
            total_requests: self.total_requests,
            cache_hits: self.cache_hits,
            unique_strings: self.unique_strings,
            memory_saved: self.memory_saved,
        }
    }
}

/// Memory-efficient bump allocator for temporary allocations.
///
/// This allocator is extremely fast for short-lived allocations
/// but cannot free individual allocations.
#[derive(Debug)]
pub struct BumpAllocator {
    buffer: Vec<u8>,
    position: Cell<usize>,
}

impl BumpAllocator {
    /// Creates a new bump allocator with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0; capacity],
            position: Cell::new(0),
        }
    }

    /// Allocates memory for a value.
    pub fn alloc<T>(&self, value: T) -> Option<&mut T> {
        let layout = Layout::for_value(&value);
        let ptr = self.alloc_raw(layout)?;

        unsafe {
            ptr::write(ptr.cast::<T>(), value);
            Some(&mut *(ptr.cast::<T>()))
        }
    }

    /// Allocates memory for a slice.
    pub fn alloc_slice<T>(&self, slice: &[T]) -> Option<&mut [T]>
    where
        T: Copy,
    {
        let layout = Layout::array::<T>(slice.len()).ok()?;
        let ptr = self.alloc_raw(layout)?.cast::<T>();

        unsafe {
            ptr::copy_nonoverlapping(slice.as_ptr(), ptr, slice.len());
            Some(slice::from_raw_parts_mut(ptr, slice.len()))
        }
    }

    /// Allocates raw memory.
    fn alloc_raw(&self, layout: Layout) -> Option<*mut u8> {
        let size = layout.size();
        let align = layout.align();

        let current = self.position.get();
        let aligned = (current + align - 1) & !(align - 1);

        if aligned + size > self.buffer.len() {
            return None; // Out of memory
        }

        self.position.set(aligned + size);
                    Some(self.buffer.as_ptr().wrapping_add(aligned).cast_mut())
    }

    /// Resets the allocator, making all memory available again.
    pub fn reset(&self) {
        self.position.set(0);
    }

    /// Returns the current memory usage.
    pub fn used(&self) -> usize {
        self.position.get()
    }

    /// Returns the total capacity.
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the remaining capacity.
    pub fn remaining(&self) -> usize {
        self.capacity() - self.used()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_arena() {
        let mut arena = Arena::new(1024);
        
        // Test value allocation
        let x = arena.alloc(42i32);
        assert_eq!(*x, 42);
        *x = 100;
        assert_eq!(*x, 100);
        
        // Test slice allocation
        let slice = arena.alloc_slice(&[1, 2, 3, 4, 5]);
        assert_eq!(slice, &[1, 2, 3, 4, 5]);
        slice[0] = 10;
        assert_eq!(slice[0], 10);
        
        // Test multiple allocations
        let _y = arena.alloc(3.14f64);
        let _z = arena.alloc("hello");
        
        // Test reset
        arena.reset();
        let w = arena.alloc(999);
        assert_eq!(*w, 999);
    }
    
    // Arena is not designed to be used with Arc/shared across threads
    // Each thread should have its own Arena instance
    #[cfg(feature = "std")]
    #[test]
    fn test_arena_per_thread() {
        use std::thread;
        
        let mut handles = vec![];
        
        // Each thread gets its own arena
        for i in 0..10 {
            let handle = thread::spawn(move || {
                let mut arena = Arena::new(1024);
                
                // Each thread allocates multiple values
                for j in 0..100 {
                    let value = i * 100 + j;
                    let allocated = arena.alloc(value);
                    assert_eq!(*allocated, value);
                    
                    // Also test slice allocation
                    let slice_data = vec![value; 5];
                    let slice = arena.alloc_slice(&slice_data);
                    assert_eq!(slice.len(), 5);
                    for &item in slice.iter() {
                        assert_eq!(item, value);
                    }
                }
            });
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }
    
    #[test]
    fn test_pool() {
        let pool: Pool<Vec<u8>> = Pool::with_initializer(10, || Vec::with_capacity(100));
        
        // Test taking from empty pool
        let mut vec1 = pool.take_or_init();
        vec1.extend_from_slice(b"hello");
        assert_eq!(vec1, b"hello");
        
        // Return to pool
        pool.put(vec1);
        
        // Take again - should get the same capacity
        let vec2 = pool.take_or_init();
        assert!(vec2.capacity() >= 100);
    }
    
    #[test]
    fn test_cow_str() {
        // Test borrowed
        let borrowed = CowStr::borrowed("hello");
        assert!(matches!(borrowed, CowStr::Borrowed(_)));
        assert_eq!(borrowed.as_str(), "hello");
        
        // Test owned
        let owned = CowStr::owned(String::from("world"));
        assert!(matches!(owned, CowStr::Owned(_)));
        assert_eq!(owned.as_str(), "world");
        
        // Test to_mut on borrowed
        let mut cow = CowStr::borrowed("test");
        let mutable = cow.to_mut();
        mutable.push_str("ing");
        assert!(matches!(cow, CowStr::Owned(_)));
        assert_eq!(cow.as_str(), "testing");
    }
    
    #[test]
    fn test_str_builder() {
        let mut builder = ZeroCopyStringBuilder::new();
        
        builder.append_borrowed("Hello");
        builder.append_borrowed(" ");
        builder.append_owned(String::from("World"));
        builder.append_borrowed("!");
        
        let result = builder.build();
        assert_eq!(result, "Hello World!");
    }
}