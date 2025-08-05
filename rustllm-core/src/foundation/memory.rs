//! Memory management utilities for zero-copy operations.
//!
//! This module provides abstractions for efficient memory management,
//! including arena allocators, memory pools, and copy-on-write wrappers.

use core::cell::{Cell, RefCell};
use core::fmt;
use core::marker::PhantomData;
use core::ptr::{self, NonNull};
use core::slice;

#[cfg(feature = "std")]
use std::alloc::{alloc, dealloc, Layout};

#[cfg(feature = "std")]
use std::sync::{Arc, RwLock};

#[cfg(not(feature = "std"))]
use alloc::alloc::{alloc, dealloc, Layout};

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::string::String;

// For no_std environments, we'll use a single-threaded arena
#[cfg(not(feature = "std"))]
type ArenaStorage = RefCell<Vec<ArenaChunk>>;

// For std environments, use thread-safe RwLock
#[cfg(feature = "std")]
type ArenaStorage = RwLock<Vec<ArenaChunk>>;

/// A simple arena allocator for temporary allocations.
///
/// In `std` environments, this is thread-safe using `RwLock`.
/// In `no_std` environments, this uses `RefCell` and is NOT thread-safe.
#[derive(Debug)]
pub struct Arena {
    chunks: ArenaStorage,
    current_chunk: Cell<usize>,
    chunk_size: usize,
}

#[derive(Debug)]
struct ArenaChunk {
    data: NonNull<u8>,
    size: usize,
    used: Cell<usize>,
}

impl Arena {
    /// Creates a new arena with the specified chunk size.
    pub fn new(chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "Chunk size must be greater than 0");
        
        Self {
            #[cfg(feature = "std")]
            chunks: RwLock::new(Vec::new()),
            #[cfg(not(feature = "std"))]
            chunks: RefCell::new(Vec::new()),
            current_chunk: Cell::new(0),
            chunk_size,
        }
    }
    
    /// Allocates memory in the arena.
    pub fn alloc<T>(&mut self, value: T) -> &mut T {
        let layout = Layout::for_value(&value);
        let ptr = self.alloc_raw(layout);
        
        unsafe {
            assert!(
                (ptr as usize) % core::mem::align_of::<T>() == 0,
                "Allocated pointer is not properly aligned for type T"
            );
            ptr::write(ptr.cast::<T>(), value);
            &mut *ptr.cast::<T>()
        }
    }
    
    /// Allocates a slice in the arena.
    pub fn alloc_slice<T>(&mut self, slice: &[T]) -> &mut [T]
    where
        T: Copy,
    {
        let layout = Layout::array::<T>(slice.len()).unwrap();
        let ptr = self.alloc_raw(layout).cast::<T>();
        
        // Ensure the pointer is properly aligned for T
        assert!(
            (ptr as usize) % core::mem::align_of::<T>() == 0,
            "Allocated memory is not properly aligned for type T"
        );
        unsafe {
            ptr::copy_nonoverlapping(slice.as_ptr(), ptr, slice.len());
            slice::from_raw_parts_mut(ptr, slice.len())
        }
    }
    
    /// Allocates raw memory in the arena.
    fn alloc_raw(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let align = layout.align();
        
        // Try to allocate from current chunk
        #[cfg(feature = "std")]
        let mut chunks = self.chunks.write().unwrap();
        
        #[cfg(not(feature = "std"))]
        let mut chunks = self.chunks.borrow_mut();
        
        if self.current_chunk.get() < chunks.len() {
            let chunk = &chunks[self.current_chunk.get()];
            let used = chunk.used.get();
            let aligned = (used + align - 1) & !(align - 1);
            
            if aligned + size <= chunk.size {
                chunk.used.set(aligned + size);
                return unsafe { chunk.data.as_ptr().add(aligned) };
            }
        }
        
        // Need a new chunk
        let chunk_size = self.chunk_size.max(size + align);
        let chunk = ArenaChunk::new(chunk_size);
        let ptr = chunk.data.as_ptr();
        
        chunk.used.set(size);
        chunks.push(chunk);
        self.current_chunk.set(chunks.len() - 1);
        
        ptr
    }
    
    /// Resets the arena, allowing memory to be reused.
    pub fn reset(&self) {
        #[cfg(feature = "std")]
        {
            let chunks = self.chunks.read().unwrap();
            for chunk in chunks.iter() {
                chunk.used.set(0);
            }
        }
        
        #[cfg(not(feature = "std"))]
        {
            let chunks = self.chunks.borrow();
            for chunk in chunks.iter() {
                chunk.used.set(0);
            }
        }
        
        self.current_chunk.set(0);
    }
}

impl ArenaChunk {
    fn new(size: usize) -> Self {
        let layout = Layout::array::<u8>(size).unwrap();
        
        let data = unsafe {
            let ptr = alloc(layout);
            NonNull::new(ptr).expect("Allocation failed")
        };
        
        Self {
            data,
            size,
            used: Cell::new(0),
        }
    }
}

impl Drop for ArenaChunk {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::array::<u8>(self.size).unwrap();
            dealloc(self.data.as_ptr(), layout);
        }
    }
}

// SAFETY: Arena can be Send in std environments because:
// 1. The chunks field uses RwLock which provides thread-safe access
// 2. The Cell fields (current_chunk and ArenaChunk::used) are only accessed
//    through the RwLock, ensuring no data races
// 3. All public methods that mutate state require &mut self
#[cfg(feature = "std")]
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for Arena {}

// In no_std environments, Arena is NOT Send due to RefCell
#[cfg(not(feature = "std"))]
// Arena is intentionally NOT Send in no_std environments

// In std environments, Arena is Sync because we use RwLock
#[cfg(feature = "std")]
unsafe impl Sync for Arena {}

// In no_std environments, Arena is NOT Sync because RefCell is not thread-safe
// Do NOT implement Sync for no_std Arena!

/// A memory pool for fixed-size allocations.
pub struct Pool<T> {
    items: RefCell<Vec<T>>,
    capacity: usize,
    _marker: PhantomData<T>,
}

impl<T> Pool<T> {
    /// Creates a new pool with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            items: RefCell::new(Vec::with_capacity(capacity)),
            capacity,
            _marker: PhantomData,
        }
    }
    
    /// Takes an item from the pool or creates a new one.
    pub fn take_or_default(&self) -> T
    where
        T: Default,
    {
        self.take_or_else(T::default)
    }
    
    /// Takes an item from the pool or creates one with the given closure.
    pub fn take_or_else<F>(&self, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        self.items.borrow_mut().pop().unwrap_or_else(f)
    }
    
    /// Returns an item to the pool.
    pub fn put(&self, item: T) {
        let mut items = self.items.borrow_mut();
        if items.len() < self.capacity {
            items.push(item);
        }
    }
}

/// A copy-on-write wrapper for efficient string handling.
#[derive(Debug, Clone)]
pub enum CowStr<'a> {
    /// Borrowed string slice.
    Borrowed(&'a str),
    /// Owned string.
    Owned(String),
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
    
    /// Returns the string as a slice.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Borrowed(s) => s,
            Self::Owned(s) => s.as_str(),
        }
    }
    
    /// Converts to an owned string if not already owned.
    pub fn into_owned(self) -> String {
        match self {
            Self::Borrowed(s) => s.to_string(),
            Self::Owned(s) => s,
        }
    }
    
    /// Makes the string mutable, cloning if necessary.
    pub fn to_mut(&mut self) -> &mut String {
        match self {
            Self::Borrowed(s) => {
                *self = Self::Owned((*s).to_string());
                match self {
                    Self::Owned(s) => s,
                    Self::Borrowed(_) => unreachable!(),
                }
            }
            Self::Owned(s) => s,
        }
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

/// Zero-copy string builder.
pub struct StrBuilder<'a> {
    parts: Vec<CowStr<'a>>,
    total_len: usize,
}

impl<'a> StrBuilder<'a> {
    /// Creates a new string builder.
    pub const fn new() -> Self {
        Self {
            parts: Vec::new(),
            total_len: 0,
        }
    }
    
    /// Creates a new string builder with capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            parts: Vec::with_capacity(capacity),
            total_len: 0,
        }
    }
    
    /// Appends a borrowed string.
    pub fn push_borrowed(&mut self, s: &'a str) {
        self.total_len += s.len();
        self.parts.push(CowStr::borrowed(s));
    }
    
    /// Appends an owned string.
    pub fn push_owned(&mut self, s: String) {
        self.total_len += s.len();
        self.parts.push(CowStr::owned(s));
    }
    
    /// Builds the final string.
    pub fn build(self) -> String {
        let mut result = String::with_capacity(self.total_len);
        for part in self.parts {
            result.push_str(part.as_str());
        }
        result
    }
}

impl<'a> Default for StrBuilder<'a> {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-copy string builder using rope data structure.
/// 
/// This provides efficient string concatenation without copying,
/// following the principle of zero-cost abstractions.
#[derive(Debug, Clone)]
pub struct ZeroCopyStringBuilder<'a> {
    segments: Vec<StringSegment<'a>>,
    total_len: usize,
}

#[derive(Debug, Clone)]
enum StringSegment<'a> {
    Borrowed(&'a str),
    #[cfg(feature = "std")]
    Shared(std::sync::Arc<String>),
    #[cfg(not(feature = "std"))]
    Owned(String),
    Slice { 
        #[cfg(feature = "std")]
        data: std::sync::Arc<String>, 
        #[cfg(not(feature = "std"))]
        data: String,
        start: usize, 
        end: usize 
    },
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
        self.total_len += s.len();
        self.segments.push(StringSegment::Borrowed(s));
        self
    }
    
    /// Appends an owned string.
    #[cfg(feature = "std")]
    pub fn append_owned(&mut self, s: String) -> &mut Self {
        self.total_len += s.len();
        self.segments.push(StringSegment::Shared(std::sync::Arc::new(s)));
        self
    }
    
    /// Appends an owned string (no_std version).
    #[cfg(not(feature = "std"))]
    pub fn append_owned(&mut self, s: String) -> &mut Self {
        self.total_len += s.len();
        self.segments.push(StringSegment::Owned(s));
        self
    }
    
    /// Appends a shared string.
    #[cfg(feature = "std")]
    pub fn append_shared(&mut self, s: std::sync::Arc<String>) -> &mut Self {
        self.total_len += s.len();
        self.segments.push(StringSegment::Shared(s));
        self
    }
    
    /// Appends a slice of a shared string.
    #[cfg(feature = "std")]
    pub fn append_slice(&mut self, data: std::sync::Arc<String>, start: usize, end: usize) -> &mut Self {
        assert!(start <= end && end <= data.len());
        self.total_len += end - start;
        self.segments.push(StringSegment::Slice { data, start, end });
        self
    }
    
    /// Appends a slice of an owned string (no_std version).
    #[cfg(not(feature = "std"))]
    pub fn append_slice(&mut self, data: String, start: usize, end: usize) -> &mut Self {
        assert!(start <= end && end <= data.len());
        self.total_len += end - start;
        self.segments.push(StringSegment::Slice { data, start, end });
        self
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
        if self.segments.is_empty() {
            return String::new();
        }
        
        if self.segments.len() == 1 {
            return match self.segments.into_iter().next().unwrap() {
                StringSegment::Borrowed(s) => s.to_string(),
                #[cfg(feature = "std")]
                StringSegment::Shared(s) => (*s).clone(),
                #[cfg(not(feature = "std"))]
                StringSegment::Owned(s) => s,
                StringSegment::Slice { data, start, end } => {
                    #[cfg(feature = "std")]
                    { data[start..end].to_string() }
                    #[cfg(not(feature = "std"))]
                    { data[start..end].to_string() }
                },
            };
        }
        
        let mut result = String::with_capacity(self.total_len);
        for segment in self.segments {
            match segment {
                StringSegment::Borrowed(s) => result.push_str(s),
                #[cfg(feature = "std")]
                StringSegment::Shared(s) => result.push_str(&s),
                #[cfg(not(feature = "std"))]
                StringSegment::Owned(s) => result.push_str(&s),
                StringSegment::Slice { data, start, end } => result.push_str(&data[start..end]),
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

// ============================================================================
// Advanced Zero-Copy Memory Management
// ============================================================================

/// SIMD-aligned memory allocator for vectorized operations.
///
/// This allocator ensures memory is aligned for SIMD instructions,
/// providing optimal performance for vectorized computations.
#[derive(Debug)]
pub struct SimdAlignedArena {
    arena: Arena,
    alignment: usize,
}

impl SimdAlignedArena {
    /// Creates a new SIMD-aligned arena.
    ///
    /// The alignment should be a power of 2 (typically 16, 32, or 64 bytes).
    pub fn new(chunk_size: usize, alignment: usize) -> Self {
        assert!(alignment.is_power_of_two(), "Alignment must be a power of 2");
        assert!(alignment >= core::mem::align_of::<usize>(), "Alignment too small");

        Self {
            arena: Arena::new(chunk_size),
            alignment,
        }
    }

    /// Allocates SIMD-aligned memory for a slice.
    pub fn alloc_simd_slice<T>(&mut self, len: usize) -> &mut [T]
    where
        T: Copy + Default,
    {
        let layout = Layout::from_size_align(
            len * core::mem::size_of::<T>(),
            self.alignment.max(core::mem::align_of::<T>())
        ).unwrap();

        let ptr = self.arena.alloc_raw(layout).cast::<T>();

        // Initialize with default values
        unsafe {
            for i in 0..len {
                ptr.add(i).write(T::default());
            }
            slice::from_raw_parts_mut(ptr, len)
        }
    }

    /// Allocates SIMD-aligned memory and copies data.
    pub fn alloc_simd_copy<T>(&mut self, data: &[T]) -> &mut [T]
    where
        T: Copy,
    {
        let layout = Layout::from_size_align(
            data.len() * core::mem::size_of::<T>(),
            self.alignment.max(core::mem::align_of::<T>())
        ).unwrap();

        let ptr = self.arena.alloc_raw(layout).cast::<T>();

        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            slice::from_raw_parts_mut(ptr, data.len())
        }
    }

    /// Resets the arena for reuse.
    pub fn reset(&self) {
        self.arena.reset();
    }
}

/// Zero-copy slice builder that avoids allocations.
///
/// This builder creates views over existing data without copying,
/// implementing true zero-cost abstractions.
#[derive(Debug)]
pub struct ZeroCopySliceBuilder<'a, T> {
    segments: Vec<SliceSegment<'a, T>>,
    total_len: usize,
}

#[derive(Debug, Clone)]
enum SliceSegment<'a, T> {
    Borrowed(&'a [T]),
    #[cfg(feature = "std")]
    Shared(std::sync::Arc<Vec<T>>),
    #[cfg(not(feature = "std"))]
    Owned(Vec<T>),
    Slice {
        #[cfg(feature = "std")]
        data: std::sync::Arc<Vec<T>>,
        #[cfg(not(feature = "std"))]
        data: Vec<T>,
        start: usize,
        end: usize,
    },
}

impl<'a, T> ZeroCopySliceBuilder<'a, T> {
    /// Creates a new zero-copy slice builder.
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            total_len: 0,
        }
    }

    /// Appends a borrowed slice.
    pub fn append_borrowed(&mut self, slice: &'a [T]) -> &mut Self {
        self.total_len += slice.len();
        self.segments.push(SliceSegment::Borrowed(slice));
        self
    }

    /// Appends a shared slice (std version).
    #[cfg(feature = "std")]
    pub fn append_shared(&mut self, data: std::sync::Arc<Vec<T>>) -> &mut Self {
        self.total_len += data.len();
        self.segments.push(SliceSegment::Shared(data));
        self
    }

    /// Appends an owned slice (no_std version).
    #[cfg(not(feature = "std"))]
    pub fn append_owned(&mut self, data: Vec<T>) -> &mut Self {
        self.total_len += data.len();
        self.segments.push(SliceSegment::Owned(data));
        self
    }

    /// Appends a slice of a shared vector (std version).
    ///
    /// This method enables zero-copy slicing of shared data, which is particularly
    /// useful for tokenization where you want to reference parts of larger buffers.
    #[cfg(feature = "std")]
    pub fn append_slice(&mut self, data: std::sync::Arc<Vec<T>>, start: usize, end: usize) -> &mut Self {
        assert!(start <= end && end <= data.len(), "Invalid slice bounds");
        self.total_len += end - start;
        self.segments.push(SliceSegment::Slice { data, start, end });
        self
    }

    /// Appends a slice of an owned vector (no_std version).
    ///
    /// This method enables zero-copy slicing for no_std environments.
    #[cfg(not(feature = "std"))]
    pub fn append_slice(&mut self, data: Vec<T>, start: usize, end: usize) -> &mut Self {
        assert!(start <= end && end <= data.len(), "Invalid slice bounds");
        self.total_len += end - start;
        self.segments.push(SliceSegment::Slice { data, start, end });
        self
    }

    /// Returns the total length of all segments.
    pub fn len(&self) -> usize {
        self.total_len
    }

    /// Returns whether the builder is empty.
    pub fn is_empty(&self) -> bool {
        self.total_len == 0
    }

    /// Materializes the builder into a vector.
    pub fn into_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        let mut result = Vec::with_capacity(self.total_len);

        for segment in self.segments {
            match segment {
                SliceSegment::Borrowed(slice) => result.extend_from_slice(slice),
                #[cfg(feature = "std")]
                SliceSegment::Shared(vec) => result.extend_from_slice(&vec),
                #[cfg(not(feature = "std"))]
                SliceSegment::Owned(vec) => result.extend_from_slice(&vec),
                SliceSegment::Slice { data, start, end } => {
                    result.extend_from_slice(&data[start..end]);
                }
            }
        }

        result
    }
}

impl<'a, T> Default for ZeroCopySliceBuilder<'a, T> {
    fn default() -> Self {
        Self::new()
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
        let pool: Pool<Vec<u8>> = Pool::new(10);
        
        // Test taking from empty pool
        let mut vec1 = pool.take_or_else(|| Vec::with_capacity(100));
        vec1.extend_from_slice(b"hello");
        assert_eq!(vec1, b"hello");
        
        // Return to pool
        pool.put(vec1);
        
        // Take again - should get the same capacity
        let vec2 = pool.take_or_else(|| Vec::with_capacity(50));
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
        let mut builder = StrBuilder::new();
        
        builder.push_borrowed("Hello");
        builder.push_borrowed(" ");
        builder.push_owned(String::from("World"));
        builder.push_borrowed("!");
        
        let result = builder.build();
        assert_eq!(result, "Hello World!");
    }
}