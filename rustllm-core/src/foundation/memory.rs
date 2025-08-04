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
use std::sync::RwLock;

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
pub struct Arena {
    chunks: ArenaStorage,
    current_chunk: Cell<usize>,
    chunk_size: usize,
}

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
    pub fn alloc<T>(&self, value: T) -> &mut T {
        let layout = Layout::for_value(&value);
        let ptr = self.alloc_raw(layout);
        
        unsafe {
            if (ptr as usize) % core::mem::align_of::<T>() != 0 {
                panic!("Allocated pointer is not properly aligned for type T");
            }
            ptr::write(ptr as *mut T, value);
            &mut *(ptr as *mut T)
        }
    }
    
    /// Allocates a slice in the arena.
    pub fn alloc_slice<T>(&self, slice: &[T]) -> &mut [T]
    where
        T: Copy,
    {
        let layout = Layout::array::<T>(slice.len()).unwrap();
        let ptr = self.alloc_raw(layout) as *mut T;
        
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
        let chunks = self.chunks.read().unwrap();
        
        #[cfg(not(feature = "std"))]
        let chunks = self.chunks.borrow();
        
        for chunk in chunks.iter() {
            chunk.used.set(0);
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

// Arena is Send because it owns its data
unsafe impl Send for Arena {}

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
    /// Creates a new borrowed CowStr.
    pub fn borrowed(s: &'a str) -> Self {
        Self::Borrowed(s)
    }
    
    /// Creates a new owned CowStr.
    pub fn owned(s: String) -> Self {
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
                *self = Self::Owned(s.to_string());
                match self {
                    Self::Owned(s) => s,
                    _ => unreachable!(),
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
    pub fn new() -> Self {
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
    pub fn new() -> Self {
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
    pub fn len(&self) -> usize {
        self.total_len
    }
    
    /// Returns whether the builder is empty.
    pub fn is_empty(&self) -> bool {
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
    pub fn len(&self) -> usize {
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
        let arena = Arena::new(1024);
        
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
    
    #[cfg(feature = "std")]
    #[test]
    fn test_arena_thread_safety() {
        use std::sync::Arc;
        use std::thread;
        
        let arena = Arc::new(Arena::new(1024));
        let mut handles = vec![];
        
        // Spawn multiple threads that allocate concurrently
        for i in 0..10 {
            let arena_clone = Arc::clone(&arena);
            let handle = thread::spawn(move || {
                // Each thread allocates multiple values
                for j in 0..100 {
                    let value = i * 100 + j;
                    let allocated = arena_clone.alloc(value);
                    assert_eq!(*allocated, value);
                    
                    // Also test slice allocation
                    let slice_data = vec![value; 5];
                    let slice = arena_clone.alloc_slice(&slice_data);
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
        
        // Arena should still be usable after concurrent access
        let final_value = arena.alloc(12345);
        assert_eq!(*final_value, 12345);
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