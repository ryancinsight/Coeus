//! Memory management utilities for zero-copy operations.
//!
//! This module provides abstractions for efficient memory management,
//! including arena allocators, memory pools, and copy-on-write wrappers.

use core::cell::{Cell, RefCell};
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