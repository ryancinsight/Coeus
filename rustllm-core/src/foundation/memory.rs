//! Memory management utilities for zero-copy operations.
//!
//! This module provides abstractions for efficient memory management,
//! including arena allocators, memory pools, and copy-on-write wrappers.

use core::alloc::{GlobalAlloc, Layout};
use core::cell::{Cell, RefCell};
use core::marker::PhantomData;
use core::ptr::{self, NonNull};
use core::slice;

#[cfg(feature = "std")]
use std::alloc::System;

/// A simple arena allocator for temporary allocations.
pub struct Arena {
    chunks: RefCell<Vec<ArenaChunk>>,
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
        
        #[cfg(feature = "std")]
        let data = unsafe {
            let ptr = System.alloc(layout);
            NonNull::new(ptr).expect("Allocation failed")
        };
        
        #[cfg(not(feature = "std"))]
        let data = panic!("Arena allocation requires std feature");
        
        Self {
            data,
            size,
            used: Cell::new(0),
        }
    }
}

impl Drop for ArenaChunk {
    fn drop(&mut self) {
        #[cfg(feature = "std")]
        unsafe {
            let layout = Layout::array::<u8>(self.size).unwrap();
            System.dealloc(self.data.as_ptr(), layout);
        }
    }
}

unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

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
    #[cfg(feature = "std")]
    fn test_arena() {
        let arena = Arena::new(1024);
        
        let x = arena.alloc(42u32);
        assert_eq!(*x, 42);
        *x = 100;
        assert_eq!(*x, 100);
        
        let slice = arena.alloc_slice(&[1, 2, 3, 4, 5]);
        assert_eq!(slice, &[1, 2, 3, 4, 5]);
        slice[0] = 10;
        assert_eq!(slice[0], 10);
        
        arena.reset();
    }
    
    #[test]
    fn test_pool() {
        let pool: Pool<Vec<u8>> = Pool::new(2);
        
        let mut buf1 = pool.take_or_default();
        buf1.extend_from_slice(b"hello");
        
        let buf2 = pool.take_or_default();
        assert!(buf2.is_empty());
        
        pool.put(buf1);
        
        let buf3 = pool.take_or_default();
        assert_eq!(buf3.len(), 5); // Reused buf1
    }
    
    #[test]
    fn test_cow_str() {
        let mut cow1 = CowStr::borrowed("hello");
        assert_eq!(cow1.as_str(), "hello");
        
        let s = cow1.to_mut();
        s.push_str(" world");
        assert_eq!(cow1.as_str(), "hello world");
        
        let cow2 = CowStr::owned(String::from("rust"));
        assert_eq!(cow2.as_str(), "rust");
    }
    
    #[test]
    fn test_str_builder() {
        let mut builder = StrBuilder::new();
        builder.push_borrowed("hello");
        builder.push_borrowed(" ");
        builder.push_owned(String::from("world"));
        
        let result = builder.build();
        assert_eq!(result, "hello world");
    }
}