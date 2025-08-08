# Iterator Implementation Fixes

## Summary of Issues Fixed

### 1. RopeIterator Infinite Loop (FIXED ✅)
**Issue**: The `next()` method could enter an infinite loop when `self.advance()` returns `None`.

**Fix**: Added proper pattern matching to handle the `None` case and return immediately:
```rust
match self.advance() {
    Some(next) => {
        self.current = Some(next);
        self.position = 0;
    }
    None => return None, // Exit when exhausted
}
```

### 2. BTreeIterator Incomplete Implementation (FIXED ✅)
**Issue**: The iterator only traversed the root node's keys and didn't visit child nodes.

**Fix**: Implemented proper B-tree traversal using a stack-based approach:
- Maintains a stack of `(node_index, key_index)` pairs
- Traverses the leftmost path first
- Properly visits all nodes in sorted order
- Handles both leaf and internal nodes correctly

### 3. SuffixArrayIterator Memory Leak (FIXED ✅)
**Issue**: Used `Box::leak` to create a `'static` lifetime, causing memory leaks.

**Fix**: 
- Changed the iterator to own the text data (`String` instead of `&'a str`)
- Added `from_str()` convenience method for creating from string slices
- Returns owned `String`s instead of borrowed `&str`s to avoid lifetime issues
- No more memory leaks!

### 4. Removal of Placeholder Iterators ✅
VEB and Wavelet Tree iterator placeholders were removed to uphold SSOT, YAGNI, and to avoid shipping stubs. They can be reintroduced as complete, literature-faithful implementations in future iterations if needed.

### 6. BloomFilterIterator Misleading Name (FIXED ✅)
**Issue**: The iterator returned bits from the internal bit vector, not elements from the set.

**Fix**: 
- Renamed to `BloomFilterBitIterator` to clearly indicate it iterates bits
- Added comprehensive documentation explaining:
  - Bloom filters cannot enumerate their elements
  - This iterator is for debugging/serialization of the filter state
  - Added utility methods: `set_bit()`, `get_bit()`, `bit_count()`, `popcount()`

## Additional Improvements

### 7. AdvancedIteratorExt Trait
- Fixed lifetime issues in `into_rope()` by creating `OwnedRopeIterator`
- Updated `into_suffix_array()` to accept non-static lifetimes
- Properly handles ownership to avoid memory issues

### 8. Test Coverage
- Added comprehensive unit tests for all fixed iterators
- Created example program that demonstrates all fixes work correctly
- Tests verify:
  - No infinite loops
  - No memory leaks
  - Correct traversal order
  - Proper documentation

## Design Principles Maintained

1. **Zero-Copy Where Possible**: Used borrowing and references where appropriate
2. **Clear Documentation**: All placeholder implementations clearly marked
3. **Type Safety**: Proper use of Rust's type system and lifetimes
4. **Performance**: Stack-based traversal for B-tree, efficient string handling
5. **Correctness**: All iterators now terminate properly and don't leak memory

## Usage Examples

```rust
// OwnedRopeIterator - for string concatenation
let strings = vec!["hello".to_string(), " ".to_string(), "world".to_string()];
let result: String = strings.into_iter().into_rope_owned().collect();

// SuffixArrayIterator - no more leaks!
let suffix_iter = SuffixArrayIterator::new("banana".to_string());
let suffixes: Vec<String> = suffix_iter.collect();

// BloomFilterBitIterator - clearly named
let mut bloom = BloomFilterBitIterator::new(100, 0.01);
bloom.set_bit(10);
let bits: Vec<bool> = bloom.collect();
```

All iterator implementations now follow Rust best practices and the library's design principles!