//! Memory arena allocation for JIT optimization
//!
//! This module provides memory pool management to eliminate intermediate
//! allocations during inference and optimize memory reuse.

use crate::error::{JitError, Result};
use crate::graph::{ComputationGraph, NodeId};
use std::collections::HashMap;

/// Lifetime information for tensor allocations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifetime {
    /// Tensor lives for the entire execution
    Static,
    /// Tensor is created and destroyed within a scope
    Scoped { start: usize, end: usize },
    /// Tensor is temporary and can be reused immediately
    Temporary,
}

/// Memory allocation record
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Allocation {
    offset: usize,
    size: usize,
    lifetime: Lifetime,
    in_use: bool,
}

#[allow(dead_code)]
impl Allocation {
    /// Check if allocation is currently in use
    fn is_in_use(&self) -> bool {
        self.in_use
    }

    /// Get the lifetime of this allocation
    fn lifetime(&self) -> Lifetime {
        self.lifetime
    }

    /// Get the memory offset
    fn offset(&self) -> usize {
        self.offset
    }

    /// Get the allocated size
    fn size(&self) -> usize {
        self.size
    }
}

/// Free memory block for reuse
#[derive(Debug, Clone)]
struct FreeBlock {
    offset: usize,
    size: usize,
}

/// Safe tensor pointer into arena memory
#[derive(Debug)]
pub struct TensorPtr<T> {
    data: *mut T,
    len: usize,
    _lifetime: std::marker::PhantomData<T>,
}

/// Lifetime analysis results
#[derive(Debug)]
pub struct LifetimeAnalysis {
    pub lifetimes: HashMap<NodeId, Lifetime>,
    pub max_concurrent_usage: usize,
    pub total_allocations: usize,
}

/// Memory arena allocator for intermediate tensors
#[derive(Debug)]
pub struct MemoryArena {
    pool: Vec<u8>,
    allocations: Vec<Allocation>,
    free_list: Vec<FreeBlock>,
    total_allocated: usize,
}

/// Lifetime tracker for analyzing tensor usage
#[derive(Debug)]
pub struct LifetimeTracker {
    node_lifetimes: HashMap<NodeId, Lifetime>,
    execution_order: Vec<NodeId>,
}

impl LifetimeTracker {
    /// Add a node to the execution order
    pub fn add_execution_step(&mut self, node_id: NodeId) {
        self.execution_order.push(node_id);
    }

    /// Set the lifetime for a node
    pub fn set_lifetime(&mut self, node_id: NodeId, lifetime: Lifetime) {
        self.node_lifetimes.insert(node_id, lifetime);
    }

    /// Get the lifetime of a node
    pub fn get_lifetime(&self, node_id: NodeId) -> Option<Lifetime> {
        self.node_lifetimes.get(&node_id).copied()
    }

    /// Get the execution order
    pub fn execution_order(&self) -> &[NodeId] {
        &self.execution_order
    }

    /// Analyze lifetimes for memory optimization
    pub fn analyze(&self, _graph: &ComputationGraph) -> LifetimeAnalysis {
        let mut max_concurrent = 0;
        let mut current_active = 0;

        // Simple lifetime analysis based on execution order
        for &node_id in &self.execution_order {
            if let Some(lifetime) = self.get_lifetime(node_id) {
                match lifetime {
                    Lifetime::Temporary => {
                        // Temporary tensors don't add to concurrent usage
                        continue;
                    }
                    Lifetime::Static => {
                        current_active += 1;
                    }
                    Lifetime::Scoped { .. } => {
                        current_active += 1;
                    }
                }
                max_concurrent = max_concurrent.max(current_active);
            }
        }

        LifetimeAnalysis {
            lifetimes: self.node_lifetimes.clone(),
            max_concurrent_usage: max_concurrent,
            total_allocations: self.node_lifetimes.len(),
        }
    }
}

impl<T> TensorPtr<T> {
    /// Create a new tensor pointer from arena memory
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn new(data: *mut u8, len: usize) -> Self {
        Self {
            data: data as *mut T,
            len,
            _lifetime: std::marker::PhantomData,
        }
    }

    /// Get a slice view of the tensor data
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn as_slice(&self) -> &[T] {
        std::slice::from_raw_parts(self.data, self.len)
    }

    /// Get a mutable slice view of the tensor data
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn as_slice_mut(&mut self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.data, self.len)
    }
}

impl MemoryArena {
    /// Create a new memory arena with initial capacity
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            pool: vec![0u8; initial_capacity],
            allocations: Vec::new(),
            free_list: vec![FreeBlock {
                offset: 0,
                size: initial_capacity,
            }],
            total_allocated: 0,
        }
    }

    /// Allocate memory for a tensor with lifetime tracking
    pub fn allocate_tensor<T>(
        &mut self,
        element_count: usize,
        lifetime: Lifetime,
    ) -> Result<TensorPtr<T>> {
        let size_bytes = element_count * std::mem::size_of::<T>();

        // Align to cache line (64 bytes)
        let aligned_size = (size_bytes + 63) & !63;

        let offset = self.find_free_block(aligned_size)?;

        // Remove the free block and create allocation
        self.free_list
            .retain(|block| block.offset != offset || block.size != aligned_size);

        let allocation = Allocation {
            offset,
            size: aligned_size,
            lifetime,
            in_use: true,
        };

        self.allocations.push(allocation);
        self.total_allocated += aligned_size;

        // Split remaining free space if any
        if aligned_size
            < self
                .free_list
                .iter()
                .find(|b| b.offset == offset)
                .map_or(0, |b| b.size)
        {
            let remaining_size = self
                .free_list
                .iter()
                .find(|b| b.offset == offset)
                .map(|b| b.size - aligned_size)
                .unwrap_or(0);

            if remaining_size > 0 {
                self.free_list.push(FreeBlock {
                    offset: offset + aligned_size,
                    size: remaining_size,
                });
            }
        }

        unsafe {
            Ok(TensorPtr::new(
                self.pool.as_mut_ptr().add(offset),
                element_count,
            ))
        }
    }

    /// Deallocate a tensor and mark its memory as free
    pub fn deallocate_tensor(&mut self, offset: usize) -> Result<()> {
        // Find and mark allocation as free
        if let Some(allocation) = self.allocations.iter_mut().find(|a| a.offset == offset) {
            allocation.in_use = false;
            self.total_allocated -= allocation.size;

            // Add to free list
            self.free_list.push(FreeBlock {
                offset: allocation.offset,
                size: allocation.size,
            });

            // Merge adjacent free blocks
            self.merge_free_blocks();

            Ok(())
        } else {
            Err(JitError::InvalidGraph {
                message: format!("No allocation found at offset {}", offset),
            })
        }
    }

    /// Analyze tensor lifetimes for optimal memory reuse
    pub fn analyze_lifetimes(&self, graph: &ComputationGraph) -> Result<LifetimeAnalysis> {
        let mut tracker = LifetimeTracker::new();

        // Get topological execution order
        let execution_order = graph.topological_order()?;

        // Analyze each node
        for &node_id in &execution_order {
            tracker.track_node_lifetime(node_id, graph)?;
        }

        let lifetimes = tracker.node_lifetimes.clone();
        let max_concurrent_usage =
            self.calculate_max_concurrent_usage(&lifetimes, &execution_order);
        let total_allocations = lifetimes.len();

        Ok(LifetimeAnalysis {
            lifetimes,
            max_concurrent_usage,
            total_allocations,
        })
    }

    /// Find a free block that can accommodate the requested size
    fn find_free_block(&mut self, size: usize) -> Result<usize> {
        // First-fit allocation strategy
        for block in &self.free_list {
            if block.size >= size {
                return Ok(block.offset);
            }
        }

        // If no suitable block found, try to grow the arena
        self.grow_pool(size)?;
        Ok(self.pool.len() - size)
    }

    /// Grow the memory pool to accommodate a new allocation
    fn grow_pool(&mut self, min_additional_size: usize) -> Result<()> {
        let current_size = self.pool.len();
        let growth_size = (min_additional_size * 2).max(1024 * 1024); // At least 1MB growth

        self.pool.resize(current_size + growth_size, 0);

        // Add new free block
        self.free_list.push(FreeBlock {
            offset: current_size,
            size: growth_size,
        });

        Ok(())
    }

    /// Merge adjacent free blocks to reduce fragmentation
    fn merge_free_blocks(&mut self) {
        self.free_list.sort_by_key(|b| b.offset);

        let mut merged = Vec::new();
        let mut current: Option<FreeBlock> = None;

        for block in &self.free_list {
            if let Some(ref mut curr) = current {
                if curr.offset + curr.size == block.offset {
                    // Adjacent blocks - merge
                    curr.size += block.size;
                } else {
                    // Non-adjacent - add current and start new
                    merged.push(curr.clone());
                    current = Some(block.clone());
                }
            } else {
                current = Some(block.clone());
            }
        }

        if let Some(curr) = current {
            merged.push(curr);
        }

        self.free_list = merged;
    }

    /// Calculate maximum concurrent memory usage
    fn calculate_max_concurrent_usage(
        &self,
        lifetimes: &HashMap<NodeId, Lifetime>,
        execution_order: &[NodeId],
    ) -> usize {
        let mut max_usage = 0;
        let mut current_usage = 0;

        for &node_id in execution_order {
            if let Some(lifetime) = lifetimes.get(&node_id) {
                match lifetime {
                    Lifetime::Temporary => {
                        // Temporary tensors are allocated and immediately freed
                        current_usage += 1; // Simplified - would use actual size
                        max_usage = max_usage.max(current_usage);
                        current_usage -= 1;
                    }
                    Lifetime::Scoped { .. } => {
                        // Scoped tensors have defined lifetimes
                        current_usage += 1;
                        max_usage = max_usage.max(current_usage);
                        // In a full implementation, would track when they go out of scope
                    }
                    Lifetime::Static => {
                        // Static tensors live for the entire execution
                        current_usage += 1;
                        max_usage = max_usage.max(current_usage);
                    }
                }
            }
        }

        max_usage
    }

    /// Get memory usage statistics
    pub fn stats(&self) -> MemoryStats {
        let total_size = self.pool.len();
        let allocated_size = self.total_allocated;
        let free_size = self.free_list.iter().map(|b| b.size).sum::<usize>();
        let fragmentation_ratio = if total_size > 0 {
            self.free_list.len() as f32 / total_size as f32
        } else {
            0.0
        };

        MemoryStats {
            total_size,
            allocated_size,
            free_size,
            fragmentation_ratio,
            allocation_count: self.allocations.len(),
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_size: usize,
    pub allocated_size: usize,
    pub free_size: usize,
    pub fragmentation_ratio: f32,
    pub allocation_count: usize,
}

impl Default for MemoryArena {
    fn default() -> Self {
        Self::new(1024 * 1024) // 1MB default
    }
}

impl LifetimeTracker {
    /// Create a new lifetime tracker
    pub fn new() -> Self {
        Self {
            node_lifetimes: HashMap::new(),
            execution_order: Vec::new(),
        }
    }

    /// Track the lifetime of a computation graph node
    pub fn track_node_lifetime(&mut self, node_id: NodeId, graph: &ComputationGraph) -> Result<()> {
        let node = graph
            .get_node(node_id)
            .ok_or_else(|| JitError::InvalidGraph {
                message: format!("Node {:?} not found in graph", node_id),
            })?;

        // Analyze dependencies to determine lifetime
        let lifetime = if node.inputs.is_empty() {
            // Input nodes are static
            Lifetime::Static
        } else if node.outputs.is_empty() {
            // Output nodes live until end
            Lifetime::Static
        } else {
            // Intermediate nodes - analyze usage pattern
            self.analyze_intermediate_lifetime(node_id, graph)
        };

        self.node_lifetimes.insert(node_id, lifetime);
        Ok(())
    }

    /// Analyze lifetime for intermediate nodes
    fn analyze_intermediate_lifetime(&self, node_id: NodeId, graph: &ComputationGraph) -> Lifetime {
        // Simplified lifetime analysis
        // In a full implementation, this would:
        // 1. Find the first and last use of the tensor
        // 2. Determine if it can be reused
        // 3. Check for data dependencies

        let node = graph.get_node(node_id).unwrap();

        // If the node has multiple outputs, it's likely needed longer
        if node.outputs.len() > 1 {
            Lifetime::Static
        } else {
            // Check if consumers are close in execution order
            Lifetime::Temporary
        }
    }

    /// Optimize memory reuse based on lifetime analysis
    pub fn optimize_reuse(self) -> Result<LifetimeAnalysis> {
        let lifetimes = self.node_lifetimes;
        let max_concurrent_usage = 0; // Would be calculated properly
        let total_allocations = lifetimes.len();

        Ok(LifetimeAnalysis {
            lifetimes,
            max_concurrent_usage,
            total_allocations,
        })
    }
}

impl Default for LifetimeTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_arena_creation() {
        let arena = MemoryArena::new(1024);
        assert_eq!(arena.pool.len(), 1024);
        assert_eq!(arena.free_list.len(), 1);
        assert_eq!(arena.free_list[0].size, 1024);
    }

    #[test]
    fn test_tensor_allocation() {
        let mut arena = MemoryArena::new(1024);

        // Allocate space for 10 f32 values (40 bytes)
        let tensor = arena
            .allocate_tensor::<f32>(10, Lifetime::Temporary)
            .unwrap();

        unsafe {
            assert_eq!(tensor.as_slice().len(), 10);
        }

        // Check that allocation was recorded
        assert_eq!(arena.allocations.len(), 1);
        assert!(arena.total_allocated > 0);
    }

    #[test]
    fn test_memory_stats() {
        let mut arena = MemoryArena::new(1024);
        let _tensor = arena
            .allocate_tensor::<f32>(10, Lifetime::Temporary)
            .unwrap();

        let stats = arena.stats();
        assert_eq!(stats.total_size, 1024);
        assert!(stats.allocated_size > 0);
        assert!(stats.free_size > 0);
        assert!(stats.allocation_count > 0);
    }

    #[test]
    fn test_lifetime_tracker() {
        let tracker = LifetimeTracker::new();
        assert!(tracker.node_lifetimes.is_empty());
        assert!(tracker.execution_order.is_empty());
    }

    #[test]
    fn test_lifetime_analysis() {
        let mut graph = ComputationGraph::new();

        let input = graph.add_node(
            crate::graph::Operation::Parameter,
            crate::graph::NodeMetadata::default(),
        );
        let temp = graph.add_node(
            crate::graph::Operation::Add,
            crate::graph::NodeMetadata::default(),
        );
        let output = graph.add_node(
            crate::graph::Operation::ReLU,
            crate::graph::NodeMetadata::default(),
        );

        graph.add_edge(input, temp).unwrap();
        graph.add_edge(temp, output).unwrap();

        graph.mark_input(input);
        graph.mark_output(output);

        let arena = MemoryArena::new(1024);
        let analysis = arena.analyze_lifetimes(&graph).unwrap();

        assert_eq!(analysis.total_allocations, 3);
        assert!(analysis.max_concurrent_usage > 0);
    }

    #[test]
    fn test_memory_reuse_optimization() {
        let tracker = LifetimeTracker::new();
        let analysis = tracker.optimize_reuse().unwrap();

        assert_eq!(analysis.total_allocations, 0);
        assert_eq!(analysis.max_concurrent_usage, 0);
    }
}
