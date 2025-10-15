//! Kernel fusion detection and optimization

use crate::error::Result;
use crate::graph::{ComputationGraph, NodeId, Operation};

/// Specification for a fused kernel operation
#[derive(Debug, Clone)]
pub struct FusedKernel {
    pub operations: Vec<Operation>,
    pub input_nodes: Vec<NodeId>,
    pub output_nodes: Vec<NodeId>,
    pub memory_layout: MemoryLayout,
    pub fusion_benefits: FusionMetrics,
}

/// Memory layout information for fusion decisions
#[derive(Debug, Clone)]
pub struct MemoryLayout {
    pub input_strides: Vec<Vec<usize>>,
    pub output_strides: Vec<Vec<usize>>,
    pub contiguous_inputs: Vec<bool>,
    pub contiguous_outputs: Vec<bool>,
}

/// Metrics for evaluating fusion benefits
#[derive(Debug, Clone)]
pub struct FusionMetrics {
    pub memory_accesses_saved: usize,
    pub computation_complexity: f32,
    pub register_pressure: usize,
    pub cache_efficiency: f32,
}

/// Fusion pattern matcher
#[derive(Debug)]
struct FusionPattern {
    operations: Vec<Operation>,
    benefit_score: f32,
}

impl FusionPattern {
    /// Check if this pattern matches a sequence of operations
    fn _matches(&self, ops: &[Operation]) -> bool {
        if ops.len() != self.operations.len() {
            return false;
        }
        ops.iter().zip(&self.operations).all(|(a, b)| a == b)
    }

    /// Get the benefit score for this pattern
    fn benefit_score(&self) -> f32 {
        self.benefit_score
    }
}

/// Kernel fusion detector
#[derive(Debug)]
pub struct FusionDetector {
    patterns: Vec<FusionPattern>,
    max_fusion_size: usize,
}

impl FusionDetector {
    /// Create a new fusion detector with default patterns
    pub fn new() -> Self {
        let mut patterns = Vec::new();

        // MatMul + ReLU fusion pattern
        patterns.push(FusionPattern {
            operations: vec![Operation::MatMul, Operation::ReLU],
            benefit_score: 2.5,
        });

        // Element-wise operations chain
        patterns.push(FusionPattern {
            operations: vec![Operation::Add, Operation::ReLU],
            benefit_score: 1.8,
        });

        // Convolution + activation
        patterns.push(FusionPattern {
            operations: vec![Operation::Conv2d, Operation::ReLU],
            benefit_score: 3.0,
        });

        Self {
            patterns,
            max_fusion_size: 4, // Maximum operations to fuse together
        }
    }

    /// Detect fusion opportunities in the computation graph
    pub fn detect_fusions(&self, graph: &ComputationGraph) -> Result<Vec<FusedKernel>> {
        let mut fused_kernels = Vec::new();
        let mut processed_nodes = std::collections::HashSet::new();

        // Get topological order to process nodes in dependency order
        let topological_order = graph.topological_order()?;

        // First try pattern-based fusion using predefined patterns
        for pattern in &self.patterns {
            if let Some(fusions) = self.detect_pattern_fusions(
                graph,
                pattern,
                &topological_order,
                &mut processed_nodes,
            ) {
                fused_kernels.extend(fusions);
            }
        }

        // Then try general fusion detection
        for &node_id in &topological_order {
            if processed_nodes.contains(&node_id) {
                continue;
            }

            if let Some(fusion) = self.try_fuse_from_node(graph, node_id, &mut processed_nodes) {
                fused_kernels.push(fusion);
            }
        }

        Ok(fused_kernels)
    }

    /// Detect fusions using predefined patterns
    fn detect_pattern_fusions(
        &self,
        graph: &ComputationGraph,
        pattern: &FusionPattern,
        topological_order: &[NodeId],
        processed: &mut std::collections::HashSet<NodeId>,
    ) -> Option<Vec<FusedKernel>> {
        let mut fusions = Vec::new();

        for &start_node in topological_order {
            if processed.contains(&start_node) {
                continue;
            }

            if let Some(chain) = self.find_pattern_chain(graph, pattern, start_node) {
                if chain.len() >= 2 {
                    // Mark all nodes as processed
                    for &node_id in &chain {
                        processed.insert(node_id);
                    }

                    // Create fused kernel
                    let operations: Vec<Operation> = chain
                        .iter()
                        .filter_map(|&node_id| graph.get_node(node_id))
                        .map(|node| node.operation.clone())
                        .collect();

                    let input_nodes = self.find_input_nodes(graph, &chain);
                    let output_nodes = self.find_output_nodes(graph, &chain);
                    let memory_layout = self.compute_memory_layout(graph, &chain)?;
                    let fusion_benefits = FusionMetrics {
                        memory_accesses_saved: (chain.len() - 1) * 20,
                        computation_complexity: pattern.benefit_score(),
                        register_pressure: chain.len(),
                        cache_efficiency: 0.85,
                    };

                    fusions.push(FusedKernel {
                        operations,
                        input_nodes,
                        output_nodes,
                        memory_layout,
                        fusion_benefits,
                    });
                }
            }
        }

        if fusions.is_empty() {
            None
        } else {
            Some(fusions)
        }
    }

    /// Find a chain of operations matching the pattern
    fn find_pattern_chain(
        &self,
        graph: &ComputationGraph,
        pattern: &FusionPattern,
        start_node: NodeId,
    ) -> Option<Vec<NodeId>> {
        let mut chain = vec![start_node];
        let mut current_node = start_node;

        for pattern_op in &pattern.operations[1..] {
            // Skip first op as it's already matched
            let mut found = false;

            if let Some(node) = graph.get_node(current_node) {
                for &output_id in &node.outputs {
                    if let Some(next_node) = graph.get_node(output_id) {
                        if next_node.operation == *pattern_op {
                            chain.push(output_id);
                            current_node = output_id;
                            found = true;
                            break;
                        }
                    }
                }
            }

            if !found {
                return None;
            }
        }

        Some(chain)
    }

    /// Try to create a fused kernel starting from a given node
    fn try_fuse_from_node(
        &self,
        graph: &ComputationGraph,
        start_node: NodeId,
        processed: &mut std::collections::HashSet<NodeId>,
    ) -> Option<FusedKernel> {
        let _start_node_data = graph.get_node(start_node)?;

        // Find fusable operation chains
        let mut fusable_chain = vec![start_node];
        let mut current_node = start_node;

        // Look for fusable successors
        while let Some(node) = graph.get_node(current_node) {
            if node.outputs.len() == 1 {
                let next_node_id = node.outputs[0];
                if let Some(next_node) = graph.get_node(next_node_id) {
                    if node.can_fuse_with(next_node) && fusable_chain.len() < self.max_fusion_size {
                        fusable_chain.push(next_node_id);
                        current_node = next_node_id;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // Only create fusion if we have more than one operation
        if fusable_chain.len() < 2 {
            processed.insert(start_node);
            return None;
        }

        // Mark all nodes in the fusion as processed
        for &node_id in &fusable_chain {
            processed.insert(node_id);
        }

        // Extract operations and determine memory layout
        let operations: Vec<Operation> = fusable_chain
            .iter()
            .filter_map(|&node_id| graph.get_node(node_id))
            .map(|node| node.operation.clone())
            .collect();

        let input_nodes = self.find_input_nodes(graph, &fusable_chain);
        let output_nodes = self.find_output_nodes(graph, &fusable_chain);

        let memory_layout = self.compute_memory_layout(graph, &fusable_chain)?;
        let fusion_benefits = self.compute_fusion_benefits(&operations, &memory_layout);

        Some(FusedKernel {
            operations,
            input_nodes,
            output_nodes,
            memory_layout,
            fusion_benefits,
        })
    }

    /// Find input nodes for a fused kernel
    fn find_input_nodes(&self, graph: &ComputationGraph, fusion_nodes: &[NodeId]) -> Vec<NodeId> {
        let fusion_set: std::collections::HashSet<_> = fusion_nodes.iter().cloned().collect();

        let mut input_nodes = Vec::new();

        for &node_id in fusion_nodes {
            if let Some(node) = graph.get_node(node_id) {
                for &input_id in &node.inputs {
                    if !fusion_set.contains(&input_id) && !input_nodes.contains(&input_id) {
                        input_nodes.push(input_id);
                    }
                }
            }
        }

        input_nodes
    }

    /// Find output nodes for a fused kernel
    fn find_output_nodes(&self, graph: &ComputationGraph, fusion_nodes: &[NodeId]) -> Vec<NodeId> {
        let fusion_set: std::collections::HashSet<_> = fusion_nodes.iter().cloned().collect();

        let mut output_nodes = Vec::new();

        for &node_id in fusion_nodes {
            if let Some(node) = graph.get_node(node_id) {
                for &output_id in &node.outputs {
                    if !fusion_set.contains(&output_id) && !output_nodes.contains(&output_id) {
                        output_nodes.push(output_id);
                    }
                }
            }
        }

        output_nodes
    }

    /// Compute memory layout for fused operations
    fn compute_memory_layout(
        &self,
        graph: &ComputationGraph,
        fusion_nodes: &[NodeId],
    ) -> Option<MemoryLayout> {
        // Simplified memory layout computation
        // In a real implementation, this would analyze tensor shapes and strides

        let mut input_strides = Vec::new();
        let mut output_strides = Vec::new();
        let mut contiguous_inputs = Vec::new();
        let mut contiguous_outputs = Vec::new();

        for &node_id in fusion_nodes {
            if let Some(node) = graph.get_node(node_id) {
                // Placeholder: assume row-major contiguous layout
                if !node.inputs.is_empty() {
                    input_strides.push(vec![1; node.inputs.len()]); // Simplified strides
                    contiguous_inputs.push(true);
                }

                if !node.outputs.is_empty() {
                    output_strides.push(vec![1; node.outputs.len()]); // Simplified strides
                    contiguous_outputs.push(true);
                }
            }
        }

        Some(MemoryLayout {
            input_strides,
            output_strides,
            contiguous_inputs,
            contiguous_outputs,
        })
    }

    /// Compute fusion benefits and metrics
    fn compute_fusion_benefits(
        &self,
        operations: &[Operation],
        layout: &MemoryLayout,
    ) -> FusionMetrics {
        let mut memory_accesses_saved = 0;
        let mut computation_complexity = 0.0;
        let mut register_pressure = 0;

        for operation in operations {
            match operation {
                Operation::MatMul => {
                    memory_accesses_saved += 100; // Significant memory savings
                    computation_complexity += 10.0;
                    register_pressure += 5;
                }
                Operation::ReLU => {
                    memory_accesses_saved += 20; // Moderate savings
                    computation_complexity += 1.0;
                    register_pressure += 1;
                }
                Operation::Add | Operation::Multiply => {
                    memory_accesses_saved += 15;
                    computation_complexity += 0.5;
                    register_pressure += 1;
                }
                _ => {
                    memory_accesses_saved += 10;
                    computation_complexity += 1.0;
                    register_pressure += 2;
                }
            }
        }

        // Cache efficiency based on memory layout
        let cache_efficiency = if layout.contiguous_inputs.iter().all(|&x| x)
            && layout.contiguous_outputs.iter().all(|&x| x)
        {
            0.9 // High efficiency for contiguous memory
        } else {
            0.6 // Lower efficiency for non-contiguous
        };

        FusionMetrics {
            memory_accesses_saved,
            computation_complexity,
            register_pressure,
            cache_efficiency,
        }
    }
}

impl Default for FusionDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::NodeMetadata;

    #[test]
    fn test_fusion_detector_creation() {
        let detector = FusionDetector::new();
        assert!(!detector.patterns.is_empty());
        assert_eq!(detector.max_fusion_size, 4);
    }

    #[test]
    fn test_fusion_detection() {
        let mut graph = ComputationGraph::new();

        // Create a simple fusion pattern: MatMul -> ReLU
        let input1 = graph.add_node(Operation::Parameter, NodeMetadata::default());
        let input2 = graph.add_node(Operation::Parameter, NodeMetadata::default());
        let matmul = graph.add_node(Operation::MatMul, NodeMetadata::default());
        let relu = graph.add_node(Operation::ReLU, NodeMetadata::default());
        let output = graph.add_node(Operation::Parameter, NodeMetadata::default());

        graph.add_edge(input1, matmul).unwrap();
        graph.add_edge(input2, matmul).unwrap();
        graph.add_edge(matmul, relu).unwrap();
        graph.add_edge(relu, output).unwrap();

        graph.mark_input(input1);
        graph.mark_input(input2);
        graph.mark_output(output);

        let detector = FusionDetector::new();
        let fusions = detector.detect_fusions(&graph).unwrap();

        // Should detect one fusion (MatMul + ReLU)
        assert_eq!(fusions.len(), 1);
        assert_eq!(fusions[0].operations.len(), 2);
        assert!(matches!(fusions[0].operations[0], Operation::MatMul));
        assert!(matches!(fusions[0].operations[1], Operation::ReLU));
    }

    #[test]
    fn test_fusion_benefits() {
        let detector = FusionDetector::new();

        let operations = vec![Operation::MatMul, Operation::ReLU];
        let layout = MemoryLayout {
            input_strides: vec![vec![1, 2], vec![1]],
            output_strides: vec![vec![1]],
            contiguous_inputs: vec![true, true],
            contiguous_outputs: vec![true],
        };

        let benefits = detector.compute_fusion_benefits(&operations, &layout);

        assert!(benefits.memory_accesses_saved > 0);
        assert!(benefits.computation_complexity > 0.0);
        assert!(benefits.cache_efficiency > 0.0);
    }
}
