//! TorchScript compatibility layer for JIT compilation
//!
//! This module provides PyTorch TorchScript-compatible tracing and execution,
//! enabling seamless model deployment and inference optimization.

use crate::compiler::CompiledKernel;
use crate::error::{JitError, Result};
use crate::graph::{ComputationGraph, NodeId, NodeMetadata, Operation};
use std::collections::HashMap;

/// TorchScript runtime for traced and scripted models
#[derive(Debug)]
pub struct TorchScript {
    tracer: Tracer,
    runtime: JitRuntime,
}

/// Execution tracer for recording model operations
#[derive(Debug)]
pub struct Tracer {
    graph: ComputationGraph,
    node_map: HashMap<String, NodeId>,
    input_shapes: Vec<Vec<usize>>,
}

/// JIT runtime for executing traced models
#[derive(Debug)]
#[allow(dead_code)]
pub struct JitRuntime {
    compiler: crate::JitCompiler,
    cache: crate::KernelCache,
}

/// Traced module wrapper with TorchScript compatibility
#[derive(Debug)]
#[allow(dead_code)]
pub struct TracedModule<M, B, T> {
    original_module: M,
    traced_graph: ComputationGraph,
    compiled_kernel: Option<CompiledKernel>,
    _phantom: std::marker::PhantomData<(B, T)>,
}

impl TorchScript {
    /// Create a new TorchScript runtime
    pub fn new() -> Self {
        Self {
            tracer: Tracer::new(),
            runtime: JitRuntime::new(),
        }
    }

    /// Trace a model's forward pass (torch.jit.trace equivalent)
    pub fn trace<M, B, S, T>(
        &mut self,
        model: &M,
        example_input: &coeus_tensor::Tensor<B, S, T>,
    ) -> Result<TracedModule<M, B, T>>
    where
        M: coeus_nn::Module<B, S, T> + Clone,
        B: coeus_backend::Backend<T>,
        S: coeus_storage::Storage<T> + coeus_tensor::StorageFromVec<T> + Clone + 'static,
        T: coeus_dtype::DataType,
    {
        // Record the execution by running forward pass with tracing
        let traced_graph = self.tracer.trace_execution(model, example_input)?;

        Ok(TracedModule {
            original_module: model.clone(), // Assuming Clone is available
            traced_graph,
            compiled_kernel: None,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Execute a traced module
    pub fn forward<M, B, T>(
        &mut self,
        traced_module: &mut TracedModule<M, B, T>,
        input: &coeus_tensor::Tensor<B, coeus_storage::DenseStorage<T>, T>,
    ) -> Result<coeus_tensor::Tensor<B, coeus_storage::DenseStorage<T>, T>>
    where
        M: coeus_nn::Module<B, coeus_storage::DenseStorage<T>, T>,
        B: coeus_backend::Backend<T> + Default,
        T: coeus_dtype::DataType,
    {
        // Compile the traced graph if not already compiled
        if traced_module.compiled_kernel.is_none() {
            let kernel = self.runtime.compile_graph(&traced_module.traced_graph)?;
            traced_module.compiled_kernel = Some(kernel);
        }

        // Execute using the compiled kernel
        self.runtime
            .execute_kernel(traced_module.compiled_kernel.as_ref().unwrap(), input)
    }
}

impl Default for TorchScript {
    fn default() -> Self {
        Self::new()
    }
}

impl Tracer {
    /// Create a new execution tracer
    pub fn new() -> Self {
        Self {
            graph: ComputationGraph::new(),
            node_map: HashMap::new(),
            input_shapes: Vec::new(),
        }
    }

    /// Trace the execution of a model by recording its operations
    pub fn trace_execution<M, B, S, T>(
        &mut self,
        _model: &M,
        input: &coeus_tensor::Tensor<B, S, T>,
    ) -> Result<ComputationGraph>
    where
        M: coeus_nn::Module<B, S, T>,
        B: coeus_backend::Backend<T>,
        S: coeus_storage::Storage<T> + coeus_tensor::StorageFromVec<T> + Clone + 'static,
        T: coeus_dtype::DataType,
    {
        // Reset tracer state
        self.graph = ComputationGraph::new();
        self.node_map.clear();
        self.input_shapes.clear();

        // Create input node
        let input_shape = input.shape().dims().to_vec();
        self.input_shapes.push(input_shape.clone());

        let input_node = self.graph.add_node(
            Operation::Parameter,
            NodeMetadata {
                shape: Some(input_shape),
                dtype: Some("f32".to_string()), // Simplified
                requires_grad: false,
                name: Some("input".to_string()),
            },
        );

        self.node_map.insert("input".to_string(), input_node);
        self.graph.mark_input(input_node);

        // Start JIT tracing context to record operations
        let tracing_guard =
            crate::TracingGuard::start_tracing().map_err(|e| crate::JitError::TracingError {
                message: format!("Failed to start tracing: {:?}", e),
            })?;

        // Record input parameter with actual tensor shape
        let input_metadata = NodeMetadata {
            shape: Some(input.shape().dims().to_vec()),
            dtype: Some("f32".to_string()), // TODO: Infer from tensor dtype
            requires_grad: false,           // Input doesn't require gradients
            name: Some("input".to_string()),
        };
        crate::tracing::record_parameter(0, input_metadata); // Use ID 0 for input

        // TODO: Execute the model's forward pass with tracing enabled
        // This would involve calling model.forward() while the tracing context is active.
        // The autograd system would automatically record operations via tracing hooks.
        //
        // For now, simulate a realistic neural network forward pass:
        // Linear(input) -> ReLU -> Linear -> ReLU

        // First layer parameters (input_dim -> hidden_dim)
        let hidden_dim = 32;
        let weight1_metadata = NodeMetadata {
            shape: Some(vec![input.shape().dims()[0], hidden_dim]),
            dtype: Some("f32".to_string()),
            requires_grad: true,
            name: Some("weight1".to_string()),
        };
        crate::tracing::record_parameter(1, weight1_metadata);

        let bias1_metadata = NodeMetadata {
            shape: Some(vec![hidden_dim]),
            dtype: Some("f32".to_string()),
            requires_grad: true,
            name: Some("bias1".to_string()),
        };
        crate::tracing::record_parameter(2, bias1_metadata);

        // First linear transformation: input @ weight1 + bias1
        let linear1_metadata = NodeMetadata {
            shape: Some(vec![hidden_dim]),
            dtype: Some("f32".to_string()),
            requires_grad: true,
            name: Some("linear1".to_string()),
        };
        crate::tracing::record_operation(
            Operation::MatMul,
            vec![0, 1], // input and weight1
            3,          // intermediate result
            linear1_metadata,
        );

        let bias_add1_metadata = NodeMetadata {
            shape: Some(vec![hidden_dim]),
            dtype: Some("f32".to_string()),
            requires_grad: true,
            name: Some("bias_add1".to_string()),
        };
        crate::tracing::record_operation(
            Operation::Add,
            vec![3, 2], // linear1 result and bias1
            4,          // intermediate result
            bias_add1_metadata,
        );

        // First ReLU activation
        let relu1_metadata = NodeMetadata {
            shape: Some(vec![hidden_dim]),
            dtype: Some("f32".to_string()),
            requires_grad: true,
            name: Some("relu1".to_string()),
        };
        crate::tracing::record_operation(
            Operation::ReLU,
            vec![4], // bias_add1 result
            5,       // intermediate result
            relu1_metadata,
        );

        // Second layer parameters (hidden_dim -> output_dim)
        let output_dim = 10;
        let weight2_metadata = NodeMetadata {
            shape: Some(vec![hidden_dim, output_dim]),
            dtype: Some("f32".to_string()),
            requires_grad: true,
            name: Some("weight2".to_string()),
        };
        crate::tracing::record_parameter(6, weight2_metadata);

        let bias2_metadata = NodeMetadata {
            shape: Some(vec![output_dim]),
            dtype: Some("f32".to_string()),
            requires_grad: true,
            name: Some("bias2".to_string()),
        };
        crate::tracing::record_parameter(7, bias2_metadata);

        // Second linear transformation: relu1 @ weight2 + bias2
        let linear2_metadata = NodeMetadata {
            shape: Some(vec![output_dim]),
            dtype: Some("f32".to_string()),
            requires_grad: true,
            name: Some("linear2".to_string()),
        };
        crate::tracing::record_operation(
            Operation::MatMul,
            vec![5, 6], // relu1 result and weight2
            8,          // intermediate result
            linear2_metadata,
        );

        let bias_add2_metadata = NodeMetadata {
            shape: Some(vec![output_dim]),
            dtype: Some("f32".to_string()),
            requires_grad: true,
            name: Some("bias_add2".to_string()),
        };
        crate::tracing::record_operation(
            Operation::Add,
            vec![8, 7], // linear2 result and bias2
            9,          // intermediate result
            bias_add2_metadata,
        );

        // Final ReLU activation
        let relu2_metadata = NodeMetadata {
            shape: Some(vec![output_dim]),
            dtype: Some("f32".to_string()),
            requires_grad: true,
            name: Some("relu2".to_string()),
        };
        crate::tracing::record_operation(
            Operation::ReLU,
            vec![9], // bias_add2 result
            10,      // final output
            relu2_metadata,
        );

        // Stop tracing and get the recorded computation graph
        let traced_graph =
            tracing_guard
                .stop_tracing()
                .map_err(|e| crate::JitError::TracingError {
                    message: format!("Failed to stop tracing: {:?}", e),
                })?;

        Ok(traced_graph)
    }
}

impl Default for Tracer {
    fn default() -> Self {
        Self::new()
    }
}

impl<M, B, T> TracedModule<M, B, T> {
    /// Create a new traced module
    pub fn new(original_module: M, traced_graph: ComputationGraph) -> Self {
        Self {
            original_module,
            traced_graph,
            compiled_kernel: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the traced computation graph
    pub fn graph(&self) -> &ComputationGraph {
        &self.traced_graph
    }

    /// Check if the module has been compiled
    pub fn is_compiled(&self) -> bool {
        self.compiled_kernel.is_some()
    }
}

impl JitRuntime {
    /// Create a new JIT runtime
    pub fn new() -> Self {
        Self {
            compiler: crate::JitCompiler::new(),
            cache: crate::KernelCache::new(),
        }
    }

    /// Compile a computation graph into an executable kernel
    pub fn compile_graph(&mut self, graph: &ComputationGraph) -> Result<CompiledKernel> {
        // Apply optimizations to the graph
        let optimizer = crate::Optimizer::new();
        let mut optimized_graph = graph.clone(); // Simplified - should deep clone

        optimizer.optimize(&mut optimized_graph)?;

        // Detect fusion opportunities
        let fusion_detector = crate::FusionDetector::new();
        let fusions = fusion_detector.detect_fusions(&optimized_graph)?;

        // Compile the primary kernel
        // In a full implementation, this would handle multiple fused kernels
        if let Some(primary_fusion) = fusions.first() {
            self.compiler.compile_fused(primary_fusion)
        } else {
            // Fallback: compile a simple kernel for the graph
            Err(JitError::CompilationFailed {
                message: "Graph compilation not yet implemented for non-fused operations"
                    .to_string(),
            })
        }
    }

    /// Execute a compiled kernel with the given input
    pub fn execute_kernel<B, T>(
        &self,
        _kernel: &CompiledKernel,
        input: &coeus_tensor::Tensor<B, coeus_storage::DenseStorage<T>, T>,
    ) -> Result<coeus_tensor::Tensor<B, coeus_storage::DenseStorage<T>, T>>
    where
        B: coeus_backend::Backend<T> + Default,
        T: coeus_dtype::DataType,
    {
        // Placeholder execution - in a real implementation, this would:
        // 1. Prepare input tensors in the kernel's expected format
        // 2. Execute the compiled machine code
        // 3. Return the results

        tracing::warn!("JIT kernel execution is not yet implemented, returning placeholder output");

        // Create a simple output tensor (placeholder)
        let output_shape = input.shape().clone();
        use coeus_tensor::Tensor;
        Tensor::zeros(output_shape.dims()).map_err(|e| JitError::ExecutionFailed {
            message: format!("Failed to create output tensor: {:?}", e),
        })
    }
}

impl Default for JitRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::NodeMetadata;

    #[test]
    fn test_tracer_creation() {
        let tracer = Tracer::new();
        assert!(tracer.node_map.is_empty());
        assert!(tracer.input_shapes.is_empty());
    }

    #[test]
    fn test_torchscript_creation() {
        let _ts = TorchScript::new();
        // Basic functionality tests would require full module implementations
        // For now, just verify creation succeeds
    }

    #[test]
    fn test_traced_module_creation() {
        let mut graph = ComputationGraph::new();
        let _input_node = graph.add_node(Operation::Parameter, NodeMetadata::default());

        // Simplified test - in reality would need a proper module type
        // let traced = TracedModule::new(mock_module, graph);
        // assert!(!traced.is_compiled());
    }

    #[test]
    fn test_jit_runtime_creation() {
        let runtime = JitRuntime::new();
        // Verify components are initialized
        let stats = runtime.cache.stats();
        assert_eq!(stats.memory_entries, 0);
    }

    #[test]
    fn test_jit_autograd_integration() {
        use crate::graph::{ComputationGraph, NodeMetadata, Operation};

        // Create a simple computation graph that could come from autograd tracing
        let mut graph = ComputationGraph::new();

        // Create input parameter
        let input = graph.add_node(
            Operation::Parameter,
            NodeMetadata {
                shape: Some(vec![4]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("input".to_string()),
            },
        );

        // Create weight parameter
        let weight = graph.add_node(
            Operation::Parameter,
            NodeMetadata {
                shape: Some(vec![4, 2]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("weight".to_string()),
            },
        );

        // Create bias parameter
        let bias = graph.add_node(
            Operation::Parameter,
            NodeMetadata {
                shape: Some(vec![2]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("bias".to_string()),
            },
        );

        // Create MatMul operation
        let matmul = graph.add_node(
            Operation::MatMul,
            NodeMetadata {
                shape: Some(vec![2]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("matmul".to_string()),
            },
        );

        // Create Add operation
        let add = graph.add_node(
            Operation::Add,
            NodeMetadata {
                shape: Some(vec![2]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("add".to_string()),
            },
        );

        // Create ReLU operation
        let relu = graph.add_node(
            Operation::ReLU,
            NodeMetadata {
                shape: Some(vec![2]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("relu".to_string()),
            },
        );

        // Set up graph connections
        graph.add_edge(input, matmul).unwrap();
        graph.add_edge(weight, matmul).unwrap();
        graph.add_edge(matmul, add).unwrap();
        graph.add_edge(bias, add).unwrap();
        graph.add_edge(add, relu).unwrap();

        // Mark inputs and outputs
        graph.mark_input(input);
        graph.mark_input(weight);
        graph.mark_input(bias);
        graph.mark_output(relu);

        // Test graph optimization
        let optimizer = crate::Optimizer::new();
        let mut optimized_graph = graph.clone();
        optimizer.optimize(&mut optimized_graph).unwrap();

        // Test fusion detection
        let fusion_detector = crate::FusionDetector::new();
        let fusions = fusion_detector.detect_fusions(&optimized_graph).unwrap();

        // Should detect Add + ReLU fusion
        assert!(!fusions.is_empty(), "Should detect at least one fusion");

        // Test kernel compilation
        let mut compiler = crate::JitCompiler::new();
        for fusion in fusions {
            let compiled = compiler.compile_fused(&fusion).unwrap();
            assert!(!compiled.kernel_id.is_empty());
            assert!(compiled.memory_requirements > 0);
            assert!(compiled.performance_estimate > 0.0);
            assert!(!compiled.machine_code.is_empty());
        }

        // Test JIT runtime compilation
        let mut runtime = JitRuntime::new();
        let kernel = runtime.compile_graph(&optimized_graph).unwrap();
        assert!(!kernel.kernel_id.is_empty());
    }

    #[test]
    fn test_jit_neural_network_workflow() {
        use crate::graph::{ComputationGraph, NodeMetadata, Operation};

        // Create a more complex neural network graph (2-layer MLP)
        let mut graph = ComputationGraph::new();

        // Input layer
        let input = graph.add_node(
            Operation::Parameter,
            NodeMetadata {
                shape: Some(vec![10]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("input".to_string()),
            },
        );

        // First layer weights and bias
        let w1 = graph.add_node(
            Operation::Parameter,
            NodeMetadata {
                shape: Some(vec![10, 32]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("w1".to_string()),
            },
        );
        let b1 = graph.add_node(
            Operation::Parameter,
            NodeMetadata {
                shape: Some(vec![32]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("b1".to_string()),
            },
        );

        // First layer: input @ w1 + b1
        let matmul1 = graph.add_node(
            Operation::MatMul,
            NodeMetadata {
                shape: Some(vec![32]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("matmul1".to_string()),
            },
        );
        let add1 = graph.add_node(
            Operation::Add,
            NodeMetadata {
                shape: Some(vec![32]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("add1".to_string()),
            },
        );
        let relu1 = graph.add_node(
            Operation::ReLU,
            NodeMetadata {
                shape: Some(vec![32]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("relu1".to_string()),
            },
        );

        // Second layer weights and bias
        let w2 = graph.add_node(
            Operation::Parameter,
            NodeMetadata {
                shape: Some(vec![32, 2]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("w2".to_string()),
            },
        );
        let b2 = graph.add_node(
            Operation::Parameter,
            NodeMetadata {
                shape: Some(vec![2]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("b2".to_string()),
            },
        );

        // Second layer: relu1 @ w2 + b2
        let matmul2 = graph.add_node(
            Operation::MatMul,
            NodeMetadata {
                shape: Some(vec![2]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("matmul2".to_string()),
            },
        );
        let add2 = graph.add_node(
            Operation::Add,
            NodeMetadata {
                shape: Some(vec![2]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("add2".to_string()),
            },
        );
        let relu2 = graph.add_node(
            Operation::ReLU,
            NodeMetadata {
                shape: Some(vec![2]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("relu2".to_string()),
            },
        );

        // Set up connections
        graph.add_edge(input, matmul1).unwrap();
        graph.add_edge(w1, matmul1).unwrap();
        graph.add_edge(matmul1, add1).unwrap();
        graph.add_edge(b1, add1).unwrap();
        graph.add_edge(add1, relu1).unwrap();

        graph.add_edge(relu1, matmul2).unwrap();
        graph.add_edge(w2, matmul2).unwrap();
        graph.add_edge(matmul2, add2).unwrap();
        graph.add_edge(b2, add2).unwrap();
        graph.add_edge(add2, relu2).unwrap();

        // Mark inputs and outputs
        graph.mark_input(input);
        graph.mark_input(w1);
        graph.mark_input(b1);
        graph.mark_input(w2);
        graph.mark_input(b2);
        graph.mark_output(relu2);

        // Test full JIT pipeline
        let optimizer = crate::Optimizer::new();
        let mut optimized_graph = graph.clone();
        optimizer.optimize(&mut optimized_graph).unwrap();

        let fusion_detector = crate::FusionDetector::new();
        let fusions = fusion_detector.detect_fusions(&optimized_graph).unwrap();

        // Should detect multiple fusions in a neural network
        assert!(
            !fusions.is_empty(),
            "Neural network should have fusion opportunities"
        );

        // Compile all fusions
        let mut compiler = crate::JitCompiler::new();
        for fusion in &fusions {
            let compiled = compiler.compile_fused(fusion).unwrap();
            assert!(compiled.memory_requirements > 0);
            assert!(compiled.performance_estimate > 0.0);
        }

        // Test runtime compilation
        let mut runtime = JitRuntime::new();
        let kernel = runtime.compile_graph(&optimized_graph).unwrap();
        assert!(!kernel.kernel_id.is_empty());
        assert!(kernel.memory_requirements > 0);
    }

    #[test]
    fn test_jit_memory_optimization() {
        use crate::graph::{ComputationGraph, NodeMetadata, Operation};
        use crate::memory::{Lifetime, LifetimeTracker, MemoryArena};

        // Create a graph with intermediate tensors that can be optimized
        let mut graph = ComputationGraph::new();

        let input = graph.add_node(
            Operation::Parameter,
            NodeMetadata {
                shape: Some(vec![4]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("input".to_string()),
            },
        );

        // Create a chain that produces intermediate results
        let add1 = graph.add_node(
            Operation::Add,
            NodeMetadata {
                shape: Some(vec![4]),
                dtype: Some("f32".to_string()),
                requires_grad: false,
                name: Some("add1".to_string()),
            },
        );
        let mul1 = graph.add_node(
            Operation::Multiply,
            NodeMetadata {
                shape: Some(vec![4]),
                dtype: Some("f32".to_string()),
                requires_grad: false,
                name: Some("mul1".to_string()),
            },
        );
        let relu1 = graph.add_node(
            Operation::ReLU,
            NodeMetadata {
                shape: Some(vec![4]),
                dtype: Some("f32".to_string()),
                requires_grad: false,
                name: Some("relu1".to_string()),
            },
        );

        // Another chain
        let add2 = graph.add_node(
            Operation::Add,
            NodeMetadata {
                shape: Some(vec![4]),
                dtype: Some("f32".to_string()),
                requires_grad: false,
                name: Some("add2".to_string()),
            },
        );
        let mul2 = graph.add_node(
            Operation::Multiply,
            NodeMetadata {
                shape: Some(vec![4]),
                dtype: Some("f32".to_string()),
                requires_grad: false,
                name: Some("mul2".to_string()),
            },
        );

        // Final operation
        let final_add = graph.add_node(
            Operation::Add,
            NodeMetadata {
                shape: Some(vec![4]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("final".to_string()),
            },
        );

        // Set up connections
        graph.add_edge(input, add1).unwrap();
        graph.add_edge(add1, mul1).unwrap();
        graph.add_edge(mul1, relu1).unwrap();

        graph.add_edge(input, add2).unwrap();
        graph.add_edge(add2, mul2).unwrap();

        graph.add_edge(relu1, final_add).unwrap();
        graph.add_edge(mul2, final_add).unwrap();

        graph.mark_input(input);
        graph.mark_output(final_add);

        // Test lifetime analysis
        let mut tracker = LifetimeTracker::new();
        tracker.add_execution_step(add1);
        tracker.add_execution_step(add2);
        tracker.add_execution_step(mul1);
        tracker.add_execution_step(mul2);
        tracker.add_execution_step(relu1);
        tracker.add_execution_step(final_add);

        // Set lifetimes
        tracker.set_lifetime(add1, Lifetime::Temporary);
        tracker.set_lifetime(add2, Lifetime::Temporary);
        tracker.set_lifetime(mul1, Lifetime::Temporary);
        tracker.set_lifetime(mul2, Lifetime::Temporary);
        tracker.set_lifetime(relu1, Lifetime::Scoped { start: 2, end: 5 });
        tracker.set_lifetime(final_add, Lifetime::Static);

        let analysis = tracker.analyze(&graph);
        assert_eq!(analysis.lifetimes.len(), 6);
        assert!(analysis.max_concurrent_usage > 0);
        assert_eq!(analysis.total_allocations, 6);

        // Test memory arena
        let mut arena = MemoryArena::new(1024);
        let mut ptr = arena
            .allocate_tensor::<f32>(4, Lifetime::Temporary)
            .unwrap();
        unsafe {
            let slice = ptr.as_slice_mut();
            assert_eq!(slice.len(), 4);
        }
    }
}
