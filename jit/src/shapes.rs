//! Dynamic shape specialization for JIT compilation
//!
//! This module handles variable tensor dimensions by analyzing shape patterns
//! and creating specialized kernels for common cases.

use crate::error::Result;
use crate::graph::ComputationGraph;
use std::collections::HashMap;

/// Shape representation for specialization
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    pub dims: Vec<usize>,
}

/// Key for shape specialization cache
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShapeKey {
    pub input_shapes: Vec<Shape>,
    pub output_shapes: Vec<Shape>,
}

/// Shape pattern observed during execution
#[derive(Debug, Clone)]
pub struct ShapePattern {
    pub input_shapes: Vec<Shape>,
    pub output_shapes: Vec<Shape>,
    pub frequency: usize,
    pub first_seen: std::time::Instant,
    pub last_seen: std::time::Instant,
}

/// Specialized kernel for specific shape combinations
#[derive(Debug, Clone)]
pub struct SpecializedKernel {
    pub shape_key: ShapeKey,
    pub kernel_id: String,
    pub performance_score: f32,
}

/// Collection of shape specializations
#[derive(Debug)]
pub struct ShapeSpecializations {
    pub specializations: Vec<SpecializedKernel>,
}

/// Shape analyzer for pattern detection
#[derive(Debug)]
pub struct ShapeAnalyzer {
    patterns: HashMap<ShapeKey, ShapePattern>,
    min_frequency_threshold: usize,
}

/// Dynamic shape specialization system
#[derive(Debug)]
pub struct ShapeSpecializer {
    analyzer: ShapeAnalyzer,
    specialization_cache: HashMap<ShapeKey, SpecializedKernel>,
    specialization_threshold: usize,
    max_specializations: usize,
}

impl Shape {
    /// Create a new shape from dimensions
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Get the total number of elements
    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    /// Check if shape is compatible with another for broadcasting
    pub fn can_broadcast_to(&self, other: &Shape) -> bool {
        if self.dims.len() != other.dims.len() {
            return false;
        }

        // Simplified broadcasting check - in reality would be more complex
        for (a, b) in self.dims.iter().zip(&other.dims) {
            if *a != *b && *a != 1 && *b != 1 {
                return false;
            }
        }

        true
    }
}

impl ShapeKey {
    /// Create a shape key from input and output shapes
    pub fn new(input_shapes: Vec<Shape>, output_shapes: Vec<Shape>) -> Self {
        Self {
            input_shapes,
            output_shapes,
        }
    }

    /// Create a shape key from runtime tensor shapes
    pub fn from_shapes(shapes: &[Shape]) -> Self {
        // Simplified: assume first half are inputs, second half are outputs
        let mid = shapes.len() / 2;
        let input_shapes = shapes[..mid].to_vec();
        let output_shapes = shapes[mid..].to_vec();

        Self::new(input_shapes, output_shapes)
    }
}

impl ShapeAnalyzer {
    /// Create a new shape analyzer
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            min_frequency_threshold: 5,
        }
    }

    /// Record a shape pattern observation
    pub fn record_pattern(&mut self, key: ShapeKey) {
        let now = std::time::Instant::now();

        let pattern = self
            .patterns
            .entry(key.clone())
            .or_insert_with(|| ShapePattern {
                input_shapes: Vec::new(),
                output_shapes: Vec::new(),
                frequency: 0,
                first_seen: now,
                last_seen: now,
            });

        pattern.frequency += 1;
        pattern.last_seen = now;

        // Copy shapes from key if not already set
        if pattern.input_shapes.is_empty() {
            pattern.input_shapes = key.input_shapes.clone();
            pattern.output_shapes = key.output_shapes.clone();
        }
    }

    /// Analyze patterns and return frequent ones
    pub fn analyze_patterns(&self, _graph: &ComputationGraph) -> Result<Vec<ShapePattern>> {
        let mut frequent_patterns = Vec::new();

        for pattern in self.patterns.values() {
            if pattern.frequency >= self.min_frequency_threshold {
                frequent_patterns.push(pattern.clone());
            }
        }

        // Sort by frequency (most frequent first)
        frequent_patterns.sort_by(|a, b| b.frequency.cmp(&a.frequency));

        Ok(frequent_patterns)
    }

    /// Get statistics about observed patterns
    pub fn stats(&self) -> ShapeStats {
        let total_patterns = self.patterns.len();
        let frequent_patterns = self
            .patterns
            .values()
            .filter(|p| p.frequency >= self.min_frequency_threshold)
            .count();

        let avg_frequency = if total_patterns > 0 {
            self.patterns.values().map(|p| p.frequency).sum::<usize>() as f32
                / total_patterns as f32
        } else {
            0.0
        };

        ShapeStats {
            total_patterns,
            frequent_patterns,
            avg_frequency,
        }
    }
}

impl Default for ShapeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about shape patterns
#[derive(Debug, Clone)]
pub struct ShapeStats {
    pub total_patterns: usize,
    pub frequent_patterns: usize,
    pub avg_frequency: f32,
}

impl ShapeSpecializer {
    /// Create a new shape specializer
    pub fn new() -> Self {
        Self {
            analyzer: ShapeAnalyzer::new(),
            specialization_cache: HashMap::new(),
            specialization_threshold: 10,
            max_specializations: 50,
        }
    }

    /// Analyze shape patterns and create specializations
    pub fn specialize_shapes(&mut self, _graph: &ComputationGraph) -> Result<ShapeSpecializations> {
        let patterns = self.analyzer.analyze_patterns(&ComputationGraph::new())?; // Use empty graph since patterns are stored in analyzer

        let mut specializations = Vec::new();

        for pattern in patterns {
            if pattern.frequency >= self.specialization_threshold
                && specializations.len() < self.max_specializations
            {
                // Create specialization key
                let key =
                    ShapeKey::new(pattern.input_shapes.clone(), pattern.output_shapes.clone());

                // Check if we already have this specialization
                if !self.specialization_cache.contains_key(&key) {
                    let specialized_kernel = self.create_specialization(&pattern)?;
                    self.specialization_cache
                        .insert(key.clone(), specialized_kernel.clone());
                    specializations.push(specialized_kernel);
                }
            }
        }

        Ok(ShapeSpecializations { specializations })
    }

    /// Select optimal specialization for runtime shapes
    pub fn select_specialization(&self, runtime_shapes: &[Shape]) -> Option<&SpecializedKernel> {
        let key = ShapeKey::from_shapes(runtime_shapes);
        self.specialization_cache.get(&key)
    }

    /// Record a runtime shape observation
    pub fn record_runtime_shapes(&mut self, shapes: &[Shape]) {
        let key = ShapeKey::from_shapes(shapes);
        self.analyzer.record_pattern(key);
    }

    /// Create a specialized kernel for a shape pattern
    fn create_specialization(&self, pattern: &ShapePattern) -> Result<SpecializedKernel> {
        // Generate a unique kernel ID based on shapes
        let mut kernel_id = "specialized_".to_string();

        for shape in &pattern.input_shapes {
            kernel_id.push_str(&format!(
                "{}x",
                shape
                    .dims
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("x")
            ));
        }
        kernel_id.push_str("_to_");

        for shape in &pattern.output_shapes {
            kernel_id.push_str(&format!(
                "{}x",
                shape
                    .dims
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("x")
            ));
        }

        // Estimate performance score based on shape properties
        let performance_score = self.estimate_performance(&pattern);

        let key = ShapeKey::new(pattern.input_shapes.clone(), pattern.output_shapes.clone());

        Ok(SpecializedKernel {
            shape_key: key,
            kernel_id,
            performance_score,
        })
    }

    /// Estimate performance score for a shape pattern
    fn estimate_performance(&self, pattern: &ShapePattern) -> f32 {
        // Simplified performance estimation
        // In reality, this would consider:
        // - Memory layout compatibility
        // - Cache efficiency
        // - SIMD utilization
        // - Memory bandwidth requirements

        let mut score = 1.0;

        // Prefer shapes that are powers of 2 (good for SIMD)
        for shape in &pattern.input_shapes {
            for &dim in &shape.dims {
                if dim & (dim - 1) == 0 {
                    // Power of 2
                    score *= 1.2;
                }
            }
        }

        // Prefer contiguous memory layouts
        for shape in &pattern.input_shapes {
            if self.is_contiguous_layout(shape) {
                score *= 1.1;
            }
        }

        // Frequency bonus
        score *= (pattern.frequency as f32).sqrt();

        score
    }

    /// Check if a shape has a contiguous memory layout
    fn is_contiguous_layout(&self, _shape: &Shape) -> bool {
        // Simplified: assume most shapes are contiguous
        // In reality, this would check strides vs shape
        true
    }

    /// Get specialization statistics
    pub fn stats(&self) -> SpecializationStats {
        SpecializationStats {
            total_specializations: self.specialization_cache.len(),
            analyzer_stats: self.analyzer.stats(),
        }
    }
}

/// Statistics about shape specialization
#[derive(Debug, Clone)]
pub struct SpecializationStats {
    pub total_specializations: usize,
    pub analyzer_stats: ShapeStats,
}

impl Default for ShapeSpecializer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_creation() {
        let shape = Shape::new(vec![32, 64, 128]);
        assert_eq!(shape.dims, vec![32, 64, 128]);
        assert_eq!(shape.size(), 32 * 64 * 128);
    }

    #[test]
    fn test_shape_broadcasting() {
        let shape1 = Shape::new(vec![1, 64, 128]);
        let shape2 = Shape::new(vec![32, 1, 128]);

        assert!(shape1.can_broadcast_to(&shape2));
        assert!(shape2.can_broadcast_to(&shape1));
    }

    #[test]
    fn test_shape_key_creation() {
        let inputs = vec![Shape::new(vec![32, 64]), Shape::new(vec![64, 128])];
        let outputs = vec![Shape::new(vec![32, 128])];

        let key = ShapeKey::new(inputs.clone(), outputs.clone());
        assert_eq!(key.input_shapes, inputs);
        assert_eq!(key.output_shapes, outputs);
    }

    #[test]
    fn test_shape_analyzer() {
        let mut analyzer = ShapeAnalyzer::new();

        let key = ShapeKey::new(
            vec![Shape::new(vec![32, 64])],
            vec![Shape::new(vec![32, 128])],
        );

        // Record the same pattern multiple times
        for _ in 0..10 {
            analyzer.record_pattern(key.clone());
        }

        let patterns = analyzer.analyze_patterns(&ComputationGraph::new()).unwrap();
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].frequency, 10);
    }

    #[test]
    fn test_shape_specializer() {
        let mut specializer = ShapeSpecializer::new();

        // Record patterns
        let key = ShapeKey::new(
            vec![Shape::new(vec![32, 64])],
            vec![Shape::new(vec![32, 128])],
        );

        for _ in 0..15 {
            // Above threshold
            specializer.analyzer.record_pattern(key.clone());
        }

        let specializations = specializer
            .specialize_shapes(&ComputationGraph::new())
            .unwrap();
        assert_eq!(specializations.specializations.len(), 1);

        let kernel = &specializations.specializations[0];
        assert!(kernel.kernel_id.contains("specialized"));
        assert!(kernel.performance_score > 0.0);
    }

    #[test]
    fn test_specialization_selection() {
        let mut specializer = ShapeSpecializer::new();

        let shapes = vec![Shape::new(vec![32, 64]), Shape::new(vec![32, 128])];

        // Record and specialize
        let key = ShapeKey::from_shapes(&shapes);
        for _ in 0..15 {
            specializer.analyzer.record_pattern(key.clone());
        }

        let _specializations = specializer
            .specialize_shapes(&ComputationGraph::new())
            .unwrap();

        // Try to select the specialization
        let selected = specializer.select_specialization(&shapes);
        assert!(selected.is_some());
    }
}
