# Coeus Architecture Documentation Index

This index provides a comprehensive guide to understanding the Coeus architecture,
its dispatch flow, and implementation tracking methodology.

## Quick Start

New to Coeus? Start here:

1. **[ARCHITECTURE_DISPATCH_FLOW.md](./ARCHITECTURE_DISPATCH_FLOW.md)** - Overview of the architecture
2. **[LAYER_HIERARCHY.md](./LAYER_HIERARCHY.md)** - Understanding each layer's responsibility
3. **[DISPATCH_EXAMPLES.md](./DISPATCH_EXAMPLES.md)** - See how operations flow through layers

## Core Concepts

### Architecture Layers

The Coeus architecture consists of 8 distinct layers, each with specific responsibilities:

```
Tensor → Autograd → Quantization → Dense/Sparse → Storage → Backend → Dtype
```

**Read these documents to understand each layer:**

- **[LAYER_HIERARCHY.md](./LAYER_HIERARCHY.md)** - Layers 1-4 (Tensor, Autograd, Quantization, Dense)
- **[LAYER_HIERARCHY_PART2.md](./LAYER_HIERARCHY_PART2.md)** - Layers 5-6 (Sparse, Storage)
- **[LAYER_HIERARCHY_PART3.md](./LAYER_HIERARCHY_PART3.md)** - Layers 7-8 (Backend, Dtype)

### Dispatch Flow

Understanding how operations dispatch through layers:

- **[DISPATCH_EXAMPLES.md](./DISPATCH_EXAMPLES.md)** - Examples 1-2 (Basic operations, Autograd)
- **[DISPATCH_EXAMPLES_PART2.md](./DISPATCH_EXAMPLES_PART2.md)** - Examples 3-4 (GPU, Sparse)
- **[DISPATCH_EXAMPLES_PART3.md](./DISPATCH_EXAMPLES_PART3.md)** - Examples 5-6 (Quantization, Complex chains)

### Parity Tracking

How we maintain consistency across backends:

- **[PARITY_TRACKING.md](./PARITY_TRACKING.md)** - File structure and organization
- **[PARITY_TRACKING_PART2.md](./PARITY_TRACKING_PART2.md)** - Parity rules and scripts
- **[PARITY_TRACKING_PART3.md](./PARITY_TRACKING_PART3.md)** - Benefits and testing

### Implementation Status

Tracking what's implemented and what's not:

- **[IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md)** - Current status and methodology
- **[IMPLEMENTATION_STATUS_PART2.md](./IMPLEMENTATION_STATUS_PART2.md)** - Tools and maintenance

## Documentation by Role

### For New Contributors

Start with these documents to understand the codebase:

1. [ARCHITECTURE_DISPATCH_FLOW.md](./ARCHITECTURE_DISPATCH_FLOW.md) - High-level overview
2. [LAYER_HIERARCHY.md](./LAYER_HIERARCHY.md) - Layer responsibilities
3. [DISPATCH_EXAMPLES.md](./DISPATCH_EXAMPLES.md) - Concrete examples
4. [PARITY_TRACKING.md](./PARITY_TRACKING.md) - File organization

### For Backend Developers

Implementing operations for specific devices (CPU, GPU, TPU, NPU):

1. [PARITY_TRACKING.md](./PARITY_TRACKING.md) - File structure requirements
2. [PARITY_TRACKING_PART2.md](./PARITY_TRACKING_PART2.md) - Parity rules
3. [IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md) - What needs implementation
4. [LAYER_HIERARCHY_PART3.md](./LAYER_HIERARCHY_PART3.md) - Backend layer details

### For Core Developers

Working on high-level APIs and architecture:

1. [LAYER_HIERARCHY.md](./LAYER_HIERARCHY.md) - All layer details
2. [DISPATCH_EXAMPLES_PART3.md](./DISPATCH_EXAMPLES_PART3.md) - Complex operation chains
3. [ARCHITECTURE_DISPATCH_FLOW.md](./ARCHITECTURE_DISPATCH_FLOW.md) - Full architecture

### For Maintainers

Managing the project and tracking progress:

1. [IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md) - Current status
2. [IMPLEMENTATION_STATUS_PART2.md](./IMPLEMENTATION_STATUS_PART2.md) - Tracking tools
3. [PARITY_TRACKING_PART2.md](./PARITY_TRACKING_PART2.md) - Parity scripts

## Key Principles

### 1. Vertical Hierarchy

Each layer delegates to lower layers only. No layer skips levels or calls upward.

**Example**: Tensor → Autograd → Dense → Storage → Backend

**Read**: [LAYER_HIERARCHY.md](./LAYER_HIERARCHY.md)

### 2. Single Responsibility

Each layer has one clear responsibility and doesn't duplicate logic from other layers.

**Example**: 
- Tensor: User-facing API
- Autograd: Gradient tracking
- Backend: Device primitives

**Read**: [LAYER_HIERARCHY.md](./LAYER_HIERARCHY.md), [LAYER_HIERARCHY_PART2.md](./LAYER_HIERARCHY_PART2.md)

### 3. Backend Parity

All backends (CPU, GPU, TPU, NPU) maintain identical file structure and APIs.

**Example**: `backend/src/cpu/arithmetic/add.rs` ↔ `backend/src/gpu/arithmetic/add.rs`

**Read**: [PARITY_TRACKING.md](./PARITY_TRACKING.md)

### 4. File-Based Tracking

Implementation status is tracked through file existence and structure.

**Example**: Missing file = Missing implementation

**Read**: [PARITY_TRACKING_PART2.md](./PARITY_TRACKING_PART2.md)

## Common Workflows

### Adding a New Operation

1. **Understand the dispatch flow**: [DISPATCH_EXAMPLES.md](./DISPATCH_EXAMPLES.md)
2. **Identify the correct layer**: [LAYER_HIERARCHY.md](./LAYER_HIERARCHY.md)
3. **Follow file structure**: [PARITY_TRACKING.md](./PARITY_TRACKING.md)
4. **Implement across backends**: [IMPLEMENTATION_STATUS_PART2.md](./IMPLEMENTATION_STATUS_PART2.md)

### Debugging an Operation

1. **Trace the dispatch flow**: [DISPATCH_EXAMPLES_PART3.md](./DISPATCH_EXAMPLES_PART3.md)
2. **Check each layer**: [LAYER_HIERARCHY.md](./LAYER_HIERARCHY.md)
3. **Verify backend implementation**: [PARITY_TRACKING.md](./PARITY_TRACKING.md)

### Checking Implementation Status

1. **Review status tables**: [IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md)
2. **Run status scripts**: [IMPLEMENTATION_STATUS_PART2.md](./IMPLEMENTATION_STATUS_PART2.md)
3. **Check parity**: [PARITY_TRACKING_PART2.md](./PARITY_TRACKING_PART2.md)

## Architecture Diagrams

### Full Layer Stack

```
┌─────────────────────────────────────────┐
│         Tensor Layer (Public API)       │
│  - Operator overloading                 │
│  - Method chaining                      │
│  - Shape inference                      │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      Autograd Layer (Gradients)         │
│  - Computational graph                  │
│  - Backward pass                        │
│  - Gradient accumulation                │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│   Quantization Layer (Precision)        │
│  - Quantization schemes                 │
│  - Dequantization                       │
│  - Quantized operations                 │
└─────────────────┬───────────────────────┘
                  │
         ┌────────┴────────┐
         │                 │
┌────────▼──────┐  ┌──────▼────────┐
│  Dense Layer  │  │ Sparse Layer  │
│  - Dense ops  │  │ - Sparse ops  │
│  - Broadcast  │  │ - Format conv │
└────────┬──────┘  └──────┬────────┘
         │                 │
         └────────┬────────┘
                  │
┌─────────────────▼───────────────────────┐
│      Storage Layer (Memory)             │
│  - Memory allocation                    │
│  - Shape/stride management              │
│  - Device transfer                      │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│     Backend Layer (Device Primitives)   │
│  ┌─────┬─────┬─────┬─────┐             │
│  │ CPU │ GPU │ TPU │ NPU │             │
│  └─────┴─────┴─────┴─────┘             │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│       Dtype Layer (Type System)         │
│  - Type definitions                     │
│  - Type conversions                     │
│  - Type traits                          │
└─────────────────────────────────────────┘
```

### Backend File Structure

```
backend/src/
├── cpu/
│   ├── arithmetic/
│   │   ├── add.rs
│   │   ├── sub.rs
│   │   └── ...
│   ├── linear_algebra/
│   │   ├── matmul.rs
│   │   └── ...
│   └── activation/
│       ├── relu.rs
│       └── ...
├── gpu/
│   ├── arithmetic/      ← Same structure as CPU
│   ├── linear_algebra/  ← Same structure as CPU
│   └── activation/      ← Same structure as CPU
├── tpu/
│   └── (same structure)
└── npu/
    └── (same structure)
```

**Read**: [PARITY_TRACKING.md](./PARITY_TRACKING.md)

## Additional Resources

### Architecture Decision Records (ADRs)

Recent architectural decisions and their rationale:

- **[ADR-036: Quantization Crate Extraction](./adr/036-quantization-crate-extraction.md)** - Extracting quantization logic to dedicated crate
- **[ADR-037: Dense Crate Creation](./adr/037-dense-crate-creation.md)** - Creating dedicated dense operations crate
- **[ADR-038: Storage Simplification](./adr/038-storage-simplification.md)** - Limiting storage to basic operations only
- **[ADR-039: Hierarchical File Structure](./adr/039-hierarchical-file-structure.md)** - Deep vertical file organization
- **[ADR-040: Domain Separation Enforcement](./adr/040-domain-separation-enforcement.md)** - Strict domain boundaries

### Developer Guides

Practical guides for working with the architecture:

- **[DEVELOPER_GUIDE_HIERARCHICAL_STRUCTURE.md](./DEVELOPER_GUIDE_HIERARCHICAL_STRUCTURE.md)** - Navigate the hierarchical file structure
- **[MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md)** - Migrate code to enhanced architecture

### Related Documentation

- **[../ARCHITECTURE.md](../ARCHITECTURE.md)** - Original architecture document
- **[../ARCHITECTURAL_ENHANCEMENT_PLAN.md](../ARCHITECTURAL_ENHANCEMENT_PLAN.md)** - Enhancement plan
- **[../README.md](../README.md)** - Project overview

### Code Examples

- **[../examples/](../examples/)** - Example usage
- **[../tests/](../tests/)** - Test suite

### Scripts

- **[../scripts/status_dashboard.py](../scripts/)** - Implementation status
- **[../scripts/check_backend_parity.sh](../scripts/)** - Parity checking
- **[../scripts/check_api_consistency.sh](../scripts/)** - API consistency

## Contributing

When contributing to Coeus:

1. **Read the architecture docs** to understand the design
2. **Follow the layer hierarchy** - don't skip layers
3. **Maintain backend parity** - implement across all backends
4. **Update documentation** - keep docs in sync with code
5. **Run parity checks** - ensure consistency

## Questions?

If you have questions about the architecture:

1. Check this index for relevant documentation
2. Read the specific document for your question
3. Look at code examples in the relevant layer
4. Ask in the project discussions

## Document Maintenance

This documentation is maintained alongside the code. When making architectural
changes:

1. Update the relevant documentation files
2. Update this index if adding new documents
3. Ensure examples reflect current implementation
4. Run documentation linters and validators

---

**Last Updated**: January 2026
**Version**: 1.0
**Maintainers**: Coeus Core Team
