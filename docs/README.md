# Coeus Architecture Documentation

Welcome to the Coeus architecture documentation! This directory contains comprehensive
documentation about the Coeus tensor library architecture, dispatch flow, and
implementation tracking methodology.

## 📚 Documentation Overview

This documentation is organized into several interconnected documents that explain
different aspects of the Coeus architecture:

### 🎯 Start Here

- **[ARCHITECTURE_INDEX.md](./ARCHITECTURE_INDEX.md)** - Complete index and navigation guide
- **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)** - Quick reference for developers

### 🏗️ Architecture Deep Dive

#### Layer Hierarchy
Understanding each layer's responsibility and how they interact:

- **[LAYER_HIERARCHY.md](./LAYER_HIERARCHY.md)** - Layers 1-4: Tensor, Autograd, Quantization, Dense
- **[LAYER_HIERARCHY_PART2.md](./LAYER_HIERARCHY_PART2.md)** - Layers 5-6: Sparse, Storage
- **[LAYER_HIERARCHY_PART3.md](./LAYER_HIERARCHY_PART3.md)** - Layers 7-8: Backend, Dtype

#### Dispatch Flow
How operations flow through the architecture:

- **[DISPATCH_EXAMPLES.md](./DISPATCH_EXAMPLES.md)** - Basic operations and autograd
- **[DISPATCH_EXAMPLES_PART2.md](./DISPATCH_EXAMPLES_PART2.md)** - GPU and sparse operations
- **[DISPATCH_EXAMPLES_PART3.md](./DISPATCH_EXAMPLES_PART3.md)** - Quantization and complex chains

### 🔍 Implementation Tracking

#### Parity Tracking
Maintaining consistency across backends:

- **[PARITY_TRACKING.md](./PARITY_TRACKING.md)** - File structure and organization
- **[PARITY_TRACKING_PART2.md](./PARITY_TRACKING_PART2.md)** - Parity rules and scripts
- **[PARITY_TRACKING_PART3.md](./PARITY_TRACKING_PART3.md)** - Benefits and testing

#### Implementation Status
Tracking what's implemented:

- **[IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md)** - Current status and methodology
- **[IMPLEMENTATION_STATUS_PART2.md](./IMPLEMENTATION_STATUS_PART2.md)** - Tools and maintenance

### 📖 Main Documentation

- **[ARCHITECTURE_DISPATCH_FLOW.md](./ARCHITECTURE_DISPATCH_FLOW.md)** - Overview document with links to all sections

## 🚀 Quick Start Guides

### For New Contributors

1. Read [ARCHITECTURE_INDEX.md](./ARCHITECTURE_INDEX.md) for an overview
2. Study [LAYER_HIERARCHY.md](./LAYER_HIERARCHY.md) to understand layer responsibilities
3. Review [DISPATCH_EXAMPLES.md](./DISPATCH_EXAMPLES.md) for concrete examples
4. Check [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) while coding

### For Backend Developers

1. Read [PARITY_TRACKING.md](./PARITY_TRACKING.md) for file structure requirements
2. Check [IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md) for what needs implementation
3. Follow [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for common patterns
4. Use scripts in [PARITY_TRACKING_PART2.md](./PARITY_TRACKING_PART2.md) to verify parity

### For Core Developers

1. Study all [LAYER_HIERARCHY*.md](./LAYER_HIERARCHY.md) documents
2. Review [DISPATCH_EXAMPLES_PART3.md](./DISPATCH_EXAMPLES_PART3.md) for complex flows
3. Use [ARCHITECTURE_INDEX.md](./ARCHITECTURE_INDEX.md) as a reference

## 📋 Document Structure

Each major topic is split into multiple parts to keep files manageable:

```
Topic
├── Main document (overview + part 1)
├── Part 2 (continuation)
└── Part 3 (continuation)
```

**Example**: Layer Hierarchy
- `LAYER_HIERARCHY.md` - Layers 1-4
- `LAYER_HIERARCHY_PART2.md` - Layers 5-6
- `LAYER_HIERARCHY_PART3.md` - Layers 7-8

## 🎓 Learning Path

### Beginner Path (Understanding the Architecture)

1. **Overview** → [ARCHITECTURE_DISPATCH_FLOW.md](./ARCHITECTURE_DISPATCH_FLOW.md)
2. **Layers** → [LAYER_HIERARCHY.md](./LAYER_HIERARCHY.md)
3. **Examples** → [DISPATCH_EXAMPLES.md](./DISPATCH_EXAMPLES.md)
4. **Reference** → [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)

### Intermediate Path (Contributing Code)

1. **Quick Reference** → [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
2. **File Structure** → [PARITY_TRACKING.md](./PARITY_TRACKING.md)
3. **Implementation Status** → [IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md)
4. **Complex Examples** → [DISPATCH_EXAMPLES_PART3.md](./DISPATCH_EXAMPLES_PART3.md)

### Advanced Path (Architecture Design)

1. **All Layer Docs** → [LAYER_HIERARCHY*.md](./LAYER_HIERARCHY.md)
2. **All Dispatch Docs** → [DISPATCH_EXAMPLES*.md](./DISPATCH_EXAMPLES.md)
3. **All Parity Docs** → [PARITY_TRACKING*.md](./PARITY_TRACKING.md)
4. **Maintenance** → [IMPLEMENTATION_STATUS_PART2.md](./IMPLEMENTATION_STATUS_PART2.md)

## 🔑 Key Concepts

### The 8 Layers

```
1. Tensor       → User-facing API
2. Autograd     → Gradient tracking
3. Quantization → Precision management
4. Dense        → Dense operations
5. Sparse       → Sparse operations
6. Storage      → Memory management
7. Backend      → Device primitives
8. Dtype        → Type system
```

### Vertical Hierarchy

Each layer only delegates to layers below it. No layer skips levels or calls upward.

**Read more**: [LAYER_HIERARCHY.md](./LAYER_HIERARCHY.md)

### Backend Parity

All backends (CPU, GPU, TPU, NPU) maintain identical file structure and APIs.

**Read more**: [PARITY_TRACKING.md](./PARITY_TRACKING.md)

### File-Based Tracking

Implementation status is tracked through file existence and structure.

**Read more**: [IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md)

## 🛠️ Tools and Scripts

The documentation references several tools for maintaining the architecture:

- **Status Dashboard** - `scripts/status_dashboard.py`
- **Parity Checker** - `scripts/check_backend_parity.sh`
- **API Consistency** - `scripts/check_api_consistency.sh`

**Read more**: [IMPLEMENTATION_STATUS_PART2.md](./IMPLEMENTATION_STATUS_PART2.md)

## 📊 Visual Guides

### Architecture Diagram

```
Tensor (High-level API)
  ↓
Autograd (Automatic differentiation)
  ↓
Quantization (Precision management)
  ↓
Dense/Sparse (Data structure operations)
  ↓
Storage (Memory layout primitives)
  ↓
Backend (Device-specific primitives)
  ↓
Dtype (Type-safe data representation)
```

### Backend Structure

```
backend/src/
├── cpu/
│   ├── arithmetic/
│   ├── linear_algebra/
│   ├── activation/
│   └── reduction/
├── gpu/        ← Same structure
├── tpu/        ← Same structure
└── npu/        ← Same structure
```

**Read more**: [PARITY_TRACKING.md](./PARITY_TRACKING.md)

## 🤝 Contributing to Documentation

When updating the architecture or adding features:

1. **Update relevant docs** - Keep documentation in sync with code
2. **Add examples** - Include concrete examples in dispatch docs
3. **Update status** - Reflect implementation status changes
4. **Check links** - Ensure all cross-references work
5. **Update index** - Add new documents to [ARCHITECTURE_INDEX.md](./ARCHITECTURE_INDEX.md)

## 📝 Documentation Standards

- **Keep files under 50 lines when possible** - Split into parts if needed
- **Use clear headings** - Make documents scannable
- **Include code examples** - Show, don't just tell
- **Cross-reference** - Link to related documents
- **Update timestamps** - Note when documents are updated

## 🔗 Related Documentation

- **[../ARCHITECTURE.md](../ARCHITECTURE.md)** - Original architecture document
- **[../ARCHITECTURAL_ENHANCEMENT_PLAN.md](../ARCHITECTURAL_ENHANCEMENT_PLAN.md)** - Enhancement plan
- **[../README.md](../README.md)** - Project overview
- **[../CONTRIBUTING.md](../CONTRIBUTING.md)** - Contribution guidelines

## ❓ Getting Help

If you can't find what you're looking for:

1. Check [ARCHITECTURE_INDEX.md](./ARCHITECTURE_INDEX.md) for a complete index
2. Use [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for quick lookups
3. Search across all docs for specific terms
4. Ask in project discussions with specific questions

## 📅 Document Maintenance

**Last Updated**: January 2026  
**Version**: 1.0  
**Maintainers**: Coeus Core Team

These documents are maintained alongside the codebase. When making architectural
changes, please update the relevant documentation files.

---

**Happy coding! 🚀**

For questions or suggestions about this documentation, please open an issue or
discussion in the project repository.
