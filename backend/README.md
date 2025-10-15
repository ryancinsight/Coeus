# Coeus Backend

Compute device abstractions for the Coeus deep learning framework.

## Overview

This crate provides backend trait abstractions for executing tensor operations on different compute substrates (CPU, GPU, NPU).

## Features

- **Backend Trait**: Zero-cost device abstraction via static dispatch
- **CpuBackend**: Native CPU execution (SIMD-ready)
- **DeviceInfo**: Runtime capability queries
- **Thread-Safe**: All backends are `Send + Sync`

## Usage

```rust
use coeus_backend::{Backend, CpuBackend};

let backend = CpuBackend::new();
assert_eq!(backend.device_name(), "cpu");
assert!(backend.supports("arithmetic"));
```

## Architecture

Backend hierarchy (ADR-003):

```
Backend
├── CpuBackend      ✅ Implemented
├── GpuBackend      Future (Sprint 4)
└── NpuBackend      Future (Sprint 5)
```

## Testing

```bash
cargo test --package coeus-backend
```

**Coverage**: 9/9 tests passing (6 unit + 3 doc)

