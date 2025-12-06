---
name: Bug Report
about: Report a bug or unexpected behavior
title: "[BUG] "
labels: ["bug", "triage"]
assignees: []
---

## Bug Report

### Description
A clear and concise description of the bug.

### Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

### Expected Behavior
A clear and concise description of what you expected to happen.

### Actual Behavior
What actually happened instead.

### Environment
- **OS**: [e.g., Windows 11, Ubuntu 22.04, macOS 13]
- **Rust Version**: [e.g., 1.70.0]
- **Coeus Version**: [e.g., v0.1.0]
- **Backend**: [CPU/GPU/TPU]
- **Python Version** (if applicable): [e.g., 3.11]

### Code Example
```rust
// Please provide a minimal code example that reproduces the issue
use coeus_tensor::Tensor;
use coeus_backend::CpuBackend;
// ... your code here
```

### Error Output
```
Paste any error messages, stack traces, or logs here
```

### Additional Context
- Is this a regression? (did it work in previous versions?)
- Are you using any specific features/flags?
- Any relevant performance metrics or benchmarks?
- Screenshots or additional files that might help

### Checklist
- [ ] I have searched existing issues for similar bugs
- [ ] I have provided a minimal reproducible example
- [ ] I have included environment details
- [ ] I have checked that this bug still exists in the latest version
