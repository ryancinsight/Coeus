# Performance Analysis: Coeus vs PyTorch

## Executive Summary

This analysis compares Coeus performance against PyTorch baselines
to validate the claimed '<5% performance overhead' target.

### Key Findings

- **Coeus faster in 9/9 benchmarks**
.2f
.1f
.1f

## Detailed Benchmark Results

| Benchmark | Coeus (ns) | PyTorch (ns) | Speedup | Overhead |
|-----------|------------|--------------|---------|----------|
| accumulate_100 | 112.8 | 1803.3 | 15.99x | -93.7% |
| accumulate_1000 | 159.6 | 1896.9 | 11.89x | -91.6% |
| accumulate_10000 | 1267.6 | 2692.4 | 2.12x | -52.9% |
| add_100 | 113.3 | 1773.7 | 15.65x | -93.6% |
| add_1000 | 155.6 | 1880.2 | 12.08x | -91.7% |
| add_10000 | 1336.7 | 2768.7 | 2.07x | -51.7% |
| broadcast_10x10 | 110.3 | 2151.8 | 19.51x | -94.9% |
| broadcast_1x100 | 302.2 | 1868.3 | 6.18x | -83.8% |
| broadcast_1x1000 | 1044.8 | 1958.3 | 1.87x | -46.6% |

## Analysis

### Performance Claim Validation

**PERFORMANCE CLAIM VALIDATED**: Maximum overhead is -46.6%, within the <5% target.

### Performance Characteristics

- **Small tensors (100-1000 elements)**: Coeus demonstrates significant performance advantages
- **Large tensors (10000+ elements)**: Performance becomes more competitive
- **Broadcasting operations**: Mixed results depending on tensor shapes

## Recommendations

### Sustained Excellence
- Current performance optimizations are effective
- Continue monitoring performance in future releases
- Consider extending benchmarks to additional operations
