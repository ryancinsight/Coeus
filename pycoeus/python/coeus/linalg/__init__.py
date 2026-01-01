"""Linear Algebra operations.

This module provides linear algebra operations compatible with torch.linalg.
"""

from .._coeus import inv, norm, vector_norm, det, solve, cholesky, qr, svd

__all__ = [
    "inv",
    "norm",
    "vector_norm",
    "det",
    "solve",
    "cholesky",
    "qr",
    "svd",
]

