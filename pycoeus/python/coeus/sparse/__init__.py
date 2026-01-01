"""Sparse tensor operations for Coeus."""

from .._coeus import SparseCsrTensor, CooTensor

def sparse_csr_tensor(data, indices, indptr, shape):
    """Create a CSR sparse tensor."""
    return SparseCsrTensor(data, indices, indptr, shape)

def sparse_coo_tensor(data, row_indices, col_indices, shape):
    """Create a COO sparse tensor."""
    return CooTensor(data, row_indices, col_indices, shape)

__all__ = [
    "SparseCsrTensor",
    "CooTensor",
    "sparse_csr_tensor",
    "sparse_coo_tensor",
]
