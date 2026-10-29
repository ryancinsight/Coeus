// ── einsum — Einstein summation notation ──
//
// Supports the following common ML patterns via direct dispatch to
// optimised coeus-ops kernels:
//
//   "ij,jk->ik"       matrix multiply  (→ matmul)
//   "bij,bjk->bik"    batched matmul   (→ bmm)
//   "bi,bj->bij"      outer product per batch (→ broadcast + mul)
//   "ij->ji"          2-D transpose    (→ permute [1,0])
//   "...ij->...ji"    last-2D transpose  (→ permute swap last two dims)
//   "i,i->"           dot product      (→ element-wise mul + sum)
//   "i,j->ij"         outer product    (→ unsqueeze + broadcast + mul)
//   "ii->"            trace            (→ diagonal sum)
//   "ij,j->i"         matrix-vector    (→ matmul with shape reshape)
//   "bik,bk->bi"      batched matvec   (→ reshape + batched matmul + squeeze)
//
// Unrecognised patterns and subscript/shape mismatches are typed
// [`BackendError`](coeus_core::BackendError) failures, never panics.
//
// The module is split by pattern arity so each leaf owns one recognition
// table: [`dispatch`] parses the subscript and counts operands, [`single`] and
// [`binary`] hold the one- and two-operand arms, [`ternary`] the three-operand
// chain decomposition, and [`subscript`] the typed rejections they share.

mod binary;
mod dispatch;
mod single;
mod subscript;
mod ternary;

pub use dispatch::einsum;
pub use ternary::einsum3;

#[cfg(test)]
mod tests;
