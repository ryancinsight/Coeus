//! Closed-set operation tags for CPU dispatch.

/// Binary element-wise operation tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// Addition: a + b.
    Add,
    /// Subtraction: a - b.
    Sub,
    /// Multiplication: a * b.
    Mul,
    /// Division: a / b.
    Div,
    /// Element-wise equality: 1 if a == b, else 0.
    Eq,
    /// Element-wise inequality: 1 if a != b, else 0.
    Ne,
    /// Element-wise less-than: 1 if a < b, else 0.
    Lt,
    /// Element-wise greater-than: 1 if a > b, else 0.
    Gt,
    /// Element-wise less-than-or-equal: 1 if a <= b, else 0.
    Le,
    /// Element-wise greater-than-or-equal: 1 if a >= b, else 0.
    Ge,
}

/// Reduction operation tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    /// Sum of all elements.
    Sum,
    /// Product of all elements.
    Prod,
    /// Arithmetic mean of all elements.
    Mean,
    /// Maximum element.
    Max,
    /// Minimum element.
    Min,
}

/// CPU unary operation dispatch tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuUnaryOp {
    /// Rectified Linear Unit: max(0, x).
    Relu,
    /// ReLU gradient: 1 if x > 0, else 0.
    ReluGrad,
    /// Logistic sigmoid: 1 / (1 + e^(-x)).
    Sigmoid,
    /// Sigmoid gradient: σ(x) * (1 - σ(x)).
    SigmoidGrad,
    /// Hyperbolic tangent: tanh(x).
    Tanh,
    /// Tanh gradient: 1 - tanh(x)^2.
    TanhGrad,
    /// Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
    Gelu,
    /// GELU gradient.
    GeluGrad,
    /// Element-wise sine: sin(x).
    Sin,
    /// Element-wise cosine: cos(x).
    Cos,
    /// Element-wise exponential: e^x.
    Exp,
    /// Element-wise natural logarithm: ln(x).
    Log,
    /// Gauss error function: erf(x).
    Erf,
    /// Complementary Gauss error function: erfc(x) = 1 - erf(x).
    Erfc,
    /// Natural logarithm of the absolute gamma function: ln|Γ(x)|.
    Lgamma,
    /// tan(x)
    Tan,
    /// arcsin(x)
    Asin,
    /// arccos(x)
    Acos,
    /// arctan(x)
    Atan,
    /// hyperbolic sine: sinh(x)
    Sinh,
    /// hyperbolic cosine: cosh(x)
    Cosh,
    /// base-2 logarithm: log2(x)
    Log2,
    /// base-10 logarithm: log10(x)
    Log10,
    /// base-2 exponential: exp2(x) = 2^x
    Exp2,
    /// inverse hyperbolic tangent: atanh(x)
    Atanh,
    /// inverse hyperbolic sine: asinh(x)
    Asinh,
    /// inverse hyperbolic cosine: acosh(x)
    Acosh,
    /// exp(x) - 1
    Expm1,
    /// ln(1 + x)
    Log1p,
    /// Element-wise negation: -x.
    Neg,
    /// Element-wise absolute value: |x|.
    Abs,
    /// Element-wise square root: sqrt(x).
    Sqrt,
    /// SiLU (Sigmoid Linear Unit): x * sigmoid(x).
    Silu,
    /// SiLU gradient.
    SiluGrad,
    /// Mish: x * tanh(softplus(x)).
    Mish,
    /// Mish gradient.
    MishGrad,
    /// ELU: x > 0 ? x : alpha * (e^x - 1).
    Elu,
    /// ELU gradient.
    EluGrad,
    /// Softplus: ln(1 + e^x).
    Softplus,
    /// Softplus gradient: sigmoid(x).
    SoftplusGrad,
    /// Tanh-approximated GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))).
    GeluTanh,
    /// Tanh-GELU gradient.
    GeluTanhGrad,
    /// Leaky ReLU: x > 0 ? x : slope * x. The packed `u64` is the negative-slope bit pattern.
    LeakyRelu(u64),
    /// Leaky ReLU gradient. The packed `u64` is the negative-slope bit pattern.
    LeakyReluGrad(u64),
    /// Hardtanh: clamp(x, min, max). The packed `u64` stores `(min, max)` as
    /// little-endian `f32` bit patterns.
    Hardtanh(u64),
    /// Hardtanh gradient: 1 inside (min, max), 0 outside. Same packed-min/max convention.
    HardtanhGrad(u64),
    /// Hardsigmoid: clamp(x/6 + 0.5, 0, 1). No parameters.
    Hardsigmoid,
    /// Hardsigmoid gradient: 1/6 inside (-3, 3), 0 outside.
    HardsigmoidGrad,
    /// Hardswish: x * ReLU6(x+3) / 6. No parameters.
    Hardswish,
    /// Hardswish gradient.
    HardswishGrad,
    /// Hardshrink: x if |x| > λ else 0. Packed `u64` is `λ.bits()`.
    Hardshrink(u64),
    /// Hardshrink gradient: 1 if |x| > λ else 0.
    HardshrinkGrad(u64),
    /// Softshrink: sign(x) * max(|x| - λ, 0). Packed `u64` is `λ.bits()`.
    Softshrink(u64),
    /// Softshrink gradient: 1 if |x| > λ else 0.
    SoftshrinkGrad(u64),
    /// Softsign: x / (1 + |x|). No parameters.
    Softsign,
    /// Softsign gradient: 1 / (1 + |x|)^2.
    SoftsignGrad,
    /// Threshold: x if x > threshold else value. The packed `u64` stores
    /// `(threshold, value)` as little-endian `f32` bit patterns.
    Threshold(u64),
    /// Threshold gradient: 1 if x > threshold else 0.
    ThresholdGrad(u64),
    /// CELU: max(0,x) + min(0, α·(exp(x/α) - 1)). Packed `u64` is `α.bits()` (default α = 1.0).
    Celu(u64),
    /// CELU gradient: 1 if x ≥ 0 else exp(x/α).
    CeluGrad(u64),
    /// Element-wise reciprocal: 1/x
    Recip,
    /// Element-wise signum: -1, 0, or 1
    Sign,
    /// Element-wise floor: largest integer ≤ x
    Floor,
    /// Element-wise ceil: smallest integer ≥ x
    Ceil,
    /// Element-wise round to nearest integer, ties to even (IEEE-754
    /// roundTiesToEven, matching torch.round)
    Round,
    /// Element-wise truncation toward zero
    Trunc,
}

impl CpuUnaryOp {
    /// Decode the low and high halves of a packed activation parameter pair.
    #[must_use]
    pub const fn decode_parameter_pair(bits: u64) -> [f32; 2] {
        [
            f32::from_bits(bits as u32),
            f32::from_bits((bits >> 32) as u32),
        ]
    }

    /// Decode the two runtime parameters carried by a parameterized activation.
    ///
    /// Hardtanh and threshold store two `f32` bit patterns. Single-parameter
    /// activations store one `f64` bit pattern, converted to the provider's
    /// `f32` parameter representation here; their unused second parameter is
    /// zero. Non-parameterized operations return `None`.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the provider parameter ABI is f32; single activation parameters enter as f64 bit patterns"
    )]
    pub const fn parameter_pair(self) -> Option<[f32; 2]> {
        let bits = match self {
            Self::Hardtanh(bits)
            | Self::HardtanhGrad(bits)
            | Self::Threshold(bits)
            | Self::ThresholdGrad(bits) => bits,
            Self::LeakyRelu(bits)
            | Self::LeakyReluGrad(bits)
            | Self::Hardshrink(bits)
            | Self::HardshrinkGrad(bits)
            | Self::Softshrink(bits)
            | Self::SoftshrinkGrad(bits)
            | Self::Celu(bits)
            | Self::CeluGrad(bits) => return Some([f64::from_bits(bits) as f32, 0.0]),
            _ => return None,
        };
        Some(Self::decode_parameter_pair(bits))
    }
}

#[cfg(test)]
mod op_tag_tests {
    use super::CpuUnaryOp;

    #[test]
    fn parameter_pairs_preserve_both_bit_patterns() {
        let first = -1.25_f32;
        let second = 2.5_f32;
        let bits = u64::from(first.to_bits()) | (u64::from(second.to_bits()) << 32);

        assert_eq!(
            CpuUnaryOp::Hardtanh(bits).parameter_pair(),
            Some([first, second])
        );
        assert_eq!(CpuUnaryOp::Relu.parameter_pair(), None);
    }
}
