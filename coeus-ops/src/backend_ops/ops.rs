#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Relu,
    ReluGrad,
    Sigmoid,
    SigmoidGrad,
    Tanh,
    TanhGrad,
    Gelu,
    GeluGrad,
    Sin,
    Cos,
    Exp,
    Log,
    Neg,
    Abs,
    Sqrt,
    Silu,
    SiluGrad,
    Mish,
    MishGrad,
    /// ELU: x >= 0 ? x : exp(x) - 1 (alpha=1.0)
    Elu,
    /// ELU gradient: x >= 0 ? 1 : exp(x)
    EluGrad,
    /// Softplus: log(1 + exp(x))
    Softplus,
    /// Softplus gradient: sigmoid(x)
    SoftplusGrad,
    /// GELU tanh approximation
    GeluTanh,
    /// GELU tanh gradient
    GeluTanhGrad,
    /// LeakyReLU: x >= 0 ? x : slope * x. Slope encoded as f64::to_bits() in u64.
    LeakyRelu(u64),
    /// LeakyReLU gradient: x >= 0 ? 1 : slope. Same slope encoding.
    LeakyReluGrad(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    Sum,
    Max,
    Min,
}
