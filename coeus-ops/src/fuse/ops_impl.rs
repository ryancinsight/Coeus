use crate::fuse::expr_node::{BinaryExpr, Expr, ScalarVal, UnaryExpr};
use crate::fuse::op_tags::{
    Abs, Add, Ceil, Cos, Div, Elu, EluGrad, Exp, Floor, Gelu, GeluGrad, GeluTanh, GeluTanhGrad,
    Log, Mish, MishGrad, Mul, Neg, Recip, Relu, Round, Sigmoid, Sign, Silu, SiluGrad, Sin,
    Softplus, SoftplusGrad, Sqrt, Sub, Tanh, Trunc,
};
use coeus_core::Scalar;

// ── Operator Overloading ──

impl<L, R> std::ops::Add<Expr<R>> for Expr<L> {
    type Output = Expr<BinaryExpr<Add, L, R>>;
    #[inline(always)]
    fn add(self, rhs: Expr<R>) -> Self::Output {
        Expr(BinaryExpr {
            op: Add,
            left: self.0,
            right: rhs.0,
        })
    }
}

impl<L, R> std::ops::Sub<Expr<R>> for Expr<L> {
    type Output = Expr<BinaryExpr<Sub, L, R>>;
    #[inline(always)]
    fn sub(self, rhs: Expr<R>) -> Self::Output {
        Expr(BinaryExpr {
            op: Sub,
            left: self.0,
            right: rhs.0,
        })
    }
}

impl<L, R> std::ops::Mul<Expr<R>> for Expr<L> {
    type Output = Expr<BinaryExpr<Mul, L, R>>;
    #[inline(always)]
    fn mul(self, rhs: Expr<R>) -> Self::Output {
        Expr(BinaryExpr {
            op: Mul,
            left: self.0,
            right: rhs.0,
        })
    }
}

impl<L, R> std::ops::Div<Expr<R>> for Expr<L> {
    type Output = Expr<BinaryExpr<Div, L, R>>;
    #[inline(always)]
    fn div(self, rhs: Expr<R>) -> Self::Output {
        Expr(BinaryExpr {
            op: Div,
            left: self.0,
            right: rhs.0,
        })
    }
}

impl<L, T: Scalar> std::ops::Add<T> for Expr<L> {
    type Output = Expr<BinaryExpr<Add, L, ScalarVal<T>>>;
    #[inline(always)]
    fn add(self, rhs: T) -> Self::Output {
        Expr(BinaryExpr {
            op: Add,
            left: self.0,
            right: ScalarVal(rhs),
        })
    }
}

impl<L, T: Scalar> std::ops::Sub<T> for Expr<L> {
    type Output = Expr<BinaryExpr<Sub, L, ScalarVal<T>>>;
    #[inline(always)]
    fn sub(self, rhs: T) -> Self::Output {
        Expr(BinaryExpr {
            op: Sub,
            left: self.0,
            right: ScalarVal(rhs),
        })
    }
}

impl<L, T: Scalar> std::ops::Mul<T> for Expr<L> {
    type Output = Expr<BinaryExpr<Mul, L, ScalarVal<T>>>;
    #[inline(always)]
    fn mul(self, rhs: T) -> Self::Output {
        Expr(BinaryExpr {
            op: Mul,
            left: self.0,
            right: ScalarVal(rhs),
        })
    }
}

impl<L, T: Scalar> std::ops::Div<T> for Expr<L> {
    type Output = Expr<BinaryExpr<Div, L, ScalarVal<T>>>;
    #[inline(always)]
    fn div(self, rhs: T) -> Self::Output {
        Expr(BinaryExpr {
            op: Div,
            left: self.0,
            right: ScalarVal(rhs),
        })
    }
}

impl<L> std::ops::Neg for Expr<L> {
    type Output = Expr<UnaryExpr<Neg, L>>;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Expr(UnaryExpr {
            op: Neg,
            child: self.0,
        })
    }
}

// ── Activation methods ──

impl<L> Expr<L> {
    /// Apply ReLU to this expression.
    #[inline(always)]
    pub fn relu(self) -> Expr<UnaryExpr<Relu, L>> {
        Expr(UnaryExpr {
            op: Relu,
            child: self.0,
        })
    }

    /// Apply sigmoid to this expression.
    #[inline(always)]
    pub fn sigmoid(self) -> Expr<UnaryExpr<Sigmoid, L>> {
        Expr(UnaryExpr {
            op: Sigmoid,
            child: self.0,
        })
    }

    /// Apply tanh to this expression.
    #[inline(always)]
    pub fn tanh(self) -> Expr<UnaryExpr<Tanh, L>> {
        Expr(UnaryExpr {
            op: Tanh,
            child: self.0,
        })
    }

    /// Apply exact GELU to this expression.
    #[inline(always)]
    pub fn gelu(self) -> Expr<UnaryExpr<Gelu, L>> {
        Expr(UnaryExpr {
            op: Gelu,
            child: self.0,
        })
    }

    /// Apply exact GELU gradient to this expression.
    #[inline(always)]
    pub fn gelu_grad(self) -> Expr<UnaryExpr<GeluGrad, L>> {
        Expr(UnaryExpr {
            op: GeluGrad,
            child: self.0,
        })
    }

    /// Apply sine to this expression.
    #[inline(always)]
    pub fn sin(self) -> Expr<UnaryExpr<Sin, L>> {
        Expr(UnaryExpr {
            op: Sin,
            child: self.0,
        })
    }

    /// Apply cosine to this expression.
    #[inline(always)]
    pub fn cos(self) -> Expr<UnaryExpr<Cos, L>> {
        Expr(UnaryExpr {
            op: Cos,
            child: self.0,
        })
    }

    /// Apply exponential to this expression.
    #[inline(always)]
    pub fn exp(self) -> Expr<UnaryExpr<Exp, L>> {
        Expr(UnaryExpr {
            op: Exp,
            child: self.0,
        })
    }

    /// Apply natural log to this expression.
    #[inline(always)]
    pub fn log(self) -> Expr<UnaryExpr<Log, L>> {
        Expr(UnaryExpr {
            op: Log,
            child: self.0,
        })
    }

    /// Apply absolute value to this expression.
    #[inline(always)]
    pub fn abs(self) -> Expr<UnaryExpr<Abs, L>> {
        Expr(UnaryExpr {
            op: Abs,
            child: self.0,
        })
    }

    /// Apply square root to this expression.
    #[inline(always)]
    pub fn sqrt(self) -> Expr<UnaryExpr<Sqrt, L>> {
        Expr(UnaryExpr {
            op: Sqrt,
            child: self.0,
        })
    }

    /// Apply SiLU to this expression.
    #[inline(always)]
    pub fn silu(self) -> Expr<UnaryExpr<Silu, L>> {
        Expr(UnaryExpr {
            op: Silu,
            child: self.0,
        })
    }

    /// Apply SiLU gradient to this expression.
    #[inline(always)]
    pub fn silu_grad(self) -> Expr<UnaryExpr<SiluGrad, L>> {
        Expr(UnaryExpr {
            op: SiluGrad,
            child: self.0,
        })
    }

    /// Apply Mish to this expression.
    #[inline(always)]
    pub fn mish(self) -> Expr<UnaryExpr<Mish, L>> {
        Expr(UnaryExpr {
            op: Mish,
            child: self.0,
        })
    }

    /// Apply Mish gradient to this expression.
    #[inline(always)]
    pub fn mish_grad(self) -> Expr<UnaryExpr<MishGrad, L>> {
        Expr(UnaryExpr {
            op: MishGrad,
            child: self.0,
        })
    }

    /// Apply ELU to this expression.
    #[inline(always)]
    pub fn elu(self) -> Expr<UnaryExpr<Elu, L>> {
        Expr(UnaryExpr {
            op: Elu,
            child: self.0,
        })
    }

    /// Apply ELU gradient to this expression.
    #[inline(always)]
    pub fn elu_grad(self) -> Expr<UnaryExpr<EluGrad, L>> {
        Expr(UnaryExpr {
            op: EluGrad,
            child: self.0,
        })
    }

    /// Apply softplus to this expression.
    #[inline(always)]
    pub fn softplus(self) -> Expr<UnaryExpr<Softplus, L>> {
        Expr(UnaryExpr {
            op: Softplus,
            child: self.0,
        })
    }

    /// Apply softplus gradient to this expression.
    #[inline(always)]
    pub fn softplus_grad(self) -> Expr<UnaryExpr<SoftplusGrad, L>> {
        Expr(UnaryExpr {
            op: SoftplusGrad,
            child: self.0,
        })
    }

    /// Apply tanh-approximation GELU to this expression.
    #[inline(always)]
    pub fn gelu_tanh(self) -> Expr<UnaryExpr<GeluTanh, L>> {
        Expr(UnaryExpr {
            op: GeluTanh,
            child: self.0,
        })
    }

    /// Apply tanh-approximation GELU gradient to this expression.
    #[inline(always)]
    pub fn gelu_tanh_grad(self) -> Expr<UnaryExpr<GeluTanhGrad, L>> {
        Expr(UnaryExpr {
            op: GeluTanhGrad,
            child: self.0,
        })
    }

    /// Apply reciprocal (1/x) to this expression.
    #[inline(always)]
    pub fn recip(self) -> Expr<UnaryExpr<Recip, L>> {
        Expr(UnaryExpr {
            op: Recip,
            child: self.0,
        })
    }

    /// Apply signum to this expression.
    #[inline(always)]
    pub fn sign(self) -> Expr<UnaryExpr<Sign, L>> {
        Expr(UnaryExpr {
            op: Sign,
            child: self.0,
        })
    }

    /// Apply floor to this expression.
    #[inline(always)]
    pub fn floor(self) -> Expr<UnaryExpr<Floor, L>> {
        Expr(UnaryExpr {
            op: Floor,
            child: self.0,
        })
    }

    /// Apply ceil to this expression.
    #[inline(always)]
    pub fn ceil(self) -> Expr<UnaryExpr<Ceil, L>> {
        Expr(UnaryExpr {
            op: Ceil,
            child: self.0,
        })
    }

    /// Apply round to this expression.
    #[inline(always)]
    pub fn round(self) -> Expr<UnaryExpr<Round, L>> {
        Expr(UnaryExpr {
            op: Round,
            child: self.0,
        })
    }

    /// Apply truncation toward zero to this expression.
    #[inline(always)]
    pub fn trunc(self) -> Expr<UnaryExpr<Trunc, L>> {
        Expr(UnaryExpr {
            op: Trunc,
            child: self.0,
        })
    }
}
