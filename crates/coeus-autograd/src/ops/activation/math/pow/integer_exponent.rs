use coeus_core::{Float, FloatElement};

// A finite binary64 exponent has at most 1024 magnitude bits. One extra
// word accommodates rounding at the upper boundary and the derivative's ±1.
// Float is sealed to the supported binary16, bfloat16, binary32 and binary64.
const MAGNITUDE_WORDS: usize = 17;

/// Exact integer control, extracted once per operation from the rounded T.
/// Tensor arithmetic stays in T; control uses the provider's native floor.
#[derive(Clone, Copy)]
pub(super) struct IntegerExponent {
    words: [u64; MAGNITUDE_WORDS],
    used: usize,
}

impl IntegerExponent {
    pub(super) fn from_value<T: Float + FloatElement>(value: T) -> Option<Self> {
        if !<T as Float>::is_finite(value) || value != <T as FloatElement>::floor(value) {
            return None;
        }
        let mut result = Self {
            words: [0; MAGNITUDE_WORDS],
            used: 0,
        };
        let mut remaining = <T as Float>::abs(value);
        let two = T::ONE + T::ONE;
        let mut bit = 0;
        // For binary integers, floor(n/2) is exact in T. Comparing n with
        // twice that quotient recovers its low bit without a numeric cast.
        // Use the provider floor: the Coeus half-format Float floor widens.
        while remaining > T::ZERO {
            let half = <T as FloatElement>::floor(remaining / two);
            if remaining != half + half {
                result.words[bit / 64] |= 1 << (bit % 64);
                result.used = bit / 64 + 1;
            }
            remaining = half;
            bit += 1;
        }
        Some(result)
    }

    pub(super) fn is_zero(&self) -> bool {
        self.used == 0
    }

    pub(super) fn is_odd(&self) -> bool {
        self.words[0] & 1 == 1
    }

    /// The caller supplies a nonzero magnitude for positive-power backward.
    pub(super) fn predecessor(mut self) -> Self {
        for word in &mut self.words {
            let (value, borrow) = word.overflowing_sub(1);
            *word = value;
            if !borrow {
                break;
            }
        }
        self.trim();
        self
    }

    pub(super) fn successor(mut self) -> Self {
        for word in &mut self.words {
            let (value, carry) = word.overflowing_add(1);
            *word = value;
            if !carry {
                break;
            }
        }
        self.trim();
        self
    }

    fn trim(&mut self) {
        self.used = self
            .words
            .iter()
            .rposition(|&word| word != 0)
            .map_or(0, |i| i + 1);
    }

    /// Least-significant-bit repeated squaring, in the original native-T order.
    #[inline]
    pub(super) fn pow_magnitude<T: Float>(&self, x: T) -> T {
        let mut acc = T::ONE;
        let mut base = x;
        for (index, &word) in self.words[..self.used].iter().enumerate() {
            let bits = if index + 1 == self.used {
                u64::BITS - word.leading_zeros()
            } else {
                u64::BITS
            };
            let mut remaining = word;
            for _ in 0..bits {
                if remaining & 1 == 1 {
                    acc *= base;
                }
                base *= base;
                remaining >>= 1;
            }
        }
        acc
    }
}
