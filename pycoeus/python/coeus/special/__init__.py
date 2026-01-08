"""
Special mathematical functions compatible with torch.special.
"""

from .._coeus import (
    erf, erfc, erfinv, ndtr,
    gamma, lgamma, digamma, polygamma,
    logit, expit, sinc,
    bessel_j0, bessel_j1
)

__all__ = [
    "erf", "erfc", "erfinv", "ndtr",
    "gamma", "lgamma", "digamma", "polygamma",
    "logit", "expit", "sinc",
    "bessel_j0", "bessel_j1"
]
