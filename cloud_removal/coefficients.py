"""Scalar reference for thesis equations (3-7) and (3-8)."""
import math


def consistency_coefficients(sigma, *, sigma_min=0.02, sigma_data=0.5):
    if not all(math.isfinite(v) for v in (sigma, sigma_min, sigma_data)):
        raise ValueError('Noise parameters must be finite')
    if sigma_min <= 0 or sigma_data <= 0 or sigma < sigma_min:
        raise ValueError('Require sigma >= sigma_min > 0 and sigma_data > 0')
    delta = sigma - sigma_min
    return (sigma_data**2 / (delta**2 + sigma_data**2),
            sigma_data * delta / math.sqrt(sigma_data**2 + sigma**2))
