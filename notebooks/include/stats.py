"""Confidence intervals and summary numbers."""
import numpy as np

from .config import DELTA


def chebyshev_eps(N, delta=DELTA, m=1):
    # accuracy on N points is a mean of N Bernoulli variables, so its variance is at most 1/(4N).
    # Chebyshev gives P(|acc_hat - acc| >= eps) <= 1/(4 N eps^2). Setting that to delta/m
    # (union bound over m models) and solving for eps:
    return 1 / (2 * np.sqrt(N * delta / m))


def mean_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1))
