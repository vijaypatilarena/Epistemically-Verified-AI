"""
Utility functions for EVA (Epistemically Verified AI).

This module contains helper utilities that support EVA's
core epistemic logic without introducing domain assumptions.
"""

import math


def adaptive_k(
    S: float,
    D: float,
    k_min: int = 3,
    k_max: int = 10,
    lam: float = 4.0,
    mu: float = 4.0,
) -> int:
    """
    Compute the adaptive number of samples to draw based on
    epistemic uncertainty.

    The rule increases sampling when:
    - stability S is low (answers diverge)
    - difficulty D is high (task is brittle)

    Mathematical form:
        k = k_min + ceil( λ(1 − S) + μD )

    Parameters
    ----------
    S : float
        Epistemic stability score S ∈ [-1, 1].
        Higher means more consistent outputs.

    D : float
        Task difficulty score D ∈ [0, 1].
        Higher means more sensitive to perturbation.

    k_min : int
        Minimum number of samples.

    k_max : int
        Maximum number of samples.

    lam : float
        Sensitivity to instability (1 − S).

    mu : float
        Sensitivity to difficulty D.

    Returns
    -------
    int
        Number of samples k to draw, clamped to [k_min, k_max].
    """

    # Raw adaptive computation
    k = k_min + math.ceil(lam * (1.0 - S) + mu * D)

    # Clamp to safe bounds
    if k < k_min:
        return k_min
    if k > k_max:
        return k_max
    return k
