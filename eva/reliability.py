"""
Reliability computation module for EVA (Epistemically Verified AI).

This module combines epistemic stability (S), task difficulty (D),
and external verification (V) into a single reliability score R.

The goal is NOT to predict correctness, but to estimate whether
an AI output is reliable enough to accept.
"""

import math


def sigmoid(x: float) -> float:
    """
    Sigmoid activation function.

    Maps real-valued input to (0, 1), allowing the reliability
    score to be interpreted as a bounded confidence-like value.
    """
    return 1.0 / (1.0 + math.exp(-x))


def compute_reliability(
    stability: float,
    verification: float,
    difficulty: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """
    Compute the EVA reliability score.

    Mathematical form:
        R = σ( α·S + β·V − γ·D )

    Parameters
    ----------
    stability : float
        Epistemic stability score S ∈ [-1, 1].
        Measures semantic self-consistency.

    verification : float
        Verification score V ∈ [0, 1].
        Fraction of outputs passing external verification.

    difficulty : float
        Task difficulty score D ∈ [0, 1].
        Higher values indicate more epistemically fragile tasks.

    alpha : float
        Weight for stability contribution.

    beta : float
        Weight for verification contribution.

    gamma : float
        Weight for difficulty penalty.

    Returns
    -------
    float
        Reliability score R ∈ (0, 1).
        Higher values indicate greater epistemic reliability.
    """

    # Linear reliability signal
    z = (alpha * stability) + (beta * verification) - (gamma * difficulty)

    # Squash to bounded range
    return sigmoid(z)
