from typing import List


def aggregate_verification(verifier_scores: List[float],
                           verifier_weights: List[float]) -> float:
    """
    Combines verification signals conservatively.

    Returns:
        V ∈ [0, 1]
    """
    if not verifier_scores:
        return 1.0  # epistemic neutrality

    weighted_mean = sum(
        s * w for s, w in zip(verifier_scores, verifier_weights)
    ) / sum(verifier_weights)

    hard_min = min(verifier_scores)

    # Conservative aggregation
    return 0.5 * weighted_mean + 0.5 * hard_min


def compute_reliability(S: float, V: float, D: float) -> float:
    """
    Final epistemic reliability score.

    R = S * V * (1 - D)
    """
    return max(0.0, min(1.0, S * V * (1.0 - D)))
