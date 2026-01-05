def adaptive_k(S: float,
               D: float,
               k_min: int = 3,
               k_max: int = 10) -> int:
    """
    Adaptive sampling size.

    More samples when:
    - stability is low
    - difficulty is high
    """
    uncertainty = (1.0 - S) + D
    alpha = uncertainty / 2.0

    k = int(k_min + alpha * (k_max - k_min))
    return max(k_min, min(k, k_max))
