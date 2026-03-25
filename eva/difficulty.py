import numpy as np
from typing import List, Optional

def compute_difficulty(embeddings: np.ndarray) -> float:
    """
    Estimate difficulty via variance in embedding space.
    Higher variance suggests higher uncertainty/difficulty.

    Args:
        embeddings: Array of embedding vectors for the sampled outputs. [k, d]

    Returns:
        float: Difficulty score D ∈ [0, 1].
    """
    k = embeddings.shape[0]
    if k <= 1:
        return 0.0

    # Calculate average Euclidean distance squared from the centroid as a measure of variance.
    centroid = np.mean(embeddings, axis=0)
    # Norm each embedding vector (assuming they are unit vectors from cosine similarity)
    # If using unit vectors, max variance is 2.0 (opposite directions).
    # d(u,v)^2 = 2 - 2*cos(u,v) = 2(1 - cos(u,v))
    # Average d^2 / 2 = 1 - average cos(u,v)
    # This is equivalent to 1 - stability for normalized embeddings.

    # We will use the average distance squared from centroid:
    # d_centroid^2 = ||x - centroid||^2
    d_squared = np.sum((embeddings - centroid)**2, axis=1) # [k]
    avg_d_squared = np.mean(d_squared)

    # Let's normalize it to [0, 1].
    # For unit vectors, max distance is 2.0, max distance squared is 4.0.
    # But for a set of unit vectors, the max variance (mean square distance from centroid)
    # occurs when they are spread out (e.g., mutually orthogonal).
    # We will clip to [0, 1] for safety.
    return max(0.0, min(1.0, float(avg_d_squared)))
