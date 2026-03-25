import numpy as np
from typing import List, Callable, Optional
from sklearn.metrics.pairwise import cosine_similarity

def compute_stability(embeddings: np.ndarray) -> float:
    """
    Compute semantic consistency across multiple sampled outputs.
    Average over all unique pairs of cosine similarity.

    Args:
        embeddings: Array of embedding vectors for the sampled outputs. [k, d]

    Returns:
        float: Stability score S ∈ [0, 1].
    """
    k = embeddings.shape[0]
    if k <= 1:
        return 1.0

    # Pairwise cosine similarity [k, k]
    similarities = cosine_similarity(embeddings)

    # Average over unique pairs (excluding diagonal)
    # sum(triu(similarities, k=1)) / (k * (k - 1) / 2)
    upper_tri_indices = np.triu_indices(k, k=1)
    if len(upper_tri_indices[0]) == 0:
        return 1.0

    avg_similarity = np.mean(similarities[upper_tri_indices])

    # Ensure result is in [0, 1] range (cosine similarity can be [-1, 1], so we normalize if needed)
    # However, for semantic similarity of related outputs, it's typically positive.
    # We'll clip or shift to ensure positivity if required by the framework.
    return max(0.0, min(1.0, float(avg_similarity)))
