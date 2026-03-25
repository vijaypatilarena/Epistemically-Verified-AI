import numpy as np
from typing import List, Callable, Optional

def compute_adaptive_k(s: float, d: float, k_min: int = 3, k_max: int = 15) -> int:
    """
    Compute adaptive sampling size k based on stability and difficulty.
    k = k_min + (k_max - k_min) * (1 - S) * (1 + D)
    
    Args:
        s: Stability score S ∈ [0, 1].
        d: Difficulty score D ∈ [0, 1].
        k_min: Minimum number of samples.
        k_max: Maximum number of samples.
        
    Returns:
        int: Adjusted sampling size k.
    """
    # Clamp to [0, 1] range.
    s = max(0.0, min(1.0, s))
    d = max(0.0, min(1.0, d))
    
    # Calculate k using formula.
    # If S=1, D=0 -> k = k_min + (k_max - k_min) * 0 * 1 = k_min
    # If S=0, D=1 -> k = k_min + (k_max - k_min) * 1 * 2 = k_min + 2 * (k_max - k_min)
    # Actually, the formula (1-S)*(1+D) can result in anything between 0 and 2.
    # To keep k within k_max, we should normalize (1-S)*(1+D) to [0, 1].
    # But for simplicity, we'll just clip it.
    
    factor = (1.0 - s) * (1.0 + d)
    # Factor is in range [0, 2].
    
    # Adjust range interpolation. Let's assume we want k in [k_min, k_max].
    # To ensure it doesn't exceed k_max:
    factor_normalized = min(1.0, factor)
    
    k = k_min + round((k_max - k_min) * factor_normalized)
    
    return int(k)

def default_embedding_fn(texts: List[str]) -> np.ndarray:
    """
    A default embedding function that uses SentenceTransformers if available,
    otherwise uses a simple placeholder embedding.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(texts)
    except ImportError:
        # Fallback to simple identity-based encoding (dummy embeddings for testing)
        if not texts:
            return np.array([])
        
        # Simple placeholder embedding based on word frequencies or similar.
        # This is strictly for keeping things runnable without external deps.
        # Real use should provide a real embedding_fn.
        dim = 384
        embeddings = []
        for text in texts:
            # Deterministic pseudo-random embedding based on text hash
            np.random.seed(hash(text) % (2**32))
            embeddings.append(np.random.randn(dim))
        
        # Unit normalization
        embeddings = np.array(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.where(norms == 0, 1, norms)
