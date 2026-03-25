import numpy as np
from typing import List, Callable, Optional, Dict, Any, Union
from eva.stability import compute_stability
from eva.difficulty import compute_difficulty
from eva.verification import BaseVerifier, AggregateVerifier
from eva.reliability import compute_reliability
from eva.utils import compute_adaptive_k, default_embedding_fn

class EVA:
    """
    EVA — Epistemically Verified AI
    
    A model-agnostic layer for evaluating LLM output reliability through 
    Stability, Difficulty, and Verification.
    """
    
    def __init__(
        self, 
        llm_fn: Callable[[str, int], List[str]], 
        verifiers: Optional[List[BaseVerifier]] = None, 
        threshold: float = 0.6,
        embedding_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        k_min: int = 3,
        k_max: int = 15
    ):
        """
        Initialize the EVA reliability engine.
        
        Args:
            llm_fn: A function that takes a prompt and an integer (n_samples) 
                   and returns a list of sampled outputs strings.
            verifiers: List of verifiers to use for output verification (V).
            threshold: Reliability score threshold below which the output is rejected.
            embedding_fn: Function to compute text embeddings for [S, D] calculation.
            k_min: Minimum number of samples for adaptive sampling.
            k_max: Maximum number of samples for adaptive sampling.
        """
        self.llm_fn = llm_fn
        self.verifiers = verifiers
        self.threshold = threshold
        self.embedding_fn = embedding_fn or default_embedding_fn
        self.k_min = k_min
        self.k_max = k_max
        self._aggregate_verifier = AggregateVerifier(verifiers or [])

    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Evaluate the reliability of an LLM prompt using a multi-sample check.
        
        Returns:
            Dict containing outputs, stability, difficulty, verification, 
            reliability, and its acceptance status.
        """
        # Step 1: Initial sampling
        # We start with k_min samples.
        k = self.k_min
        outputs = self.llm_fn(prompt, k)
        
        if not outputs:
            return {
                "outputs": [],
                "stability": 0.0,
                "difficulty": 1.0,
                "verification": 0.0,
                "reliability": 0.0,
                "accepted": False
            }
            
        # Step 2: Compute initial signals
        embeddings = self.embedding_fn(outputs)
        s = compute_stability(embeddings)
        d = compute_difficulty(embeddings)
        
        # Step 3: Adaptive sampling
        # If stability is low or difficulty is high, we take more samples.
        k_adj = compute_adaptive_k(s, d, self.k_min, self.k_max)
        
        if k_adj > k:
            # We need additional samples
            additional_outputs = self.llm_fn(prompt, k_adj - k)
            outputs.extend(additional_outputs)
            embeddings = self.embedding_fn(outputs)
            s = compute_stability(embeddings)
            d = compute_difficulty(embeddings)
            
        # Step 4: Verification
        # Apply external validation to all sampled outputs.
        v = self._aggregate_verifier.verify(outputs)
        
        # Step 5: Reliability computation
        # R = V * S / (1 + D)
        reliability = compute_reliability(v, s, d)
        
        # Decision
        accepted = reliability >= self.threshold
        
        return {
          "outputs": outputs,
          "stability": s,
          "difficulty": d,
          "verification": v,
          "reliability": reliability,
          "accepted": accepted
        }
