"""
Stability estimation module for EVA (Epistemically Verified AI).

This module measures epistemic stability by evaluating the
semantic self-consistency of multiple AI-generated outputs.

The key idea:
If a model truly "knows" an answer, repeated samples should
be semantically consistent. If it hallucinates, samples diverge.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class StabilityEstimator:
    """
    Estimates epistemic stability via semantic self-consistency.

    Stability is computed as the average pairwise cosine similarity
    between embeddings of multiple AI-generated outputs.

    Output range:
        S ∈ [-1, 1]
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Parameters
        ----------
        model_name : str
            Name of the sentence-transformers model used to embed outputs.
        """
        self.embedder = SentenceTransformer(model_name)

    def compute(self, outputs: List[str]) -> float:
        """
        Compute the stability score S.

        Parameters
        ----------
        outputs : List[str]
            Multiple outputs generated from the same prompt.

        Returns
        -------
        float
            Stability score in [-1, 1].
        """

        n = len(outputs)
        if n < 2:
            return 0.0

        embeddings = self.embedder.encode(
            outputs,
            normalize_embeddings=True
        )

        sim_matrix = cosine_similarity(embeddings)

        score = 0.0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                score += sim_matrix[i, j]
                count += 1

        return score / count
