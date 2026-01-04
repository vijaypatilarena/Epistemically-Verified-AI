"""
Difficulty estimation module for EVA (Epistemically Verified AI).

This module estimates task difficulty by measuring how sensitive
model outputs are to controlled perturbations of the prompt.

Core idea:
Easy tasks are robust to perturbations.
Hard tasks show semantic divergence under small changes.
"""

from typing import Callable, List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class DifficultyEstimator:
    """
    Estimates intrinsic task difficulty via perturbation sensitivity.

    Difficulty is defined as:
        D = 1 - mean semantic similarity across perturbed outputs

    Output range:
        D ∈ [0, 1]
    """

    def __init__(
        self,
        generator_fn: Callable[[str], str],
        embedder_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Parameters
        ----------
        generator_fn : Callable[[str], str]
            Function that takes a prompt and returns a model output.
            This abstracts away the LLM backend.

        embedder_model : str
            Sentence-transformer model for semantic embedding.
        """
        self.generator_fn = generator_fn
        self.embedder = SentenceTransformer(embedder_model)

    def _perturb_prompt(self, prompt: str) -> List[str]:
        """
        Generate semantically equivalent prompt perturbations.

        NOTE:
        This is intentionally simple and interpretable.
        Contributors may replace this with richer perturbation schemes.
        """

        return [
            prompt,
            f"Please answer the following: {prompt}",
            f"In simple terms, {prompt}",
            f"From a theoretical perspective, {prompt}",
            f"Explain concisely: {prompt}",
        ]

    def compute(self, prompt: str) -> float:
        """
        Compute difficulty score D for a given prompt.

        Parameters
        ----------
        prompt : str
            Original task prompt.

        Returns
        -------
        float
            Difficulty score D ∈ [0, 1].
        """

        # Generate perturbed prompts
        perturbed_prompts = self._perturb_prompt(prompt)

        # Generate outputs
        outputs = [
            self.generator_fn(p) for p in perturbed_prompts
        ]

        # Embed outputs
        embeddings = self.embedder.encode(
            outputs,
            normalize_embeddings=True
        )

        # Compute pairwise similarity
        sim_matrix = cosine_similarity(embeddings)

        # Average similarity (excluding diagonal)
        score = 0.0
        count = 0

        n = len(outputs)
        for i in range(n):
            for j in range(i + 1, n):
                score += sim_matrix[i, j]
                count += 1

        mean_similarity = score / count if count > 0 else 0.0

        # Difficulty is inverse of stability
        difficulty = 1.0 - mean_similarity

        # Clamp for numerical safety
        return float(np.clip(difficulty, 0.0, 1.0))
