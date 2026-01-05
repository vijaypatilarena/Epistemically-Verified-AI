import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class StabilityEstimator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def compute(self, outputs: List[str]) -> float:
        if len(outputs) < 2:
            return 1.0

        embeddings = self.model.encode(outputs)
        sims = cosine_similarity(embeddings)

        upper = sims[np.triu_indices(len(outputs), k=1)]
        mean_sim = float(np.mean(upper))

        # Entropy penalty
        probs = np.clip(upper, 1e-6, 1.0)
        entropy = -np.mean(probs * np.log(probs))

        # Normalize entropy to [0,1]
        entropy_penalty = np.tanh(entropy)

        return max(0.0, mean_sim * (1.0 - entropy_penalty))
