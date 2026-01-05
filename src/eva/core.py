from typing import Callable, List, Optional, Dict
from .stability import StabilityEstimator
from .difficulty import DifficultyEstimator
from .reliability import compute_reliability, aggregate_verification
from .utils import adaptive_k
from .verification import Verifier


class EVA:
    def __init__(
        self,
        llm_fn: Callable[[str], str],
        verifiers: Optional[List[Verifier]] = None,
        threshold: float = 0.6,
        k_min: int = 3,
        k_max: int = 10,
    ):
        self.llm_fn = llm_fn
        self.verifiers = verifiers or []
        self.threshold = threshold
        self.k_min = k_min
        self.k_max = k_max

        self.stability_estimator = StabilityEstimator()
        self.difficulty_estimator = DifficultyEstimator(llm_fn)

    def run(self, prompt: str) -> Dict:
        # 1. Difficulty
        D = self.difficulty_estimator.compute(prompt)

        # 2. Initial sampling
        k = adaptive_k(S=0.5, D=D, k_min=self.k_min, k_max=self.k_max)
        outputs = [self.llm_fn(prompt) for _ in range(k)]

        # 3. Stability
        S = self.stability_estimator.compute(outputs)

        # 4. Verification
        scores = []
        weights = []

        for verifier in self.verifiers:
            scores.append(verifier.verify(prompt, outputs))
            weights.append(verifier.weight)

        V = aggregate_verification(scores, weights)

        # 5. Reliability
        R = compute_reliability(S, V, D)

        return {
            "prompt": prompt,
            "outputs": outputs,
            "stability": S,
            "difficulty": D,
            "verification": V,
            "reliability": R,
            "accepted": R >= self.threshold,
        }
