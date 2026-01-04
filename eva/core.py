"""
Core engine for EVA (Epistemically Verified AI).

This module orchestrates:
- multiple stochastic generations
- epistemic stability estimation
- task difficulty estimation
- external verification
- reliability-based acceptance

EVA acts as a control layer on top of any AI model.
"""

from typing import Callable, List, Dict, Any

from eva.stability import StabilityEstimator
from eva.difficulty import DifficultyEstimator
from eva.reliability import compute_reliability
from eva.verification import Verifier, NoOpVerifier
from eva.utils import adaptive_k


class EVA:
    """
    Epistemically Verified AI (EVA)

    EVA evaluates whether an AI output should be accepted
    based on epistemic reliability rather than surface confidence.
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        verifiers: List[Verifier] | None = None,
        threshold: float = 0.75,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        k_min: int = 3,
        k_max: int = 10,
    ):
        """
        Parameters
        ----------
        llm_fn : Callable[[str], str]
            Function that takes a prompt and returns a model output.
            This abstracts away the underlying LLM.

        verifiers : List[Verifier], optional
            External verifiers used to validate outputs.
            If None, a NoOpVerifier is used.

        threshold : float
            Reliability threshold above which outputs are accepted.

        alpha, beta, gamma : float
            Weights for stability, verification, and difficulty.

        k_min, k_max : int
            Minimum and maximum number of samples used.
        """

        self.llm_fn = llm_fn
        self.verifiers = verifiers or [NoOpVerifier()]

        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.k_min = k_min
        self.k_max = k_max

        # Core estimators
        self.stability_estimator = StabilityEstimator()
        self.difficulty_estimator = DifficultyEstimator(llm_fn)

    def _verification_score(self, outputs: List[str]) -> float:
        """
        Compute average verification score V over outputs.
        """

        total = 0.0
        for output in outputs:
            score = sum(
                1.0 if verifier.verify(output) else 0.0
                for verifier in self.verifiers
            )
            total += score / len(self.verifiers)

        return total / len(outputs)

    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Run EVA on a given prompt.

        Returns a diagnostic dictionary containing:
        - acceptance decision
        - reliability score
        - stability, difficulty, verification metrics
        - number of samples used
        - generated outputs
        """

        # Step 1: Initial sampling
        outputs = [self.llm_fn(prompt) for _ in range(self.k_min)]
        stability = self.stability_estimator.compute(outputs)

        # Step 2: Difficulty estimation
        difficulty = self.difficulty_estimator.compute(prompt)

        # Step 3: Adaptive sampling
        k = adaptive_k(
            S=stability,
            D=difficulty,
            k_min=self.k_min,
            k_max=self.k_max,
        )

        if k > self.k_min:
            outputs.extend(self.llm_fn(prompt) for _ in range(k - self.k_min))
            stability = self.stability_estimator.compute(outputs)

        # Step 4: Verification
        verification = self._verification_score(outputs)

        # Step 5: Reliability computation
        reliability = compute_reliability(
            stability=stability,
            verification=verification,
            difficulty=difficulty,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )

        accepted = reliability >= self.threshold

        return {
            "accepted": accepted,
            "reliability": reliability,
            "stability": stability,
            "verification": verification,
            "difficulty": difficulty,
            "samples_used": k,
            "outputs": outputs,
        }
