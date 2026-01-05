"""
Task difficulty estimation.
"""

from typing import Callable
import numpy as np


class DifficultyEstimator:
    """
    Estimates difficulty via output variability.
    """

    def __init__(self, llm_fn: Callable[[str], str], samples: int = 5):
        self.llm_fn = llm_fn
        self.samples = samples

    def compute(self, prompt: str) -> float:
        outputs = [self.llm_fn(prompt) for _ in range(self.samples)]
        unique = len(set(outputs))
        return min(1.0, unique / self.samples)
