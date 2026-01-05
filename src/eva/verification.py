from abc import ABC, abstractmethod
from typing import List


class Verifier(ABC):
    """Abstract base class for epistemic verifiers."""

    weight: float = 1.0

    @abstractmethod
    def verify(self, prompt: str, outputs: List[str]) -> float:
        """
        Returns a verification score in [0, 1].
        """
        pass


class KeywordVerifier(Verifier):
    def __init__(self, keywords: List[str], weight: float = 1.0):
        self.keywords = keywords
        self.weight = weight

    def verify(self, prompt: str, outputs: List[str]) -> float:
        hits = 0
        for out in outputs:
            if any(k.lower() in out.lower() for k in self.keywords):
                hits += 1
        return hits / max(len(outputs), 1)
