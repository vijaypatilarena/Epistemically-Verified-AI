"""
Verification module for EVA (Epistemically Verified AI).

This module defines the verification interface used to assess
whether a model output is externally grounded, correct, or valid
with respect to a task-specific verifier.

Verification is intentionally abstract:
- EVA does not assume what "truth" is
- EVA only asks whether verification succeeded
"""

from abc import ABC, abstractmethod
from typing import Any


class Verifier(ABC):
    """
    Abstract base class for verifiers.

    A verifier checks whether a model output satisfies
    some external correctness criterion.
    """

    @abstractmethod
    def verify(self, output: str, context: Any = None) -> bool:
        """
        Verify a single model output.

        Parameters
        ----------
        output : str
            The model-generated output to verify.

        context : Any, optional
            Optional task-specific context (e.g. ground truth,
            constraints, retrieved documents, test cases).

        Returns
        -------
        bool
            True if verification passes, False otherwise.
        """
        pass


class NoOpVerifier(Verifier):
    """
    A default verifier that always returns True.

    This allows EVA to run even when no external
    verification mechanism is available.
    """

    def verify(self, output: str, context: Any = None) -> bool:
        return True


class KeywordVerifier(Verifier):
    """
    Simple rule-based verifier.

    Verification passes if all required keywords
    appear in the output.
    """

    def __init__(self, required_keywords):
        self.required_keywords = required_keywords

    def verify(self, output: str, context: Any = None) -> bool:
        output_lower = output.lower()
        return all(
            keyword.lower() in output_lower
            for keyword in self.required_keywords
        )
