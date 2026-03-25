from abc import ABC, abstractmethod
from typing import List, Any

class BaseVerifier(ABC):
    """Base interface for all verifiers."""
    
    @abstractmethod
    def verify(self, outputs: List[str]) -> float:
        """
        Verify the given sampled outputs.
        
        Args:
            outputs: List of sampled LLM outputs.
            
        Returns:
            float: Verification score V ∈ [0, 1].
        """
        pass

class KeywordVerifier(BaseVerifier):
    """
    Simple verifier that checks for the presence of required keywords
    or absence of forbidden keywords.
    """
    
    def __init__(self, required: List[str] = None, forbidden: List[str] = None):
        self.required = required or []
        self.forbidden = forbidden or []
        
    def verify(self, outputs: List[str]) -> float:
        """
        Verify outputs based on keyword presence.
        Returns the proportion of samples that pass the keyword check.
        """
        if not outputs:
            return 0.0
            
        pass_count = 0
        for out in outputs:
            out_lower = out.lower()
            
            # Check required
            has_required = all(req.lower() in out_lower for req in self.required)
            
            # Check forbidden
            has_forbidden = any(forb.lower() in out_lower for forb in self.forbidden)
            
            if has_required and not has_forbidden:
                pass_count += 1
                
        return float(pass_count) / len(outputs)

class AggregateVerifier(BaseVerifier):
    """
    Combines multiple verifiers using conservative aggregation (minimum score).
    """
    
    def __init__(self, verifiers: List[BaseVerifier]):
        self.verifiers = verifiers
        
    def verify(self, outputs: List[str]) -> float:
        if not self.verifiers:
            return 1.0  # Default to 1 if no verifiers
            
        scores = [v.verify(outputs) for v in self.verifiers]
        return min(scores)
