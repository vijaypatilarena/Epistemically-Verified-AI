def compute_reliability(v: float, s: float, d: float) -> float:
    """
    Compute reliability score R = V * S / (1 + D).
    
    Properties:
    - R ∈ [0, 1]
    - ∂R/∂S > 0
    - ∂R/∂D < 0
    
    Args:
        v: Verification score V ∈ [0, 1].
        s: Stability score S ∈ [0, 1].
        d: Difficulty score D ∈ [0, 1].
        
    Returns:
        float: Reliability score R.
    """
    # Ensure inputs are in [0, 1]
    v = max(0.0, min(1.0, v))
    s = max(0.0, min(1.0, s))
    d = max(0.0, min(1.0, d))
    
    # R = V * S / (1 + D)
    # Max R when V=1, S=1, D=0 -> R = 1 / (1 + 0) = 1.0
    # Min R when V=0 or S=0 -> R = 0
    reliability = (v * s) / (1.0 + d)
    
    return float(reliability)
