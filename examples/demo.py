import time
import numpy as np
from eva import EVA, KeywordVerifier

# Mock LLM function that simulates variable reliability
def mock_llm_fn(prompt: str, k: int):
    """
    Returns k samples for the given prompt.
    Simulates high stability for common questions and low for others.
    """
    if "capital of France" in prompt:
        return ["Paris is the capital of France."] * k
    elif "fact about space" in prompt:
        # High variance simulation
        return [
            "The Sun is a star.",
            "Jupiter is the largest planet.",
            "Space is mostly empty.",
            "Black holes are dense.",
            "The Moon orbits Earth."
        ][:k] if k <= 5 else ["Space is vast."] * k
    else:
        return ["I don't know."] * k

# Mock embedding function for demo
def mock_embedding_fn(texts):
    # Map messages to simple unit vectors for predictability
    embs = []
    for t in texts:
        t_low = t.lower()
        if "paris" in t_low:
            embs.append([1.0, 0.0, 0.0])
        elif "sun" in t_low or "jupiter" in t_low or "space" in t_low:
            # Different space facts get different vectors to lower stability
            if "sun" in t_low: embs.append([0.0, 1.0, 0.0])
            elif "jupiter" in t_low: embs.append([0.0, 0.0, 1.0])
            else: embs.append([0.5, 0.5, 0.5])
        else:
            embs.append([1.0, 1.0, 1.0])
    
    # Normalize
    embs = np.array(embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.where(norms == 0, 1, norms)

# Initialize EVA with a keyword verifier
verifiers = [KeywordVerifier(required=["Paris"])]

eva = EVA(
    llm_fn=mock_llm_fn, 
    verifiers=verifiers, 
    threshold=0.7,
    embedding_fn=mock_embedding_fn
)

def run_demo():
    print("--- EVA Demo ---")
    
    # Case 1: High Reliability
    print("\nCase 1: Stable and Verified (Capital of France)")
    result = eva.run("What is the capital of France?")
    print(f"Reliability: {result['reliability']:.4f}")
    print(f"Stability: {result['stability']:.4f}")
    print(f"Difficulty: {result['difficulty']:.4f}")
    print(f"Verification: {result['verification']:.4f}")
    print(f"Accepted: {result['accepted']}")
    
    # Case 2: Low Reliability Due to Verification Failure
    print("\nCase 2: Stable but Not Verified (Space fact with 'Paris' required)")
    result = eva.run("Tell me a fact about space.")
    print(f"Reliability: {result['reliability']:.4f}")
    print(f"Verification: {result['verification']:.4f}")
    print(f"Accepted: {result['accepted']}")

    # Case 3: Low Reliability Due to Stability/Difficulty
    print("\nCase 3: Unstable and Unverified")
    # Using a new EVA instance without a required keyword to see stability effect
    eva_generic = EVA(llm_fn=mock_llm_fn, threshold=0.5, embedding_fn=mock_embedding_fn)
    result = eva_generic.run("Tell me a fact about space.")
    print(f"Reliability: {result['reliability']:.4f}")
    print(f"Stability: {result['stability']:.4f}")
    print(f"Difficulty: {result['difficulty']:.4f}")
    print(f"Accepted: {result['accepted']}")

if __name__ == "__main__":
    run_demo()
