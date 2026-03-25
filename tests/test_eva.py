import pytest
import numpy as np
from eva import EVA, KeywordVerifier, compute_stability, compute_difficulty, compute_reliability, compute_adaptive_k

def test_stability_identical():
    # Identical embeddings should have stability 1.0
    embeddings = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    s = compute_stability(embeddings)
    assert s == pytest.approx(1.0)

def test_stability_orthogonal():
    # Orthogonal embeddings should have low stability (0.0 for cosine similarity if clamped)
    embeddings = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    s = compute_stability(embeddings)
    assert s == pytest.approx(0.0)

def test_difficulty_identical():
    # Identical embeddings should have difficulty 0.0
    embeddings = np.array([[1, 0, 0], [1, 0, 0]])
    d = compute_difficulty(embeddings)
    assert d == pytest.approx(0.0)

def test_reliability_formula():
    # R = V * S / (1 + D)
    # V=1, S=1, D=0 -> R=1
    assert compute_reliability(1.0, 1.0, 0.0) == pytest.approx(1.0)
    # V=1, S=0.5, D=0 -> R=0.5
    assert compute_reliability(1.0, 0.5, 0.0) == pytest.approx(0.5)
    # V=1, S=1, D=1 -> R=0.5
    assert compute_reliability(1.0, 1.0, 1.0) == pytest.approx(0.5)
    # V=0, any S, D -> R=0
    assert compute_reliability(0.0, 1.0, 0.0) == pytest.approx(0.0)

def test_adaptive_k():
    # S=1, D=0 -> k_min
    assert compute_adaptive_k(1.0, 0.0, k_min=3, k_max=10) == 3
    # S=0, D=1 -> k_max
    assert compute_adaptive_k(0.0, 1.0, k_min=3, k_max=10) == 10
    
def test_keyword_verifier():
    verifier = KeywordVerifier(required=["test"], forbidden=["error"])
    assert verifier.verify(["this is a test"]) == 1.0
    assert verifier.verify(["this is a test with error"]) == 0.0
    assert verifier.verify(["no such keyword"]) == 0.0
    # Average verification score across multiple samples
    assert verifier.verify(["test", "other", "test"]) == pytest.approx(2/3)

def test_eva_e2e():
    def mock_llm(prompt, n):
        if "correct" in prompt:
            return ["Correct answer"] * n
        return ["Wrong answer"] * n

    # Mock embedding function for testing:
    # "Correct answer" returns [1, 0], "Wrong answer" returns [0, 1]
    def mock_emb(texts):
        embs = []
        for t in texts:
            if "Correct" in t:
                embs.append([1.0, 0.0])
            else:
                embs.append([0.0, 1.0])
        return np.array(embs)
        
    verifier = KeywordVerifier(required=["Correct"])
    eva = EVA(llm_fn=mock_llm, verifiers=[verifier], embedding_fn=mock_emb, threshold=0.8)
    
    # Positive case
    result_pos = eva.run("Give me the correct answer")
    assert result_pos["accepted"] is True
    assert result_pos["reliability"] == pytest.approx(1.0)

    # Negative case (Verification failure)
    result_neg = eva.run("Give me the wrong answer")
    assert result_neg["accepted"] is False
    assert result_neg["verification"] == 0.0
    assert result_neg["reliability"] == 0.0

if __name__ == "__main__":
    pytest.main([__file__])
