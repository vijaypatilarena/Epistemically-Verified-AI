from eva import EVA
from eva.stability import StabilityEstimator
from eva.difficulty import DifficultyEstimator
from eva.reliability import compute_reliability
from eva.utils import adaptive_k
from eva.verification import KeywordVerifier


def constant_llm(prompt: str) -> str:
    return "Gradient descent converges for convex functions."


def unstable_llm(prompt: str) -> str:
    if "simple" in prompt.lower():
        return "It works because convexity."
    return "I'm not sure, maybe it diverges."


def test_stability_high_for_consistent_outputs():
    outputs = [
        "A convex function has one minimum.",
        "Convex functions guarantee a single minimum.",
        "For convex objectives, the minimum is unique."
    ]

    S = StabilityEstimator().compute(outputs)
    assert S > 0.5


def test_stability_low_for_divergent_outputs():
    outputs = [
        "The sky is blue.",
        "Gradient descent converges.",
        "Bananas are fruits."
    ]

    S = StabilityEstimator().compute(outputs)
    assert S < 0.5


def test_difficulty_low_for_easy_task():
    D = DifficultyEstimator(constant_llm).compute("Explain convex optimization")
    assert 0.0 <= D <= 0.5


def test_adaptive_k_bounds():
    assert adaptive_k(1.0, 0.0, 3, 10) == 3
    k = adaptive_k(0.0, 1.0, 3, 10)
    assert 3 <= k <= 10


def test_reliability_monotonicity():
    r1 = compute_reliability(S=0.8, V=1.0, D=0.2)
    r2 = compute_reliability(S=0.4, V=1.0, D=0.2)
    assert r1 > r2


def test_eva_end_to_end_reliability_nonzero():
    eva = EVA(
        llm_fn=constant_llm,
        verifiers=[KeywordVerifier(["convex"])],
        threshold=0.6
    )

    result = eva.run("Explain why gradient descent converges.")

    assert isinstance(result, dict)
    assert "reliability" in result
    assert result["reliability"] > 0.0


def test_eva_runs_without_verifiers():
    eva = EVA(llm_fn=constant_llm)
    result = eva.run("Any prompt")
    assert result["reliability"] >= 0.0
