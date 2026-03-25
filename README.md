# EVA — Epistemically Verified AI

EVA is a model-agnostic reliability layer for LLM outputs. It evaluates the reliability of a generated response using three core signals:

1.  **Stability (S):** Measures semantic consistency across multiple sampled outputs. If multiple generations lead to similar semantic results, the model is more likely to be correct.
2.  **Difficulty (D):** Intrinsic task uncertainty based on the variance of sampled outputs. Higher variance suggests a more difficult or ambiguous task.
3.  **Verification (V):** External validation via pluggable verifiers (e.g., keyword checks, schema validation, code execution).

## Mathematical Framework

The reliability score **R** is computed as:

$R = \frac{V \cdot S}{1 + D}$

where:
- $R \in [0, 1]$
- $S \in [0, 1]$ (Average pairwise cosine similarity of embeddings)
- $D \in [0, 1]$ (Normalised variance in embedding space)
- $V \in [0, 1]$ (Aggregation of verification scores)

### Adaptive Sampling
To optimize cost and performance, EVA uses adaptive sampling for the number of outputs $k$:

$k = k_{min} + (k_{max} - k_{min}) \cdot (1 - S) \cdot (1 + D)$

## Installation

It is recommended to use a virtual environment:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package in editable mode with test dependencies
pip install -e ".[test]"
```

## Quick Start

```python
from eva import EVA

# Example LLM function
def my_llm_fn(prompt, k=3):
    # k is the number of samples requested
    return ["Paris is the capital of France."] * k

# Initialize EVA
eva = EVA(llm_fn=my_llm_fn)

# Run evaluation
result = eva.run("What is the capital of France?")

print(f"Reliability: {result['reliability']}")
if result['accepted']:
    print("Output accepted!")
```

## Features
- **Model-Agnostic**: Works with any LLM by providing a function wrapper.
- **Semantic Stability**: Uses text embeddings to measure consistency.
- **Pluggable Verifiers**: Add custom verification logic for your specific use-case.
- **Adaptive Efficiency**: Dynamically adjusts sampling based on confidence.

## Testing

```bash
pytest
```
