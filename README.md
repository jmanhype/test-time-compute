# test-time-compute

Implements test-time compute optimization strategies for language models. Classifies tasks, selects reasoning strategies, and allocates compute adaptively during inference.

## Components

| File | Purpose |
|---|---|
| `src/task_classifier.py` | Classifies prompts into domains (math, coding, commonsense) |
| `src/dynamic_prompt.py` | Generates chain-of-thought prompts based on task type |
| `src/reasoning_strategies.py` | Divide-and-conquer, self-refinement, best-of-N |
| `src/inference_with_gists.py` | Intermediate output generation during inference |
| `src/test_time_compute.py` | Adaptive batch sizing and compute allocation |
| `src/benchmarking.py` | Evaluation across metrics |
| `run_benchmark.py` | CLI benchmark runner |

## Requirements

- Python 3.8+
- Hugging Face Transformers
- A model (default: `Qwen/Qwen2.5-Coder-0.5B-Instruct`)

## Setup

```bash
git clone https://github.com/jmanhype/test-time-compute.git
cd test-time-compute
pip install -r requirements.txt
```

## Usage

```bash
# Default settings
python run_benchmark.py

# Custom model and parameters
python run_benchmark.py \
    --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --baseline "Qwen/Qwen2.5-Coder-0.5B" \
    --num-runs 5 \
    --warmup-runs 2 \
    --max-batch-size 8 \
    --test-cases test_cases.json
```

## Tests

```bash
pytest tests/
```

8 unit test files covering each module.

## Status

Research prototype. The benchmark runner works but published results are not included in `benchmark_results/`. The strategies implement the concepts from test-time compute papers but have not been validated against published baselines. The project references NLRL (Natural Language Reinforcement Learning) in its description but the RL component is limited to prompt-based strategy selection.

## License

MIT
