# Test Time Compute Optimization for NLRL

This project implements Natural Language Reinforcement Learning (NLRL) with optimized test-time compute strategies. It focuses on enhancing model performance through efficient compute allocation and adaptive batch sizing.

## Features

- **Task Classification**: Automatically classifies input prompts into domains (math, coding, commonsense)
- **Dynamic Prompting**: Generates context-aware prompts with chain-of-thought reasoning
- **Reasoning Strategies**: Implements divide-and-conquer, self-refinement, and best-of-N approaches
- **Inference with Gists**: Provides intermediate outputs during generation for debugging
- **Test Time Compute Optimization**: Implements adaptive batch sizing and compute allocation
- **Comprehensive Benchmarking**: Evaluates model performance across various metrics

## Installation

```bash
git clone https://github.com/yourusername/test_time_compute.git
cd test_time_compute
pip install -r requirements.txt
```

## Usage

Run benchmarks with default settings:
```bash
python run_benchmark.py
```

Run benchmarks with custom settings:
```bash
python run_benchmark.py \
    --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --baseline "Qwen/Qwen2.5-Coder-0.5B" \
    --num-runs 5 \
    --warmup-runs 2 \
    --max-batch-size 8 \
    --test-cases test_cases.json
```

## Project Structure

```
test_time_compute/
├── src/                    # Source code
│   ├── task_classifier.py  # Task classification module
│   ├── dynamic_prompt.py   # Dynamic prompting module
│   ├── reasoning_strategies.py  # Reasoning strategies
│   ├── inference_with_gists.py  # Inference with gists
│   └── test_time_compute.py     # Test time optimization
├── tests/                  # Test suite
│   └── unit/              # Unit tests
├── benchmark_results/      # Benchmark results
├── run_benchmark.py        # Benchmark runner
└── test_cases.json        # Test cases for benchmarking
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on research in Natural Language Reinforcement Learning
- Uses the Hugging Face Transformers library
- Implements concepts from recent papers on test-time compute optimization
