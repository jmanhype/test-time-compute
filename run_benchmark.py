"""
Run benchmarks for the NLRL system with optimized test-time compute.
"""

import argparse
import json
import logging
from pathlib import Path

from src.benchmarking import BenchmarkConfig, Benchmarker
from src.test_time_compute import ComputeConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Sample test cases
DEFAULT_TEST_CASES = [
    {
        "prompt": "Write a Python function to calculate the factorial of a number.",
        "expected": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)",
    },
    {
        "prompt": "Create a list comprehension that generates squares of even numbers from 1 to 10.",
        "expected": "[x**2 for x in range(1,11) if x % 2 == 0]",
    },
    {
        "prompt": "Write a function to check if a string is a palindrome.",
        "expected": "def is_palindrome(s):\n    return s == s[::-1]",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NLRL benchmarks with optimized compute"
    )

    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct", help="Model to benchmark"
    )
    parser.add_argument(
        "--baseline",
        default="Qwen/Qwen2.5-Coder-0.5B",
        help="Baseline model to compare against",
    )
    parser.add_argument(
        "--num-runs", type=int, default=3, help="Number of benchmark runs"
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=1, help="Number of warmup runs"
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Maximum batch size for compute optimization",
    )
    parser.add_argument(
        "--test-cases", type=str, help="Path to JSON file containing test cases"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results",
    )

    return parser.parse_args()


def load_test_cases(path=None):
    """Load test cases from file or use defaults."""
    if path:
        with open(path) as f:
            return json.load(f)
    return DEFAULT_TEST_CASES


def main():
    args = parse_args()

    # Load test cases
    test_cases = load_test_cases(args.test_cases)
    logger.info(f"Loaded {len(test_cases)} test cases")

    # Configure benchmarking
    compute_config = ComputeConfig(
        max_batch_size=args.max_batch_size, min_batch_size=1, target_latency_ms=200
    )

    benchmark_config = BenchmarkConfig(
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        compute_config=compute_config,
        save_results=True,
        results_dir=args.output_dir,
    )

    # Initialize benchmarker
    logger.info(f"Initializing benchmarker with model: {args.model}")
    benchmarker = Benchmarker(args.model)

    # Run benchmarks
    logger.info("Starting benchmark run...")
    results = benchmarker.run_benchmark(test_cases, benchmark_config)

    # Compare with baseline if specified
    if args.baseline:
        logger.info(f"Comparing with baseline model: {args.baseline}")
        comparison = benchmarker.compare_with_baseline(
            test_cases, args.baseline, benchmark_config
        )

        # Save comparison results
        output_dir = Path(args.output_dir)
        comparison_file = output_dir / "model_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Saved comparison results to {comparison_file}")

    # Print summary
    logger.info("\nBenchmark Summary:")
    logger.info(f"Model: {args.model}")
    logger.info(f"Number of test cases: {len(test_cases)}")
    logger.info(f"Number of runs: {args.num_runs}")

    for metric, values in results["metrics"].items():
        if isinstance(values, dict):
            logger.info(f"\n{metric.upper()}:")
            for k, v in values.items():
                logger.info(f"  {k}: {v:.2f}")
        else:
            logger.info(f"\n{metric.upper()}: {values:.2f}")

    logger.info(f"\nDetailed results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
