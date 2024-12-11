"""
Benchmarking Module for NLRL System.
Handles evaluation metrics, performance tracking, and model comparisons.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .model_init import ModelInitializer
from .test_time_compute import ComputeConfig, TestTimeCompute

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

    num_runs: int = 5
    warmup_runs: int = 2
    compute_config: Optional[ComputeConfig] = None
    save_results: bool = True
    results_dir: str = "benchmark_results"
    baseline_model: Optional[str] = None
    metrics: List[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.metrics is None:
            self.metrics = [
                "latency",
                "throughput",
                "compute_efficiency",
                "memory_usage",
                "task_accuracy",
            ]
        if self.compute_config is None:
            self.compute_config = ComputeConfig()


class Benchmarker:
    """Handles benchmarking and evaluation."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        """Initialize benchmarker with model."""
        self.model_name = model_name
        self.model_init = ModelInitializer(model_name=model_name)
        self.compute_optimizer = TestTimeCompute(model=self.model_init.model)

    def run_benchmark(
        self, test_cases: List[Dict[str, Any]], config: Optional[BenchmarkConfig] = None
    ) -> Dict[str, Any]:
        """
        Run benchmarking suite.

        Args:
            test_cases: List of test cases with prompts and expected outputs
            config: Benchmark configuration

        Returns:
            Dictionary of benchmark results
        """
        if config is None:
            config = BenchmarkConfig()

        results = {
            "model_name": self.model_name,
            "timestamp": time.time(),
            "config": {
                "num_runs": config.num_runs,
                "warmup_runs": config.warmup_runs,
                "metrics": config.metrics,
            },
            "metrics": {},
            "per_task_results": [],
        }

        # Warmup runs
        logger.info(f"Performing {config.warmup_runs} warmup runs...")
        warmup_prompts = [case["prompt"] for case in test_cases[:2]]
        for _ in range(config.warmup_runs):
            self.compute_optimizer.generate_optimized(
                warmup_prompts, config=config.compute_config
            )
        self.compute_optimizer.reset_metrics()

        # Main benchmark runs
        logger.info(f"Starting {config.num_runs} benchmark runs...")
        for run in range(config.num_runs):
            for case in test_cases:
                start_time = time.time()

                # Generate response
                response = self.compute_optimizer.generate_optimized(
                    case["prompt"], config=config.compute_config
                )

                # Calculate metrics
                latency = (time.time() - start_time) * 1000
                performance_stats = self.compute_optimizer.get_performance_stats()

                # Record results
                task_result = {
                    "prompt": case["prompt"],
                    "expected": case.get("expected"),
                    "response": response,
                    "latency_ms": latency,
                    "run_number": run,
                    **performance_stats,
                }
                results["per_task_results"].append(task_result)

        # Aggregate metrics
        all_latencies = [r["latency_ms"] for r in results["per_task_results"]]
        all_throughputs = [r.get("avg_throughput", 0) for r in results["per_task_results"]]

        results["metrics"] = {
            "latency": {
                "mean": np.mean(all_latencies),
                "std": np.std(all_latencies),
                "p50": np.percentile(all_latencies, 50),
                "p95": np.percentile(all_latencies, 95),
                "p99": np.percentile(all_latencies, 99),
            },
            "throughput": {
                "mean": np.mean(all_throughputs),
                "std": np.std(all_throughputs),
            },
        }

        # Add compute efficiency if available
        if all(
            "total_compute_flops" in r and "avg_throughput" in r
            for r in results["per_task_results"]
        ):
            results["metrics"]["compute_efficiency"] = np.mean(
                [
                    r["avg_throughput"] / r["total_compute_flops"]
                    for r in results["per_task_results"]
                ]
            )

        # Save results if configured
        if config.save_results:
            self._save_results(results, config.results_dir)

        return results

    def _save_results(self, results: Dict[str, Any], results_dir: str) -> None:
        """Save benchmark results to file."""
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"benchmark_{self.model_name.replace('/', '_')}_{timestamp}.json"
        filepath = results_path / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved benchmark results to {filepath}")

    def compare_with_baseline(
        self, current_results: Dict[str, Any], test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare current results with baseline model."""
        if not current_results or "metrics" not in current_results:
            raise ValueError("Invalid current results")

        baseline_config = BenchmarkConfig(
            num_runs=current_results["config"]["num_runs"],
            warmup_runs=current_results["config"]["warmup_runs"],
        )

        baseline_results = self.run_benchmark(test_cases, baseline_config)

        comparison = {
            "current_model": self.model_name,
            "baseline_model": baseline_config.baseline_model,
            "metrics_comparison": {},
        }

        for metric in ["latency", "throughput"]:
            if metric in current_results["metrics"] and metric in baseline_results["metrics"]:
                current_mean = current_results["metrics"][metric]["mean"]
                baseline_mean = baseline_results["metrics"][metric]["mean"]
                improvement = ((baseline_mean - current_mean) / baseline_mean) * 100

                comparison["metrics_comparison"][f"{metric}_diff"] = {
                    "current": current_mean,
                    "baseline": baseline_mean,
                    "improvement_percent": improvement,
                }

        return comparison
