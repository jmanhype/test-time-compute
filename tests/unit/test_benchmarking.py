"""Tests for the benchmarking module."""

import pytest
import tempfile
from pathlib import Path
from src.benchmarking import Benchmarker, BenchmarkConfig

@pytest.fixture
def test_cases():
    """Sample test cases for benchmarking."""
    return [
        {
            "prompt": "Write a function to add two numbers.",
            "expected": "def add(a, b):\n    return a + b"
        },
        {
            "prompt": "Create a list comprehension that squares numbers from 1 to 5.",
            "expected": "[x**2 for x in range(1, 6)]"
        }
    ]

def test_benchmark_config():
    """Test benchmark configuration."""
    config = BenchmarkConfig(
        num_runs=3,
        warmup_runs=1,
        save_results=True
    )
    
    assert config.num_runs == 3
    assert config.warmup_runs == 1
    assert config.save_results == True
    assert isinstance(config.metrics, list)
    assert "latency" in config.metrics
    assert "throughput" in config.metrics

def test_benchmarker_initialization():
    """Test benchmarker initialization."""
    benchmarker = Benchmarker()
    assert benchmarker.model_name == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    assert benchmarker.model_init is not None
    assert benchmarker.compute_optimizer is not None

def test_run_benchmark(test_cases):
    """Test running benchmarks."""
    benchmarker = Benchmarker()
    config = BenchmarkConfig(num_runs=2, warmup_runs=1)
    
    results = benchmarker.run_benchmark(test_cases, config)
    
    assert "model_name" in results
    assert "timestamp" in results
    assert "metrics" in results
    assert "per_task_results" in results
    
    # Check metrics
    metrics = results["metrics"]
    assert "latency" in metrics
    assert "throughput" in metrics
    assert "compute_efficiency" in metrics
    
    # Check per-task results
    task_results = results["per_task_results"]
    assert len(task_results) == len(test_cases) * config.num_runs
    
    for result in task_results:
        assert "prompt" in result
        assert "response" in result
        assert "latency_ms" in result
        assert "run_number" in result

def test_save_results(test_cases):
    """Test saving benchmark results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = BenchmarkConfig(
            num_runs=1,
            warmup_runs=1,
            save_results=True,
            results_dir=temp_dir
        )
        
        benchmarker = Benchmarker()
        benchmarker.run_benchmark(test_cases, config)
        
        # Check if results file was created
        results_files = list(Path(temp_dir).glob("benchmark_*.json"))
        assert len(results_files) == 1
        
        # Check if file is readable
        result_file = results_files[0]
        assert result_file.stat().st_size > 0

def test_compare_with_baseline(test_cases):
    """Test comparing with baseline model."""
    benchmarker = Benchmarker()
    config = BenchmarkConfig(num_runs=1, warmup_runs=1)
    
    comparison = benchmarker.compare_with_baseline(
        test_cases,
        baseline_model="Qwen/Qwen2.5-Coder-0.5B",
        config=config
    )
    
    assert "current_model" in comparison
    assert "baseline_model" in comparison
    assert "metrics_comparison" in comparison
    
    metrics_comparison = comparison["metrics_comparison"]
    for metric in ["latency", "throughput", "compute_efficiency"]:
        assert metric in metrics_comparison
        assert "current" in metrics_comparison[metric]
        assert "baseline" in metrics_comparison[metric]
        assert "improvement_percent" in metrics_comparison[metric]
