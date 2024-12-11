"""Tests for the benchmarking module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.benchmarking import BenchmarkConfig, Benchmarker


@pytest.fixture
def test_cases():
    """Sample test cases for benchmarking."""
    return [
        {
            "prompt": "Write a function to add two numbers.",
            "expected": "def add(a, b):\n    return a + b",
        },
        {
            "prompt": "Create a list comprehension that squares numbers from 1 to 5.",
            "expected": "[x**2 for x in range(1, 6)]",
        },
    ]


@pytest.fixture
def mock_model_init():
    """Mock model initializer."""
    mock = MagicMock()
    mock.generate_response.return_value = "def mock_response():\n    pass"
    return mock


@pytest.fixture
def mock_compute_optimizer():
    """Mock compute optimizer."""
    mock = MagicMock()
    mock.optimize_batch_size.return_value = 4
    return mock


@pytest.fixture
def mock_performance_stats():
    """Mock performance statistics."""
    return {
        "avg_throughput": 100.0,
        "avg_latency": 50.0,
        "compute_efficiency": 0.95,
        "memory_usage": 1024,
        "task_accuracy": 0.98,
        "total_compute_flops": 1000000
    }


def test_benchmark_config():
    """Test benchmark configuration."""
    config = BenchmarkConfig(num_runs=3, warmup_runs=1, save_results=True)

    assert config.num_runs == 3
    assert config.warmup_runs == 1
    assert config.save_results == True
    assert isinstance(config.metrics, list)
    assert "latency" in config.metrics
    assert "throughput" in config.metrics


@patch("src.benchmarking.ModelInitializer")
@patch("src.benchmarking.TestTimeCompute")
def test_benchmarker_initialization(mock_ttc, mock_model_init):
    """Test benchmarker initialization."""
    mock_model = MagicMock()
    mock_model_init.return_value = MagicMock(model=mock_model)
    mock_ttc.return_value = MagicMock()
    
    benchmarker = Benchmarker()
    assert benchmarker.model_name == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    assert benchmarker.model_init is not None
    assert benchmarker.compute_optimizer is not None


@patch("src.benchmarking.ModelInitializer")
@patch("src.benchmarking.TestTimeCompute")
def test_run_benchmark(mock_ttc, mock_model_init, test_cases, mock_performance_stats):
    """Test running benchmarks."""
    mock_model = MagicMock()
    mock_model_init.return_value = MagicMock(model=mock_model)
    mock_compute = MagicMock()
    
    # Setup mock return values
    mock_ttc.return_value = mock_compute
    mock_compute.generate_optimized.return_value = "def mock_response():\n    pass"
    mock_compute.get_performance_stats.return_value = mock_performance_stats
    
    benchmarker = Benchmarker()
    config = BenchmarkConfig(num_runs=2, warmup_runs=1)
    results = benchmarker.run_benchmark(test_cases, config)
    
    assert isinstance(results, dict)
    assert "model_name" in results
    assert "timestamp" in results
    assert "metrics" in results
    assert "per_task_results" in results
    
    # Check if performance stats were properly recorded
    for result in results["per_task_results"]:
        assert "avg_throughput" in result
        assert "avg_latency" in result
        assert "compute_efficiency" in result
        assert result["avg_throughput"] == mock_performance_stats["avg_throughput"]


@patch("src.benchmarking.ModelInitializer")
@patch("src.benchmarking.TestTimeCompute")
def test_save_results(mock_ttc, mock_model_init, test_cases, mock_performance_stats):
    """Test saving benchmark results."""
    mock_model = MagicMock()
    mock_model_init.return_value = MagicMock(model=mock_model)
    mock_compute = MagicMock()
    
    # Setup mock return values
    mock_ttc.return_value = mock_compute
    mock_compute.generate_optimized.return_value = "def mock_response():\n    pass"
    mock_compute.get_performance_stats.return_value = mock_performance_stats
    
    with tempfile.TemporaryDirectory() as tmpdir:
        benchmarker = Benchmarker()
        config = BenchmarkConfig(
            num_runs=2,
            warmup_runs=1,
            save_results=True,
            results_dir=tmpdir
        )
        
        results = benchmarker.run_benchmark(test_cases, config)
        result_files = list(Path(tmpdir).glob("*.json"))
        
        assert len(result_files) > 0
        assert result_files[0].exists()


@patch("src.benchmarking.ModelInitializer")
@patch("src.benchmarking.TestTimeCompute")
def test_compare_with_baseline(mock_ttc, mock_model_init, test_cases, mock_performance_stats):
    """Test comparing with baseline model."""
    mock_model = MagicMock()
    mock_model_init.return_value = MagicMock(model=mock_model)
    mock_compute = MagicMock()
    
    # Setup mock return values
    mock_ttc.return_value = mock_compute
    mock_compute.generate_optimized.return_value = "def mock_response():\n    pass"
    mock_compute.get_performance_stats.return_value = mock_performance_stats
    
    benchmarker = Benchmarker()
    config = BenchmarkConfig(
        num_runs=2,
        warmup_runs=1,
        baseline_model="Qwen/Qwen2.5-Coder-0.5B"
    )
    
    results = benchmarker.run_benchmark(test_cases, config)
    comparison = benchmarker.compare_with_baseline(results, test_cases)
    
    assert isinstance(comparison, dict)
    assert "metrics_comparison" in comparison
    assert "latency_diff" in comparison["metrics_comparison"]
    assert "throughput_diff" in comparison["metrics_comparison"]
