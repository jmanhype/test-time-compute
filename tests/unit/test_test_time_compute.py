"""
Unit tests for the test time compute module.
"""

import pytest
import torch

from src.test_time_compute import TestTimeCompute
from src.model_init import has_accelerate


@pytest.fixture(scope="module")
def compute_setup():
    """Set up test fixtures that can be reused across all tests."""
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    compute = TestTimeCompute(model_name=model_name)
    return {"compute": compute, "model_name": model_name}


def test_optimize_batch_size(compute_setup):
    """Test batch size optimization."""
    compute = compute_setup["compute"]
    prompts = ["Hello world" for _ in range(5)]
    
    optimal_batch_size = compute.optimize_batch_size(prompts)
    assert isinstance(optimal_batch_size, int)
    assert optimal_batch_size > 0
    assert optimal_batch_size <= len(prompts)


def test_generate_optimized_single(compute_setup):
    """Test optimized generation for a single prompt."""
    compute = compute_setup["compute"]
    prompt = "Write a hello world program"
    
    result = compute.generate_optimized(prompt)
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_optimized_batch(compute_setup):
    """Test optimized generation for multiple prompts."""
    compute = compute_setup["compute"]
    prompts = [
        "Write a hello world program",
        "Calculate factorial of n",
        "Print numbers 1 to 10"
    ]
    
    results = compute.generate_optimized(prompts)
    assert isinstance(results, list)
    assert len(results) == len(prompts)
    for result in results:
        assert isinstance(result, str)
        assert len(result) > 0


def test_performance_metrics(compute_setup):
    """Test performance metrics collection."""
    compute = compute_setup["compute"]
    prompt = "Write a simple function"
    
    # Generate some text to collect metrics
    compute.generate_optimized(prompt)
    
    metrics = compute.get_performance_metrics()
    assert isinstance(metrics, dict)
    assert "avg_tokens_per_second" in metrics
    assert "total_tokens_generated" in metrics
    assert metrics["total_tokens_generated"] >= 0
    assert metrics["avg_tokens_per_second"] >= 0


def test_reset_metrics(compute_setup):
    """Test resetting performance metrics."""
    compute = compute_setup["compute"]
    
    # Generate some text to collect metrics
    compute.generate_optimized("Test prompt")
    
    # Get metrics before reset
    metrics_before = compute.get_performance_metrics()
    assert metrics_before["total_tokens_generated"] > 0
    
    # Reset metrics
    compute.reset_metrics()
    
    # Get metrics after reset
    metrics_after = compute.get_performance_metrics()
    assert metrics_after["total_tokens_generated"] == 0
    assert metrics_after["avg_tokens_per_second"] == 0.0


def test_memory_management(compute_setup):
    """Test memory management during generation."""
    compute = compute_setup["compute"]
    
    # Get initial memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
    
    # Generate some text
    compute.generate_optimized("Test prompt")
    
    # Check memory after generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        # Memory usage should be reasonable
        assert (final_memory - initial_memory) < 1e10  # Less than 10GB difference
