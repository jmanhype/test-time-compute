import pytest

from src.model_init import ModelInitializer
from src.test_time_compute import ComputeConfig, TestTimeCompute


def test_optimize_batch_size():
    """Test batch size optimization."""
    model_init = ModelInitializer()
    model_init.initialize_model()
    compute = TestTimeCompute(model_init)

    config = ComputeConfig(max_batch_size=8, min_batch_size=1, target_latency_ms=200)

    sample_inputs = ["Write a simple function.", "Calculate 2+2.", "What is Python?"]

    optimal_size = compute.optimize_batch_size(sample_inputs, config)
    assert isinstance(optimal_size, int)
    assert 1 <= optimal_size <= 8


def test_generate_optimized_single():
    """Test optimized generation with single prompt."""
    model_init = ModelInitializer()
    model_init.initialize_model()
    compute = TestTimeCompute(model_init)

    prompt = "Write a hello world program."
    result = compute.generate_optimized(prompt)

    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_optimized_batch():
    """Test optimized generation with multiple prompts."""
    model_init = ModelInitializer()
    model_init.initialize_model()
    compute = TestTimeCompute(model_init)

    prompts = ["Write a for loop.", "Print hello world.", "Define a function."]

    results = compute.generate_optimized(prompts)

    assert isinstance(results, list)
    assert len(results) == len(prompts)
    for result in results:
        assert isinstance(result, str)
        assert len(result) > 0


def test_performance_metrics():
    """Test performance metrics tracking."""
    model_init = ModelInitializer()
    model_init.initialize_model()
    compute = TestTimeCompute(model_init)

    # Generate some outputs to collect metrics
    prompts = ["Test prompt 1.", "Test prompt 2."]
    _ = compute.generate_optimized(prompts)

    stats = compute.get_performance_stats()
    assert isinstance(stats, dict)
    assert "avg_latency_ms" in stats
    assert "avg_throughput" in stats
    assert "total_compute_flops" in stats
    assert "avg_batch_size" in stats


def test_compute_config():
    """Test compute configuration options."""
    config = ComputeConfig(
        max_batch_size=16, target_latency_ms=150, adaptive_batching=True
    )

    assert config.max_batch_size == 16
    assert config.target_latency_ms == 150
    assert config.adaptive_batching == True


def test_reset_metrics():
    """Test resetting performance metrics."""
    model_init = ModelInitializer()
    model_init.initialize_model()
    compute = TestTimeCompute(model_init)

    # Generate some outputs
    _ = compute.generate_optimized("Test prompt")

    # Verify metrics were collected
    assert len(compute.performance_metrics["latencies"]) > 0

    # Reset metrics
    compute.reset_metrics()

    # Verify metrics were reset
    assert len(compute.performance_metrics["latencies"]) == 0
    assert len(compute.performance_metrics["batch_sizes"]) == 0
    assert len(compute.performance_metrics["throughputs"]) == 0
    assert len(compute.performance_metrics["compute_used"]) == 0
