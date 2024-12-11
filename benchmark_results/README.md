# Benchmark Results

This directory contains benchmark results from running the test-time compute optimization system.

## File Format

Each benchmark run creates a JSON file with the following naming convention:
```
benchmark_{model_name}_{timestamp}.json
```

The model comparison results are stored in `model_comparison.json`.

## Result Structure

Each benchmark result file contains:
- Model configuration
- Test case details
- Performance metrics:
  - Latency (mean, std, p50, p95, p99)
  - Throughput (tokens/second)
  - Compute efficiency
- Batch size statistics
- Memory usage

## Interpreting Results

- Lower latency values indicate better performance
- Higher throughput values indicate better performance
- Compute efficiency is measured as throughput/FLOPS
- Batch size statistics show how well the adaptive batching worked

## Baseline Comparison

The `model_comparison.json` file shows how the optimized model performs compared to the baseline model across all metrics.
