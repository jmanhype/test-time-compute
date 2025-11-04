"""
Test-Time Compute Optimization Module.
Implements strategies from "Scaling LLM Test-Time Compute Optimally" paper.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch

from .model_init import ModelInitializer

logger = logging.getLogger(__name__)


@dataclass
class ComputeConfig:
    """Configuration for test-time compute optimization."""

    max_batch_size: int = 32
    min_batch_size: int = 1
    target_latency_ms: float = 100
    warmup_iterations: int = 5
    adaptive_batching: bool = True
    compute_budget: Optional[float] = None  # In FLOPS
    optimization_strategy: str = "dynamic"  # "dynamic" or "static"


class TestTimeCompute:
    """Handles test-time compute optimization for inference."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        """Initialize with a model name."""
        self.model_initializer = ModelInitializer(model_name=model_name)
        self.model_initializer.initialize_model()
        self.reset_metrics()

    def reset_metrics(self):
        """Reset performance metrics."""
        self.performance_metrics = {
            "batch_sizes": [],
            "latencies": [],
            "throughputs": [],
            "compute_used": [],
            "total_tokens": [],
        }

    def get_performance_metrics(self) -> Dict[str, Union[float, int, List[float]]]:
        """Get performance metrics for the test run."""
        if not self.performance_metrics["latencies"]:
            return {
                "latencies": [],
                "throughputs": [],
                "batch_sizes": [],
                "compute_used": [],
                "total_tokens_generated": 0,
                "avg_tokens_per_second": 0.0,
            }

        avg_latency = sum(self.performance_metrics["latencies"]) / len(
            self.performance_metrics["latencies"]
        )
        avg_throughput = sum(self.performance_metrics["throughputs"]) / len(
            self.performance_metrics["throughputs"]
        )
        total_tokens = sum(self.performance_metrics["total_tokens"])
        avg_tokens_per_second = (
            total_tokens / sum(self.performance_metrics["latencies"])
            if self.performance_metrics["latencies"]
            else 0.0
        )

        return {
            "latencies": self.performance_metrics["latencies"],
            "throughputs": self.performance_metrics["throughputs"],
            "batch_sizes": self.performance_metrics["batch_sizes"],
            "compute_used": self.performance_metrics["compute_used"],
            "total_tokens_generated": total_tokens,
            "avg_tokens_per_second": avg_tokens_per_second,
        }

    def optimize_batch_size(
        self, sample_inputs: List[str], config: Optional[ComputeConfig] = None
    ) -> int:
        """
        Determine optimal batch size based on latency constraints.

        Args:
            sample_inputs: List of sample inputs to test with
            config: Configuration for optimization, uses default if None

        Returns:
            Optimal batch size

        Raises:
            ValueError: If sample_inputs is empty or invalid
            RuntimeError: If model is not initialized
        """
        if not sample_inputs:
            raise ValueError("sample_inputs cannot be empty")

        if not isinstance(sample_inputs, list):
            raise ValueError("sample_inputs must be a list of strings")

        if not all(isinstance(s, str) for s in sample_inputs):
            raise ValueError("All items in sample_inputs must be strings")

        if self.model_initializer.model is None:
            raise RuntimeError("Model not initialized")

        if config is None:
            config = ComputeConfig()

        # Start with minimum batch size
        current_batch_size = config.min_batch_size
        best_batch_size = current_batch_size
        min_latency = float("inf")

        while current_batch_size <= min(config.max_batch_size, len(sample_inputs)):
            # Measure latency for current batch size
            batch = sample_inputs[:current_batch_size]
            start_time = time.time()

            # Run a few iterations to get stable measurements
            for _ in range(config.warmup_iterations):
                with torch.no_grad():
                    _ = self.model_initializer.model.generate(
                        self.model_initializer.tokenizer(
                            batch,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        ).input_ids.to(self.model_initializer.model.device)
                    )

            latency = (time.time() - start_time) / config.warmup_iterations * 1000  # ms

            # Update metrics
            self.performance_metrics["batch_sizes"].append(current_batch_size)
            self.performance_metrics["latencies"].append(latency)
            self.performance_metrics["throughputs"].append(current_batch_size / latency)

            # Check if this batch size meets our constraints
            if latency <= config.target_latency_ms and latency < min_latency:
                min_latency = latency
                best_batch_size = current_batch_size

            # If we're already over target latency, no point trying larger batches
            if latency > config.target_latency_ms:
                break

            current_batch_size *= 2

        return best_batch_size

    def generate_optimized(
        self, prompts: Union[str, List[str]], config: Optional[ComputeConfig] = None
    ) -> Union[str, List[str]]:
        """
        Generate text with optimized batch size.

        Args:
            prompts: Single prompt or list of prompts
            config: Configuration for optimization, uses default if None

        Returns:
            Generated text or list of generated texts

        Raises:
            ValueError: If prompts is empty or invalid
            RuntimeError: If model is not initialized
        """
        if not prompts:
            raise ValueError("prompts cannot be empty")

        if self.model_initializer.model is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        if config is None:
            config = ComputeConfig()

        # Handle single prompt case
        if isinstance(prompts, str):
            if not prompts.strip():
                raise ValueError("prompt cannot be empty or whitespace")
            prompts = [prompts]
            return_single = True
        else:
            if not isinstance(prompts, list):
                raise ValueError("prompts must be a string or list of strings")
            if not all(isinstance(p, str) and p.strip() for p in prompts):
                raise ValueError("All prompts must be non-empty strings")
            return_single = False

        # Get optimal batch size
        optimal_batch_size = self.optimize_batch_size(prompts, config)

        # Generate in batches
        results = []
        for i in range(0, len(prompts), optimal_batch_size):
            batch = prompts[i : i + optimal_batch_size]
            inputs = self.model_initializer.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).input_ids.to(self.model_initializer.model.device)

            with torch.no_grad():
                outputs = self.model_initializer.model.generate(inputs)

            texts = self.model_initializer.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            results.extend(texts)
            self.performance_metrics["total_tokens"].append(
                sum(len(text.split()) for text in texts)
            )

        return results[0] if return_single else results
