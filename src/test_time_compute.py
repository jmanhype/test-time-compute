"""
Test-Time Compute Optimization Module.
Implements strategies from "Scaling LLM Test-Time Compute Optimally" paper.
"""

from typing import Dict, List, Optional, Union, Tuple
import torch
import time
import logging
from dataclasses import dataclass
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
    
    model_initializer = None
    performance_metrics = {
        "batch_sizes": [],
        "latencies": [],
        "throughputs": [],
        "compute_used": []
    }
    
    @staticmethod
    def setup_class(model_initializer: ModelInitializer):
        """Set up test class."""
        TestTimeCompute.model_initializer = model_initializer
        
    def optimize_batch_size(self, 
                          sample_inputs: List[str], 
                          config: ComputeConfig) -> int:
        """
        Determine optimal batch size based on latency constraints.
        
        Args:
            sample_inputs: List of sample inputs for calibration
            config: Compute configuration
            
        Returns:
            Optimal batch size
        """
        if not config.adaptive_batching:
            return config.max_batch_size
            
        # Binary search for optimal batch size
        left, right = config.min_batch_size, config.max_batch_size
        optimal_size = config.min_batch_size
        
        while left <= right:
            mid = (left + right) // 2
            batch = sample_inputs[:mid]
            
            # Measure latency
            start_time = time.time()
            _ = self.model_initializer.generate_response(
                batch[0] if len(batch) == 1 else batch
            )
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            if latency <= config.target_latency_ms:
                optimal_size = mid
                left = mid + 1
            else:
                right = mid - 1
                
        return optimal_size
        
    def estimate_compute(self, 
                        num_tokens: int, 
                        batch_size: int) -> float:
        """
        Estimate compute cost in FLOPS.
        
        Args:
            num_tokens: Number of tokens to generate
            batch_size: Batch size
            
        Returns:
            Estimated FLOPS
        """
        # Simplified estimate based on model size and generation length
        num_params = sum(p.numel() for p in self.model_initializer.model.parameters())
        return num_params * num_tokens * batch_size * 2  # *2 for forward and backward pass
        
    def generate_optimized(self,
                          prompts: Union[str, List[str]],
                          config: Optional[ComputeConfig] = None) -> Union[str, List[str]]:
        """
        Generate responses with optimized compute allocation.
        
        Args:
            prompts: Input prompt or list of prompts
            config: Compute configuration
            
        Returns:
            Generated text or list of generated texts
        """
        if config is None:
            config = ComputeConfig()
            
        if isinstance(prompts, str):
            prompts = [prompts]
            
        # Optimize batch size
        optimal_batch_size = self.optimize_batch_size(prompts[:10], config)
        logger.info(f"Using optimal batch size: {optimal_batch_size}")
        
        # Process in batches
        results = []
        for i in range(0, len(prompts), optimal_batch_size):
            batch = prompts[i:i + optimal_batch_size]
            
            start_time = time.time()
            batch_results = self.model_initializer.generate_response(
                batch[0] if len(batch) == 1 else batch
            )
            end_time = time.time()
            
            # Record metrics
            latency = (end_time - start_time) * 1000
            throughput = len(batch) / (end_time - start_time)
            compute = self.estimate_compute(
                num_tokens=100,  # Approximate tokens per response
                batch_size=len(batch)
            )
            
            self.performance_metrics["batch_sizes"].append(len(batch))
            self.performance_metrics["latencies"].append(latency)
            self.performance_metrics["throughputs"].append(throughput)
            self.performance_metrics["compute_used"].append(compute)
            
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
                
        return results[0] if len(prompts) == 1 else results
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Get statistics about inference performance."""
        if not self.performance_metrics["latencies"]:
            return {}
            
        return {
            "avg_latency_ms": sum(self.performance_metrics["latencies"]) / len(self.performance_metrics["latencies"]),
            "avg_throughput": sum(self.performance_metrics["throughputs"]) / len(self.performance_metrics["throughputs"]),
            "total_compute_flops": sum(self.performance_metrics["compute_used"]),
            "avg_batch_size": sum(self.performance_metrics["batch_sizes"]) / len(self.performance_metrics["batch_sizes"])
        }
        
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {
            "batch_sizes": [],
            "latencies": [],
            "throughputs": [],
            "compute_used": []
        }
