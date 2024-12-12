"""
Module for handling inference with gists and analysis.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .model_init import ModelInitializer

logger = logging.getLogger(__name__)


class StopReason(str, Enum):
    """Reasons for stopping text generation."""

    MAX_LENGTH = "max_length"
    STOP_PHRASE = "stop_phrase"
    LOW_PROBABILITY = "low_probability"
    TIMEOUT = "timeout"


@dataclass
class Gist:
    """A gist of generated text with metadata."""

    text: str
    token_probability: float
    cumulative_prob: float
    cumulative_text: str
    start_pos: int
    end_pos: int
    token_position: int
    timestamp: float


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_length: int = 200
    min_token_probability: float = 0.1
    temperature: float = 0.7
    top_p: float = 0.9
    stop_phrases: Set[str] = field(
        default_factory=lambda: {
            "Final Answer:",
            "Let's approach this step by step:",
            "Here's what we'll do:",
        }
    )
    timeout_seconds: float = 10.0
    batch_size: int = 1

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if not (0.0 <= self.min_token_probability <= 1.0):
            raise ValueError("min_token_probability must be between 0 and 1")
        if not (0.0 < self.temperature <= 2.0):
            raise ValueError("temperature must be between 0 and 2")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be between 0 and 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


class InferenceWithGists:
    """Handles inference with gist analysis."""

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    ):
        """Initialize with model and tokenizer."""
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.model_initializer = ModelInitializer(model_name=model_name)
            self.model_initializer.initialize_model()
            self.model = self.model_initializer.model
            self.tokenizer = self.model_initializer.tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate_with_gists(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> Tuple[str, List[Gist], StopReason]:
        """
        Generate text with gist analysis.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            Tuple of (generated text, list of gists, stop reason)
        """
        if config is None:
            config = GenerationConfig()

        start_time = time.time()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        generated_text = ""
        gists = []
        cumulative_text = prompt
        token_position = 0

        while len(input_ids[0]) < config.max_length:
            current_time = time.time()
            # Check timeout
            if current_time - start_time > config.timeout_seconds:
                return cumulative_text, gists, StopReason.TIMEOUT

            # Generate next token
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[:, -1, :] / config.temperature
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                
                # Apply top-p sampling
                sorted_probs, sorted_indices = torch.sort(next_token_probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumsum_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_probs = next_token_probs.masked_fill(indices_to_remove, 0.0)
                
                # Renormalize probabilities
                next_token_probs = next_token_probs / next_token_probs.sum()
                
                # Sample next token
                next_token = torch.multinomial(next_token_probs, num_samples=1)

            # Get token probability
            token_prob = next_token_probs[0, next_token[0, 0]].item()

            # Check for low probability
            if token_prob < config.min_token_probability:
                return cumulative_text, gists, StopReason.LOW_PROBABILITY

            # Decode token
            next_token_text = self.tokenizer.decode(next_token[0])
            generated_text += next_token_text
            cumulative_text += next_token_text

            # Create gist for the new token
            gist = Gist(
                text=next_token_text,
                token_probability=token_prob,
                cumulative_prob=token_prob,
                cumulative_text=cumulative_text,
                start_pos=len(cumulative_text) - len(next_token_text),
                end_pos=len(cumulative_text),
                token_position=token_position,
                timestamp=current_time - start_time
            )
            gists.append(gist)
            token_position += 1

            # Check for stop phrases
            for stop_phrase in config.stop_phrases:
                if stop_phrase in cumulative_text:
                    return cumulative_text, gists, StopReason.STOP_PHRASE

            # Update input_ids for next iteration
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Add a small delay to prevent infinite loops
            time.sleep(0.001)

        return cumulative_text, gists, StopReason.MAX_LENGTH

    def generate_batch(
        self, prompts: List[str], config: Optional[GenerationConfig] = None
    ) -> List[Tuple[str, List[Gist], StopReason]]:
        """
        Generate text for multiple prompts in batches.

        Args:
            prompts: List of input prompts
            config: Generation configuration

        Returns:
            List of (generated text, gists, stop reason) tuples
        """
        if config is None:
            config = GenerationConfig()

        results = []
        for i in range(0, len(prompts), config.batch_size):
            batch = prompts[i : i + config.batch_size]
            batch_results = [
                self.generate_with_gists(prompt, config) for prompt in batch
            ]
            results.extend(batch_results)

        return results

    def analyze_gists(
        self, gists: List[Gist], window_size: int = 5
    ) -> Dict[str, float]:
        """
        Analyze gists to extract insights.

        Args:
            gists: List of gists to analyze
            window_size: Size of sliding window for local analysis

        Returns:
            Dictionary of analysis metrics
        """
        if not gists:
            return {
                "avg_token_probability": 0.0,
                "min_token_probability": 0.0,
                "max_token_probability": 0.0,
                "average_local_coherence": 0.0,
                "total_tokens": 0,
                "generation_time": 0.0,
                "tokens_per_second": 0.0,
            }

        # Calculate basic statistics
        token_probs = [gist.token_probability for gist in gists]
        avg_prob = sum(token_probs) / len(token_probs)
        min_prob = min(token_probs)
        max_prob = max(token_probs)

        # Calculate local coherence using sliding window
        local_coherence = []
        for i in range(len(gists) - window_size + 1):
            window = gists[i : i + window_size]
            window_probs = [gist.token_probability for gist in window]
            local_coherence.append(sum(window_probs) / len(window_probs))

        avg_local_coherence = (
            sum(local_coherence) / len(local_coherence) if local_coherence else 0
        )

        # Calculate generation speed
        total_time = gists[-1].timestamp - gists[0].timestamp if gists else 0
        tokens_per_second = len(gists) / total_time if total_time > 0 else 0

        return {
            "avg_token_probability": avg_prob,
            "min_token_probability": min_prob,
            "max_token_probability": max_prob,
            "average_local_coherence": avg_local_coherence,
            "total_tokens": len(gists),
            "generation_time": total_time,
            "tokens_per_second": tokens_per_second,
        }
