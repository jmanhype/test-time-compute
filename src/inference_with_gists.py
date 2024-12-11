"""
Inference with Gists Module for NLRL System.
Handles token-by-token generation with intermediate outputs and adaptive stopping.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.model_init import ModelInitializer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StopReason(Enum):
    """Enumeration of possible reasons for stopping generation."""

    MAX_LENGTH = "max_length"
    STOP_PHRASE = "stop_phrase"
    EOS_TOKEN = "eos_token"
    LOW_PROBABILITY = "low_probability"
    TIMEOUT = "timeout"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_length: int = 200
    min_length: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 1
    do_sample: bool = True
    early_stopping: bool = True
    timeout_seconds: float = 30.0
    min_token_probability: float = 0.05
    stop_phrases: Set[str] = None

    def __post_init__(self):
        """Initialize default stop phrases if none provided."""
        if self.stop_phrases is None:
            self.stop_phrases = {
                "Final Answer:",
                "Let's approach this step by step:",
                "Here's what we'll do:",
            }

        # Validate parameters
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.min_length <= 0:
            raise ValueError("min_length must be positive")
        if self.min_length > self.max_length:
            raise ValueError("min_length cannot be greater than max_length")
        if not (0.0 < self.temperature <= 2.0):
            raise ValueError("temperature must be between 0 and 2")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be between 0 and 1")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not (0.0 <= self.min_token_probability <= 1.0):
            raise ValueError("min_token_probability must be between 0 and 1")


@dataclass
class Gist:
    """A snapshot of the generation process at a particular token."""

    token: str
    token_probability: float
    cumulative_text: str
    timestamp: float
    token_position: int


class InferenceWithGists:
    """Handles token-by-token generation with intermediate outputs."""

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    ):
        """
        Initialize the inference handler.

        Args:
            model: Optional pre-initialized model
            tokenizer: Optional pre-initialized tokenizer
            model_name: Name of the model to use if model/tokenizer not provided
        """
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            initializer = ModelInitializer(model_name=model_name)
            initializer.initialize_model()
            self.model = initializer.model
            self.tokenizer = initializer.tokenizer

        self.device = next(self.model.parameters()).device

    def _prepare_input(self, prompt: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Prepare input for the model.

        Args:
            prompt: Single prompt or list of prompts

        Returns:
            Dictionary of input tensors
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _get_token_probability(self, logits: torch.Tensor, token_id: int) -> float:
        """
        Calculate probability of the selected token.

        Args:
            logits: Token logits from model
            token_id: ID of token to get probability for

        Returns:
            Probability of the token
        """
        probs = torch.softmax(logits, dim=-1)
        return probs[0, token_id].item()

    def _should_stop(
        self,
        current_text: str,
        token_prob: float,
        token_id: int,
        start_time: float,
        config: GenerationConfig,
    ) -> Tuple[bool, Optional[StopReason]]:
        """
        Determine if generation should stop.

        Returns:
            Tuple of (should_stop, reason)
        """
        # Check max length
        if len(current_text.split()) >= config.max_length:
            return True, StopReason.MAX_LENGTH

        # Check for stop phrases
        for phrase in config.stop_phrases:
            if phrase in current_text:
                return True, StopReason.STOP_PHRASE

        # Check for EOS token
        if token_id == self.tokenizer.eos_token_id:
            return True, StopReason.EOS_TOKEN

        # Check token probability
        if token_prob < config.min_token_probability:
            return True, StopReason.LOW_PROBABILITY

        # Check timeout
        if time.time() - start_time > config.timeout_seconds:
            return True, StopReason.TIMEOUT

        return False, None

    def generate_with_gists(
        self, prompt: str, config: GenerationConfig
    ) -> Tuple[str, List[Gist], StopReason]:
        """
        Generate text with intermediate gists.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            Tuple of (final text, list of gists, stop reason)
        """
        gists = []
        start_time = time.time()
        current_text = prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        for token_position in range(config.max_length):
            # Check timeout first
            elapsed_time = time.time() - start_time
            if elapsed_time > config.timeout_seconds:
                return current_text, gists, StopReason.TIMEOUT

            # Generate next token probabilities
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[0, -1, :]
                next_token_probs = torch.softmax(next_token_logits / config.temperature, dim=-1)

            # Sample next token
            if config.do_sample:
                filtered_probs = next_token_probs.clone()
                # Apply top-k filtering
                if config.top_k > 0:
                    top_k_values, _ = torch.topk(filtered_probs, config.top_k)
                    min_value = top_k_values[-1]
                    filtered_probs[filtered_probs < min_value] = 0.0
                # Apply top-p filtering
                if config.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(filtered_probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=0)
                    mask = cumsum_probs > config.top_p
                    mask[1:] = mask[:-1].clone()
                    mask[0] = False
                    filtered_probs[sorted_indices[mask]] = 0.0
                # Renormalize probabilities
                if filtered_probs.sum() > 0:
                    filtered_probs = filtered_probs / filtered_probs.sum()
                next_token_id = torch.multinomial(filtered_probs, num_samples=1)
            else:
                next_token_id = torch.argmax(next_token_probs).unsqueeze(0)

            # Get token probability and text
            token_prob = next_token_probs[next_token_id].item()
            token_text = self.tokenizer.decode(next_token_id)
            
            # Update current text and create gist
            current_text += token_text
            gist = Gist(
                token=token_text,
                token_probability=token_prob,
                cumulative_text=current_text,  # Use the full cumulative text
                timestamp=time.time() - start_time,
                token_position=token_position,
            )
            gists.append(gist)

            # Check stopping conditions
            if token_prob < config.min_token_probability:
                return current_text, gists, StopReason.LOW_PROBABILITY

            # Check for stop phrases
            for phrase in config.stop_phrases:
                if phrase in current_text[-len(phrase) * 2:]:  # Look at recent text for efficiency
                    return current_text, gists, StopReason.STOP_PHRASE

            if next_token_id.item() == self.tokenizer.eos_token_id:
                return current_text, gists, StopReason.EOS_TOKEN

            # Update input_ids for next iteration
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

        return current_text, gists, StopReason.MAX_LENGTH

    def generate_batch(
        self, prompts: List[str], config: GenerationConfig
    ) -> List[Tuple[str, List[Gist], StopReason]]:
        """
        Generate text for multiple prompts in parallel.

        Args:
            prompts: List of input prompts
            config: Generation configuration

        Returns:
            List of (final text, gists, stop reason) tuples
        """
        results = []
        for prompt in prompts:
            result = self.generate_with_gists(prompt, config)
            results.append(result)
        return results

    def analyze_gists(self, gists: List[Gist]) -> Dict[str, float]:
        """
        Analyze generation gists to provide insights about the generation process.

        Args:
            gists: List of generation gists

        Returns:
            Dictionary with analysis metrics
        """
        if not gists:
            return {}

        # Calculate various metrics
        token_probs = [g.token_probability for g in gists]
        generation_time = gists[-1].timestamp - gists[0].timestamp

        analysis = {
            "avg_token_probability": sum(token_probs) / len(token_probs),
            "min_token_probability": min(token_probs),
            "max_token_probability": max(token_probs),
            "total_tokens": len(gists),
            "generation_time": generation_time,
            "tokens_per_second": (
                len(gists) / generation_time if generation_time > 0 else 0
            ),
        }

        return analysis

    def get_generation_trace(
        self, gists: List[Gist], interval_seconds: float = 0.1
    ) -> List[Dict]:
        """
        Get a time-based trace of the generation process.

        Args:
            gists: List of generation gists
            interval_seconds: Time interval for snapshots

        Returns:
            List of dictionaries containing generation states at regular intervals
        """
        if not gists:
            return []

        trace = []
        current_time = gists[0].timestamp
        last_gist_idx = 0

        while current_time <= gists[-1].timestamp:
            # Find all gists up to current_time
            while (
                last_gist_idx < len(gists)
                and gists[last_gist_idx].timestamp <= current_time
            ):
                last_gist_idx += 1

            if last_gist_idx > 0:
                trace.append(
                    {
                        "cumulative_text": gists[last_gist_idx - 1].cumulative_text,
                        "timestamp": current_time,
                    }
                )

            current_time += interval_seconds

        return trace

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        try:
            if hasattr(self, "model_initializer"):
                del self.model_initializer
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "tokenizer"):
                del self.tokenizer
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
