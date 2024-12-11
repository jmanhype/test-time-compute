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
                "Therefore, the answer is",
                "In conclusion,",
                "###",
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

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        """
        Initialize the inference handler with a Qwen model.

        Args:
            model_name: Name of the model to use
        """
        self.model_initializer = ModelInitializer(model_name)
        self.model_initializer.initialize_model()
        self.model = self.model_initializer.model
        self.tokenizer = self.model_initializer.tokenizer
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
        self, prompt: Union[str, List[str]], config: Optional[GenerationConfig] = None
    ) -> Union[
        Tuple[str, List[Gist], StopReason], List[Tuple[str, List[Gist], StopReason]]
    ]:
        """
        Generate text with intermediate gists.

        Args:
            prompt: Input prompt or list of prompts
            config: Generation configuration

        Returns:
            For single prompt: Tuple of (final_text, gists, stop_reason)
            For multiple prompts: List of tuples, each containing (final_text, gists, stop_reason)
        """
        if config is None:
            config = GenerationConfig()

        # Input validation
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        if isinstance(prompt, list):
            if not all(p.strip() for p in prompt):
                raise ValueError("All prompts in batch must be non-empty")
            return [self.generate_with_gists(p, config) for p in prompt]

        # Prepare input
        input_ids = self._prepare_input(prompt)
        current_ids = input_ids["input_ids"]
        attention_mask = input_ids.get("attention_mask", None)

        # Initialize tracking variables
        gists = []
        start_time = time.time()
        current_text = ""

        try:
            # Generate tokens one at a time
            for step in range(config.max_length):
                # Get model output for next token
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=current_ids,
                        attention_mask=(
                            attention_mask if attention_mask is not None else None
                        ),
                    )

                # Get next token probabilities
                next_token_logits = outputs.logits[:, -1, :]

                # Apply temperature and top-p sampling
                if config.do_sample:
                    next_token_logits = next_token_logits / config.temperature
                    if config.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(
                            next_token_logits, descending=True
                        )
                        cumulative_probs = torch.cumsum(
                            torch.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        sorted_indices_to_remove = cumulative_probs > config.top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                            ..., :-1
                        ].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        next_token_logits[indices_to_remove] = float("-inf")

                # Sample next token
                if config.do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                # Get token probability
                token_prob = self._get_token_probability(
                    next_token_logits, next_token.item()
                )

                # Decode token
                token_text = self.tokenizer.decode(next_token[0])
                current_text += token_text

                # Create gist
                gist = Gist(
                    token=token_text,
                    token_probability=token_prob,
                    cumulative_text=current_text,
                    timestamp=time.time() - start_time,
                    token_position=step,
                )
                gists.append(gist)

                # Check if we should stop
                should_stop, reason = self._should_stop(
                    current_text, token_prob, next_token.item(), start_time, config
                )

                if should_stop:
                    return current_text, gists, reason

                # Update input ids for next iteration
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            attention_mask.new_ones((attention_mask.shape[0], 1)),
                        ],
                        dim=-1,
                    )

        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            return current_text, gists, StopReason.MAX_LENGTH

        return current_text, gists, StopReason.MAX_LENGTH

    def analyze_gists(self, gists: List[Gist]) -> Dict[str, float]:
        """
        Analyze gists to provide insights about the generation process.

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
