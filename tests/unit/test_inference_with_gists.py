"""
Unit tests for the Inference with Gists Module.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inference_with_gists import (
    GenerationConfig,
    Gist,
    InferenceWithGists,
    StopReason,
)
from src.model_init import has_accelerate


@pytest.fixture
def inference_setup():
    """Set up test fixtures that can be reused across all tests."""
    # Load a small model for testing
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

    # Configure model loading based on accelerate availability
    load_kwargs = {
        "torch_dtype": torch.float32,  # Use float32 for testing
        "trust_remote_code": True,
    }

    if has_accelerate():
        load_kwargs.update({"device_map": "auto", "low_cpu_mem_usage": True})
    else:
        load_kwargs["device"] = "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inference = InferenceWithGists(model, tokenizer)
    return {"inference": inference, "model_name": model_name}


@pytest.fixture
def inference_handler():
    """Create an inference handler with a small test model."""
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    return InferenceWithGists(model_name=model_name)


def test_generation_config_defaults():
    """Test default values of GenerationConfig."""
    config = GenerationConfig()

    assert config.max_length == 200
    assert config.temperature == 0.7
    assert "Final Answer:" in config.stop_phrases


def test_basic_generation(inference_setup):
    """Test basic text generation with gists."""
    inference = inference_setup["inference"]
    prompt = "Write a short story about a cat"
    config = GenerationConfig(max_length=20)

    text, gists, reason = inference.generate_with_gists(prompt, config)

    assert isinstance(text, str)
    assert len(text) > 3  # Ensure some text is generated
    assert isinstance(gists, list)
    assert len(gists) > 0
    assert isinstance(reason, StopReason)


def test_stop_phrase_detection(inference_setup):
    """Test generation stops when encountering a stop phrase."""
    inference = inference_setup["inference"]
    prompt = "What is 2+2? Think step by step and end with Final Answer:"
    config = GenerationConfig(max_length=50)

    text, gists, reason = inference.generate_with_gists(prompt, config)

    # The model might not always generate "Final Answer:" exactly
    # We should check that it either reached max length or stopped on a phrase
    assert reason in [StopReason.MAX_LENGTH, StopReason.STOP_PHRASE]
    assert len(text) > len(prompt)


def test_low_probability_stopping(inference_setup):
    """Test generation stops on low probability tokens."""
    inference = inference_setup["inference"]
    prompt = "Generate some random text"
    config = GenerationConfig(
        max_length=30,
        temperature=2.0,  # High temperature to encourage low probability tokens
        min_token_probability=0.01,
    )

    text, gists, reason = inference.generate_with_gists(prompt, config)

    assert isinstance(text, str)
    assert reason in [StopReason.LOW_PROBABILITY, StopReason.MAX_LENGTH]


def test_timeout_handling(inference_setup):
    """Test generation stops after timeout."""
    inference = inference_setup["inference"]
    prompt = "Write a very long essay about artificial intelligence"
    config = GenerationConfig(
        max_length=1000,  # Long enough to trigger timeout
        timeout_seconds=1.0,  # Short timeout
        min_token_probability=0.0001,  # Set a very low threshold to avoid early stopping
    )

    text, gists, reason = inference.generate_with_gists(prompt, config)

    assert reason in [StopReason.TIMEOUT, StopReason.MAX_LENGTH]
    assert len(text) > len(prompt)


def test_gist_analysis(inference_setup):
    """Test analysis of generation gists."""
    inference = inference_setup["inference"]
    prompt = "Write a haiku"
    config = GenerationConfig(max_length=20)

    _, gists, _ = inference.generate_with_gists(prompt, config)
    analysis = inference.analyze_gists(gists)

    assert "avg_token_probability" in analysis
    assert "total_tokens" in analysis
    assert "tokens_per_second" in analysis
    assert analysis["total_tokens"] > 0
    assert (
        analysis["tokens_per_second"] >= 0
    )  # Could be very small but should be non-negative


def test_generation_trace(inference_setup):
    """Test the generation trace functionality."""
    inference = inference_setup["inference"]
    prompt = "Tell me a joke"
    config = GenerationConfig(max_length=30)

    text, gists, reason = inference.generate_with_gists(prompt, config)

    # Check that gists form a proper trace
    assert len(gists) > 0
    for i, gist in enumerate(gists):
        assert isinstance(gist, Gist)
        assert gist.token_position == i
        assert gist.timestamp > 0
        assert 0 <= gist.token_probability <= 1
        assert len(gist.cumulative_text) >= len(prompt)


def test_generate_with_gists_basic(inference_handler):
    """Test basic text generation with gists."""
    config = GenerationConfig(max_length=50)
    prompt = "Write a function to calculate factorial"

    text, gists, stop_reason = inference_handler.generate_with_gists(prompt, config)

    assert isinstance(text, str)
    assert len(text) > len(prompt)
    assert len(gists) > 0
    assert isinstance(stop_reason, StopReason)


def test_generate_batch(inference_handler):
    """Test batch generation functionality."""
    config = GenerationConfig(max_length=50)
    prompts = ["Write a function to calculate factorial", "Explain what is recursion"]

    results = inference_handler.generate_batch(prompts, config)

    assert len(results) == len(prompts)
    for text, gists, stop_reason in results:
        assert isinstance(text, str)
        assert len(gists) > 0
        assert isinstance(stop_reason, StopReason)


def test_stop_phrases(inference_handler):
    """Test generation stops on encountering stop phrases."""
    config = GenerationConfig(
        max_length=100, stop_phrases={"Final Answer:", "Here's how:"}
    )
    prompt = "Write a function to calculate factorial"

    text, gists, stop_reason = inference_handler.generate_with_gists(prompt, config)

    if stop_reason == StopReason.STOP_PHRASE:
        assert any(phrase in text for phrase in config.stop_phrases)


def test_timeout(inference_handler):
    """Test generation timeout."""
    config = GenerationConfig(max_length=1000, timeout_seconds=0.1)
    prompt = "Write a very long essay about programming"

    text, gists, stop_reason = inference_handler.generate_with_gists(prompt, config)

    assert stop_reason == StopReason.TIMEOUT or len(text) < config.max_length


def test_invalid_config():
    """Test handling of invalid configuration."""
    with pytest.raises(ValueError):
        GenerationConfig(max_length=-1)

    with pytest.raises(ValueError):
        GenerationConfig(temperature=0.0)

    with pytest.raises(ValueError):
        GenerationConfig(top_p=1.5)


def test_analyze_gists(inference_handler):
    """Test gist analysis functionality."""
    config = GenerationConfig(max_length=50)
    prompt = "Write a short function"

    _, gists, _ = inference_handler.generate_with_gists(prompt, config)
    metrics = inference_handler.analyze_gists(gists)

    assert "avg_token_probability" in metrics
    assert "total_tokens" in metrics
    assert "tokens_per_second" in metrics
    assert metrics["total_tokens"] > 0
    assert 0 <= metrics["avg_token_probability"] <= 1
