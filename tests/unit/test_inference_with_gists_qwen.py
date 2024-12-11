"""
Unit tests for the Inference with Gists Module using Qwen model.
"""

import pytest
import torch

from src.inference_with_gists import GenerationConfig, InferenceWithGists, StopReason
from src.model_init import has_accelerate


@pytest.fixture(scope="module")
def inference_setup():
    """Set up test fixtures that can be reused across all tests."""
    inference = InferenceWithGists(model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    return {"inference": inference}


def test_inference_initialization(inference_setup):
    """Test initialization of inference with Qwen model."""
    inference = inference_setup["inference"]
    assert inference is not None


def test_generation_with_gists(inference_setup):
    """Test generating text with gists using Qwen model."""
    inference = inference_setup["inference"]
    prompt = "Write a function to calculate fibonacci numbers"
    config = GenerationConfig(max_length=100)

    text, gists, reason = inference.generate_with_gists(prompt, config)

    assert isinstance(text, str)
    assert len(text) > len(prompt)
    assert isinstance(gists, list)
    assert len(gists) > 0
    assert isinstance(reason, StopReason)


def test_batch_generation(inference_setup):
    """Test batch generation with Qwen model."""
    inference = inference_setup["inference"]
    prompts = [
        "Write a hello world program",
        "Calculate factorial of n",
    ]
    config = GenerationConfig(max_length=50)

    results = inference.generate_batch(prompts, config)

    assert len(results) == len(prompts)
    for text, gists, reason in results:
        assert isinstance(text, str)
        assert len(text) > 0
        assert isinstance(gists, list)
        assert len(gists) > 0
        assert isinstance(reason, StopReason)


def test_gist_analysis(inference_setup):
    """Test gist analysis with Qwen model."""
    inference = inference_setup["inference"]
    prompt = "Explain what is recursion"
    config = GenerationConfig(max_length=50)

    _, gists, _ = inference.generate_with_gists(prompt, config)
    analysis = inference.analyze_gists(gists)

    assert "avg_token_probability" in analysis
    assert "total_tokens" in analysis
    assert "tokens_per_second" in analysis
    assert analysis["total_tokens"] > 0
    assert analysis["tokens_per_second"] >= 0


def test_generation_trace(inference_setup):
    """Test generation trace with Qwen model."""
    inference = inference_setup["inference"]
    prompt = "Write a sorting algorithm"
    config = GenerationConfig(max_length=50)

    _, gists, _ = inference.generate_with_gists(prompt, config)

    assert len(gists) > 0
    for gist in gists:
        assert hasattr(gist, "token")
        assert hasattr(gist, "token_probability")
        assert hasattr(gist, "cumulative_text")
        assert hasattr(gist, "timestamp")
        assert hasattr(gist, "token_position")


def test_error_handling(inference_setup):
    """Test error handling in generation."""
    inference = inference_setup["inference"]
    
    # Test with invalid max_length
    with pytest.raises(ValueError):
        config = GenerationConfig(max_length=-1)
        inference.generate_with_gists("test", config)

    # Test with invalid temperature
    with pytest.raises(ValueError):
        config = GenerationConfig(temperature=2.5)
        inference.generate_with_gists("test", config)

    # Test with invalid top_p
    with pytest.raises(ValueError):
        config = GenerationConfig(top_p=1.5)
        inference.generate_with_gists("test", config)
