import pytest

from src.inference_with_gists import GenerationConfig, InferenceWithGists, StopReason


def test_inference_initialization():
    """Test that the inference module initializes correctly with Qwen model."""
    inference = InferenceWithGists()
    assert inference.model is not None
    assert inference.tokenizer is not None
    assert inference.device is not None


def test_generation_with_gists():
    """Test generation with gists using Qwen model."""
    inference = InferenceWithGists()
    prompt = "Write a function to calculate factorial."
    config = GenerationConfig(
        max_length=100, temperature=0.7, stop_phrases={"```", "def"}
    )

    final_text, gists, stop_reason = inference.generate_with_gists(prompt, config)

    assert isinstance(final_text, str)
    assert len(final_text) > 0
    assert len(gists) > 0
    assert isinstance(stop_reason, StopReason)


def test_batch_generation():
    """Test batch generation with multiple prompts."""
    inference = InferenceWithGists()
    prompts = ["Write a simple for loop.", "Print hello world."]
    config = GenerationConfig(max_length=30)

    results = inference.generate_with_gists(prompts, config)

    assert isinstance(results, list)
    assert len(results) == len(prompts)
    for result in results:
        assert isinstance(result, tuple)
        text, gists, reason = result
        assert isinstance(text, str)
        assert len(text) > 0


def test_gist_analysis():
    """Test analysis of generation gists."""
    inference = InferenceWithGists()
    prompt = "Write a simple for loop."
    config = GenerationConfig(max_length=50)

    _, gists, _ = inference.generate_with_gists(prompt, config)
    analysis = inference.analyze_gists(gists)

    assert isinstance(analysis, dict)
    assert "avg_token_probability" in analysis
    assert "generation_time" in analysis
    assert "min_token_probability" in analysis
    assert "max_token_probability" in analysis


def test_generation_trace():
    """Test getting time-based generation trace."""
    inference = InferenceWithGists()
    prompt = "Print hello world."
    config = GenerationConfig(max_length=30)

    _, gists, _ = inference.generate_with_gists(prompt, config)
    # Add a small delay to ensure we have some time difference
    import time

    time.sleep(0.2)
    trace = inference.get_generation_trace(gists, interval_seconds=0.1)

    assert isinstance(trace, list)
    if len(gists) > 0:  # Only check trace if we have gists
        assert len(trace) > 0
        for state in trace:
            assert isinstance(state, dict)
            assert "cumulative_text" in state
            assert "timestamp" in state


def test_error_handling():
    """Test error handling during generation."""
    inference = InferenceWithGists()

    # Test with empty prompt
    with pytest.raises(Exception):
        inference.generate_with_gists("")

    # Test with invalid config
    with pytest.raises(Exception):
        inference.generate_with_gists("test", GenerationConfig(max_length=-1))
