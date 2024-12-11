import pytest

from src.model_init import ModelInitializer


def test_model_initialization():
    """Test that the model initializer can load the model and tokenizer."""
    initializer = ModelInitializer()
    initializer.initialize_model()

    assert initializer.model is not None
    assert initializer.tokenizer is not None


def test_generate_response():
    """Test that the model can generate responses."""
    initializer = ModelInitializer()
    initializer.initialize_model()

    prompt = "Write a function to add two numbers."
    response = initializer.generate_response(prompt)

    assert isinstance(response, str)
    assert len(response) > 0


def test_generate_response_without_init():
    """Test that generating without initialization raises an error."""
    initializer = ModelInitializer()

    with pytest.raises(RuntimeError):
        initializer.generate_response("test prompt")


def test_custom_system_prompt():
    """Test generation with a custom system prompt."""
    initializer = ModelInitializer()
    initializer.initialize_model()

    system_prompt = "You are a Python coding expert."
    prompt = "Write a simple hello world program."
    response = initializer.generate_response(prompt, system_prompt=system_prompt)

    assert isinstance(response, str)
    assert len(response) > 0
