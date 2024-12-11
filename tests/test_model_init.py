"""
Tests for the model initialization module.
"""

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.model_init import ModelInitializer, has_accelerate


@pytest.fixture
def model_initializer():
    return ModelInitializer(
        model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        device="cpu",  # Use CPU for testing
    )


def test_model_initializer_init():
    """Test ModelInitializer initialization."""
    initializer = ModelInitializer()
    assert initializer.model_name == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    assert initializer.device in ["cpu", "cuda"]
    assert initializer.model is None
    assert initializer.tokenizer is None


def test_model_initialization(model_initializer, monkeypatch):
    """Test model and tokenizer initialization."""

    # Mock the model and tokenizer classes
    class MockModel:
        def __init__(self, *args, **kwargs):
            self.config = type("Config", (), {"is_decoder": True})()

        def to(self, device):
            return self

        def eval(self):
            return self

    class MockTokenizer:
        def __init__(self, *args, **kwargs):
            self.pad_token = None
            self.eos_token = "[EOS]"
            self.padding_side = "right"

    def mock_from_pretrained(*args, **kwargs):
        return MockModel()

    def mock_tokenizer_from_pretrained(*args, **kwargs):
        return MockTokenizer()

    # Mock has_accelerate to return False for testing
    monkeypatch.setattr("src.model_init.has_accelerate", lambda: False)

    # Apply the mocks
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", mock_from_pretrained
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", mock_tokenizer_from_pretrained
    )

    # Initialize model
    model_initializer.initialize_model()

    # Check that model and tokenizer are initialized
    assert model_initializer.model is not None
    assert model_initializer.tokenizer is not None
    assert model_initializer.tokenizer.padding_side == "left"
    assert model_initializer.tokenizer.pad_token == "[EOS]"


def test_generate_response(model_initializer, monkeypatch):
    """Test response generation."""

    # Mock the necessary components
    class MockOutput:
        def __init__(self):
            self.sequences = torch.tensor([[1, 2, 3]])

    class MockModel:
        def __init__(self):
            self.config = type("Config", (), {"is_decoder": True})()
            self.device = "cpu"

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3]])

        def to(self, device):
            return self

    class MockTokenizer:
        def __init__(self):
            self.pad_token = "[PAD]"
            self.eos_token = "[EOS]"
            self.padding_side = "left"
            self.pad_token_id = 0

        def __call__(self, text, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

        def decode(self, *args, **kwargs):
            return f"You are a helpful AI assistant.\n\nUser: Test prompt\n\nAssistant: Test response"

    # Mock has_accelerate to return False for testing
    monkeypatch.setattr("src.model_init.has_accelerate", lambda: False)

    # Set up the mocks
    model_initializer.model = MockModel()
    model_initializer.tokenizer = MockTokenizer()

    # Test generate_response
    response = model_initializer.generate_response("Test prompt")
    assert isinstance(response, str)
    assert len(response) > 0
    assert response == "Test response"


def test_generate_response_without_init():
    """Test that generating response without initialization raises error."""
    initializer = ModelInitializer()
    with pytest.raises(RuntimeError):
        initializer.generate_response("Test prompt")


def test_has_accelerate(monkeypatch):
    """Test has_accelerate function."""

    def mock_import_error(*args, **kwargs):
        raise ImportError("No module named 'accelerate'")

    # Test when accelerate is not available
    monkeypatch.setattr("builtins.__import__", mock_import_error)
    assert not has_accelerate()

    # Test when accelerate is available
    monkeypatch.undo()
    try:
        import accelerate

        assert has_accelerate()
    except ImportError:
        pytest.skip("accelerate package not installed")
