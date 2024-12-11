"""
Model Initialization Module for NLRL System.
Handles loading and configuring language models.
"""

import logging
from typing import Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def has_accelerate() -> bool:
    """Check if accelerate package is available."""
    try:
        import accelerate
        return True
    except ImportError:
        return False


def get_device() -> str:
    """Get the appropriate device for model execution."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelInitializer:
    """Handles model initialization and configuration."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        device: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the model handler.

        Args:
            model_name: Name of the model to use
            device: Device to load model on ('cpu', 'cuda', 'mps', or specific cuda device)
            system_prompt: Optional system prompt to use
        """
        self.model_name = model_name
        self.device = device or get_device()
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    def initialize_model(self) -> None:
        """Initialize the model and tokenizer."""
        try:
            logger.info(f"Initializing model {self.model_name}")

            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with appropriate device configuration
            load_kwargs = {
                "torch_dtype": torch.float32,  # Use float32 for better compatibility
                "trust_remote_code": True,
            }

            # Only use device_map and low_cpu_mem_usage if accelerate is available
            if has_accelerate():
                load_kwargs.update({
                    "device_map": "auto",
                    "low_cpu_mem_usage": True
                })
            else:
                logger.warning("Accelerate package not found, using basic device placement")
                if self.device == "mps":
                    # Load on CPU first for MPS compatibility
                    load_kwargs["device"] = "cpu"
                else:
                    load_kwargs["device"] = self.device

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )

            # Move model to MPS if needed
            if self.device == "mps" and not has_accelerate():
                self.model = self.model.to(self.device)

            # Set left padding for decoder-only models
            if self.model.config.is_decoder:
                self.tokenizer.padding_side = "left"
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Successfully initialized model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def generate_response(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            system_prompt: Optional system prompt to override the default

        Returns:
            Generated text response
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        try:
            # Prepare input
            full_prompt = f"{system_prompt or self.system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            inputs = self.tokenizer(full_prompt, return_tensors="pt", padding=True)
            
            # Move inputs to the same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode and clean up response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(full_prompt, "").strip()

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
