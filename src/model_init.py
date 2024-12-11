import logging
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInitializer:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        """Initialize the model initializer.

        Args:
            model_name: Name of the model to use from Hugging Face
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def initialize_model(
        self, device_map: str = "auto", torch_dtype: str = "auto"
    ) -> None:
        """Initialize the model and tokenizer.

        Args:
            device_map: Device mapping strategy ('auto', 'cpu', 'cuda', etc.)
            torch_dtype: Data type for model weights

        Raises:
            RuntimeError: If model initialization fails
        """
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch_dtype, device_map=device_map
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

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
        prompt: Union[str, List[str]],
        system_prompt: str = "You are Qwen, a helpful AI assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> Union[str, List[str]]:
        """Generate a response for the given prompt.

        Args:
            prompt: The input prompt or list of prompts
            system_prompt: System message to control model behavior
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            num_return_sequences: Number of sequences to return

        Returns:
            Generated text or list of generated texts

        Raises:
            RuntimeError: If model is not initialized or generation fails
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be initialized first")

        try:
            # Handle both single prompt and list of prompts
            if isinstance(prompt, str):
                prompts = [prompt]
            else:
                prompts = prompt

            # Create messages for each prompt
            all_texts = []
            for p in prompts:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": p},
                ]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                all_texts.append(text)

            # Prepare model inputs
            model_inputs = self.tokenizer(
                all_texts, return_tensors="pt", padding=True
            ).to(self.model.device)

            # Generate
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # Process outputs
            outputs = []
            input_len = model_inputs.input_ids.size(1)
            for i in range(len(prompts)):
                batch_start = i * num_return_sequences
                batch_end = (i + 1) * num_return_sequences
                batch_outputs = generated_ids[batch_start:batch_end, input_len:]
                decoded = self.tokenizer.batch_decode(
                    batch_outputs, skip_special_tokens=True
                )
                if num_return_sequences == 1:
                    outputs.append(decoded[0])
                else:
                    outputs.append(decoded)

            # Return single string for single prompt, list for multiple prompts
            if isinstance(prompt, str) and num_return_sequences == 1:
                return outputs[0]
            return outputs

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise RuntimeError(f"Generation failed: {str(e)}")

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
