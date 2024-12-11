"""
Unit tests for the Inference with Gists Module.
"""
import unittest
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from inference_with_gists import (
    InferenceWithGists,
    GenerationConfig,
    Gist,
    StopReason
)

class TestInferenceWithGists(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that can be reused across all tests."""
        # Load a small model for testing
        cls.model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_name)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        if cls.tokenizer.pad_token_id is None:
            cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id
        
        cls.inference = InferenceWithGists(cls.model, cls.tokenizer)
    
    def test_generation_config_defaults(self):
        """Test default values of GenerationConfig."""
        config = GenerationConfig()
        
        self.assertEqual(config.max_length, 200)
        self.assertEqual(config.temperature, 0.7)
        self.assertTrue("Final Answer:" in config.stop_phrases)
    
    def test_basic_generation(self):
        """Test basic text generation with gists."""
        prompt = "Write a short story about a cat"
        config = GenerationConfig(max_length=20)
        
        text, gists, reason = self.inference.generate_with_gists(prompt, config)
        
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), len(prompt))
        self.assertIsInstance(gists, list)
        self.assertGreater(len(gists), 0)
        self.assertIsInstance(reason, StopReason)
    
    def test_stop_phrase_detection(self):
        """Test generation stops when encountering a stop phrase."""
        prompt = "What is 2+2? Think step by step and end with Final Answer:"
        config = GenerationConfig(max_length=50)
        
        text, gists, reason = self.inference.generate_with_gists(prompt, config)
        
        self.assertIn("Final Answer:", text)
        self.assertEqual(reason, StopReason.STOP_PHRASE)
    
    def test_low_probability_stopping(self):
        """Test generation stops on low probability tokens."""
        prompt = "Generate some random text"
        config = GenerationConfig(
            max_length=30,
            temperature=2.0,  # High temperature to encourage low probability tokens
            min_token_probability=0.01
        )
        
        text, gists, reason = self.inference.generate_with_gists(prompt, config)
        
        self.assertIsInstance(text, str)
        self.assertTrue(
            reason in [StopReason.LOW_PROBABILITY, StopReason.MAX_LENGTH]
        )
    
    def test_timeout_handling(self):
        """Test generation stops after timeout."""
        prompt = "Write a very long essay about artificial intelligence"
        config = GenerationConfig(
            max_length=1000,  # Long enough to trigger timeout
            timeout_seconds=1.0  # Short timeout
        )
        
        text, gists, reason = self.inference.generate_with_gists(prompt, config)
        
        self.assertEqual(reason, StopReason.TIMEOUT)
    
    def test_gist_analysis(self):
        """Test analysis of generation gists."""
        prompt = "Write a haiku"
        config = GenerationConfig(max_length=20)
        
        _, gists, _ = self.inference.generate_with_gists(prompt, config)
        analysis = self.inference.analyze_gists(gists)
        
        self.assertIn("avg_token_probability", analysis)
        self.assertIn("total_tokens", analysis)
        self.assertIn("tokens_per_second", analysis)
        self.assertGreater(analysis["total_tokens"], 0)
        self.assertGreater(analysis["tokens_per_second"], 0)
    
    def test_generation_trace(self):
        """Test generation trace creation."""
        prompt = "Write a short poem"
        config = GenerationConfig(max_length=20)
        
        _, gists, _ = self.inference.generate_with_gists(prompt, config)
        trace = self.inference.get_generation_trace(gists, interval_seconds=0.1)
        
        self.assertGreater(len(trace), 0)
        self.assertTrue(all(isinstance(t, str) for t in trace))
        self.assertEqual(trace[-1], gists[-1].cumulative_text)

if __name__ == '__main__':
    unittest.main()
