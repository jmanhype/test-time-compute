"""
Unit tests for the Dynamic Prompting Module.
"""

import os
import sys
import unittest

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from dynamic_prompt import get_dynamic_prompt, get_prompt_config


class TestDynamicPrompt(unittest.TestCase):
    def test_math_prompt_generation(self):
        """Test generation of math prompts."""
        problem = "Find the derivative of x^3."
        prompt = get_dynamic_prompt("math", problem, difficulty="basic")

        # Check essential components
        self.assertIn(problem, prompt)
        self.assertIn("Chain-of-Thought", prompt)
        self.assertIn("Final Answer:", prompt)
        self.assertIn("Difficulty Level:", prompt)

    def test_coding_prompt_generation(self):
        """Test generation of coding prompts."""
        problem = "Write a function to reverse a string."
        prompt = get_dynamic_prompt("coding", problem, difficulty="medium")

        # Check essential components
        self.assertIn(problem, prompt)
        self.assertIn("Language: Python", prompt)
        self.assertIn("Requirements:", prompt)
        self.assertIn("Final Answer:", prompt)

    def test_commonsense_prompt_generation(self):
        """Test generation of commonsense prompts."""
        problem = "Explain how rainbows form."
        prompt = get_dynamic_prompt("commonsense", problem)

        # Check essential components
        self.assertIn(problem, prompt)
        self.assertIn("Requirements:", prompt)
        self.assertIn("Final Answer:", prompt)

    def test_difficulty_affects_steps(self):
        """Test that difficulty level affects the number of steps."""
        config_basic = get_prompt_config("math", "basic")
        config_advanced = get_prompt_config("math", "advanced")

        self.assertLess(config_basic["steps"], config_advanced["steps"])

    def test_example_inclusion(self):
        """Test example inclusion/exclusion based on configuration."""
        # Basic coding should not include example
        prompt_basic = get_dynamic_prompt(
            "coding", "Write a hello world program", difficulty="basic"
        )
        self.assertNotIn("Example Problem", prompt_basic)

        # Advanced coding should include example
        prompt_advanced = get_dynamic_prompt(
            "coding", "Implement a red-black tree", difficulty="advanced"
        )
        self.assertIn("Example Problem", prompt_advanced)

    def test_invalid_task_type_fallback(self):
        """Test fallback to commonsense for invalid task type."""
        problem = "Invalid task type test"
        prompt = get_dynamic_prompt("invalid_type", problem)

        # Should use commonsense template as fallback
        self.assertIn(problem, prompt)
        self.assertIn("Requirements:", prompt)
        self.assertIn("Final Answer:", prompt)


if __name__ == "__main__":
    unittest.main()
