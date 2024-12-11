"""
Unit tests for the Task Classification Module.
"""

import os
import sys
import unittest

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from task_classifier import classify_task


class TestTaskClassifier(unittest.TestCase):
    def test_math_classification(self):
        """Test classification of mathematical prompts."""
        test_cases = [
            "Solve the equation x^2 - 4 = 0",
            "Calculate the integral of x^2 dx",
            "What is the derivative of sin(x)?",
            "Find the sum of the series 1 + 2 + 3 + ... + n",
        ]
        for prompt in test_cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(classify_task(prompt), "math")

    def test_coding_classification(self):
        """Test classification of coding prompts."""
        test_cases = [
            "Write a Python function to sort a list",
            "Debug this code snippet: for i in range(10): print(i)",
            "Implement a binary search algorithm",
            "Create a class for handling user authentication",
        ]
        for prompt in test_cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(classify_task(prompt), "coding")

    def test_commonsense_classification(self):
        """Test classification of commonsense prompts."""
        test_cases = [
            "What is the capital of France?",
            "Explain the process of photosynthesis",
            "Why is the sky blue?",
            "What causes ocean tides?",
        ]
        for prompt in test_cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(classify_task(prompt), "commonsense")


if __name__ == "__main__":
    unittest.main()
