"""
Unit tests for the Reasoning Strategies Module.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

import torch

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from reasoning_strategies import (
    DivideAndConquer,
    ReasoningStrategy,
    SelfConsistency,
    SelfRefinement,
    get_strategy,
)


class TestReasoningStrategies(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()

        # Configure mock tokenizer
        self.mock_tokenizer.encode.return_value = torch.tensor([1, 2, 3])
        self.mock_tokenizer.decode.return_value = "Mock decoded text"
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        # Configure mock model
        self.mock_model.generate.return_value = torch.tensor([[4, 5, 6]])

    def test_divide_and_conquer_decomposition(self):
        """Test problem decomposition in Divide and Conquer strategy."""
        strategy = DivideAndConquer(self.mock_model, self.mock_tokenizer)

        # Configure mock for specific test
        self.mock_tokenizer.decode.return_value = """
        Step 1: First subtask
        Step 2: Second subtask
        Step 3: Third subtask
        """

        subtasks = strategy.decompose_problem("Test problem")

        self.assertEqual(len(subtasks), 3)
        self.assertTrue(all(task.startswith("Step") for task in subtasks))

    def test_self_refinement_improvement(self):
        """Test solution refinement in Self-Refinement strategy."""
        # Mock reward function that accepts first solution after one refinement
        mock_reward = Mock(side_effect=[0.5, 0.95])

        strategy = SelfRefinement(self.mock_model, self.mock_tokenizer, mock_reward)

        # Configure mock responses
        self.mock_tokenizer.decode.side_effect = [
            "Initial critique",
            "Improved solution with Final Answer: 42",
        ]

        final_solution, history = strategy.refine(
            "Initial solution", max_iterations=3, score_threshold=0.9
        )

        self.assertIn("42", final_solution)
        self.assertEqual(len(history), 2)  # Initial + 1 refinement

    def test_self_consistency_selection(self):
        """Test solution selection in Self-Consistency strategy."""
        strategy = SelfConsistency(self.mock_model, self.mock_tokenizer)

        # Test solutions with different final answers
        solutions = [
            "Solution 1\nFinal Answer: 42",
            "Solution 2\nFinal Answer: 42",
            "Solution 3\nFinal Answer: 7",
            "Solution 4\nFinal Answer: 42",
        ]

        selected = strategy.select_most_consistent(solutions)

        # Should select one of the solutions with answer "42"
        self.assertIn("42", selected)

    def test_strategy_factory(self):
        """Test the strategy factory function."""
        # Test valid strategy names
        for strategy_name in [
            "divide_and_conquer",
            "self_refinement",
            "self_consistency",
        ]:
            strategy = get_strategy(strategy_name, self.mock_model, self.mock_tokenizer)
            self.assertIsInstance(strategy, ReasoningStrategy)

        # Test invalid strategy name
        with self.assertRaises(ValueError):
            get_strategy("invalid_strategy", self.mock_model, self.mock_tokenizer)

    def test_extract_final_answer(self):
        """Test final answer extraction from solutions."""
        strategy = SelfConsistency(self.mock_model, self.mock_tokenizer)

        # Test various formats of final answers
        test_cases = [
            ("Some reasoning\nFinal Answer: 42", "42"),
            ("Final Answer: multi\nline\nanswer", "multi\nline\nanswer"),
            ("No final answer marker", "No final answer marker"),
        ]

        for solution, expected in test_cases:
            result = strategy.extract_final_answer(solution)
            self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
