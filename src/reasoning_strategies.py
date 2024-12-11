"""
Reasoning Strategies Module for NLRL System.
Implements various strategies to enhance the quality and correctness of model outputs.
"""

from typing import List, Dict, Any, Tuple, Optional, Callable
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
import re

class ReasoningStrategy:
    """Base class for reasoning strategies."""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Initialize the reasoning strategy.
        
        Args:
            model: The language model to use
            tokenizer: The tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer

class DivideAndConquer(ReasoningStrategy):
    """Strategy that breaks down complex problems into smaller sub-tasks."""
    
    def decompose_problem(self, problem: str, num_subtasks: int = 3) -> List[str]:
        """
        Decompose a complex problem into smaller sub-tasks.
        
        Args:
            problem: The problem to decompose
            num_subtasks: Number of sub-tasks to create
            
        Returns:
            List of sub-task descriptions
        """
        # Create a prompt for decomposition
        decomposition_prompt = f"""
        Break down this problem into {num_subtasks} smaller, manageable sub-tasks:
        
        Problem: {problem}
        
        Output each sub-task on a new line starting with 'Step X:' where X is the step number.
        """
        
        inputs = self.tokenizer(decomposition_prompt, return_tensors="pt", padding=True)
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7
        )
        
        # Decode and split into sub-tasks
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        subtasks = [line.strip() for line in result.split('\n') 
                   if line.strip().startswith('Step')]
        
        return subtasks[:num_subtasks]  # Ensure we return exactly num_subtasks
    
    def solve(self, problem: str, num_subtasks: int = 3) -> str:
        """
        Solve a problem using divide and conquer strategy.
        
        Args:
            problem: The problem to solve
            num_subtasks: Number of sub-tasks to create
            
        Returns:
            Combined solution with reasoning steps
        """
        subtasks = self.decompose_problem(problem, num_subtasks)
        solutions = []
        
        for subtask in subtasks:
            # Create a prompt for solving the sub-task
            solution_prompt = f"""
            Solve this sub-task step by step:
            
            {subtask}
            
            Show your reasoning and end with 'Solution:'.
            """
            
            inputs = self.tokenizer(solution_prompt, return_tensors="pt", padding=True)
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7
            )
            
            solution = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            solutions.append(f"{subtask}\n{solution}")
        
        # Combine solutions with clear separation
        return "\n\n".join([
            "# Divide and Conquer Solution",
            "## Problem Decomposition",
            *solutions,
            "## Final Combined Solution",
            "Combining the above solutions..."
        ])

class SelfRefinement(ReasoningStrategy):
    """Strategy that iteratively refines solutions based on self-critique."""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 reward_fn: Optional[Callable[[str], float]] = None):
        """
        Initialize the self-refinement strategy.
        
        Args:
            model: The language model to use
            tokenizer: The tokenizer for the model
            reward_fn: Optional function that returns a score for a solution
        """
        super().__init__(model, tokenizer)
        self.reward_fn = reward_fn or (lambda x: 0.5)  # Default dummy reward function
    
    def generate_critique(self, solution: str) -> str:
        """Generate a critique of the current solution."""
        critique_prompt = f"""
        Analyze this solution and identify potential improvements:
        
        {solution}
        
        Provide specific suggestions for improvement:
        1. Correctness:
        2. Completeness:
        3. Clarity:
        """
        
        inputs = self.tokenizer(critique_prompt, return_tensors="pt", padding=True)
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def refine(self, solution: str, max_iterations: int = 3,
               score_threshold: float = 0.9) -> Tuple[str, List[str]]:
        """
        Iteratively refine a solution through self-critique.
        
        Args:
            solution: Initial solution to refine
            max_iterations: Maximum number of refinement iterations
            score_threshold: Score threshold to stop refinement
            
        Returns:
            Tuple of (final solution, list of intermediate solutions)
        """
        solutions = [solution]
        
        for i in range(max_iterations):
            current_score = self.reward_fn(solutions[-1])
            if current_score >= score_threshold:
                break
                
            critique = self.generate_critique(solutions[-1])
            
            # Generate improved solution based on critique
            improvement_prompt = f"""
            Original solution:
            {solutions[-1]}
            
            Critique and suggestions:
            {critique}
            
            Provide an improved solution addressing these points.
            End with 'Final Answer:'.
            """
            
            inputs = self.tokenizer(improvement_prompt, return_tensors="pt", padding=True)
            outputs = self.model.generate(
                **inputs,
                max_length=300,
                num_return_sequences=1,
                temperature=0.7
            )
            
            improved = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            solutions.append(improved)
        
        return solutions[-1], solutions

class SelfConsistency(ReasoningStrategy):
    """Strategy that generates multiple solutions and selects the most consistent one."""
    
    def generate_solutions(self, prompt: str, n: int = 5) -> List[str]:
        """Generate multiple solutions for the same prompt."""
        solutions = []
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=n,
            temperature=0.8,
            top_p=0.95
        )
        
        for output in outputs:
            solution = self.tokenizer.decode(output, skip_special_tokens=True)
            solutions.append(solution)
        
        return solutions
    
    def extract_final_answer(self, solution: str) -> str:
        """Extract the final answer from a solution."""
        # Look for "Final Answer:" followed by any text until the end or a double newline
        match = re.search(r"Final Answer:(.+?)(?=\n\n|$)", solution, re.DOTALL)
        if match:
            # Remove leading/trailing whitespace but preserve internal newlines
            return match.group(1).strip()
        return solution.strip()
    
    def select_most_consistent(self, solutions: List[str]) -> str:
        """
        Select the most consistent solution based on final answers.
        Uses a voting mechanism among the solutions.
        """
        final_answers = [self.extract_final_answer(sol) for sol in solutions]
        
        # Count occurrences of each answer
        answer_counts = {}
        for answer in final_answers:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        # Find the most common answer
        most_common = max(answer_counts.items(), key=lambda x: x[1])
        
        # Return the full solution corresponding to the most common answer
        for solution, answer in zip(solutions, final_answers):
            if answer == most_common[0]:
                return solution
        
        return solutions[0]  # Fallback to first solution
    
    def solve(self, prompt: str, n: int = 5) -> Tuple[str, List[str]]:
        """
        Generate multiple solutions and select the most consistent one.
        
        Args:
            prompt: The problem prompt
            n: Number of solutions to generate
            
        Returns:
            Tuple of (selected solution, all generated solutions)
        """
        solutions = self.generate_solutions(prompt, n)
        selected = self.select_most_consistent(solutions)
        return selected, solutions

def get_strategy(strategy_name: str, model: PreTrainedModel,
                tokenizer: PreTrainedTokenizer,
                reward_fn: Optional[Callable] = None) -> ReasoningStrategy:
    """
    Factory function to get a reasoning strategy.
    
    Args:
        strategy_name: Name of the strategy to use
        model: The language model to use
        tokenizer: The tokenizer for the model
        reward_fn: Optional reward function for self-refinement
        
    Returns:
        An instance of the requested strategy
    """
    strategies = {
        "divide_and_conquer": DivideAndConquer,
        "self_refinement": lambda m, t: SelfRefinement(m, t, reward_fn),
        "self_consistency": SelfConsistency
    }
    
    strategy_class = strategies.get(strategy_name.lower())
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategy_class(model, tokenizer)
