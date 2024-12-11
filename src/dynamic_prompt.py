"""
Dynamic Prompting Module for NLRL System.
Generates tailored prompts incorporating context, parameters, and chain-of-thought instructions.
"""

from typing import Any, Dict, Optional

# Example problems and solutions for each domain
EXAMPLES = {
    "math": {
        "basic": """
# Example Problem
Find the integral of x^2 dx.

# Chain-of-Thought Reasoning:
1. The integral of x^2 is x^3/3 + C by the power rule
2. No boundary conditions are given, so we include + C
3. Double-check: derivative of x^3/3 is x^2, confirming our answer

# Final Answer:
(x^3 / 3) + C
""",
        "advanced": """
# Example Problem
Solve the differential equation dy/dx + 2y = x^2

# Chain-of-Thought Reasoning:
1. This is a first-order linear differential equation
2. The integrating factor is e^(∫2dx) = e^(2x)
3. Multiply both sides: e^(2x)(dy/dx + 2y) = x^2 * e^(2x)
4. Solve for y using integration by parts
5. Simplify and verify the solution

# Final Answer:
y = (x^2/2 - x/2 + 1/4)e^(-2x) + Ce^(-2x)
""",
    },
    "coding": {
        "basic": """
# Example Problem
Write a function to find the maximum element in a list.

# Chain-of-Thought Reasoning:
1. Need to handle empty list case
2. Initialize max with first element
3. Iterate through remaining elements
4. Update max if current element is larger

# Final Answer:
def find_maximum(numbers):
    if not numbers:
        raise ValueError("Cannot find maximum of empty list")
    max_num = numbers[0]
    for num in numbers[1:]:
        if num > max_num:
            max_num = num
    return max_num
""",
        "advanced": """
# Example Problem
Implement a binary search tree insertion method.

# Chain-of-Thought Reasoning:
1. Check if root exists, if not create new node
2. Compare value with current node
3. Recursively insert into left/right subtree
4. Return the modified tree

# Final Answer:
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def insert(root, value):
    if root is None:
        return Node(value)
    if value < root.value:
        root.left = insert(root.left, value)
    else:
        root.right = insert(root.right, value)
    return root
""",
    },
}

# Template strings for different task types
PROMPT_TEMPLATES = {
    "math": """You are a mathematical reasoning assistant. Use step-by-step chain-of-thought to solve the problem.

Context and Example:
{context}

Now solve this new math problem:
{problem}

Requirements:
- Show your complete reasoning process
- Explain each step clearly
- State any assumptions made
- Verify your solution if possible
- End with "Final Answer:" followed by the result

Difficulty Level: {difficulty}
Expected Steps: {steps}
""",
    "coding": """You are an expert software developer. Write clean, efficient, and well-documented code.

Context and Example:
{context}

Task:
{problem}

Requirements:
- Write production-quality code
- Include proper error handling
- Add clear comments explaining the logic
- Consider edge cases
- End with "Final Answer:" followed by the implementation

Language: Python
Difficulty Level: {difficulty}
Expected Steps: {steps}
""",
    "commonsense": """You are a knowledgeable assistant. Think through the problem step by step.

Question:
{problem}

Requirements:
- Break down your reasoning process
- Consider multiple perspectives
- Support your answer with logic
- End with "Final Answer:" followed by a concise response

Difficulty Level: {difficulty}
Expected Steps: {steps}
""",
}


def get_dynamic_prompt(
    task_type: str,
    problem: str,
    difficulty: str = "medium",
    steps: int = 3,
    include_example: bool = True,
) -> str:
    """
    Generate a dynamic prompt that includes contextual examples and instructions.

    Args:
        task_type (str): Type of task ('math', 'coding', or 'commonsense')
        problem (str): The problem to solve
        difficulty (str): Difficulty level ('basic', 'medium', 'advanced')
        steps (int): Expected number of reasoning steps
        include_example (bool): Whether to include an example in the prompt

    Returns:
        str: The generated prompt with appropriate context and instructions
    """
    # Get the template for the task type
    template = PROMPT_TEMPLATES.get(task_type, PROMPT_TEMPLATES["commonsense"])

    # Get appropriate example based on difficulty and task type
    context = ""
    if include_example and task_type in EXAMPLES:
        # For basic coding tasks, don't include examples
        if task_type == "coding" and difficulty == "basic":
            context = ""
        else:
            if difficulty == "advanced":
                context = EXAMPLES[task_type]["advanced"]
            else:
                context = EXAMPLES[task_type]["basic"]

    # Format the template with all parameters
    prompt = template.format(
        context=context, problem=problem, difficulty=difficulty, steps=steps
    )

    return prompt.strip()


def get_prompt_config(task_type: str, difficulty: str = "medium") -> Dict[str, Any]:
    """
    Get configuration parameters for prompt generation based on task type and difficulty.

    Args:
        task_type (str): Type of task ('math', 'coding', or 'commonsense')
        difficulty (str): Difficulty level ('basic', 'medium', 'advanced')

    Returns:
        Dict[str, Any]: Configuration parameters including number of steps and example inclusion
    """
    config = {"include_example": True, "steps": 3}  # default

    # Adjust steps based on difficulty
    if difficulty == "basic":
        config["steps"] = 2
    elif difficulty == "advanced":
        config["steps"] = 5

    # Task-specific adjustments
    if task_type == "math":
        config["steps"] += 1  # Math usually needs more steps for verification
    elif task_type == "coding":
        config["include_example"] = (
            difficulty != "basic"
        )  # Skip examples for basic coding

    return config
