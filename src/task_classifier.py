"""
Task Classification Module for NLRL System.
Classifies incoming prompts into predefined domains (math, coding, commonsense).
"""


def classify_task(prompt: str) -> str:
    """
    Classify the task into domains: math, coding, commonsense.

    Args:
        prompt (str): The input prompt to classify

    Returns:
        str: The classified domain ('math', 'coding', or 'commonsense')

    Raises:
        ValueError: If prompt is None, empty, or not a string
    """
    if prompt is None:
        raise ValueError("prompt cannot be None")

    if not isinstance(prompt, str):
        raise ValueError(f"prompt must be a string, got {type(prompt).__name__}")

    if not prompt.strip():
        raise ValueError("prompt cannot be empty or whitespace")

    lower_p = prompt.lower()

    math_keywords = [
        "sum",
        "integral",
        "solve",
        "math",
        "equation",
        "derivative",
        "calculate",
        "arithmetic",
        "algebra",
        "geometric",
        "number",
    ]

    coding_keywords = [
        "code",
        "function",
        "debug",
        "algorithm",
        "python",
        "program",
        "implement",
        "class",
        "method",
        "variable",
        "loop",
    ]

    if any(keyword in lower_p for keyword in math_keywords):
        return "math"
    elif any(keyword in lower_p for keyword in coding_keywords):
        return "coding"
    else:
        return "commonsense"
