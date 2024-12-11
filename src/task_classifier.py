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
    """
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
