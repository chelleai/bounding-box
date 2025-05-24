#

import base64
from pydantic import BaseModel
from typing import  List

from library.opensource.aikernel import LLMModelName
from library.opensource.aikernel import (
    LLMMessageContentType,
    LLMMessagePart,
    LLMModelName,
    LLMSystemMessage,
    LLMUserMessage,
    get_router,
    llm_structured
)



class SolutionNumbers(BaseModel):
    """Problem numbers for problems or subproblems."""
    solution_numbers: List[str]

class SolutionUpperBounds(BaseModel):
    """Upper bounds for solutions to problems or subproblems."""
    upper_bounds: List[int]

class NumberedSolutionUpperBound(BaseModel):
    solution_number: str
    upper_bound: int

class NumberedSolutionUpperBounds(BaseModel):
    """Upper bounds for solutions to problems or subproblems."""
    upper_bounds: List[NumberedSolutionUpperBound]

async def get_solution_numbers(image_bytes: bytes, example_image_bytes: bytes|None=None):

    solution_numbers = await llm_structured(
        messages=[
            LLMSystemMessage(
            parts=[
                LLMMessagePart(
                    content_type=LLMMessageContentType.TEXT,
                    content=f"""
                    Return the number of the problems or subproblems in the image as an array of strings.
                    """,
                )
            ]
        ),
        # Detect the number denoting each problem. If a problem has multiple parts, detect the letters or number and letter denoting each subproblem in the image.
        LLMUserMessage(
            parts=[
                LLMMessagePart(
                    content_type=LLMMessageContentType.TEXT,
                    content=f"""
                    Detect the marking denoting each problem or problem part in the image of an assignment.
                    Markings are usually a number or letter, and are typically found to the left near the top of the the problem or problem part.
                    Problems are typically denoted by a number, sometimes with a period or parenthesis (for example, "1", "2.", "3)")
                    Problem parts are typically denoted by a letter, sometimes with a period or parenthesis (for example, "a", "b.", "c)")
                    Problem parts may also be denoted by a number and letter (for example, "1a", "1b", "1c", "2a", "2b")
                    Return the numbers for problems, and combination of number and letter for subproblems (e.g., "1a", "1b", "2", "3a", "3b", "3c").
                    """,
                )
            ]
        ),
        LLMUserMessage(
            parts=[
                LLMMessagePart(
                    content_type=LLMMessageContentType.JPEG,
                    content=base64.b64encode(image_bytes).decode()
                ),
            ]
        ),
        ],
        # model="gemini-2.0-flash-001",
        router=get_router(models=(LLMModelName.GEMINI_20_FLASH,)),
        # router=get_router(models=("gemini-2.0-flash-001")),
        response_model=SolutionNumbers
    )
    
    return solution_numbers


async def get_solution_upper_bounds(image_bytes: bytes, example_image_bytes: bytes|None=None):

    solution_upper_bounds = await llm_structured(
        messages=[
            LLMSystemMessage(
                parts=[
                    LLMMessagePart(
                        content_type=LLMMessageContentType.TEXT,
                        content=f"""
                        Return the y-axis positions at the top of each problem or subproblem solutions as an array of integers. Never return masks.
                        """,
                    )
                ]
            ),
            LLMUserMessage(
                parts=[
                    LLMMessagePart(
                        content_type=LLMMessageContentType.TEXT,
                        content=f"""
                        Detect the upper bound of each of the solutions to the problems or subproblems in the image.
                        Include the problem statement and/or prompt when calculating the upper bounds.
                        If a problem has multiple parts (e.g., 1a, 1b, 1c), treat each part as a separate problem, and return the upper bound for each part.
                        Problems are typically denoted by a number, sometimes with a period or parenthesis (e.g., "1", "2.", "3)")
                        Subproblems are typically denoted by a letter, sometimes with a period or parenthesis (e.g., "a", "b.", "c)")
                        The upper bounds are the y-axis positions at the top of each problem or subproblem solutions.
                        """,
                    )
                ]
            ),
            LLMUserMessage(
                parts=[
                    LLMMessagePart(
                        content_type=LLMMessageContentType.JPEG,
                        content=base64.b64encode(image_bytes).decode()
                    ),
                ]
            ),
        ],
        # router=get_router(models=("gemini-2.0-flash-001")),
        router=get_router(models=(LLMModelName.GEMINI_20_FLASH,)),
        temperature=0.0,
        response_model=SolutionUpperBounds
    )
    
    return solution_upper_bounds

        
async def get_numbered_solution_upper_bounds(image_bytes: bytes, system_prompt:str, user_prompt: str, solution_numbers: List[str], example_image_bytes: bytes|None=None):

    solution_upper_bounds = await llm_structured(
        messages=[
            LLMSystemMessage(
            parts=[
                LLMMessagePart(
                    content_type=LLMMessageContentType.TEXT,
                    content=system_prompt
                    # f"""
                    # Return the y-axis positions at the top of each problem or subproblem solutions. Never return masks.
                    # """,
                )
            ]
        ),
        LLMUserMessage(
            parts=[
                LLMMessagePart(
                    content_type=LLMMessageContentType.TEXT,
                    # content=f"""
                    # Detect the upper bound of solutions to the problems or subproblems in the image for the provided problem numbers.
                    # Include the problem statement and/or prompt when calculating the upper bounds.
                    # If a problem has multiple parts (e.g., 1a, 1b, 1c), treat each part as a separate problem, and return the upper bound for each part.
                    # Problems are typically denoted by a number, sometimes with a period or parenthesis (e.g., "1", "2.", "3)")
                    # Subproblems are typically denoted by a letter, sometimes with a period or parenthesis (e.g., "a", "b.", "c)")
                    # The upper bounds are the y-axis positions at the top of each problem or subproblem solutions.
                    # If you cannot detect the upper bound for a problem or subproblem, return -1.
                    # The problem numbers are {[f"{k1}" for k1 in solution_numbers]}.
                    # Return a dictionary with the problem numbers as keys and the upper bounds as values.
                    # The dictionary should be in the format {{ "1": 100, "2a": 200, "2b": 300, "2c": -1, "3": 400 }}.
                    # """,
                    # content=f"""
                    # Detect the upper boundary y-coordinate of solutions to problems in the provided image.

                    # TASK DEFINITION:
                    # - For each problem identifier provided in the input list, identify where the solution area begins (upper boundary).
                    # - Return the y-coordinate value (in pixels from the top of the image) for this upper boundary.
                    # - The y-coordinate represents the vertical position where the solution to the current problem begins.

                    # PROBLEM NUMBERING CONVENTIONS:
                    # - Problem identifiers are provided as strings in the solution_numbers list.
                    # - These identifiers represent main problems (e.g., "1", "2.", "3)", "4.5")

                    # OUTPUT FORMAT:
                    # - Use -1 for any problem whose boundary cannot be confidently determined, or if the problem is not present in the image.

                    # HANDLING EDGE CASES:
                    # - If problem boundaries are unclear or solutions overlap, use visual cues such as whitespace, horizontal lines, or changes in formatting to determine boundaries.
                    # - If multiple possible boundaries exist, choose the most visually distinct one.

                    # The problem identifiers to detect are: {solution_numbers}
                    # """
                    # content=f"""
                    # Detect the y-coordinate of the upper boundary problems in the provided image, including the problem statement.

                    # TASK DEFINITION:
                    # - For each problem identifier provided in the input list, identify the upper boundary of the entire problem, including the problem statement.
                    # - Return the y-coordinate value (in pixels from the top of the image) for this upper boundary.
                    # - The y-coordinate represents the vertical position where the identifier to a problem begins.

                    # PROBLEM NUMBERING CONVENTIONS:
                    # - Problem identifiers are provided as strings in the solution_numbers list.
                    # - These identifiers represent main problems (e.g., "1", "2.", "3)", "4.5")

                    # OUTPUT FORMAT:
                    # - Use -1 for any problem whose boundary cannot be confidently determined, or if the problem is not present in the image.

                    # HANDLING EDGE CASES:
                    # - If problem boundaries are unclear or solutions overlap, use visual cues such as whitespace, horizontal lines, or changes in formatting to determine boundaries.
                    # - If multiple possible boundaries exist, choose the most visually distinct one.

                    # The problem identifiers to detect are: {solution_numbers}
                    # """
                    content=user_prompt
                )
            ]
        ),
        LLMUserMessage(
            parts=[
                LLMMessagePart(
                    content_type=LLMMessageContentType.JPEG,
                    content=base64.b64encode(image_bytes).decode()
                ),
            ]
        ),
        ],
        # model="gemini-2.0-flash-001",
        # router=get_router(models=(LLMModelName.GEMINI_20_FLASH,)),
        router=get_router(models=(LLMModelName.GEMINI_25_FLASH,)),
        # router=get_router(models=("gemini-2.0-flash-001")),
        response_model=NumberedSolutionUpperBounds
    )
    
    return solution_upper_bounds
