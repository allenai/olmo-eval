# ruff: noqa: E501
"""Prompt construction for SciCode sequential sub-step generation.

Templates mirror scicode-bench/SciCode/eval/data/{multistep_template.txt,
background_comment_template.txt}. The hardcoded snippets are gold code the
upstream repo ships for three sub-steps (13.6, 62.1, 76.3) rather than asking
the model to generate them.
"""

from __future__ import annotations

import re
from typing import Any

from olmo_eval.evals.extract import extract_code as _extract_code_first

_HARDCODED_SNIPPETS: dict[str, dict[int, str]] = {
    "13": {
        5: '''class Maxwell:
    """ The base class for evolution of Maxwell's equations.
    """

    def __init__(self, n_grid, x_out):
        """Constructor sets up coordinates, memory for variables."""

        self.n_grid = n_grid
        self.n_vars = 7
        self.delta = float(x_out) / (n_grid - 2.0)
        delta = self.delta

        _axis = np.linspace(-self.delta * 0.5, x_out + 0.5 * self.delta, self.n_grid)
        self.x = _axis[:, None, None]
        self.y = _axis[None, :, None]
        self.z = _axis[None, None, :]
        self.r = np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

        self.E_x = zeros((n_grid, n_grid, n_grid))
        self.E_y = zeros((n_grid, n_grid, n_grid))
        self.E_z = zeros((n_grid, n_grid, n_grid))
        self.A_x = zeros((n_grid, n_grid, n_grid))
        self.A_y = zeros((n_grid, n_grid, n_grid))
        self.A_z = zeros((n_grid, n_grid, n_grid))
        self.phi = zeros((n_grid, n_grid, n_grid))
        self.constraint = zeros((n_grid, n_grid, n_grid))

        self.t = 0.0
''',
    },
    "62": {
        0: """class Block:
    def __init__(self, length, basis_size, operator_dict):
        self.length = length
        self.basis_size = basis_size
        self.operator_dict = operator_dict


class EnlargedBlock:
    def __init__(self, length, basis_size, operator_dict):
        self.length = length
        self.basis_size = basis_size
        self.operator_dict = operator_dict
""",
    },
    "76": {
        2: '''def generate_dna(N: int, PWM: dict) -> tuple:
    """Generate a random DNA sequence with a motif inserted at a random position."""
    p = random.randint(0, N - 1)

    nucleotide = "ACGT"
    uni_weights = [0.25, 0.25, 0.25, 0.25]
    dna_string = "".join(random.choices(nucleotide, uni_weights, k=N))

    spiked_seq = "".join(
        random.choices(nucleotide, weights=[PWM[nuc][i] for nuc in nucleotide], k=1)[0]
        for i in range(len(PWM["A"]))
    )

    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    reversed_seq = dna_string[::-1]
    reverse_complement = "".join(complement[nuc] for nuc in reversed_seq if nuc in complement)

    new_seq = dna_string[:p] + spiked_seq + dna_string[p:]
    new_seq_rc = reverse_complement[: N - p] + spiked_seq + reverse_complement[N - p :]

    return p, new_seq, new_seq_rc
''',
    },
}


_WITH_BACKGROUND_PROMPT_TEMPLATE = """\
PROBLEM DESCRIPTION:
You will be provided with problem steps along with background knowledge necessary for solving the problem. Your task will be to develop a Python solution focused on the next step of the problem-solving process.

PROBLEM STEPS AND FUNCTION CODE:
Here, you'll find the Python code for the initial steps of the problem-solving process. This code is integral to building the solution.

{problem_steps_str}

NEXT STEP - PROBLEM STEP AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. A function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.

{dependencies}

RESPONSE GUIDELINES:
Now, based on the instructions and information provided above, write the complete and executable Python program for the next step in a single block.
Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
Your response should NOT include the dependencies and functions of all previous steps. If your next step function calls functions from previous steps, please make sure it uses the headers provided without modification.
DO NOT generate EXAMPLE USAGE OR TEST CODE in your response. Please make sure your response python code in format of ```python```."""


_DEFAULT_PROMPT_TEMPLATE = """\
PROBLEM DESCRIPTION:
You will be provided with the main description of the problem, previous steps, and the next step. Your task will be to generate the disciplinary knowledge necessary for solving the next step and then develop a Python solution focused on this step.

PREVIOUS STEPS DESCRIPTION:
{problem_steps_str}

NEXT STEP - PROBLEM DESCRIPTION AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. First, provide the necessary scientific background knowledge as a comment at the beginning of your response, starting with 'Background: '. Then, a function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.
{dependencies}

RESPONSE GUIDELINES:
1. Start with the scientific background required for the next step, formatted as a comment.
2. Then write the complete and executable Python program for the next step in a single block.
3. Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
4. DO NOT include previous function code, example usage or test code in your response.
5. Ensure your response is in the format of ```python``` and includes the necessary background as a comment at the top.

Example:
```python
# Background: [Here, insert the necessary scientific knowledge required for the next step.]

[Insert the Python code here based on the provided function header and dependencies.]
```
"""


def _step_description(step: dict[str, Any], with_background: bool) -> str:
    desc = step["step_description_prompt"]
    if with_background and step.get("step_background"):
        return desc + "\n" + step["step_background"]
    return desc


def _step_header_and_return(step: dict[str, Any]) -> str:
    header = step["function_header"]
    return_line = step.get("return_line") or ""
    return f"{header}\n\n{return_line}"


def build_step_prompt(
    sub_steps: list[dict[str, Any]],
    required_dependencies: str,
    step_idx: int,
    previous_llm_code: list[str | None],
    with_background: bool,
) -> str:
    """Build the prompt asking the model to generate sub-step ``step_idx``.

    ``previous_llm_code[i]`` must contain the code for sub-step ``i`` for all
    ``i < step_idx`` (either model-generated from an earlier call, or a
    hardcoded snippet).
    """
    output_lines: list[str] = []
    for i in range(step_idx):
        output_lines.append(_step_description(sub_steps[i], with_background))
        prev_code = previous_llm_code[i]
        if prev_code is not None:
            output_lines.append(prev_code)
        output_lines.append("------")
    if output_lines:
        output_lines = output_lines[:-1]

    next_step_lines = [
        _step_description(sub_steps[step_idx], with_background),
        _step_header_and_return(sub_steps[step_idx]),
    ]

    template = _WITH_BACKGROUND_PROMPT_TEMPLATE if with_background else _DEFAULT_PROMPT_TEMPLATE
    return template.format(
        problem_steps_str="\n\n".join(output_lines),
        next_step_str="\n\n".join(next_step_lines),
        dependencies=required_dependencies,
    )


def strip_import_lines(code: str) -> str:
    kept = []
    for line in code.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        kept.append(line)
    return "\n".join(kept)


_PYTHON_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_GENERIC_BLOCK_RE = re.compile(r"```\n?(.*?)```", re.DOTALL)


def extract_step_code(text: str) -> str:
    """Extract the final python code block from a model response.

    Reasoning models interleave scratch code blocks with their final answer, so
    we take the LAST fenced block, not the first.
    """
    if not text:
        return ""
    matches = _PYTHON_BLOCK_RE.findall(text)
    if not matches:
        matches = _GENERIC_BLOCK_RE.findall(text)
    code = _extract_code_first(text) if not matches else matches[-1]
    if not code:
        return ""
    return strip_import_lines(code)


def hardcoded_for_problem(problem_id: str) -> dict[int, str]:
    return _HARDCODED_SNIPPETS.get(problem_id, {})
