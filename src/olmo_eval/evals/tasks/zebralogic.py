"""ZebraLogic task implementation."""

import json
from collections.abc import Iterator, Sequence
from typing import Any

from olmo_eval.core import (
    Instance,
    LMOutput,
    LMRequest,
    Metric,
    RequestType,
    Response,
    Scorer,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.core import Task, TaskConfig, register

# Grid sizes categorized by difficulty
EASY_SIZES = ["2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3"]
HARD_SIZES = [
    "3*4",
    "3*5",
    "4*2",
    "3*6",
    "4*3",
    "4*4",
    "5*2",
    "6*2",
    "4*5",
    "4*6",
    "5*3",
    "5*4",
    "5*5",
    "5*6",
    "6*3",
    "6*4",
    "6*5",
    "6*6",
]

# Prompt template for ZebraLogic
# fmt: off
ZEBRA_GRID_TEMPLATE = """
# Example Puzzle

There are 3 houses, numbered 1 to 3 from left to right, as seen from across \
the street. Each house is occupied by a different person. Each house has a \
unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to \
Clue 2, Arnold is directly left of the one who only drinks water. The person \
in House 3 cannot be on the left of anyone, so Arnold must be in House 1. \
Thus, Peter drinks water, and Eric lives in House 3. Then, according to \
Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {
        "House 1": {
            "Name": "Arnold",
            "Drink": "tea"
        },
        "House 2": {
            "Name": "Peter",
            "Drink": "water"
        },
        "House 3": {
            "Name": "Eric",
            "Drink": "milk"
        }
    }
}

# Puzzle to Solve

{puzzle}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}

"""
# fmt: on


def extract_last_complete_json(text: str) -> dict | None:
    """Extract the last complete JSON object from text.

    Source: https://github.com/WildEval/ZeroEval/blob/main/src/evaluation/eval_utils.py
    """
    stack = []
    last_json_start = None
    last_json_str = None

    for i, char in enumerate(text):
        if char == "{":
            stack.append(i)
            if last_json_start is None:
                last_json_start = i
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and last_json_start is not None:
                    # Complete JSON object found
                    last_json_str = text[last_json_start : i + 1]
                    last_json_start = None

    if last_json_str:
        try:
            return json.loads(last_json_str.replace("\n", ""))
        except json.JSONDecodeError:
            pass
    return None


class ZebraLogicScorer(Scorer):
    """Custom scorer for ZebraLogic that computes puzzle and cell accuracy."""

    name: str = "zebralogic"

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score the output against the solution table."""
        solution_table = instance.metadata.get("solution_table", {})
        total_cells = instance.metadata.get("total_cells", 0)

        if output.extracted_answer is None:
            return 0.0

        pred_table = output.extracted_answer
        if not isinstance(pred_table, dict):
            return 0.0

        # Count correct cells
        correct_cells = 0
        for house, attrs in solution_table.items():
            if house in pred_table:
                for attr_name, attr_value in attrs.items():
                    if pred_table[house].get(attr_name) == attr_value:
                        correct_cells += 1

        # Cell accuracy
        cell_acc = correct_cells / total_cells if total_cells > 0 else 0.0

        # Puzzle accuracy (all or nothing)
        puzzle_acc = 1.0 if correct_cells == total_cells else 0.0

        # Store both in metadata for metric computation
        output.metadata = output.metadata or {}
        output.metadata["cell_accuracy"] = cell_acc
        output.metadata["puzzle_accuracy"] = puzzle_acc
        output.metadata["parsed"] = 1.0 if pred_table else 0.0

        return puzzle_acc


class ZebraLogicMetric(Metric):
    """Metric for ZebraLogic that computes puzzle_accuracy, cell_accuracy, and parsed rate."""

    name: str = "zebralogic"

    def __init__(self, metric_type: str = "puzzle_accuracy"):
        self.metric_type = metric_type
        self.name = metric_type

    def compute(self, responses: Sequence[Response]) -> float:
        """Compute the metric across all responses."""
        values = []
        for response in responses:
            for output in response.outputs:
                if output.metadata and self.metric_type in output.metadata:
                    values.append(output.metadata[self.metric_type])

        if not values:
            return 0.0
        return sum(values) / len(values)


class ZebraLogicTask(Task):
    """ZebraLogic logic puzzle task.

    ZebraLogic is a benchmark for evaluating logical reasoning capabilities
    on grid-based constraint satisfaction puzzles (similar to Einstein's riddle).

    See: https://huggingface.co/blog/yuchenlin/zebra-logic
    Part of "ZeroEval" (https://github.com/WildEval/ZeroEval)
    """

    default_hf_path: str = "allenai/ZebraLogicBench-private"
    default_subset: str = "grid_mode"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the test split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("test")
            for idx, doc in enumerate(loader.load(source)):
                self._instances_cache.append(self.process_doc(doc, idx))
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split)
        except ValueError:
            return DataSource(
                path=self.default_hf_path,
                subset=self.default_subset,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any], index: int) -> Instance:
        """Convert a dataset document to an Instance."""
        size = doc["size"]
        puzzle = doc["puzzle"]
        solution = doc["solution"]

        # Build the prompt
        prompt_str = ZEBRA_GRID_TEMPLATE.replace("{puzzle}", puzzle)

        # Create JSON template for expected output format
        json_template: dict[str, Any] = {"reasoning": "___", "solution": {}}
        num_houses = len(solution["rows"])
        columns = solution["header"]

        for i in range(num_houses):
            json_template["solution"][f"House {i + 1}"] = {
                columns[j]: "___" for j in range(1, len(columns))
            }

        json_str = json.dumps(json_template, indent=4)
        prompt_str = prompt_str.replace("{json_template}", json_str)

        # Construct solution table for scoring
        solution_table: dict[str, dict[str, str]] = {}
        total_cells = 0

        for i in range(num_houses):
            solution_table[f"House {i + 1}"] = {
                columns[j]: solution["rows"][i][j] for j in range(1, len(columns))
            }
            total_cells += len(columns) - 1

        difficulty = "easy" if size in EASY_SIZES else "hard"

        return Instance(
            question=prompt_str,
            gold_answer=json.dumps(solution_table),
            metadata={
                "index": index,
                "puzzle": puzzle,
                "size": size,
                "solution_table": solution_table,
                "total_cells": total_cells,
                "difficulty": difficulty,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=instance.question,
        )

    def extract_answer(self, output: LMOutput) -> dict | None:
        """Extract the solution JSON from model output."""
        if not output.text:
            return None

        # Try to extract JSON from the output
        parsed = extract_last_complete_json(output.text)
        if parsed and "solution" in parsed:
            return parsed["solution"]

        return None


def _zebralogic_config() -> TaskConfig:
    return TaskConfig(
        name="zebralogic",
        data_source=DataSource(path="allenai/ZebraLogicBench-private", subset="grid_mode"),
        scorers=(ZebraLogicScorer(),),
        metrics=(
            ZebraLogicMetric(metric_type="puzzle_accuracy"),
            ZebraLogicMetric(metric_type="cell_accuracy"),
            ZebraLogicMetric(metric_type="parsed"),
        ),
    )


@register("zebralogic", _zebralogic_config)
class ZebraLogic(ZebraLogicTask):
    """ZebraLogic task."""

    pass
