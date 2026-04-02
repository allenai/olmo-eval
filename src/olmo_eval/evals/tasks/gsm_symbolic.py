import re
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import AccuracyMetric, PassAtKMetric
from olmo_eval.common.scorers import ExactMatchScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

# fmt: off
GSM8K_FIXED_FEWSHOT = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.",
        "short_answer": "6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.",
        "short_answer": "5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.",
        "short_answer": "39",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.",
        "short_answer": "8",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.",
        "short_answer": "9",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.",
        "short_answer": "29",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.",
        "short_answer": "33",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.",
        "short_answer": "8",
    },
]
# fmt: on


def _extract_last_number(text: str) -> str | None:
    output = re.sub(r"(\d),(\d)", r"\1\2", text)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    return numbers[-1] if numbers else None


def _clean_short_answer(text: str) -> str:
    output = re.sub(r"(\d),(\d)", r"\1\2", text)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    return numbers[-1] if numbers else text


@register("gsm_symbolic")
class GSMSymbolic(Task):
    data_source = DataSource(path="apple/GSM-Symbolic", subset="main")
    metrics = (AccuracyMetric(scorer=ExactMatchScorer),)
    num_fewshot = 8
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0,
        stop_sequences=("Question:", "\n\n"),
    )

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc["question"]
        answer = doc["answer"]
        short_answer = answer.split("####")[-1].strip()
        cleaned = _clean_short_answer(short_answer)

        return Instance(
            question=question,
            gold_answer=cleaned,
            metadata={
                "id": index,
                "answer": answer,
                "short_answer": short_answer,
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        instances = []
        for doc in GSM8K_FIXED_FEWSHOT:
            instances.append(
                Instance(
                    question=doc["question"],
                    gold_answer=doc["short_answer"],
                    metadata={
                        "answer": doc["answer"],
                        "short_answer": doc["short_answer"],
                    },
                )
            )
        num = self.config.num_fewshot
        if num and num < len(instances):
            instances = instances[:num]
        return instances

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()

        parts: list[str] = []
        for ex in fewshot:
            parts.append(f"Question: {ex.question}\nAnswer: {ex.metadata['answer']}")
        parts.append(f"Question: {instance.question}\nAnswer:")
        prompt = "\n\n".join(parts)

        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)

    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_last_number(output.text)


register_variant(
    "gsm_symbolic",
    "p1",
    data_source=DataSource(path="apple/GSM-Symbolic", subset="p1"),
)

register_variant(
    "gsm_symbolic",
    "p2",
    data_source=DataSource(path="apple/GSM-Symbolic", subset="p2"),
)

register_variant(
    "gsm_symbolic",
    "olmo3base",
    metrics=(
        AccuracyMetric(scorer=ExactMatchScorer),
        PassAtKMetric(k=1, scorer=ExactMatchScorer),
        PassAtKMetric(k=2, scorer=ExactMatchScorer),
        PassAtKMetric(k=4, scorer=ExactMatchScorer),
        PassAtKMetric(k=8, scorer=ExactMatchScorer),
    ),
    primary_metric=PassAtKMetric(k=1, scorer=ExactMatchScorer),
    sampling_params=SamplingParams(
        max_tokens=1024,
        temperature=0.6,
        top_p=0.6,
        stop_sequences=("Question:", "\n\n"),
        num_samples=8,
    ),
)
