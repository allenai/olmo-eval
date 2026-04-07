from __future__ import annotations

import re
import string
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import permutations
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import AccuracyMetric, F1Metric, LogprobMCAccuracyMetric
from olmo_eval.common.scorers import Scorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.constants.drop import DROP_FIXED_FEWSHOT, DROP_MC_FIXED_FEWSHOT

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _remove_articles(text: str) -> str:
    return _ARTICLES_RE.sub(" ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


def _remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in exclude)
    return text


def _fix_number(text: str) -> str:
    return str(float(text)) if _is_number(text) else text


def _tokenize(text: str) -> list[str]:
    return re.split(r" |-", text)


def _normalize(answer: str) -> str:
    tokens = [
        _white_space_fix(_remove_articles(_fix_number(_remove_punc(token.lower()))))
        for token in _tokenize(answer)
    ]
    tokens = [token for token in tokens if token.strip()]
    return " ".join(tokens).strip()


def _answer_to_bags(
    answer: str | list[str] | tuple[str, ...],
) -> tuple[list[str], list[set[str]]]:
    raw_spans = list(answer) if isinstance(answer, (list, tuple)) else [answer]
    normalized_spans: list[str] = []
    token_bags: list[set[str]] = []
    for raw_span in raw_spans:
        normalized_span = _normalize(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _compute_f1(predicted_bag: set[str], gold_bag: set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    precision = 1.0 if not predicted_bag else intersection / float(len(predicted_bag))
    recall = 1.0 if not gold_bag else intersection / float(len(gold_bag))
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _match_numbers_if_present(gold_bag: set[str], predicted_bag: set[str]) -> bool:
    gold_numbers = {w for w in gold_bag if _is_number(w)}
    predicted_numbers = {w for w in predicted_bag if _is_number(w)}
    return (not gold_numbers) or bool(gold_numbers.intersection(predicted_numbers))


def _align_bags(predicted_bags: list[set[str]], gold_bags: list[set[str]]) -> list[float]:
    n_gold = len(gold_bags)
    n_pred = len(predicted_bags)
    max_dim = max(n_gold, n_pred)

    if n_gold == 0 or n_pred == 0:
        return [0.0] * max_dim

    score_mat = [[0.0] * n_pred for _ in range(n_gold)]
    for gi, g in enumerate(gold_bags):
        for pi, p in enumerate(predicted_bags):
            if _match_numbers_if_present(g, p):
                score_mat[gi][pi] = _compute_f1(p, g)

    best_scores = [0.0] * max_dim
    best_total = -1.0

    if n_gold <= n_pred:
        # More (or equal) predicted than gold: assign each gold to a unique predicted
        for perm in permutations(range(n_pred), n_gold):
            total = sum(score_mat[gi][pi] for gi, pi in enumerate(perm))
            if total > best_total:
                best_total = total
                best_scores = [0.0] * max_dim
                for gi, pi in enumerate(perm):
                    best_scores[gi] = score_mat[gi][pi]
    else:
        # More gold than predicted: assign each predicted to a unique gold
        for perm in permutations(range(n_gold), n_pred):
            total = sum(score_mat[gi][pi] for pi, gi in enumerate(perm))
            if total > best_total:
                best_total = total
                best_scores = [0.0] * max_dim
                for pi, gi in enumerate(perm):
                    best_scores[gi] = score_mat[gi][pi]

    return best_scores


def _get_drop_metrics(
    predicted: str | list[str] | tuple[str, ...],
    gold: str | list[str] | tuple[str, ...],
) -> tuple[float, float]:
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = sum(f1_per_bag) / len(f1_per_bag) if f1_per_bag else 0.0
    f1 = round(f1, 2)
    return exact_match, f1


def _score_drop(predicted_text: str, answers: list[tuple[str, ...]]) -> tuple[float, float]:
    max_em = 0.0
    max_f1 = 0.0
    for gold_answer in answers:
        if gold_answer[0].strip():
            em, f1 = _get_drop_metrics(predicted_text, gold_answer)
            max_em = max(max_em, em)
            max_f1 = max(max_f1, f1)
    return max_em, max_f1


@dataclass(frozen=True, slots=True)
class DROPF1Scorer(Scorer):
    name: str = "drop_f1"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.extracted_answer is None:
            return 0.0
        answers = instance.metadata.get("answers", [])
        if not answers:
            return 0.0
        _, f1 = _score_drop(str(output.extracted_answer).strip(), answers)
        return f1


@dataclass(frozen=True, slots=True)
class DROPExactMatchScorer(Scorer):
    name: str = "drop_exact_match"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.extracted_answer is None:
            return 0.0
        answers = instance.metadata.get("answers", [])
        if not answers:
            return 0.0
        em, _ = _score_drop(str(output.extracted_answer).strip(), answers)
        return em


_DESCRIPTION = (
    "The following are reading comprehension questions, where the answer to each "
    "question is either a segment of text from the corresponding passage, a number, "
    "or a date (containing any of the date, month, and/or year components). Some "
    "questions may require you to pull together information pieces from the passage "
    "and reason over them.\n\n"
)


def _parse_answer(answer: dict[str, Any]) -> tuple[str, ...]:
    if answer["number"] != "":
        return (str(answer["number"]),)
    if answer["spans"] != []:
        return tuple(answer["spans"])
    return (
        " ".join([answer["date"]["day"], answer["date"]["month"], answer["date"]["year"]]).strip(),
    )


def _format_drop_mc(
    passage: str, question: str, choices: tuple[str, ...], answer: str | None = None
) -> str:
    choices_text = "\n".join(f" {chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
    prompt = f"Passage: {passage}\nQuestion: {question}\n{choices_text}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


def _get_answers(doc: dict[str, Any]) -> list[tuple[str, ...]]:
    def _flatten_validated_answers(validated_answers: dict[str, Any]) -> list[dict[str, Any]]:
        valid_answers = []
        for i in range(len(validated_answers["number"])):
            valid_answers.append(
                {
                    "number": validated_answers["number"][i],
                    "date": validated_answers["date"][i],
                    "spans": validated_answers["spans"][i],
                }
            )
        return valid_answers

    answers: list[tuple[str, ...]] = []
    answers_set: set[tuple[str, ...]] = set()
    candidates = [doc["answer"]] + _flatten_validated_answers(doc["validated_answers"])
    for candidate in candidates:
        answer = _parse_answer(candidate)
        if answer in answers_set:
            continue
        answers_set.add(answer)
        answers.append(answer)
    return answers


@register("drop")
class Drop(Task):
    data_source = DataSource(path="EleutherAI/drop", split="validation")
    split = Split.VALIDATION
    metrics = (
        F1Metric(scorer=DROPF1Scorer),
        AccuracyMetric(scorer=DROPExactMatchScorer),
    )
    primary_metric = F1Metric(scorer=DROPF1Scorer)
    sampling_params = SamplingParams(
        max_tokens=100,
        temperature=0,
        stop_sequences=("Passage:", "Question:", "\n\n"),
    )
    num_fewshot = 0

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached(split="validation")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        is_mc = "passage_original" in doc

        if is_mc:
            passage = doc["passage_original"].strip()
            question = doc["question_original"]
            choices = tuple(doc["choices"]["text"])
            answer_key = doc["answerKey"]
            gold_idx = ord(answer_key) - ord("A")
            gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

            return Instance(
                question=question,
                choices=choices,
                gold_answer=answer_key,
                metadata={
                    "id": doc.get("id", f"drop_mc_{index}"),
                    "index": index,
                    "gold_idx": gold_idx,
                    "gold_text": gold_text,
                    "passage": passage,
                },
            )

        passage = doc["passage"]
        question = doc["question"]
        answers = _get_answers(doc)
        gold_answer = " " + ", ".join(answers[0])

        formatted_question = f"Passage: {passage}\nQuestion: {question}\nAnswer:"

        return Instance(
            question=formatted_question,
            gold_answer=gold_answer,
            metadata={
                "id": doc.get("query_id", f"drop_{index}"),
                "answers": answers,
                "index": index,
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        if self.config.fewshot_source == "olmes_drop_fixed":
            return self._build_fixed_fewshot()
        if self.config.fewshot_source == "olmes_drop_mc_fixed":
            return self._build_mc_fixed_fewshot()
        return super()._build_fewshot()

    def _build_mc_fixed_fewshot(self) -> list[Instance]:
        instances = []
        for doc in DROP_MC_FIXED_FEWSHOT:
            passage = str(doc["passage_original"]).strip()
            question = str(doc["question_original"])
            choices_data = doc["choices"]
            assert isinstance(choices_data, dict)
            choices = tuple(choices_data["text"])
            answer_key = str(doc["answerKey"])
            gold_idx = ord(answer_key) - ord("A")
            gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

            instances.append(
                Instance(
                    question=question,
                    choices=choices,
                    gold_answer=gold_text,
                    metadata={
                        "gold_idx": gold_idx,
                        "gold_text": gold_text,
                        "mc_answer": answer_key,
                        "passage": passage,
                    },
                )
            )

        num = self.config.num_fewshot
        if num and num < len(instances):
            instances = instances[:num]
        return instances

    def _build_fixed_fewshot(self) -> list[Instance]:
        instances = []
        for doc in DROP_FIXED_FEWSHOT:
            answers = _get_answers(doc)
            gold_answer = " " + ", ".join(answers[0])
            formatted_question = f"Passage: {doc['passage']}\nQuestion: {doc['question']}\nAnswer:"
            instances.append(
                Instance(
                    question=formatted_question,
                    gold_answer=gold_answer,
                    metadata={"answers": answers},
                )
            )

        num = self.config.num_fewshot
        if num and num < len(instances):
            instances = instances[:num]
        return instances

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()
        is_mc = self.config.formatter is not None

        if is_mc:
            parts: list[str] = []
            for ex in fewshot:
                answer = ex.metadata.get("mc_answer", "")
                passage = ex.metadata.get("passage", "")
                parts.append(_format_drop_mc(passage, ex.question, ex.choices or (), answer))

            passage = instance.metadata.get("passage", "")
            parts.append(_format_drop_mc(passage, instance.question, instance.choices or ()))

            continuations = tuple(
                f" {chr(ord('A') + i)}" for i in range(len(instance.choices or ()))
            )
            prompt = "\n\n".join(parts)
            return LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt=prompt,
                continuations=continuations,
            )

        parts = []
        for ex in fewshot:
            parts.append(ex.question + (ex.gold_answer or ""))
        parts.append(instance.question)
        prompt = "\n\n".join(parts)

        if fewshot:
            prompt = _DESCRIPTION + prompt

        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)

    def extract_answer(self, output: LMOutput) -> str:
        return output.text.strip()


register_variant("drop", "gen")
register_variant(
    "drop",
    "mc",
    data_source=DataSource(path="allenai/drop_mc", split="validation"),
    formatter=MultipleChoiceFormatter(),
    metrics=(LogprobMCAccuracyMetric(),),
    primary_metric=LogprobMCAccuracyMetric(),
)
register_variant(
    "drop",
    "olmo3base",
    num_fewshot=5,
    fewshot_source="olmes_drop_fixed",
)
