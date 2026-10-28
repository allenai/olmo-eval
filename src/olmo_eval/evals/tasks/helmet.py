"""HELMET tasks: all seven categories at HELMET's standard context lengths.

HELMET (https://github.com/princeton-nlp/HELMET) evaluates seven task
categories at 4k-128k tokens of context: recall, RAG, re-ranking, LongQA,
summarization, ICL, and citation (ALCE). All seven are registered here.

Pre-generated and pre-retrieved data lives on the ai2-internal
`allenai/helmet-plus` Hub dataset; the remaining tasks pull their sources
from the Hub at load time. `narrativeqa` and both summarization tasks are
graded by an LLM judge (see `helmet_judge.py`) and therefore need a judge
configured -- the `helmet_nojudge__*` suites exclude them.

Upstream reference: princeton-nlp/HELMET at commit aeadacc6.
"""

import os
import re
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics import AccuracyMetric, NDCGMetric, RecallMetric, RougeLF1Metric
from olmo_eval.common.scorers import Scorer
from olmo_eval.common.scorers.alce import AlceQampariRecTop5Scorer, AlceStrEmScorer
from olmo_eval.common.scorers.base import _squad_normalize_answer
from olmo_eval.common.scorers.substring import SubstringExactMatchScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType
from olmo_eval.data import (
    helmet_alce_loader,
    helmet_icl_loader,
    helmet_infbench_loader,
    helmet_kilt_loader,
    helmet_loader,
    helmet_msmarco_loader,
    helmet_multilexsum_loader,
    helmet_narrativeqa_loader,
)
from olmo_eval.data.helmet_alce_loader import load_alce_dataset
from olmo_eval.data.helmet_icl_loader import ICL_DATASETS, load_icl_dataset
from olmo_eval.data.helmet_infbench_loader import (
    INFBENCH_SUBSETS,
    INFINITEBENCH_REPO,
    INFINITEBENCH_REVISION,
    REFERENCE_TOKENIZER,
    REFERENCE_TOKENIZER_REVISION,
    load_infbench_dataset,
)
from olmo_eval.data.helmet_kilt_loader import load_kilt_dataset
from olmo_eval.data.helmet_loader import (
    HELMET_PLUS_REPO_ID,
    HELMET_PLUS_REVISION,
    load_json_kv_dataset,
)
from olmo_eval.data.helmet_msmarco_loader import load_msmarco_dataset, parse_rankings
from olmo_eval.data.helmet_multilexsum_loader import (
    MIN_SOURCE_TOKENS,
    MULTI_LEXSUM_REPO,
    MULTI_LEXSUM_REVISION,
    load_multi_lexsum_dataset,
)
from olmo_eval.data.helmet_narrativeqa_loader import (
    NARRATIVEQA_REPO,
    NARRATIVEQA_REVISION,
    load_narrativeqa_dataset,
)
from olmo_eval.data.helmet_tasks import HELMET_TASKS
from olmo_eval.data.ruler_loader import (
    RULER_DATA_REPO,
    RULER_DATA_REVISION,
    download_ruler_data,
    get_ruler_templates,
    load_ruler_dataset,
)
from olmo_eval.data.ruler_tasks import RULER_TASKS
from olmo_eval.evals.tasks.common.registry import register
from olmo_eval.evals.tasks.helmet_judge import (
    LONGQA_METRICS,
    SUMM_METRICS,
    HelmetLongQAJudgedTask,
    HelmetSummBookJudgedTask,
    HelmetSummJudgedTask,
)
from olmo_eval.evals.tasks.ruler import RulerTask, make_ruler_task_class


class HelmetTask(RulerTask):
    """Shared plumbing for HELMET-plus tasks.

    Builds on `RulerTask`, which supplies the per-variant config lookup, the
    instance cache, the completion request with its answer prefix, and raw-text
    answers. HELMET differs in where data comes from -- subclasses implement
    `_load_dataset` -- in always rendering prompts from its templates, in the
    scoring inputs it keeps in metadata, and in sending its chat-format categories
    as chat requests.
    """

    _tasks_registry: dict[str, dict[str, Any]] = HELMET_TASKS
    _name_prefix: str = "helmet_"

    def _load_dataset(self) -> dict[str, Any]:
        """Return the loader payload: `data` plus the HELMET prompt templates."""
        raise NotImplementedError

    def _load_data(self) -> None:
        if self._dataset is not None:
            return

        loaded = self._load_dataset()

        self._dataset = loaded["data"]
        self._templates = {
            "user": loaded["user_template"],
            "system": loaded["system_template"],
        }

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        if self._templates is None:
            raise RuntimeError("Templates not loaded. Call _load_data() first.")

        question = self._templates["user"].format(**doc)

        # With a chat template the answer prefix is left off: HELMET wraps the
        # user turn alone in the chat template and adds no partial assistant turn.
        # RULER's answer prefixes name the needle and query, so the system
        # template is filled from the row too; for every other task it has no
        # fields and formatting leaves it unchanged
        prepend_text = (
            ""
            if self.ruler_config.get("use_chat_template")
            else self._templates["system"].format(**doc)
        )

        answer = doc.get("answer")

        metadata: dict[str, Any] = {
            "id": index,
            "task_type": self.task_type,
            "context_size": self.context_size,
            "prepend_text": prepend_text,
            "tag": self.ruler_config["tag"],
        }
        if isinstance(answer, list):
            metadata["all_gold_answers"] = answer
        for scoring_field in ("qa_pairs", "qampari_answers"):
            if scoring_field in doc:
                # ALCE scoring inputs, kept out of the prompt
                metadata[scoring_field] = doc[scoring_field]
        for judge_field in ("keypoints", "expert_summary"):
            if judge_field in doc:
                # inputs the summarization judge needs; extracted ahead of time
                # and shipped with the data rather than derived at scoring time
                metadata[judge_field] = doc[judge_field]
        if "qrel" in doc:
            # graded relevance judgements for the re-ranking scorer
            metadata["qrel"] = doc["qrel"]
        if "question" in doc:
            # The bare question, kept separate from `Instance.question`, which is
            # the whole rendered prompt. An LLM judge needs this one -- handing it
            # the prompt would ship a book-length context to the judge.
            metadata["judge_question"] = doc["question"]

        return Instance(
            question=question,
            gold_answer=answer,
            metadata=metadata,
        )

    @property
    def request_type(self) -> RequestType:
        # HELMET wraps its chat-format categories (LongQA, summarization, cite) in
        # the model's chat template as a single user turn; the rest are completions
        if self.ruler_config.get("use_chat_template"):
            return RequestType.CHAT
        return RequestType.COMPLETION

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is None and self.request_type == RequestType.CHAT:
            return LMRequest(
                request_type=RequestType.CHAT,
                messages=({"role": "user", "content": instance.question},),
            )
        return super().format_request(instance)


class HelmetJsonKvTask(HelmetTask):
    """HELMET-plus json_kv task: extract a value for a given key from a long JSON blob."""

    def _load_dataset(self) -> dict[str, Any]:
        """Load the helmet-plus json_kv data for this task's length tier.

        HELMET-plus data is pre-generated at specific context lengths and
        published as JSONL files on the Hub, so it's downloaded and cached via
        huggingface_hub rather than the standard dataset pipeline.
        """
        return load_json_kv_dataset(
            length_name=self.ruler_config["length_name"],
            shots=self.ruler_config["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )


def _parse_labeled_output(text: str, prefix: str) -> str | None:
    """Pull the answer out of a `label: N`-style completion.

    Mirrors HELMET's `parse_output` (utils.py): prefer the text following the
    prefix, otherwise fall back to the first line, then strip a repeated
    prefix that chat-style models often echo back.
    """
    patterns = [
        re.compile(f"(?:{re.escape(prefix)})(.*)(?:\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*)(?:\n|$)"),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match is not None:
            return re.sub(
                f"^{re.escape(prefix)}", "", match[1].strip(), flags=re.IGNORECASE
            ).strip()
    return None


class HelmetIclTask(HelmetTask):
    """HELMET ICL task: label a text given many labelled demonstrations in context."""

    def _load_dataset(self) -> dict[str, Any]:
        return load_icl_dataset(
            icl_dataset=self.ruler_config["icl_dataset"],
            shots=self.ruler_config["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )

    def extract_answer(self, output: LMOutput) -> Any:
        return _parse_labeled_output(output.text, prefix="label:")


class HelmetInfbenchTask(HelmetTask):
    """HELMET LongQA task: answer a question about a book-length story."""

    def _load_dataset(self) -> dict[str, Any]:
        return load_infbench_dataset(
            subset=self.ruler_config["infbench_subset"],
            max_context_tokens=self.ruler_config["max_context_tokens"],
            shots=self.ruler_config["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )

    def extract_answer(self, output: LMOutput) -> Any:
        # HELMET scores the raw generation and the "Answer:"-stripped version,
        # keeping whichever is better; stripping here is the closer of the two
        # for a chat model that echoes the prefix, and a no-op otherwise.
        parsed = _parse_labeled_output(output.text, prefix="Answer:")
        return parsed if parsed else output.text


@dataclass(frozen=True, slots=True)
class HelmetExactMatchScorer(Scorer):
    """Exact match under HELMET's answer normalization.

    HELMET's `exact_match` (drqa_exact_match_score in utils.py) compares
    answers after lowercasing, stripping punctuation, dropping articles, and
    collapsing whitespace. The generic ExactMatchScorer only lowercases and
    strips, so a model answering `label: 42.` or `"42"` would score 0 here but
    1 under HELMET -- a real divergence on realistic ICL outputs, since the
    stop-at-newline sampling leaves trailing punctuation intact.
    """

    name: str = "exact_match"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.extracted_answer is None:
            return 0.0

        golds = (instance.metadata or {}).get("all_gold_answers")
        if not golds:
            gold = instance.gold_answer
            if gold is None:
                return 0.0
            golds = list(gold) if isinstance(gold, (list, tuple)) else [gold]

        prediction = _squad_normalize_answer(str(output.extracted_answer))
        return float(any(_squad_normalize_answer(str(g)) == prediction for g in golds))


@dataclass(frozen=True, slots=True)
class InfbenchChoiceScorer(Scorer):
    """Exact match for InfiniteBench multiple choice, following HELMET.

    HELMET accepts the answer in any of the forms a model realistically emits,
    so this counts a response correct when the raw generation, or its
    "Answer:"-stripped form, normalizes to either the bare letter ("B") or the
    letter with its option text ("B. Paris") -- or when the latter appears
    anywhere in the generation, which is what catches a verbose model that
    answers in a sentence.

    It reads `output.text` rather than only `extracted_answer` because that
    last substring rule is defined against the untouched generation.
    """

    name: str = "exact_match"

    def score(self, instance: Instance, output: LMOutput) -> float:
        metadata = instance.metadata or {}
        golds = metadata.get("all_gold_answers") or []
        if not golds:
            gold = instance.gold_answer
            if gold is None:
                return 0.0
            golds = list(gold) if isinstance(gold, (list, tuple)) else [gold]
        golds = [str(g) for g in golds]

        raw = output.text or ""
        candidates = [raw]
        parsed = _parse_labeled_output(raw, prefix="Answer:")
        if parsed:
            candidates.append(parsed)

        normalized_golds = {_squad_normalize_answer(g) for g in golds}
        if any(_squad_normalize_answer(c) in normalized_golds for c in candidates):
            return 1.0

        # golds[1] is the "letter. option text" form
        if len(golds) > 1 and golds[1].lower() in raw.lower():
            return 1.0

        return 0.0


class HelmetNarrativeQaTask(HelmetTask):
    """HELMET LongQA task: answer a question about a novel or movie script.

    Graded by an LLM judge (see `HelmetLongQAJudgedTask`), so running it needs
    a judge configured -- `OPENAI_API_KEY` for the default OpenAI judge.
    """

    def _load_dataset(self) -> dict[str, Any]:
        return load_narrativeqa_dataset(
            max_context_tokens=self.ruler_config["max_context_tokens"],
            shots=self.ruler_config["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )


class HelmetKiltTask(HelmetTask):
    """HELMET RAG task: answer an open-domain question from retrieved passages.

    The question cap is `max_questions`, not `config.limit`: every question
    appears once per gold-passage depth, and the runner applies `limit` to
    instances, which would keep some depths of a question and drop others.
    These tasks leave `limit` unset by default; setting one (e.g. for a smoke
    test) caps instances like any other task, at the cost of the depth sweep.
    """

    def _load_dataset(self) -> dict[str, Any]:
        return load_kilt_dataset(
            task=self.ruler_config["kilt_task"],
            length_name=self.ruler_config["length_name"],
            shots=self.ruler_config["shots"],
            max_samples=self.ruler_config["max_questions"],
            seed=self.config.seed,
            popularity_threshold=self.ruler_config.get("popularity_threshold"),
            max_prompt_tokens=self.ruler_config["max_prompt_tokens"],
        )

    def extract_answer(self, output: LMOutput) -> Any:
        # the prompt asks for "Answer: [answer]", and HELMET scores the parsed
        # form as well as the raw text, keeping whichever is better
        parsed = _parse_labeled_output(output.text, prefix="Answer:")
        return parsed if parsed else output.text


class HelmetMsMarcoTask(HelmetTask):
    """HELMET re-ranking task: order candidate passages by relevance to a query."""

    def _load_dataset(self) -> dict[str, Any]:
        return load_msmarco_dataset(
            length_name=self.ruler_config["length_name"],
            shots=self.ruler_config["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )

    def extract_answer(self, output: LMOutput) -> Any:
        # a ranked list of document ids, which NDCGScorer scores against qrel
        return parse_rankings(output.text or "")


class HelmetMultiLexSumTask(HelmetTask):
    """HELMET summarization task: summarize the filings of a civil rights lawsuit."""

    def _load_dataset(self) -> dict[str, Any]:
        return load_multi_lexsum_dataset(
            max_context_tokens=self.ruler_config["max_context_tokens"],
            shots=self.ruler_config["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )


class HelmetAlceTask(HelmetTask):
    """HELMET Cite task: answer a question and cite the documents used.

    Only ALCE's answer-correctness metrics are scored here. Whether the
    citations actually support the claims is AutoAIS's job, which needs an NLI
    model this task does not yet wire up.
    """

    def _load_dataset(self) -> dict[str, Any]:
        return load_alce_dataset(
            task=self.ruler_config["alce_task"],
            length_name=self.ruler_config["length_name"],
            shots=self.ruler_config["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )


class HelmetRulerTask(HelmetTask):
    """HELMET recall task: one of the RULER needle-in-a-haystack variants.

    Reads the same pre-generated RULER files as the `ruler_*` tasks, but builds
    the prompt the way HELMET does: from its templates and the row's fields,
    with the answer prefix on its own line. The files' prebuilt `input`, which
    the `ruler_*` tasks use, joins the two with a space instead.
    """

    def _load_dataset(self) -> dict[str, Any]:
        ruler_name = f"{self.ruler_config['ruler_task']}__{self.context_size}"
        return load_ruler_dataset(
            task_name=ruler_name,
            data_path=os.path.join(download_ruler_data(), RULER_TASKS[ruler_name]["data"]),
            max_samples=self.config.limit,
            seed=self.config.seed,
        )


_TASK_CLASSES: dict[str, type[HelmetTask]] = {
    "json_kv": HelmetJsonKvTask,
    "ruler": HelmetRulerTask,
    "icl": HelmetIclTask,
    "infbench": HelmetInfbenchTask,
    "narrativeqa": HelmetNarrativeQaTask,
    "kilt": HelmetKiltTask,
    "msmarco": HelmetMsMarcoTask,
    "multi_lexsum": HelmetMultiLexSumTask,
    "alce": HelmetAlceTask,
}

# Per-kind metric configuration. HELMET scores json_kv with substring exact
# match (RecallMetric's substring scorer is equivalent for a single gold
# answer) and ICL with exact match; see HELMET's scripts/collect_results.py.
_TASK_METRICS: dict[str, tuple] = {
    "json_kv": ((RecallMetric(),), "recall"),
    # HELMET's ruler_recall: the fraction of gold values found in the output
    "ruler": ((RecallMetric(),), "recall"),
    # HELMET-normalized exact match, not the generic scorer -- see
    # HelmetExactMatchScorer for why the difference is load-bearing
    "icl": ((AccuracyMetric(name="exact_match", scorer=HelmetExactMatchScorer),), "exact_match"),
    "infbench": ((RougeLF1Metric(),), "rougeL_f1"),
    "infbench_choice": (
        (AccuracyMetric(name="exact_match", scorer=InfbenchChoiceScorer),),
        "exact_match",
    ),
    # HELMET's headline re-ranking metric
    "msmarco": ((NDCGMetric(),), "ndcg_at_10"),
    # HELMET scores RAG with substring exact match over the answer aliases
    "kilt": (
        (AccuracyMetric(name="substring_exact_match", scorer=SubstringExactMatchScorer),),
        "substring_exact_match",
    ),
    # ALCE answer correctness. The citation-grounding half (citation_rec /
    # citation_prec) needs AutoAIS and is not wired up yet.
    "alce_asqa": ((AccuracyMetric(name="str_em", scorer=AlceStrEmScorer),), "str_em"),
    "alce_qampari": (
        (AccuracyMetric(name="qampari_rec_top5", scorer=AlceQampariRecTop5Scorer),),
        "qampari_rec_top5",
    ),
    # HELMET's gpt-4-f1: fluency-gated F1 over key points, via three judge calls
    "summ_book": (SUMM_METRICS, "gpt4_f1"),
    "summ_lawsuit": (SUMM_METRICS, "gpt4_f1"),
    # HELMET's gpt-4-score, normalized to [0, 1]; see HelmetLongQAJudgedTask
    "narrativeqa": (LONGQA_METRICS, "gpt4_score"),
}

# The judge each judged metric family grades with, mixed in ahead of the task kind.
_JUDGED_TASK_CLASSES: dict[str, type[HelmetLongQAJudgedTask | HelmetSummJudgedTask]] = {
    "summ_book": HelmetSummBookJudgedTask,
    "summ_lawsuit": HelmetSummJudgedTask,
    "narrativeqa": HelmetLongQAJudgedTask,
}


# The loader module behind each task kind, whose prompt templates feed the task hash.
_LOADER_MODULES = {
    "json_kv": helmet_loader,
    "icl": helmet_icl_loader,
    "infbench": helmet_infbench_loader,
    "narrativeqa": helmet_narrativeqa_loader,
    "kilt": helmet_kilt_loader,
    "msmarco": helmet_msmarco_loader,
    "multi_lexsum": helmet_multilexsum_loader,
    "alce": helmet_alce_loader,
}

# Kinds that truncate or trim with the reference tokenizer.
_USES_REFERENCE_TOKENIZER = {"infbench", "narrativeqa", "kilt", "multi_lexsum"}


def _task_settings(task_cfg: dict) -> dict[str, Any]:
    """Everything that shapes a task's prompts but has no TaskConfig field of its own.

    Serialized into the task hash, so a change to shots, token budgets, prompt
    templates or a pinned source revision gives the task a new hash.
    """
    kind = task_cfg["kind"]
    sources: dict[str, Any] = {}
    if kind == "icl":
        sources["icl"] = ICL_DATASETS[task_cfg["icl_dataset"]]
    elif kind == "ruler":
        sources["ruler_data"] = f"{RULER_DATA_REPO}@{RULER_DATA_REVISION}"
    elif kind == "narrativeqa":
        sources["narrativeqa"] = f"{NARRATIVEQA_REPO}@{NARRATIVEQA_REVISION}"
    else:
        sources["helmet_plus"] = f"{HELMET_PLUS_REPO_ID}@{HELMET_PLUS_REVISION}"
    if kind == "infbench":
        sources["infinitebench"] = f"{INFINITEBENCH_REPO}@{INFINITEBENCH_REVISION}"
        sources["infinitebench_subset"] = INFBENCH_SUBSETS[task_cfg["infbench_subset"]]
    if kind == "multi_lexsum":
        sources["multi_lexsum"] = f"{MULTI_LEXSUM_REPO}@{MULTI_LEXSUM_REVISION}"
        sources["min_source_tokens"] = MIN_SOURCE_TOKENS
    if kind in _USES_REFERENCE_TOKENIZER:
        sources["reference_tokenizer"] = f"{REFERENCE_TOKENIZER}@{REFERENCE_TOKENIZER_REVISION}"

    if kind == "ruler":
        user, system, _ = get_ruler_templates(task_cfg["ruler_task"])
        templates = {"user": user, "system": system}
    else:
        templates = {
            name: value
            for name, value in vars(_LOADER_MODULES[kind]).items()
            if "TEMPLATE" in name and isinstance(value, str)
        }
    return {"helmet": task_cfg, "sources": sources, "templates": templates}


# Upstream's vLLM and API backends stop on "\n" and "\n\n"; its HF backend stops on
# the token ids of the newline spellings in RULER_STOP_SEQUENCES. As stop strings,
# the newline alone covers them all.
HELMET_STOP_SEQUENCES = ("\n",)


def _make_helmet_task_class(task_name: str, task_cfg: dict) -> type[RulerTask]:
    """Create a task subclass for a HELMET-plus task variant.

    The kind's task class, behind its judge for the judged tasks, with HELMET's
    metrics, stop sequences and task settings.
    """
    metrics, primary_name = _TASK_METRICS[task_cfg["metrics_key"]]
    judged_cls = _JUDGED_TASK_CLASSES.get(task_cfg["metrics_key"])
    return make_ruler_task_class(
        task_name,
        task_cfg,
        base_cls=_TASK_CLASSES[task_cfg["kind"]],
        class_prefix="Helmet",
        limit=task_cfg["limit"],
        metrics=metrics,
        # Hand over the Metric itself, not its name: TaskConfig only resolves a
        # Metric instance, and an unresolved primary drops the task out of its
        # suite's average-of-averages.
        primary_metric=next(m for m in metrics if m.name == primary_name),
        stop_sequences=HELMET_STOP_SEQUENCES,
        extra_bases=() if judged_cls is None else (judged_cls,),
        attrs={"task_settings": _task_settings(task_cfg)},
    )


# Dynamically register all HELMET-plus tasks
for _task_name, _task_config in HELMET_TASKS.items():
    _cls = _make_helmet_task_class(_task_name, _task_config)
    # Inject into module globals so pickle can find the class by name
    globals()[_cls.__name__] = _cls
    register(f"helmet_{_task_name}")(_cls)
