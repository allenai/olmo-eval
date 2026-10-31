"""HELMET tasks: all seven categories at HELMET's standard context lengths.

HELMET (https://github.com/princeton-nlp/HELMET) evaluates seven task
categories at 4k-128k tokens of context: recall, RAG, re-ranking, LongQA,
summarization, ICL, and citation (ALCE). All seven are registered here.

Pre-generated and pre-retrieved data lives on the ai2-internal
`allenai/helmet-plus` Hub dataset; the remaining tasks pull their sources
from the Hub at load time. `narrativeqa` and both summarization tasks are
graded by an LLM judge (see `helmet_judge.py`) and therefore need a judge
configured -- the `helmet_nojudge__*` suites exclude them.

The Cite tasks check citations with AutoAIS's T5 NLI model, served on a GPU of
its own by the `helmet` harness preset, so runs that include them launch with
`-H helmet`. `helmet_alce_*:no_autoais` scores answer correctness alone for runs
without that GPU; those scores are not HELMET's Cite scores.

Upstream reference: princeton-nlp/HELMET at commit aeadacc6.
"""

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from olmo_eval.common.metrics import (
    AccuracyMetric,
    Metric,
    NDCGMetric,
    RecallMetric,
    RougeLF1Metric,
)
from olmo_eval.common.scorers import Scorer
from olmo_eval.common.scorers.alce import (
    AUTOAIS_KEY,
    AUTOAIS_MODEL,
    AlceQampariRecTop5Scorer,
    AlceStrEmScorer,
    AutoAisScorer,
)
from olmo_eval.common.scorers.base import _squad_normalize_answer
from olmo_eval.common.scorers.substring import SubstringExactMatchScorer
from olmo_eval.common.types import (
    CONTEXT_CUT_TOKENS_KEY,
    ContextFit,
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
)
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
    ANSWER_PREFIX_SEPARATOR,
    HELMET_PLUS_REPO_ID,
    HELMET_PLUS_REVISION,
    load_json_kv_dataset,
)
from olmo_eval.data.helmet_msmarco_loader import load_msmarco_dataset, parse_rankings
from olmo_eval.data.helmet_multilexsum_loader import (
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
from olmo_eval.evals.tasks.common.base import TaskConfig
from olmo_eval.evals.tasks.common.registry import register, register_variant
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

    The `:base` variant turns `use_chat_template` off, as upstream's
    `--use_chat_template False` does for models without instruction tuning: the
    chat-format categories then go out as completions ending in their answer
    prefix, and the prefix is put back on each output before it is scored.

    Like upstream's `tokenize`, every request also carries its document context as
    a `ContextFit` with a budget of the tier length less the generation budget, so
    a provider that supports it cuts the end of the context, measured with the
    evaluated model's own tokenizer, until the prompt fits; the reference-tokenizer
    truncation in the loaders alone lets a model whose tokenizer splits text more
    finely overflow its window.
    """

    _tasks_registry: dict[str, dict[str, Any]] = HELMET_TASKS
    _name_prefix: str = "helmet_"
    #: The task's default settings (see `_task_settings`), set on each task class.
    task_settings: dict[str, Any]

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self.settings = self._measured_settings(config)
        constants = type(self).task_settings["constants"]
        self.answer_prefix_separator = constants["answer_prefix_separator"]

    def _measured_settings(self, config: TaskConfig) -> dict[str, Any]:
        """Return the HELMET settings this run uses: `config.task_settings["helmet"]`.

        The task settings are the task hash's record of the run, so the run reads
        them rather than HELMET_TASKS. Their other parts (the upstream commit,
        sources and prompt constants) are fixed by the code, and an override that
        changes them is refused, as is one of the TaskConfig fields HELMET has no
        use for, since either would change the hash and not the run.
        """
        default = getattr(type(self), "task_settings", None)
        settings = config.task_settings
        if settings is None or default is None:
            raise ValueError(f"{config.name} has no task_settings")
        fixed = {key: value for key, value in default.items() if key != "helmet"}
        given = {key: value for key, value in settings.items() if key != "helmet"}
        if _canonical(given) != _canonical(fixed):
            raise ValueError(
                f"{config.name}: only task_settings['helmet'] can be overridden; "
                f"{sorted(fixed)} are set by the code"
            )
        helmet = settings.get("helmet")
        if not isinstance(helmet, dict) or set(helmet) != set(default["helmet"]):
            raise ValueError(
                f"{config.name}: task_settings['helmet'] must set exactly "
                f"{sorted(default['helmet'])}"
            )
        for name in ("num_fewshot", "fewshot_seed", "split"):
            if getattr(config, name) != TaskConfig.__dataclass_fields__[name].default:
                raise ValueError(
                    f"{config.name} takes its demonstrations and data from HELMET's own "
                    f"files and task_settings, so {name} cannot be overridden"
                )
        return helmet

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
            if self.settings.get("use_chat_template")
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
        for scoring_field in ("qa_pairs", "qampari_answers", "citation_docs"):
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
        context = doc.get("context")
        if isinstance(context, str) and context and question.count(context) == 1:
            # where the document sits in the prompt, so the request can name it
            # as the text to cut when the prompt is too long for the model
            start = question.index(context)
            metadata["context_span"] = [start, start + len(context)]

        return Instance(
            question=question,
            gold_answer=answer,
            metadata=metadata,
        )

    @property
    def request_type(self) -> RequestType:
        # HELMET wraps its chat-format categories (LongQA, summarization, cite) in
        # the model's chat template as a single user turn; the rest are completions
        if self.settings.get("use_chat_template"):
            return RequestType.CHAT
        return RequestType.COMPLETION

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            return super().format_request(instance)
        if self.request_type == RequestType.CHAT:
            request = LMRequest(
                request_type=RequestType.CHAT,
                messages=({"role": "user", "content": instance.question},),
            )
        else:
            request = super().format_request(instance)
        span = (instance.metadata or {}).get("context_span")
        if not span:
            return request
        # Upstream's tier budget is the whole sequence, generation included; with
        # CONTEXT_CUT "model_window" the provider cuts only to fit the model's window.
        max_tokens = self.config.sampling_params.max_tokens or 0
        start, end = span
        return replace(
            request,
            context_fit=ContextFit(
                text=instance.question[start:end],
                max_prompt_tokens=(
                    self.context_size - max_tokens if CONTEXT_CUT == "tier" else None
                ),
            ),
        )

    @property
    def prefixes_outputs(self) -> bool:
        """Whether outputs are scored with the answer prefix put back in front.

        Upstream prepends the answer prefix to every completion before parsing it.
        For the chat-format categories run as completions, the `:base` variant,
        that decides what the parsers and judges see: LongQA's parser keys on the
        leading "Answer:", and the summarization judge reads "Summary: ...". The
        completion-format categories keep scoring the bare output, which their
        parsers and substring scorers were validated on.
        """
        default = type(self).task_settings["helmet"]
        return bool(default.get("use_chat_template")) and not self.settings.get("use_chat_template")

    def strip_thinking_traces(self, responses: Sequence[Response]) -> None:
        """Strip thinking traces, then put the answer prefix back where upstream does.

        Runners call this on every response before scoring, so the prefix reaches
        `extract_answer`, the scorers that read `output.text`, and the judges. The
        generation itself stays in `metadata["original_text"]`. Repeated calls
        leave already prefixed outputs unchanged.
        """
        super().strip_thinking_traces(responses)
        if not self.prefixes_outputs:
            return
        for response in responses:
            prefix = (response.instance.metadata or {}).get("prepend_text", "")
            for output in response.outputs:
                if not prefix or output.metadata.get("answer_prefixed"):
                    continue
                output.metadata.setdefault("original_text", output.text)
                output.text = prefix + (output.text or "")
                output.metadata["answer_prefixed"] = True


class HelmetJsonKvTask(HelmetTask):
    """HELMET-plus json_kv task: extract a value for a given key from a long JSON blob."""

    def _load_dataset(self) -> dict[str, Any]:
        """Load the helmet-plus json_kv data for this task's length tier.

        HELMET-plus data is pre-generated at specific context lengths and
        published as JSONL files on the Hub, so it's downloaded and cached via
        huggingface_hub rather than the standard dataset pipeline.
        """
        return load_json_kv_dataset(
            length_name=self.settings["length_name"],
            shots=self.settings["shots"],
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
            icl_dataset=self.settings["icl_dataset"],
            shots=self.settings["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )

    def extract_answer(self, output: LMOutput) -> Any:
        return _parse_labeled_output(output.text, prefix="label:")


class HelmetInfbenchTask(HelmetTask):
    """HELMET LongQA task: answer a question about a book-length story."""

    def _load_dataset(self) -> dict[str, Any]:
        return load_infbench_dataset(
            subset=self.settings["infbench_subset"],
            max_context_tokens=self.settings["max_context_tokens"],
            shots=self.settings["shots"],
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
class ContextCutScorer(Scorer):
    """1.0 when the provider cut the instance's context to fit the model, else 0.0."""

    name: str = "context_cut"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return 1.0 if output.metadata.get(CONTEXT_CUT_TOKENS_KEY) else 0.0


@dataclass(frozen=True, slots=True)
class ContextTruncationRateMetric(Metric):
    """Share of instances whose context was cut to fit the evaluated model.

    Read from the outputs rather than a score channel, so the judged tasks, which
    skip the scorers, report it too. Lower is better: a cut instance was answered
    from less of its document than the tier promises.
    """

    name: str = "context_truncation_rate"
    scorer: type[Scorer] | Scorer = ContextCutScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(self.compute_instance(response) for response in responses) / len(responses)

    def compute_instance(self, response: Response) -> float:
        cut = any(output.metadata.get(CONTEXT_CUT_TOKENS_KEY) for output in response.outputs)
        return 1.0 if cut else 0.0

    def higher_is_better(self) -> bool:
        return False

    def display_format(self) -> str:
        return "percentage"


CONTEXT_TRUNCATION_RATE = ContextTruncationRateMetric()


@dataclass(frozen=True, slots=True)
class CitationMetric(Metric):
    """ALCE's ``citation_rec`` or ``citation_prec``, read from each output's AutoAIS result.

    Averaged over the instances that made a claim: upstream skips an output with
    no claims when it averages both.
    """

    name: str = "citation_rec"
    scorer: type[Scorer] | Scorer = AutoAisScorer

    def compute(self, responses: Sequence[Response]) -> float:
        values = [v for r in responses if (v := self.compute_instance(r)) is not None]
        return sum(values) / len(values) if values else 0.0

    def compute_instance(self, response: Response) -> float | None:
        if not response.outputs:
            return None
        return response.outputs[0].metadata.get(AUTOAIS_KEY, {}).get(self.name)

    def supports_scorer_fallback(self) -> bool:
        return False

    def display_format(self) -> str:
        return "percentage"


@dataclass(frozen=True, slots=True)
class CiteScoreMetric(Metric):
    """The mean of a Cite task's answer, citation-recall and citation-precision metrics.

    HELMET's Cite score averages these three for ASQA and for QAMPARI, so the
    category suite's mean of the two tasks' scores is HELMET's six-number mean.
    """

    name: str = "cite_score"
    scorer: type[Scorer] | Scorer = AutoAisScorer
    parts: tuple[Metric, ...] = ()

    def compute(self, responses: Sequence[Response]) -> float:
        return sum(part.compute(responses) for part in self.parts) / len(self.parts)

    def compute_instance(self, response: Response) -> float | None:
        values = [v for part in self.parts if (v := part.compute_instance(response)) is not None]
        return sum(values) / len(values) if values else None

    def supports_scorer_fallback(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {**Metric.to_dict(self), "parts": [part.to_dict() for part in self.parts]}

    def display_format(self) -> str:
        return "percentage"


def _cite_metrics(answer: Metric, *, qampari: bool) -> tuple[tuple[Metric, ...], str]:
    """A Cite task's metrics: answer correctness, AutoAIS, and their mean as primary."""
    autoais = AutoAisScorer(qampari=qampari)
    recall = CitationMetric(name="citation_rec", scorer=autoais)
    precision = CitationMetric(name="citation_prec", scorer=autoais)
    return (answer, recall, precision, CiteScoreMetric(parts=(answer, recall, precision))), (
        "cite_score"
    )


STR_EM = AccuracyMetric(name="str_em", scorer=AlceStrEmScorer)
QAMPARI_REC_TOP5 = AccuracyMetric(name="qampari_rec_top5", scorer=AlceQampariRecTop5Scorer)


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
            max_context_tokens=self.settings["max_context_tokens"],
            shots=self.settings["shots"],
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
            task=self.settings["kilt_task"],
            length_name=self.settings["length_name"],
            shots=self.settings["shots"],
            max_samples=self.settings["max_questions"],
            seed=self.config.seed,
            popularity_threshold=self.settings.get("popularity_threshold"),
            max_prompt_tokens=self.settings["max_prompt_tokens"],
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
            length_name=self.settings["length_name"],
            shots=self.settings["shots"],
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
            max_context_tokens=self.settings["max_context_tokens"],
            shots=self.settings["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )


class HelmetAlceTask(HelmetTask):
    """HELMET Cite task: answer a question and cite the documents used.

    Scored as HELMET scores it: ALCE's answer correctness, and AutoAIS's
    citation recall and precision, judged by the T5 NLI model an auxiliary
    provider named "autoais" serves (the `helmet` harness preset sets one up).
    The nocite ablations, and the `:no_autoais` variant for runs without that
    model, score answer correctness alone.
    """

    def _load_dataset(self) -> dict[str, Any]:
        return load_alce_dataset(
            task=self.settings["alce_task"],
            length_name=self.settings["length_name"],
            shots=self.settings["shots"],
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
        ruler_name = f"{self.settings['ruler_task']}__{self.context_size}"
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
    # ALCE as HELMET's Cite category scores it: answer correctness and AutoAIS
    "alce_asqa_cite": _cite_metrics(STR_EM, qampari=False),
    "alce_qampari_cite": _cite_metrics(QAMPARI_REC_TOP5, qampari=True),
    # answer correctness alone, for the nocite ablations, which upstream scores
    # without --citations, and for the :no_autoais variant
    "alce_asqa": ((STR_EM,), "str_em"),
    "alce_qampari": ((QAMPARI_REC_TOP5,), "qampari_rec_top5"),
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


# The loader module behind each task kind, whose MEASURED_CONSTANTS feed the task hash.
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

#: The HELMET_TASKS keys that change what a task measures, and the only ones its
#: run reads from the task settings. The rest pick the task class (`kind`,
#: `metrics_key`), are suite flags (`tag`, `judged`, `ablation`), or are already
#: TaskConfig fields (`limit`, and `max_gen_toks` and `stop_new_line` through the
#: sampling parameters), so hashing them here would move every hash when a suite
#: changes, or record a value an override ignores.
MEASURED_KEYS = (
    "length_name",
    "shots",
    "use_chat_template",
    "max_context_tokens",
    "max_prompt_tokens",
    "max_questions",
    "popularity_threshold",
    "ruler_task",
    "kilt_task",
    "icl_dataset",
    "infbench_subset",
    "alce_task",
)

#: The reference implementation these tasks port.
HELMET_UPSTREAM = "princeton-nlp/HELMET@aeadacc6"

#: How far a provider cuts a prompt's context. HELMET sizes each tier's text in
#: Llama-2 tokens. Upstream then cuts it to the tier again in the evaluated model's
#: own tokens ("tier"), which drops the end of the text, a needle or gold passage
#: included, for any tokenizer that splits it into more tokens than Llama-2's, even
#: when the model has room for it. "model_window" cuts only as far as the model's
#: context window requires, so every model with room reads the same text. Part of
#: the task hash.
CONTEXT_CUT = "model_window"


def _canonical(value: Any) -> str:
    """A comparable form of task settings, which may have come through JSON."""
    return json.dumps(value, sort_keys=True)


def _task_settings(task_cfg: dict) -> dict[str, Any]:
    """Everything that shapes a task's data and prompts but has no TaskConfig field.

    Serialized into the task hash, and the only place a task's run reads these
    settings from (see `HelmetTask`), so a change to shots, token budgets, prompt
    text or a pinned source revision both changes the run and gives the task a
    new hash.
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
    if kind in _USES_REFERENCE_TOKENIZER:
        sources["reference_tokenizer"] = f"{REFERENCE_TOKENIZER}@{REFERENCE_TOKENIZER_REVISION}"

    if kind == "ruler":
        user, system, _ = get_ruler_templates(task_cfg["ruler_task"])
        constants: dict[str, Any] = {"user_template": user, "system_template": system}
    else:
        constants = dict(_LOADER_MODULES[kind].MEASURED_CONSTANTS)
    constants.setdefault("answer_prefix_separator", ANSWER_PREFIX_SEPARATOR)

    return {
        "upstream": HELMET_UPSTREAM,
        "context_cut": CONTEXT_CUT,
        "helmet": {key: task_cfg[key] for key in MEASURED_KEYS if key in task_cfg},
        "sources": sources,
        "constants": constants,
    }


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
        metrics=(*metrics, CONTEXT_TRUNCATION_RATE),
        # Hand over the Metric itself, not its name: TaskConfig only resolves a
        # Metric instance, and an unresolved primary drops the task out of its
        # suite's average-of-averages.
        primary_metric=next(m for m in metrics if m.name == primary_name),
        stop_sequences=HELMET_STOP_SEQUENCES,
        extra_bases=() if judged_cls is None else (judged_cls,),
        attrs=_task_attrs(task_cfg),
    )


#: The answer-only metrics key of each Cite metrics key, for `:no_autoais`.
_NO_AUTOAIS_METRICS = {"alce_asqa_cite": "alce_asqa", "alce_qampari_cite": "alce_qampari"}


def _task_attrs(task_cfg: dict) -> dict[str, Any]:
    """Task configuration the factory sets on each class besides its metrics."""
    attrs: dict[str, Any] = {"task_settings": _task_settings(task_cfg)}
    if task_cfg["metrics_key"] in _NO_AUTOAIS_METRICS:
        # the model the "autoais" provider must serve, checked before generation
        attrs["judge_model"] = AUTOAIS_MODEL
    return attrs


def _no_autoais_overrides(metrics_key: str) -> dict[str, Any]:
    """The `:no_autoais` variant: a Cite task scored on answer correctness alone.

    For runs without a GPU for the NLI model. Its scores are not HELMET's Cite
    scores, so it is a variant and stays out of the category suites.
    """
    metrics, primary_name = _TASK_METRICS[_NO_AUTOAIS_METRICS[metrics_key]]
    return {
        "metrics": (*metrics, CONTEXT_TRUNCATION_RATE),
        "primary_metric": next(m for m in metrics if m.name == primary_name),
        "judge_model": None,
    }


def _base_task_settings(task_settings: dict[str, Any]) -> dict[str, Any]:
    """A task's settings with the chat template off, for models without instruction tuning.

    The completion-format categories already run without one, so their `:base`
    variant is the task itself and shares its hash.
    """
    return {**task_settings, "helmet": {**task_settings["helmet"], "use_chat_template": False}}


# Dynamically register all HELMET-plus tasks
for _task_name, _task_config in HELMET_TASKS.items():
    _cls = _make_helmet_task_class(_task_name, _task_config)
    # Inject into module globals so pickle can find the class by name
    globals()[_cls.__name__] = _cls
    register(f"helmet_{_task_name}")(_cls)
    if _task_config.get("use_chat_template"):
        register_variant(
            f"helmet_{_task_name}",
            "base",
            task_settings=_base_task_settings(_task_settings(_task_config)),
        )
    else:
        register_variant(f"helmet_{_task_name}", "base")
    if _task_config["metrics_key"] in _NO_AUTOAIS_METRICS:
        register_variant(
            f"helmet_{_task_name}",
            "no_autoais",
            **_no_autoais_overrides(_task_config["metrics_key"]),
        )
