"""HELMET InfiniteBench (LongQA) data loading.

Ports HELMET's `load_infbench` (https://github.com/princeton-nlp/HELMET,
data.py): a book-length story plus a question about it. Unlike the recall
tasks, length here comes from truncating a real document, so these tasks stop
at standard HELMET's 128k -- the books themselves set the ceiling.

Contexts are truncated with a *fixed reference tokenizer* (Llama 2, as
upstream) rather than the tokenizer of whichever model is being evaluated.
That is deliberate and load-bearing: it is what makes every model see exactly
the same text, and it keeps our numbers comparable to published HELMET
results. It does mean the realized length under Olmo 3 differs from nominal,
the same trade already taken for the ICL tasks -- see `helmet_tasks.py`.
"""

import logging
from typing import Any

from datasets import Features, Sequence, Value, load_dataset

logger = logging.getLogger(__name__)

# HELMET truncates against meta-llama/Llama-2-7b-hf; this is an ungated mirror
# of the same tokenizer, so the repo doesn't require gated-model access.
REFERENCE_TOKENIZER = "NousResearch/Llama-2-7b-hf"

# Appended to a context after cutting it, so the model can tell the document
# was truncated rather than simply ending. Verbatim from HELMET.
TRUNCATION_POSTFIX = " ... [the rest of the text is omitted]"

# HELMET passes explicit features when loading InfiniteBench to work around a
# dataset hashing bug that otherwise surfaces between online/offline runs.
_INFBENCH_FEATURES = Features(
    {
        "id": Value("int64"),
        "context": Value("string"),
        "input": Value("string"),
        "answer": Sequence(Value("string")),
        "options": Sequence(Value("string")),
    }
)

INFBENCH_SUBSETS: dict[str, dict[str, Any]] = {
    "qa_eng": {
        "split": "longbook_qa_eng",
        "user_template": (
            "You are given a story and a question. Answer the question as concisely as "
            "you can, using a single phrase if possible.\n\n{demo}{context}\n\n"
            "Question: {question}"
        ),
        "system_template": "Answer:",
        "demo_template": "[story text]\nQuestion: {question}\nAnswer: {answer}",
    },
}


def _load_reference_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def _truncate_context(context: str, max_tokens: int, tokenizer, separator_length: int) -> str:
    """Cut `context` to `max_tokens` reference-tokenizer tokens, HELMET-style.

    Mirrors HELMET's `truncate_llama2`: the budget is spent on the document
    itself, with room reserved for the truncation notice appended afterwards.
    """
    encoded = tokenizer(context, return_offsets_mapping=True)
    if len(encoded["input_ids"]) <= max_tokens:
        return context
    cut_at = encoded["offset_mapping"][max_tokens - separator_length][1]
    return context[:cut_at] + TRUNCATION_POSTFIX


def load_infbench_dataset(
    subset: str,
    max_context_tokens: int,
    shots: int = 2,
    max_samples: int | None = None,
    seed: int = 42,
    reference_tokenizer: str = REFERENCE_TOKENIZER,
) -> dict[str, Any]:
    """Load a HELMET InfiniteBench subset truncated to a target context length.

    Args:
        subset: Key into `INFBENCH_SUBSETS` (e.g. "qa_eng").
        max_context_tokens: Token budget for the story, measured with the
            reference tokenizer. HELMET reserves room for the prompt and the
            generation, so this is smaller than the task's nominal length.
        shots: Number of worked examples to prepend. The demos show only a
            `[story text]` placeholder rather than a real story, so they cost
            almost nothing against the budget -- as in HELMET.
        max_samples: Cap on the number of instances.
        seed: Seed for instance sampling and demo selection.
        reference_tokenizer: Tokenizer used for truncation. Changing this
            changes how much text every model sees, so it should stay fixed.

    Returns:
        Dictionary with `data` (processed records) and the HELMET prompt templates.
    """
    if subset not in INFBENCH_SUBSETS:
        raise ValueError(
            f"Unknown InfiniteBench subset '{subset}'. Available: {sorted(INFBENCH_SUBSETS)}"
        )

    spec = INFBENCH_SUBSETS[subset]
    data = load_dataset("xinrongzhang2022/infinitebench", features=_INFBENCH_FEATURES)[
        spec["split"]
    ]

    data = data.map(lambda example: {"question": example["input"], "demo": ""})

    # HELMET filters to contexts of >=65536 reference tokens before sampling.
    # Every InfiniteBench story clears that bar (the shortest is ~78k tokens),
    # which HELMET itself notes is "just a sanity step", so the filter is a
    # no-op and is skipped here -- letting us sample first and truncate only
    # the rows we keep, instead of tokenizing all 351 book-length contexts.
    demo_pool = data
    if max_samples is not None and len(data) > max_samples:
        data = data.shuffle(seed=seed).select(range(max_samples))

    if shots > 0:
        demo_template = spec["demo_template"]

        def add_demos(example: dict[str, Any]) -> dict[str, Any]:
            # drawn from the full pool, not the sampled subset, so demos don't
            # depend on how many instances are being evaluated
            demos = demo_pool.filter(lambda x: x["id"] != example["id"])
            demos = demos.shuffle(seed=seed).select(range(shots))
            demo_text = "\n\n".join(
                demo_template.format(question=d["question"], answer=d["answer"][0]) for d in demos
            )
            return {"demo": f"For example:\n\n{demo_text}\n\nNow, read the following story:\n\n"}

        data = data.map(add_demos)

    tokenizer = _load_reference_tokenizer(reference_tokenizer)
    separator_length = len(tokenizer(TRUNCATION_POSTFIX)["input_ids"])

    # InfiniteBench asks several questions about each book (351 rows over 69
    # stories), so truncation results are memoized per distinct context.
    truncation_cache: dict[str, str] = {}

    def truncate(example: dict[str, Any]) -> dict[str, Any]:
        context = example["context"]
        if context not in truncation_cache:
            truncation_cache[context] = _truncate_context(
                context, max_context_tokens, tokenizer, separator_length
            )
        return {"context": truncation_cache[context]}

    data = data.map(truncate)

    return {
        "data": data.to_list(),
        "prompt_template": spec["user_template"] + "\n\n" + spec["system_template"],
        "user_template": spec["user_template"],
        "system_template": spec["system_template"],
    }
