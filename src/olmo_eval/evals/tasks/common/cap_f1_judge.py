"""
GPT-judged caption F1 ("cap F1") — image-side, port of the protocol in
``mm_olmo/olmo/eval/vixmo_caption_utils.py``.

The metric has two pieces:

- **Recall**: For each gold atomic statement in the reference, ask the
  judge "Is this stated in the model's caption?"; recall is the fraction
  marked *Stated*.
- **Consistency** (precision proxy): Extract atomic statements from the
  model's caption with the judge, then for each one ask "Is this Stated
  in the gold transcripts?"; consistency is the fraction marked *Stated*.

The reported F1 is the harmonic mean. Both judge calls use the same
prompts as the Molmo internal eval so a number from this module is
directly comparable to the Molmo paper's "cap F1".

The judge is OpenAI's chat API. A disk cache keyed by
``(model, system_prompt, user_prompt)`` makes re-running idempotent — only
new prompts hit the network.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)


__all__ = [
    "CapF1Score",
    "CapF1Judge",
    "DEFAULT_JUDGE_MODEL",
]


#: Default judge model. Matches the current default in
#: ``mm_olmo/olmo/eval/vixmo_caption_utils.query_gpt``.
DEFAULT_JUDGE_MODEL = "gpt-4.1-2025-04-14"


# ---------------------------------------------------------------------------
# Prompts (copied verbatim from mm_olmo so the protocol matches)
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT_DEFAULT = "You are an AI assistant for question answering."
_SYSTEM_PROMPT_RECALL = "You are an AI assistant for evaluating caption recall."

_STATEMENT_EXTRACTION_PROMPT = (
    "Based on the description of the image, come up with a list of the MOST canonical "
    "statements that are mentioned in it.\n"
    "Each statement should be self-contained and broken down as much as possible.\n"
    "The statements should be an ordered list, where each item is separated a newline. "
    "For instance, the response may look like:\n\n"
    "1. Statement A\n2. Statement B\n3. Statement C\n\n"
    "Here is the image description:\n\n{caption}"
)


def _check_prompt(gold_statements_str: str, caption: str) -> str:
    return (
        "Here are statements about an image.\n\n"
        + gold_statements_str.strip()
        + "\n\n#####\n\n"
        + "Next, consider the following caption of the image. For each statement above, "
        'state whether the fact is "Stated" or "Not Stated" in the caption. The output '
        "should be in the form\n\n1. Stated\n2. Not Stated\n3. Stated\n\nDo not output "
        "anything other than an ordered list of Stated and Not Stated.\n\n Here is the "
        "caption:\n\n" + (caption.strip() if caption else "No caption provided.")
    )


# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------


@dataclass
class CapF1Score:
    """Per-example judged scores."""

    image_id: str
    recall: float
    consistency: float
    n_gold_statements: int
    n_pred_statements: int
    recall_judgements: List[Tuple[str, Optional[bool]]] = field(default_factory=list)
    consistency_judgements: List[Tuple[str, Optional[bool]]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def f1(self) -> float:
        if self.recall + self.consistency <= 0.0:
            return 0.0
        return 2 * self.recall * self.consistency / (self.recall + self.consistency)


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


class CapF1Judge:
    """OpenAI-backed caption F1 judge with disk caching."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_JUDGE_MODEL,
        cache_dir: Optional[str] = None,
        max_retries: int = 8,
        retry_sleep: float = 5.0,
        request_timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.model = model
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.request_timeout = request_timeout
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        # Lazy import so the module is importable without `openai` installed.
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, timeout=request_timeout)

    # ------------------------------------------------------------------
    # Low-level OpenAI call with cache
    # ------------------------------------------------------------------

    def _cache_key(self, system: str, user: str) -> str:
        h = hashlib.sha256()
        h.update(self.model.encode())
        h.update(b"\n##SYS##\n")
        h.update(system.encode())
        h.update(b"\n##USR##\n")
        h.update(user.encode())
        return h.hexdigest()

    def _query(self, system: str, user: str) -> str:
        if self.cache_dir:
            key = self._cache_key(system, user)
            path = os.path.join(self.cache_dir, key + ".json")
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        return json.load(f)["response"]
                except (OSError, json.JSONDecodeError, KeyError):
                    pass  # corrupted entry — re-query

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                )
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    if self.cache_dir:
                        with open(path, "w") as f:
                            json.dump({"response": text, "model": self.model}, f)
                    return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning(
                    "judge call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                time.sleep(self.retry_sleep)
        raise RuntimeError(f"judge call failed after {self.max_retries} attempts: {last_err}")

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def extract_statements(self, caption: str) -> List[str]:
        """Ask the judge to break ``caption`` into atomic statements."""
        if not caption.strip():
            return []
        user = _STATEMENT_EXTRACTION_PROMPT.format(caption=caption)
        raw = self._query(_SYSTEM_PROMPT_DEFAULT, user)
        return _parse_numbered_list(raw)

    def judge_stated(
        self, gold_statements: Sequence[str], caption: str
    ) -> List[Tuple[str, Optional[bool]]]:
        """For each gold statement, ask the judge if it's "Stated" or "Not
        Stated" in ``caption``. Returns ``[(statement, verdict_or_None)]``.

        ``None`` means the judge produced a line we couldn't parse as either
        Stated or Not Stated; that statement is excluded from the score.
        """
        if not gold_statements:
            return []
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(gold_statements))
        user = _check_prompt(numbered, caption)
        raw = self._query(_SYSTEM_PROMPT_RECALL, user)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        out: List[Tuple[str, Optional[bool]]] = []
        for i, stmt in enumerate(gold_statements):
            line = lines[i] if i < len(lines) else ""
            verdict = _parse_stated_line(line)
            out.append((stmt, verdict))
        return out

    # ------------------------------------------------------------------
    # Main entry point per example
    # ------------------------------------------------------------------

    def score_example(
        self,
        *,
        image_id: str,
        prediction: str,
        gold_statements: Sequence[str],
        gold_caption: str,
    ) -> CapF1Score:
        """Compute the (recall, consistency, F1) judgment for one example."""
        try:
            recall_judgements = self.judge_stated(gold_statements, prediction)
            pred_statements = self.extract_statements(prediction)
            consistency_judgements = self.judge_stated(pred_statements, gold_caption)
        except Exception as e:  # noqa: BLE001
            return CapF1Score(
                image_id=image_id,
                recall=0.0,
                consistency=0.0,
                n_gold_statements=len(gold_statements),
                n_pred_statements=0,
                error=str(e),
            )

        recall = _mean_valid([v for _, v in recall_judgements])
        consistency = _mean_valid([v for _, v in consistency_judgements])
        return CapF1Score(
            image_id=image_id,
            recall=recall,
            consistency=consistency,
            n_gold_statements=len(gold_statements),
            n_pred_statements=len(consistency_judgements),
            recall_judgements=recall_judgements,
            consistency_judgements=consistency_judgements,
        )

    # ------------------------------------------------------------------
    # Batch scoring (ThreadPool — IO-bound)
    # ------------------------------------------------------------------

    def score_batch(
        self,
        items: Sequence[Dict],
        *,
        n_threads: int = 16,
    ) -> List[CapF1Score]:
        """Score multiple examples in parallel.

        :param items: list of dicts with keys ``image_id``, ``prediction``,
            ``gold_statements``, ``gold_caption``.
        """
        scores: Dict[int, CapF1Score] = {}
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = {
                ex.submit(
                    self.score_example,
                    **{
                        k: it[k]
                        for k in (
                            "image_id",
                            "prediction",
                            "gold_statements",
                            "gold_caption",
                        )
                    },
                ): i
                for i, it in enumerate(items)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                scores[i] = fut.result()
        return [scores[i] for i in range(len(items))]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


_NUM_PREFIX_RE = re.compile(r"^\s*\d+[\.\)]\s*")


def _parse_numbered_list(raw: str) -> List[str]:
    """Strip leading ``"N. "`` / ``"N) "`` markers, drop empty lines."""
    out: List[str] = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        ln = _NUM_PREFIX_RE.sub("", ln)
        if ln:
            out.append(ln)
    return out


def _parse_stated_line(line: str) -> Optional[bool]:
    """Map a judge line to ``True``/``False``/``None``. Mirrors mm_olmo's
    regex: lines ending with "not st…" (handles typos like 'not staed') are
    ``False``; lines containing the word " stated" anywhere are ``True``;
    otherwise unparseable."""
    line_l = line.lower()
    if re.fullmatch(r".*\bnot st[a-z]+$", line_l):
        return False
    if " stated" in line_l:
        return True
    return None


def _mean_valid(values: Sequence[Optional[bool]]) -> float:
    valid = [v for v in values if v is not None]
    if not valid:
        return 0.0
    return sum(1 for v in valid if v) / len(valid)
