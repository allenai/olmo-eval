"""Scorers for ALCE (HELMET's Cite category).

Ports the metrics of ALCE's `eval_alce.py` as HELMET runs it. `str_em` for ASQA
and `qampari_rec_top5` for QAMPARI score *what* the model answered, against a
normalized generation: newlines collapsed to spaces, chat end-markers dropped,
and inline `[1][2]` citation markers stripped, so a well-cited answer is not
penalized for carrying its citations inline.

`AutoAisScorer` scores whether each claim is supported by the documents it
cites, with the T5 NLI model AutoAIS uses, served by an auxiliary provider. It
gives `citation_rec` and `citation_prec`.

Scores are proportions in [0, 1]; ALCE reports the same numbers scaled to
0-100, so multiply by 100 to compare against published figures.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.scorers.base import Scorer, _squad_normalize_answer
from olmo_eval.common.scorers.execution import ContextScorer, ScoringIncompleteError
from olmo_eval.common.scoring_time import measure_scoring
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams


def remove_citations(text: str) -> str:
    """Strip inline `[1]`-style citation markers. Verbatim from ALCE's utils."""
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", text)).replace(" |", "").replace("]", "")


def normalize_generation(text: str) -> str:
    """Apply ALCE's pre-scoring cleanup to a generation.

    Mirrors the preprocessing in `eval_alce.py`'s main(): collapse newlines,
    drop the chat end-marker, then remove citations.
    """
    text = re.sub(r"\n+", " ", text or "")
    text = text.replace("<|im_end|>", "")
    return remove_citations(text)


def exact_presence(short_answers: list[str], context: str) -> bool:
    """True if any acceptable short answer appears in the context.

    Port of ALCE's `exact_presence`; the normalization matches HELMET's
    `normalize_answer`.
    """
    normalized_context = _squad_normalize_answer(context)
    return any(_squad_normalize_answer(sa) in normalized_context for sa in short_answers)


@dataclass(frozen=True, slots=True)
class AlceStrEmScorer(Scorer):
    """ALCE's STR-EM: the fraction of an ASQA question's sub-answers covered.

    ASQA questions are ambiguous and have several disambiguated sub-questions,
    each with its own acceptable short answers. A response scores the fraction
    of those sub-questions it answers, so partial coverage gets partial credit
    rather than being all-or-nothing.
    """

    name: str = "str_em"

    def score(self, instance: Instance, output: LMOutput) -> float:
        qa_pairs: list[dict[str, Any]] = (instance.metadata or {}).get("qa_pairs") or []
        if not qa_pairs:
            return 0.0

        generation = normalize_generation(output.text or "")
        hits = sum(
            1.0 for pair in qa_pairs if exact_presence(pair.get("short_answers") or [], generation)
        )
        return hits / len(qa_pairs)


@dataclass(frozen=True, slots=True)
class AlceQampariRecTop5Scorer(Scorer):
    """ALCE's QAMPARI recall@top-5.

    QAMPARI questions have many valid answers, so the model emits a
    comma-separated list. Recall is capped at five: a question with twenty gold
    answers is satisfied by finding five, which keeps questions with long answer
    sets from dominating the average.
    """

    name: str = "qampari_rec_top5"

    def score(self, instance: Instance, output: LMOutput) -> float:
        answers: list[list[str]] = (instance.metadata or {}).get("qampari_answers") or []
        if not answers:
            return 0.0

        generation = normalize_generation(output.text or "")
        # the model answers as a comma-separated list; trailing punctuation is
        # stripped before splitting, as upstream does
        predictions = [
            _squad_normalize_answer(part.strip())
            for part in generation.rstrip().rstrip(".").rstrip(",").split(",")
        ]
        predictions = [p for p in predictions if p]

        normalized_answers = [[_squad_normalize_answer(a) for a in alias] for alias in answers]
        found = sum(
            1 for alias_group in normalized_answers if any(a in predictions for a in alias_group)
        )
        return min(5, found) / min(5, len(normalized_answers))


#: The NLI model AutoAIS checks citations with (``AUTOAIS_MODEL`` in eval_alce.py).
AUTOAIS_MODEL = "google/t5_xxl_true_nli_mixture"
#: Its Hub commit, unchanged since before HELMET's release.
AUTOAIS_REVISION = "aa6cfe1dd4257853bfdd772992045f41bfc14988"
#: The auxiliary provider a task names to serve ``AUTOAIS_MODEL``.
AUTOAIS_PROVIDER = "autoais"
#: Upstream's ``--at_most_citations`` default: later citations of a sentence are ignored.
AT_MOST_CITATIONS = 3
#: ``LMOutput.metadata`` key holding an output's citation scores and NLI calls.
AUTOAIS_KEY = "autoais"

# the T5 model answers "1" for entailment; upstream allows it a few tokens
_NLI_PARAMS = SamplingParams(temperature=0.0, max_tokens=10)


def citation_sentences(text: str, question: str, *, qampari: bool) -> list[str]:
    """Split a generation into the claims AutoAIS checks, as upstream does.

    The generation gets ALCE's newline and end-marker cleanup first, but keeps its
    citations. ASQA is split into sentences with NLTK, dropping fragments under five
    characters (such as a stray "1."); QAMPARI is a comma-separated list, and each
    item is checked as the question followed by the item.
    """
    text = re.sub(r"\n+", " ", text or "").replace("<|im_end|>", "")
    if qampari:
        items = text.rstrip().rstrip(".").rstrip(",").split(",")
        return [question + " " + item.strip() for item in items]
    return [s for s in _sent_tokenize(text) if len(s.strip()) >= 5]


def _sent_tokenize(text: str) -> list[str]:
    """NLTK's sentence splitter, fetching its Punkt model the first time it is missing."""
    import nltk

    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        return nltk.sent_tokenize(text)


def _cited_ids(sentence: str) -> list[int]:
    """Zero-based ids of the documents a sentence cites, in citation order."""
    return [int(ref[1:]) - 1 for ref in re.findall(r"\[\d+", sentence)]


def _passage(docs: Sequence[dict[str, str]], ids: Sequence[int]) -> str:
    return "\n".join(f"Title: {docs[i]['title']}\n{docs[i]['text']}" for i in ids)


@dataclass(frozen=True, slots=True)
class AutoAisScorer(ContextScorer):
    """ALCE's AutoAIS: whether each cited claim is entailed by its citations.

    Port of `compute_autoais` in HELMET's eval_alce.py. Each claim's citations,
    up to three, are joined into a premise, and the NLI model judges whether it
    entails the claim with its citations removed.

    - ``citation_rec``: the share of an output's claims whose citations entail
      them. A claim that cites nothing, or a document the prompt did not show,
      is not entailed.
    - ``citation_prec``: the share of an output's citations that are needed. A
      citation of an entailed claim with one citation counts; of a claim with
      several, it counts unless it neither entails the claim alone nor is
      missed when left out.

    The score is ``citation_rec``. Both values, and every NLI call, are recorded
    in ``metadata["autoais"]`` for `CitationMetric` to read. An output with no
    claims has no citation scores, and upstream leaves it out of both averages.
    """

    name: str = "autoais"
    provider_name: str = AUTOAIS_PROVIDER
    qampari: bool = False
    at_most_citations: int = AT_MOST_CITATIONS

    async def ascore_with_context(
        self, instance: Instance, output: LMOutput, context: ScoringContext
    ) -> float:
        metadata = instance.metadata or {}
        docs = metadata.get("citation_docs")
        if docs is None:
            raise ValueError("AutoAIS needs the documents the prompt showed in citation_docs")
        sentences = citation_sentences(
            output.text or "", metadata.get("judge_question", ""), qampari=self.qampari
        )
        calls: list[dict[str, Any]] = []
        result: dict[str, Any] = {"claims": len(sentences), "nli_calls": calls}
        output.metadata[AUTOAIS_KEY] = result
        if not sentences:
            return 0.0

        async def entails(ids: Sequence[int], claim: str) -> int:
            verdict = await self._entails(_passage(docs, ids), claim, context)
            calls.append({"claim": claim, "cited": [i + 1 for i in ids], "entails": verdict})
            return verdict

        entailed = needed = total_citations = 0
        for sentence in sentences:
            claim = remove_citations(sentence).strip()
            ids = _cited_ids(sentence)
            if not ids or any(i >= len(docs) for i in ids):
                continue
            ids = ids[: self.at_most_citations]
            total_citations += len(ids)
            joint = await entails(ids, claim)
            entailed += joint
            if not (joint and len(ids) > 1):
                needed += joint
                continue
            for doc_id in ids:
                if await entails([doc_id], claim) or not await entails(
                    [i for i in ids if i != doc_id], claim
                ):
                    needed += 1

        result["citation_rec"] = entailed / len(sentences)
        result["citation_prec"] = needed / total_citations if total_citations else 0.0
        return result["citation_rec"]

    async def _entails(self, passage: str, claim: str, context: ScoringContext) -> int:
        """Ask the NLI model whether ``passage`` entails ``claim``: 1 if it says "1"."""
        request = LMRequest(
            request_type=RequestType.COMPLETION, prompt=f"premise: {passage} hypothesis: {claim}"
        )
        from olmo_eval.inference.errors import first_output

        try:
            provider = context.get_provider(self.provider_name)
            with measure_scoring("judge"):
                results = await provider.agenerate([request], _NLI_PARAMS)
            reply = first_output(results[0] if results else []).text
        except Exception as exc:
            raise ScoringIncompleteError(
                f"{self.name} provider {self.provider_name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
        return 1 if reply == "1" else 0
