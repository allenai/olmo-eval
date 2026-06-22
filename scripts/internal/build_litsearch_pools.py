"""Build BM25 candidate pools for the fixed-corpus LitSearch reranking eval.

Run once on a machine with network + the corpus available; commit (or HF-host)
the output JSONL. This is the offline build for plan 005: it downloads
``princeton-nlp/LitSearch`` ``corpus_clean`` (64,183 papers) and ``query`` (597),
builds a BM25 index over title+abstract, retrieves the top candidates per query,
reports the BM25 retriever Recall@k baseline (the reproducible retriever
number), and writes a self-contained rerank pool per query for the
``litsearch_rerank`` task.

The retriever baseline is computed over the full retrieved ranking (top
``--baseline-depth``). The stored pool keeps the top ``--pool-size`` candidates
with title + truncated abstract baked in, so the eval needs no corpus download
at run time.

Usage:
    uv run python scripts/internal/build_litsearch_pools.py
        [--pool-size 50] [--abstract-chars 500] [--revision <hf-revision>]
        [--output <path>]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_litsearch_pools")

LITSEARCH_REPO = "princeton-nlp/LitSearch"
RECALL_KS = (5, 20)
BASELINE_DEPTH = 100

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "src/olmo_eval/evals/tasks/data/litsearch_rerank_pools.jsonl"
)


def recall_at_k(gold: list[int], ranked: list[int], k: int) -> float | None:
    """Fraction of gold IDs present in the top-k of a ranked ID list."""
    if not gold:
        return None
    return len(set(gold) & set(ranked[:k])) / len(gold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool-size",
        type=int,
        default=50,
        help="Candidates kept (with text) in each rerank pool.",
    )
    parser.add_argument(
        "--abstract-chars",
        type=int,
        default=500,
        help="Truncate stored abstracts to this many characters.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Pin a specific HF dataset revision for reproducibility.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        default=None,
        help="Upload the built artifact to this HF dataset repo (e.g. "
        "allenai/litsearch-rerank-pools). Requires HF write access.",
    )
    args = parser.parse_args()

    import bm25s
    from datasets import load_dataset

    logger.info("Loading corpus_clean (this is the large download)...")
    corpus = load_dataset(LITSEARCH_REPO, "corpus_clean", split="full", revision=args.revision)

    corpus_ids: list[int] = []
    titles: list[str] = []
    abstracts: list[str] = []
    texts: list[str] = []
    for row in corpus:
        title = (row.get("title") or "").strip()
        abstract = (row.get("abstract") or "").strip()
        corpus_ids.append(int(row["corpusid"]))
        titles.append(title)
        abstracts.append(abstract)
        texts.append(f"{title}\n\n{abstract}".strip())
    id_to_pos = {cid: i for i, cid in enumerate(corpus_ids)}
    logger.info("Corpus loaded: %d papers", len(corpus_ids))

    logger.info("Tokenizing and indexing with BM25...")
    corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=True)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=True)

    logger.info("Loading queries...")
    queries = load_dataset(LITSEARCH_REPO, "query", split="full", revision=args.revision)

    rows: list[dict] = []
    recalls: dict[int, list[float]] = {k: [] for k in RECALL_KS}
    pool_ceiling: dict[int, list[float]] = {k: [] for k in RECALL_KS}
    for idx, query in enumerate(queries):
        query_text = query.get("query") or ""
        gold = [int(c) for c in (query.get("corpusids") or []) if c is not None]
        if not query_text or not gold:
            continue

        query_tokens = bm25s.tokenize([query_text], stopwords="en", show_progress=False)
        results, _scores = retriever.retrieve(query_tokens, k=BASELINE_DEPTH, show_progress=False)
        ranked_ids = [corpus_ids[i] for i in results[0].tolist()]

        # Retriever baseline over the full retrieved ranking.
        for k in RECALL_KS:
            value = recall_at_k(gold, ranked_ids, k)
            if value is not None:
                recalls[k].append(value)

        pool_ids = ranked_ids[: args.pool_size]
        # Ceiling the reranker can reach: a perfect reranker promotes every gold
        # paper present in the pool into its top-k, so recall@k tops out at
        # min(#gold-in-pool, k) / #gold.
        gold_in_pool = len(set(gold) & set(pool_ids))
        for k in RECALL_KS:
            pool_ceiling[k].append(min(gold_in_pool, k) / len(gold))

        candidates = []
        for cid in pool_ids:
            pos = id_to_pos[cid]
            candidates.append(
                {
                    "corpusid": cid,
                    "title": titles[pos],
                    "abstract": abstracts[pos][: args.abstract_chars],
                }
            )

        rows.append(
            {
                "query_id": idx,
                "query": query_text,
                "gold_corpusids": gold,
                "candidates": candidates,
                "query_set": query.get("query_set", ""),
                "specificity": query.get("specificity"),
                "quality": query.get("quality"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    logger.info("Wrote %d pools to %s", len(rows), args.output)

    def report(label: str, table: dict[int, list[float]]) -> None:
        for k in RECALL_KS:
            values = table[k]
            mean = sum(values) / len(values) if values else 0.0
            logger.info("%s Recall@%d: %.4f (over %d queries)", label, k, mean, len(values))

    logger.info("--- BM25 retriever baseline (reproducible retriever number) ---")
    report("BM25", recalls)
    logger.info(
        "--- Rerank pool ceiling (best a perfect reranker over the top-%d pool can reach) ---",
        args.pool_size,
    )
    report("Pool", pool_ceiling)

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(args.output),
            path_in_repo=args.output.name,
            repo_id=args.push_to_hub,
            repo_type="dataset",
            commit_message=f"Update rerank pools ({len(rows)} queries, pool size {args.pool_size})",
        )
        logger.info(
            "Pushed %s to https://huggingface.co/datasets/%s",
            args.output.name,
            args.push_to_hub,
        )


if __name__ == "__main__":
    main()
