"""The DeepScholar-Bench scorer, pinned, so its contract is tested and not assumed.

Verbatim copy of the parser package from
github.com/guestrin-lab/deepscholar at commit
c95413b3b2f3255b461b90d0ce650f685ae2d1ff (``eval/parsers/``), by way of
``ai2-multi-agent/integrated/tests/fixtures/deepscholar_c95413b`` which
reformatted the imports without changing behaviour.

It is here because the export writes for this parser and nothing else: it
credits a citation only as a markdown link whose URL matches arxiv.org/abs, and
returns no documents for a query that has none. Asserting that against a
hand-written imitation would only prove the imitation self-consistent, so the
end-to-end test runs the real thing.

Do not edit. Re-copy from the pinned commit if the benchmark is ever re-pinned.
"""
