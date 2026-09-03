"""Tests for the DeepResearch Bench RACE + FACT task."""

import builtins
import hashlib
import json
import re

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks import deepresearch_bench
from olmo_eval.evals.tasks.common import get_task, task_exists
from olmo_eval.evals.tasks.deepresearch_bench import (
    DEEPRESEARCH_DIMENSIONS,
    DEEPRESEARCH_GENERATION_INSTRUCTIONS,
    FACT_DEDUPLICATION_PROMPT_EN,
    FACT_DEDUPLICATION_PROMPT_ZH,
    FACT_EXTRACTION_PROMPT_EN,
    FACT_EXTRACTION_PROMPT_ZH,
    FACT_VALIDATION_PROMPT_EN,
    FACT_VALIDATION_PROMPT_ZH,
    RACE_SCORE_PROMPT_EN,
    RACE_SCORE_PROMPT_ZH,
    build_deepresearch_fact_judge_fn,
    build_deepresearch_race_judge_fn,
    calculate_fact_statistics,
    calculate_weighted_scores,
    clean_citation_url,
    clean_urls,
    fetch_crawl4ai_page,
    format_criteria_list,
    group_citations_by_url,
    normalize_comparative_score,
    normalize_weighted_scores,
    parse_dedup_indices,
    remove_urls,
    scrape_failure_unknown_results,
)


def _criterion(text, explanation, weight):
    return {"criterion": text, "explanation": explanation, "weight": weight}


def _criteria():
    return {
        "dimension_weight": {
            "comprehensiveness": 0.4,
            "insight": 0.3,
            "instruction_following": 0.2,
            "readability": 0.1,
        },
        "criterions": {
            "comprehensiveness": [
                _criterion("Population projections", "Covers 2020 through 2050.", 0.25),
                _criterion("Market segmentation", "Covers all requested sectors.", 0.75),
            ],
            "insight": [_criterion("Deep synthesis", "Explains non-obvious drivers.", 1.0)],
            "instruction_following": [
                _criterion("Answers the question", "Directly responds.", 0.2),
                _criterion("Uses the timeframe", "Uses the requested years.", 0.8),
            ],
            "readability": [_criterion("Clear structure", "Uses navigable sections.", 1.0)],
        },
    }


def _en_doc():
    # Trimmed from official query id 51, with the audited joined-row shape.
    return {
        "id": 51,
        "language": "en",
        "topic": "Finance & Business",
        "prompt": (
            "From 2020 to 2050, how many elderly people will there be in Japan? "
            "What is their consumption potential across clothing, food, housing, and "
            "transportation?"
        ),
        "criteria": _criteria(),
        "reference_article": (
            "# Japan's Silver Tsunami\n\nOfficial projections show a sustained aging trend."
        ),
    }


def _zh_doc():
    # Trimmed from official query id 1.
    return {
        "id": 1,
        "language": "zh",
        "topic": "Finance & Business",
        "prompt": "收集整理目前中国9阶层实际收入和财务状况，特别研究中国中产的特点和人数。",
        "criteria": _criteria(),
        "reference_article": "# 当代中国社会阶层结构\n\n本报告分析收入、财富与中产阶层。",
    }


def _response(instance: Instance, text: str) -> Response:
    return Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.CHAT, messages=()),
        outputs=[LMOutput(text=text)],
        scores={},
    )


@pytest.fixture
def task():
    return get_task("deepresearch_bench")


class TestRegistrationAndDocs:
    def test_registration_and_metrics(self):
        assert task_exists("deepresearch_bench")
        assert task_exists("deepresearch_bench:en")
        task = get_task("deepresearch_bench")
        assert task.config.data_source.path == "allenai/deepresearch-bench"
        assert task.config.data_source.subset == "default"
        assert task.config.data_source.split == "test"
        assert not hasattr(task.config, "language")
        assert task.config.sampling_params.temperature == 0.0
        assert task.config.sampling_params.max_tokens == 16_384
        assert task.config.get_primary_metric().name == "race_overall"
        assert {metric.name for metric in task.config.metrics} == {
            "race_overall",
            "race_comprehensiveness",
            "race_insight",
            "race_instruction_following",
            "race_readability",
            "fact_citation_accuracy",
            "fact_avg_effective_citations",
            "fact_avg_citations",
        }

    @pytest.mark.parametrize("doc", [_en_doc(), _zh_doc()])
    def test_process_doc_preserves_official_fields_and_appends_instruction(self, task, doc):
        instance = task.process_doc(doc, index=7)

        assert instance is not None
        assert instance.question.startswith(doc["prompt"] + "\n\n")
        assert instance.question.endswith(DEEPRESEARCH_GENERATION_INSTRUCTIONS[doc["language"]])
        assert instance.metadata["id"] == doc["id"]
        assert instance.metadata["language"] == doc["language"]
        assert instance.metadata["topic"] == doc["topic"]
        assert instance.metadata["prompt"] == doc["prompt"]
        assert instance.metadata["criteria"] == doc["criteria"]
        assert instance.metadata["reference_article"] == doc["reference_article"]
        assert instance.metadata["index"] == 7

        request = task.format_request(instance)
        assert request.request_type == RequestType.CHAT
        assert request.messages == ({"role": "user", "content": instance.question},)

    def test_english_variant_uses_hub_subset(self):
        task = get_task("deepresearch_bench:en")
        assert task.config.data_source.path == "allenai/deepresearch-bench"
        assert task.config.data_source.subset == "en"
        assert task.config.data_source.split == "test"
        assert task.process_doc(_en_doc()) is not None

    def test_think_strip_matches_researchqa_pattern(self, task):
        output = LMOutput(text="<think>private reasoning</think>\n\nFinal report.")
        assert task.extract_answer(output) == "Final report."

    @pytest.mark.parametrize(
        (
            "language",
            "bare_placeholder",
            "citation_format",
            "inline_citation",
            "replacement_clause",
        ),
        [
            (
                "en",
                "[source title]",
                "[<the source's actual title>](<the source's actual URL>)",
                "inline Markdown citation",
                "Replace both placeholders with the real page title and URL",
            ),
            (
                "zh",
                "[来源标题]",
                "[<来源的真实标题>](<来源的真实链接>)",
                "行内 Markdown 引用",
                "请将两个占位符替换为真实网页标题和链接",
            ),
        ],
    )
    def test_generation_instruction_uses_noncopyable_citation_placeholders(
        self,
        language,
        bare_placeholder,
        citation_format,
        inline_citation,
        replacement_clause,
    ):
        instruction = DEEPRESEARCH_GENERATION_INSTRUCTIONS[language]

        assert instruction
        assert bare_placeholder not in instruction
        assert citation_format in instruction
        assert inline_citation in instruction
        assert replacement_clause in instruction
        assert "](<" in instruction
        assert re.search(r"https?://", instruction, flags=re.IGNORECASE) is None


class TestGeneratedReportCleaning:
    def test_dangling_nid_76_closing_tags_are_removed_and_report_survives(self, task):
        report = (
            "# Findings\n\nThe evidence supports the conclusion.\n"
            "The final sources were MDPI and Healthline."
        )
        generated = f"{report}\n</parameter>\n</function>\n</tool_call>"

        cleaned = task.extract_answer(LMOutput(text=generated))

        assert cleaned == report
        assert cleaned.endswith("The final sources were MDPI and Healthline.")
        assert "<" not in cleaned

    def test_channel_and_harmony_markers_are_removed_on_production_path(self, task):
        report = "# Findings\n\nA legitimate report sentence."
        generated = f"<|channel>thought\n<channel|>{report}<|im_end|>"

        assert task.extract_answer(LMOutput(text=generated)) == report

    def test_pure_gemma_channel_and_tool_call_scaffold_becomes_empty(self, task):
        generated = "<|channel>thought<tool_call|>"

        assert task.extract_answer(LMOutput(text=generated)) == ""

    def test_complete_tool_call_wrapper_loses_markup_but_keeps_prose(self, task):
        generated = '<tool_call name="write_report">\nA legitimate report sentence.\n</tool_call>'

        assert task.extract_answer(LMOutput(text=generated)) == "A legitimate report sentence."

    def test_complete_tool_response_wrapper_loses_markup_but_keeps_prose(self, task):
        generated = "<tool_response>\nA sourced report sentence.\n</tool_response>"

        assert task.extract_answer(LMOutput(text=generated)) == "A sourced report sentence."

    @pytest.mark.parametrize("family", ["tool_call", "tool_response", "function", "parameter"])
    @pytest.mark.parametrize("quote", ['"', "'"])
    @pytest.mark.parametrize("value", ["write=report", "write report"])
    def test_quoted_scaffold_attributes_accept_spaces_and_equals_on_production_path(
        self, task, family, quote, value
    ):
        generated = f"<{family} name={quote}{value}{quote}>Final report.</{family}>"

        assert task.extract_answer(LMOutput(text=generated)) == "Final report."

    @pytest.mark.parametrize("value", ["write=report", "write report"])
    def test_unquoted_scaffold_attributes_remain_identifier_shaped(self, task, value):
        generated = f"<tool_call name={value}>Final report."

        assert task.extract_answer(LMOutput(text=generated)) == generated

    @pytest.mark.parametrize(
        "generated",
        [
            '<tool_call name="write report>Visible prose > remains visible.',
            "<tool_call name='write report>Visible prose > remains visible.",
            '<tool_call name="write report\nVisible prose > remains visible.',
            "<tool_call name='write report\nVisible prose > remains visible.",
        ],
    )
    def test_unterminated_quoted_opener_cannot_cross_angle_or_newline(self, task, generated):
        assert task.extract_answer(LMOutput(text=generated)) == generated

    def test_double_quoted_scaffold_attribute_cannot_consume_quoted_prose(self, task):
        generated = '<function name="a" and "b" > tail.'

        assert task.extract_answer(LMOutput(text=generated)) == generated

    def test_single_quoted_scaffold_attribute_cannot_consume_apostrophe_prose(self, task):
        generated = "<function name='x' the model's output '> tail."

        assert task.extract_answer(LMOutput(text=generated)) == generated

    def test_function_opener_with_double_quoted_spaced_value_is_stripped(self, task):
        generated = '<function name="write report">Final report.'

        assert task.extract_answer(LMOutput(text=generated)) == "Final report."

    def test_tool_call_opener_with_multiple_quoted_attributes_is_stripped(self, task):
        generated = '<tool_call name="a b" id="c d">Final report.'

        assert task.extract_answer(LMOutput(text=generated)) == "Final report."

    @pytest.mark.parametrize(
        "generated",
        [
            (
                "<|start|>assistant<|channel|>analysis<|message|>Private reasoning."
                "<|end|><|start|>assistant<|channel|>final<|message|>Final report."
                "<|return|>"
            ),
            (
                "<start|>assistant<|channel>thought<message|>Private reasoning."
                "<end|><start|>assistant<|channel>final<message|>Final report.<return|>"
            ),
        ],
    )
    def test_reasoning_channel_content_is_dropped_when_final_channel_exists(self, task, generated):
        assert task.extract_answer(LMOutput(text=generated)) == "Final report."

    def test_report_content_before_first_channel_is_retained_with_final_content(self, task):
        generated = (
            "Report body comes first."
            "<|channel|>analysis<|message|>Private reasoning."
            "<|channel|>final<|message|>Final report."
        )

        assert task.extract_answer(LMOutput(text=generated)) == (
            "Report body comes first. Final report."
        )

    def test_special_token_removal_separates_adjacent_words(self, task):
        generated = "First<|im_end|>second"

        assert task.extract_answer(LMOutput(text=generated)) == "First second"

    def test_think_block_is_removed_before_scaffold_tags(self):
        generated = (
            "<think>private reasoning with <tool_call></think>\n"
            "<tool_call><function=write_report><parameter=report>"
            "Final report sentence.</parameter></function></tool_call>"
        )

        assert deepresearch_bench._clean_generated_report(generated) == ("Final report sentence.")

    def test_text_without_tags_is_unchanged_modulo_strip(self):
        report = "A plain report with no scaffold markup."

        assert deepresearch_bench._clean_generated_report(f" \n{report}\n ") == report

    def test_unpaired_bare_function_keyword_in_prose_is_preserved(self):
        report = "C++ does not use the `<function>` keyword for function declarations."

        assert deepresearch_bench._clean_generated_report(report) == report

    @pytest.mark.parametrize(
        "code",
        [
            "`<function>example</function>`",
            '```xml\n<tool_call name="example">\n</tool_call>\n```',
        ],
    )
    def test_scaffold_examples_in_markdown_code_are_preserved(self, code):
        report = f"The documentation shows {code} as an example."

        assert deepresearch_bench._clean_generated_report(report) == report

    def test_paired_bare_function_tags_in_prose_lose_markup_but_keep_content(self):
        report = "The prose uses <function>call</function> to discuss the wrapper."

        assert deepresearch_bench._clean_generated_report(report) == (
            "The prose uses call to discuss the wrapper."
        )

    def test_bare_parameter_pairing_crosses_preserved_inline_code_on_production_path(self, task):
        generated = "<parameter>A report cites `x = <parameter>` verbatim.</parameter>"

        assert task.extract_answer(LMOutput(text=generated)) == (
            "A report cites `x = <parameter>` verbatim."
        )

    def test_malformed_opener_is_preserved_on_production_path(self, task):
        generated = '<function name="write_report"\n# Findings\n\nThe report starts here.'

        assert task.extract_answer(LMOutput(text=generated)) == generated

    def test_malformed_opener_cannot_consume_prose_at_bare_greater_than(self, task):
        generated = (
            '<parameter name="report"\n'
            "# Findings\n\nGlobal spending > 100 trillion dollars.\n"
            "The result was not significant (p > 0.05)."
        )

        assert task.extract_answer(LMOutput(text=generated)) == generated

    def test_large_bare_tag_sequence_is_cleaned_without_pairwise_lookahead(self):
        generated = "<function>" * 10_000 + "report" + "</function>" * 10_000

        assert deepresearch_bench._clean_generated_report(generated) == "report"


class TestRaceScoring:
    def test_format_criteria_hides_all_weights(self):
        rendered = format_criteria_list(_criteria())
        parsed = json.loads(rendered)

        assert "dimension_weight" not in parsed
        assert parsed["comprehensiveness"][0] == {
            "criterion": "Population projections",
            "explanation": "Covers 2020 through 2050.",
        }
        assert "weight" not in rendered

    def test_weighted_math_exact_case_substring_fallback_and_single_score(self):
        judge_output = {
            "comprehensiveness": [
                {
                    "criterion": "Population projections",
                    "article_1_score": 8,
                    "article_2_score": 4,
                },
                {
                    "criterion": "MARKET SEGMENTATION",
                    "article_1_score": 6,
                    "article_2_score": 2,
                },
            ],
            "insight": [
                {
                    "criterion": "Deep synthesis plus",
                    "article_1_score": 9,
                    "article_2_score": 3,
                }
            ],
            "instruction_following": [
                {
                    "criterion": "Answers the question",
                    "article_1_score": 3,
                    "article_2_score": 1,
                },
                {
                    "criterion": "A criterion the generator paraphrased beyond matching",
                    "article_1_score": 7,
                    "article_2_score": 5,
                },
            ],
            "readability": [
                {
                    "criterion": "Clear structure",
                    "target_score": 4,
                }
            ],
        }

        weighted = calculate_weighted_scores(judge_output, _criteria())

        assert weighted["target"]["dims"] == pytest.approx(
            {
                "comprehensiveness_weighted_avg": 6.5,
                "insight_weighted_avg": 9.0,
                "instruction_following_weighted_avg": 41 / 7,
                "readability_weighted_avg": 4.0,
            }
        )
        assert weighted["reference"]["dims"] == pytest.approx(
            {
                "comprehensiveness_weighted_avg": 2.5,
                "insight_weighted_avg": 3.0,
                "instruction_following_weighted_avg": 27 / 7,
                "readability_weighted_avg": 0.0,
            }
        )
        assert weighted["target"]["total"] == pytest.approx(481 / 70)
        assert weighted["reference"]["total"] == pytest.approx(187 / 70)

        normalized = normalize_weighted_scores(weighted)
        assert normalized == pytest.approx(
            {
                "race_overall": 481 / 668,
                "race_comprehensiveness": 6.5 / 9,
                "race_insight": 0.75,
                "race_instruction_following": 41 / 68,
                "race_readability": 1.0,
            }
        )

    def test_normalization_zero_guard(self):
        assert normalize_comparative_score(3.0, 1.0) == 0.75
        assert normalize_comparative_score(0.0, 0.0) == 0.0


class TestFactHelpersAndMetrics:
    def test_url_cleaning_and_fact_url_removal(self):
        url = "https://example.test/page#:~:text=quoted%20passage"
        report = f"A claim [Example]({url}) follows."

        assert clean_citation_url(url) == "https://example.test/page"
        assert clean_urls(report) == "A claim [Example](https://example.test/page) follows."
        assert remove_urls(report) == "A claim [Example] follows."

    def test_url_cleaning_removes_commonmark_autolink_brackets(self):
        url = " <https://example.test/page#:~:text=quoted%20passage> "
        report = f"A claim [Example]({url}) follows."

        assert clean_citation_url(url) == "https://example.test/page"
        assert clean_urls(report) == "A claim [Example](https://example.test/page) follows."

    def test_grouping_uses_cleaned_url(self):
        citations = [
            {
                "fact": "First claim",
                "ref_idx": 0,
                "url": "https://example.test/page#:~:text=first",
            },
            {"fact": "Second claim", "ref_idx": 0, "url": "https://example.test/page"},
            {"fact": "Other", "ref_idx": 0, "url": "https://other.test"},
        ]

        groups = group_citations_by_url(citations)

        assert list(groups) == ["https://example.test/page", "https://other.test"]
        assert [item["fact"] for item in groups["https://example.test/page"]] == [
            "First claim",
            "Second claim",
        ]

    def test_dedup_index_parsing_and_official_fallback(self):
        assert parse_dedup_indices("```json\n[1, 3]\n```", 3) == [1, 3]
        assert parse_dedup_indices("[-1]", 3) == [-1]
        assert parse_dedup_indices("[0, 2]", 3) == [1, 2, 3]
        assert parse_dedup_indices("[4]", 3) == [1, 2, 3]
        assert parse_dedup_indices("[-3]", 3) == [1, 2, 3]
        assert parse_dedup_indices("not json", 2) == [1, 2]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("judge_result", "expected"),
        [
            ("[-1]", ["second"]),
            ("[]", ["first", "second", "third"]),
            ("[0, 2]", ["first", "second", "third"]),
            ("[4]", ["first", "second", "third"]),
        ],
    )
    async def test_dedup_success_uses_official_guard_without_retry(
        self, task, judge_result, expected
    ):
        calls = 0

        async def judge(_prompt):
            nonlocal calls
            calls += 1
            return judge_result

        facts = ["first", "second", "third"]
        assert await task._deduplicate_facts(facts, "en", judge) == expected
        assert calls == 1

    @pytest.mark.anyio
    async def test_dedup_retries_only_json_parse_failure(self, task, monkeypatch):
        judge_results = iter(["not json", "still not json", "[-1]"])
        delays = []

        async def judge(_prompt):
            return next(judge_results)

        async def sleep(delay):
            delays.append(delay)

        monkeypatch.setattr(deepresearch_bench.asyncio, "sleep", sleep)
        facts = ["first", "second", "third"]
        assert await task._deduplicate_facts(facts, "en", judge) == ["second"]
        assert delays == [1, 2]

    @pytest.mark.anyio
    async def test_dedup_judge_failure_returns_all_facts(self, task, monkeypatch):
        calls = 0
        delays = []

        async def judge(_prompt):
            nonlocal calls
            calls += 1
            raise RuntimeError("transient judge failure")

        async def sleep(delay):
            delays.append(delay)

        monkeypatch.setattr(deepresearch_bench.asyncio, "sleep", sleep)
        facts = ["first", "second", "third"]

        assert await task._deduplicate_facts(facts, "en", judge) == facts
        assert calls == 3
        assert delays == [1, 2]

    def test_stat_semantics_exclude_unknown_and_skip_no_citation_tasks(self):
        statistics = calculate_fact_statistics(
            [
                "supported",
                "unknown",
                "unsupported",
                "supported",
                "unknown",
                "supported",
                "arbitrary-label",
            ],
            n_tasks=2,
        )
        assert statistics == {
            "fact_citation_accuracy": 0.6,
            "fact_avg_effective_citations": 1.5,
            "fact_avg_citations": 2.5,
        }

    def test_metric_averages_use_only_citation_bearing_responses(self, task):
        responses = [
            _response(Instance(question="one"), ""),
            _response(Instance(question="two"), ""),
        ]
        responses[0].scores.update(
            {
                "fact_avg_effective_citations": 2.0,
                "fact_avg_citations": 3.0,
                "fact_citation_accuracy": 2 / 3,
                "fact_has_citations": 1.0,
            }
        )
        responses[1].scores.update(
            {
                "fact_avg_effective_citations": 0.0,
                "fact_avg_citations": 0.0,
                "fact_citation_accuracy": 0.0,
                "fact_has_citations": 0.0,
            }
        )
        metrics = {metric.name: metric.compute(responses) for metric in task.config.metrics}
        assert metrics["fact_citation_accuracy"] == 2 / 3
        assert metrics["fact_avg_effective_citations"] == 2.0
        assert metrics["fact_avg_citations"] == 3.0

        responses[0].scores["fact_has_citations"] = 0.0
        fact_metrics = {metric.name: metric for metric in task.config.metrics}
        assert fact_metrics["fact_avg_effective_citations"].compute(responses) == 0.0
        assert fact_metrics["fact_avg_citations"].compute(responses) == 0.0

    def test_scrape_failure_short_circuits_to_unknown(self):
        assert scrape_failure_unknown_results("Error fetching webpage: timeout", 2) == [
            {"idx": 0, "result": "unknown"},
            {"idx": 1, "result": "unknown"},
        ]
        assert scrape_failure_unknown_results("A valid source page", 2) is None

    @pytest.mark.anyio
    async def test_scrape_cache_fetches_each_url_once(self, task, monkeypatch):
        calls = []

        async def fetch_page(url):
            calls.append(url)
            return "A valid source page"

        monkeypatch.setattr(deepresearch_bench, "fetch_crawl4ai_page", fetch_page)
        assert await task._fetch_page("https://example.test/page") == "A valid source page"
        assert await task._fetch_page("https://example.test/page") == "A valid source page"
        assert calls == ["https://example.test/page"]

    @pytest.mark.anyio
    async def test_crawl4ai_import_error_names_install_extra(self, monkeypatch):
        real_import = builtins.__import__

        def import_without_crawl4ai(name, *args, **kwargs):
            if name == "crawl4ai":
                raise ImportError
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", import_without_crawl4ai)
        with pytest.raises(
            RuntimeError,
            match=r"pip install 'olmo-eval\[crawl4ai\]'",
        ):
            await fetch_crawl4ai_page("https://example.test/page")

    @pytest.mark.anyio
    async def test_validation_failure_returns_all_unknown(self, task, monkeypatch):
        calls = 0
        delays = []

        async def judge(_prompt):
            nonlocal calls
            calls += 1
            return "not json"

        async def sleep(delay):
            delays.append(delay)

        monkeypatch.setattr(deepresearch_bench.asyncio, "sleep", sleep)
        results = await task._validate_facts(["first", "second"], "valid page", "en", judge)

        assert calls == 3
        assert delays == [1, 2]
        assert results == [
            {"idx": 0, "result": "unknown"},
            {"idx": 1, "result": "unknown"},
        ]

    @pytest.mark.parametrize("missing_key", ["result", "idx"])
    @pytest.mark.anyio
    async def test_validation_missing_required_key_retries_then_returns_all_unknown(
        self, task, monkeypatch, missing_key
    ):
        calls = 0
        delays = []

        async def judge(_prompt):
            nonlocal calls
            calls += 1
            items = [
                {"idx": 1, "result": "supported"},
                {"idx": 2, "result": "unsupported"},
            ]
            del items[1][missing_key]
            return json.dumps(items)

        async def sleep(delay):
            delays.append(delay)

        monkeypatch.setattr(deepresearch_bench.asyncio, "sleep", sleep)
        results = await task._validate_facts(["first", "second"], "valid page", "en", judge)

        assert calls == 3
        assert delays == [1, 2]
        assert results == [
            {"idx": 0, "result": "unknown"},
            {"idx": 1, "result": "unknown"},
        ]

    @pytest.mark.anyio
    async def test_validation_blindly_decrements_idx_and_accepts_arbitrary_label(self, task):
        async def judge(_prompt):
            return json.dumps([{"idx": -7, "result": "arbitrary-label"}])

        assert await task._validate_facts(["fact"], "valid page", "en", judge) == [
            {"idx": -8, "result": "arbitrary-label"}
        ]

    @pytest.mark.anyio
    async def test_fact_metadata_pairs_by_returned_idx_then_position(self, task, monkeypatch):
        citations = [
            {"fact": "first", "url": "https://example.test"},
            {"fact": "second", "url": "https://example.test"},
            {"fact": "third", "url": "https://example.test"},
        ]

        async def extract(*_args):
            return citations

        async def deduplicate(facts, *_args):
            return facts

        async def fetch(_url):
            return "valid page"

        async def validate(*_args):
            return [
                {"idx": 2, "result": "supported"},
                {"idx": 99, "result": "arbitrary-label"},
                {"idx": 0, "result": "unknown"},
            ]

        async def unused_judge(_prompt):
            return ""

        monkeypatch.setattr(task, "_extract_citations", extract)
        monkeypatch.setattr(task, "_deduplicate_facts", deduplicate)
        monkeypatch.setattr(task, "_fetch_page", fetch)
        monkeypatch.setattr(task, "_validate_facts", validate)
        instance = task.process_doc(_en_doc())
        assert instance is not None

        scores, details = await task._score_fact(instance, "report", unused_judge)

        assert [detail["fact"] for detail in details] == ["third", "second", "first"]
        assert scores == {
            "fact_citation_accuracy": 0.5,
            "fact_avg_effective_citations": 1.0,
            "fact_avg_citations": 2.0,
            "fact_has_citations": 1.0,
        }


class TestPromptsAndJudges:
    @pytest.mark.parametrize(
        ("prompt", "digest"),
        [
            (
                RACE_SCORE_PROMPT_EN,
                "29c0dc8365d047b293c6e6dce80c840402003aa5171c47e2a6f19a064ca77a1f",
            ),
            (
                RACE_SCORE_PROMPT_ZH,
                "d324077bbe7218ad1df5c49cf3d0207f66416526c73063be883203d7e80610c4",
            ),
            (
                FACT_EXTRACTION_PROMPT_EN,
                "3f78bbd60b6a78147bed7c51b5f056e2ba990d299c1064318ef63349009e6e8e",
            ),
            (
                FACT_EXTRACTION_PROMPT_ZH,
                "2b3ce39d315d8f0637601bbe0262887a0f0e2909bd39b7b5e8fb4b3222f56bcd",
            ),
            (
                FACT_DEDUPLICATION_PROMPT_EN,
                "7c7965e51c48189f49267b6681efce5dcd6f4638cb331a2e63b937d317c2e93b",
            ),
            (
                FACT_DEDUPLICATION_PROMPT_ZH,
                "e12080068b5fe54893ee896cdf3d9efe8a0fb221a725d3551f8c8c6b3ed4509c",
            ),
            (
                FACT_VALIDATION_PROMPT_EN,
                "3d8f531bbad26b004faf2284ccd58ce561bf827d117cae3d2475c9a61e5cb71c",
            ),
            (
                FACT_VALIDATION_PROMPT_ZH,
                "81358173e19cdee1f560e1445048d800d6d455616782f14574eb4dc0920307d4",
            ),
        ],
    )
    def test_official_prompt_sha256(self, prompt, digest):
        assert hashlib.sha256(prompt.encode()).hexdigest() == digest

    def test_official_prompt_distinctive_substrings(self):
        assert 'Start with "Standard 1"' in RACE_SCORE_PROMPT_EN
        assert '从"标准1"开始' in RACE_SCORE_PROMPT_ZH
        assert "A segment of text + [number†" in FACT_EXTRACTION_PROMPT_EN
        assert "一段文字+[（一个或多个)数字†" in FACT_EXTRACTION_PROMPT_ZH
        assert "*exactly the same thing*" in FACT_DEDUPLICATION_PROMPT_EN
        assert "只有表达完全一致的事情时" in FACT_DEDUPLICATION_PROMPT_ZH
        assert "such as a 'page not found' message" in FACT_VALIDATION_PROMPT_EN
        assert '如"page not found"页面' in FACT_VALIDATION_PROMPT_ZH

    @pytest.mark.parametrize(
        ("env_spec", "expected_models", "expected_efforts"),
        [
            (None, ["gpt-5.5", "gpt-5.4-mini"], ["medium", "low"]),
            ("judge-override", ["judge-override", "judge-override"], ["medium", "low"]),
            ("gpt-5.5:high", ["gpt-5.5", "gpt-5.5"], ["high", "high"]),
        ],
    )
    def test_single_override_applies_to_both_judges(
        self, monkeypatch, env_spec, expected_models, expected_efforts
    ):
        captured = []

        async def sentinel(_prompt):
            return ""

        def fake_builder(**kwargs):
            captured.append(kwargs)
            return sentinel

        if env_spec is None:
            monkeypatch.delenv("OLMO_EVAL_JUDGE", raising=False)
        else:
            monkeypatch.setenv("OLMO_EVAL_JUDGE", env_spec)
        monkeypatch.setattr(deepresearch_bench, "build_openai_judge_fn", fake_builder)

        assert build_deepresearch_race_judge_fn() is sentinel
        assert build_deepresearch_fact_judge_fn() is sentinel
        assert [call["model"] for call in captured] == expected_models
        assert [call["reasoning_effort"] for call in captured] == expected_efforts
        assert [call["scorer_name"] for call in captured] == [
            "DeepResearchBenchRACE",
            "DeepResearchBenchFACT",
        ]
        assert all(call["temperature"] == 0.0 for call in captured)
        assert all(call["max_tokens"] == 64_000 for call in captured)


@pytest.mark.anyio
async def test_score_responses_runs_race_and_fact_without_network(task, monkeypatch):
    instance = task.process_doc(_en_doc())
    assert instance is not None
    response = _response(
        instance,
        "<think>research plan</think>A supported fact [Source](https://example.test/page).",
    )
    race_calls = []
    fact_calls = []

    race_payload = {
        dimension: [
            {
                "criterion": _criteria()["criterions"][dimension][0]["criterion"],
                "article_1_score": 8,
                "article_2_score": 2,
            }
        ]
        for dimension in DEEPRESEARCH_DIMENSIONS
    }

    async def race_judge(prompt):
        race_calls.append(prompt)
        return f"```json\n{json.dumps(race_payload)}\n```"

    async def fact_judge(prompt):
        fact_calls.append(prompt)
        if "Here is the main text" in prompt:
            return json.dumps(
                [
                    {
                        "fact": "A supported fact [Source](https://example.test/page).",
                        "ref_idx": 0,
                        "url": "https://example.test/page",
                    }
                ]
            )
        assert "A valid source page" in prompt
        return json.dumps([{"idx": 1, "result": "supported"}])

    async def fetch_page(url):
        assert url == "https://example.test/page"
        return "A valid source page containing the supported fact."

    monkeypatch.setattr(deepresearch_bench, "build_deepresearch_race_judge_fn", lambda: race_judge)
    monkeypatch.setattr(deepresearch_bench, "build_deepresearch_fact_judge_fn", lambda: fact_judge)
    monkeypatch.setattr(deepresearch_bench, "fetch_crawl4ai_page", fetch_page)

    await task.score_responses([response])

    assert len(race_calls) == 1
    assert len(fact_calls) == 2
    assert response.scores["race_overall"] == pytest.approx(0.8)
    for dimension in DEEPRESEARCH_DIMENSIONS:
        assert response.scores[f"race_{dimension}"] == pytest.approx(0.8)
    assert response.scores["fact_citation_accuracy"] == 1.0
    assert response.scores["fact_avg_effective_citations"] == 1.0
    assert response.scores["fact_avg_citations"] == 1.0
    assert response.scores["fact_has_citations"] == 1.0
    assert response.outputs[0].extracted_answer.startswith("A supported fact")
    assert response.outputs[0].metadata["deepresearch_fact"] == [
        {
            "url": "https://example.test/page",
            "fact": "A supported fact [Source].",
            "result": "supported",
        }
    ]


@pytest.mark.anyio
async def test_unexpected_fact_failure_preserves_race_scores(task, monkeypatch, caplog):
    instance = task.process_doc(_en_doc())
    assert instance is not None
    response = _response(instance, "A complete research report.")
    race_payload = {
        dimension: [
            {
                "criterion": _criteria()["criterions"][dimension][0]["criterion"],
                "article_1_score": 8,
                "article_2_score": 2,
            }
        ]
        for dimension in DEEPRESEARCH_DIMENSIONS
    }

    async def race_judge(_prompt):
        return json.dumps(race_payload)

    async def unused_fact_judge(_prompt):
        return "[]"

    async def fail_fact(*_args):
        raise RuntimeError("unexpected FACT failure")

    monkeypatch.setattr(deepresearch_bench, "build_deepresearch_race_judge_fn", lambda: race_judge)
    monkeypatch.setattr(
        deepresearch_bench, "build_deepresearch_fact_judge_fn", lambda: unused_fact_judge
    )
    monkeypatch.setattr(task, "_score_fact", fail_fact)

    scored = await task.score_responses([response])

    assert scored == [response]
    assert response.scores["race_overall"] == pytest.approx(0.8)
    for dimension in DEEPRESEARCH_DIMENSIONS:
        assert response.scores[f"race_{dimension}"] == pytest.approx(0.8)
    assert {
        metric: response.scores[metric]
        for metric in (
            "fact_citation_accuracy",
            "fact_avg_effective_citations",
            "fact_avg_citations",
            "fact_has_citations",
        )
    } == task._zero_fact_scores()
    assert response.outputs[0].metadata["deepresearch_fact"] == []
    assert "FACT scoring failed for id 51" in caplog.text
    assert "FACT scoring failed for 1 instance(s)" in caplog.text


@pytest.mark.anyio
async def test_empty_answer_sets_fact_has_citations_to_zero(task, monkeypatch):
    instance = task.process_doc(_en_doc())
    assert instance is not None
    response = _response(instance, "<think>research plan only</think>")
    judge_calls = 0

    async def judge(_prompt):
        nonlocal judge_calls
        judge_calls += 1
        return ""

    monkeypatch.setattr(deepresearch_bench, "build_deepresearch_race_judge_fn", lambda: judge)
    monkeypatch.setattr(deepresearch_bench, "build_deepresearch_fact_judge_fn", lambda: judge)

    await task.score_responses([response])

    assert judge_calls == 0
    assert response.scores["fact_has_citations"] == 0.0
    assert response.scores["fact_avg_effective_citations"] == 0.0
    assert response.scores["fact_avg_citations"] == 0.0


@pytest.mark.anyio
async def test_race_parse_failure_retries_and_keeps_zero_instance(task, monkeypatch):
    instance = task.process_doc(_en_doc())
    assert instance is not None
    response = _response(instance, "A report without citations.")
    race_calls = 0
    delays = []

    async def race_judge(_prompt):
        nonlocal race_calls
        race_calls += 1
        return "malformed"

    async def fact_judge(_prompt):
        return "[]"

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(deepresearch_bench, "build_deepresearch_race_judge_fn", lambda: race_judge)
    monkeypatch.setattr(deepresearch_bench, "build_deepresearch_fact_judge_fn", lambda: fact_judge)
    monkeypatch.setattr(deepresearch_bench.asyncio, "sleep", sleep)

    scored = await task.score_responses([response])

    assert scored == [response]
    assert race_calls == 3
    assert delays == [1, 2]
    assert response.scores["race_overall"] == 0.0
    assert response.scores["fact_has_citations"] == 0.0
    assert task.config.get_primary_metric().compute(scored) == 0.0
