"""DeepResearch Bench long-form research evaluation (arXiv 2506.11763).

The task generates citation-rich research reports and applies both official
scoring frameworks: RACE for criteria-weighted report quality and FACT for
citation trustworthiness. Official bilingual judge prompts and score math are
ported verbatim/faithfully from the benchmark repository.

This integration deliberately deviates from the official implementation in
six ways: (1) FACT scrapes with olmo-eval's crawl4ai-backed browsing logic
instead of Jina Reader; (2) generated articles receive deterministic ``<think>``
stripping and tool/channel-scaffold marker removal instead of the optional LLM
ArticleCleaner; (3) both judge models can be overridden with
``OLMO_EVAL_JUDGE``; and (4) the appended report-generation instruction is ours
because the official benchmark leaves each deep-research system's generation
prompting unspecified; (5) ``clean_urls`` is applied to every article rather
than conditionally by source model, and text-fragment URLs are also normalized
while grouping, merging citations to fragments of the same page; and (6) judge
failures use three attempts with ``2**n`` backoff, while the official RACE
runner uses ten attempts with ``1.5**n`` backoff. After final RACE judge/parse
failure, an instance receives zero RACE scores and remains in every mean.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from abc import ABC
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlsplit

from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.llm_judge import JudgeFn, build_openai_judge_fn
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, TaskConfig, register, register_variant

logger = logging.getLogger(__name__)

DEEPRESEARCH_BENCH_REPO = "allenai/deepresearch-bench"
DEEPRESEARCH_RACE_DEFAULT_JUDGE_SPEC = "gpt-5.5"
DEEPRESEARCH_FACT_DEFAULT_JUDGE_SPEC = "gpt-5.4-mini"
DEEPRESEARCH_JUDGE_MAX_TOKENS = 64_000
DEEPRESEARCH_JUDGE_ATTEMPTS = 3
DEEPRESEARCH_SCRAPE_ATTEMPTS = 3
DEEPRESEARCH_PAGE_CONTENT_LIMIT = 200_000
DEEPRESEARCH_DIMENSIONS = (
    "comprehensiveness",
    "insight",
    "instruction_following",
    "readability",
)

DEEPRESEARCH_GENERATION_INSTRUCTIONS = {
    "en": (
        "Write a comprehensive research report in English. Research the task using the "
        "available web-search and webpage-browsing tools before writing. Place an inline "
        "Markdown citation in the exact form `[<the source's actual "
        "title>](<the source's actual URL>)` immediately after every factual claim it "
        "supports. Replace both placeholders with the real page title and URL; never output "
        "either placeholder itself. Synthesize the sources "
        "into a clear, detailed report."
    ),
    "zh": (
        "请用中文撰写一份全面的研究报告。写作前请使用可用的网页搜索和网页浏览工具调研任务。"
        "每项事实性主张后都应紧跟一个格式严格为 `[<来源的真实标题>]"
        "(<来源的真实链接>)` 的行内 Markdown 引用。请将两个占位符替换为真实网页标题和链接，"
        "绝不要原样输出任何一个占位符。请综合各项来源，形成清晰、详尽的报告。"
    ),
}

# Verbatim from prompt/score_prompt_en.py:generate_merged_score_prompt.
RACE_SCORE_PROMPT_EN = """
<system_role>You are a strict, meticulous, and objective research article evaluation expert. You excel at using specific assessment criteria to deeply compare two articles on the same task, providing precise scores and clear justifications.</system_role>

<user_prompt>
**Task Background**
There is a deep research task, and you need to evaluate two research articles written for this task. We will assess the articles across four dimensions: Comprehensiveness, Insight, Instruction Following, and Readability. The content is as follows:
<task>
"{task_prompt}"
</task>

**Articles to Evaluate**
<article_1>
"{article_1}"
</article_1>

<article_2>
"{article_2}"
</article_2>

**Evaluation Criteria**
Now, you need to evaluate and compare these two articles based on the following **evaluation criteria list**, providing comparative analysis and scoring each on a scale of 0-10. Each criterion includes an explanation, please understand carefully.

<criteria_list>
{criteria_list}
</criteria_list>

<Instruction>
**Your Task**
Please strictly evaluate and compare `<article_1>` and `<article_2>` based on **each criterion** in the `<criteria_list>`. You need to:
1.  **Analyze Each Criterion**: Consider how each article fulfills the requirements of each criterion.
2.  **Comparative Evaluation**: Analyze how the two articles perform on each criterion, referencing the content and criterion explanation.
3.  **Score Separately**: Based on your comparative analysis, score each article on each criterion (0-10 points).

**Scoring Rules**
For each criterion, score both articles on a scale of 0-10 (continuous values). The score should reflect the quality of performance on that criterion:
*   0-2 points: Very poor performance. Almost completely fails to meet the criterion requirements.
*   2-4 points: Poor performance. Minimally meets the criterion requirements with significant deficiencies.
*   4-6 points: Average performance. Basically meets the criterion requirements, neither good nor bad.
*   6-8 points: Good performance. Largely meets the criterion requirements with notable strengths.
*   8-10 points: Excellent/outstanding performance. Fully meets or exceeds the criterion requirements.

**Output Format Requirements**
Please **strictly** follow the `<output_format>` below for each criterion evaluation. **Do not include any other unrelated content, introduction, or summary**. Start with "Standard 1" and proceed sequentially through all criteria:
</Instruction>

<output_format>
{{
    "comprehensiveness": [
        {{
            "criterion": [Text content of the first comprehensiveness evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
}},
{{
            "criterion": [Text content of the second comprehensiveness evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
        }},
        ...
    ],
    "insight": [
        {{
            "criterion": [Text content of the first insight evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
        }},
        ...
    ],
    ...
}}
</output_format>

Now, please evaluate the two articles based on the research task and criteria, providing detailed comparative analysis and scores according to the requirements above. Ensure your output follows the specified `<output_format>` and that the JSON format is parsable, with all characters that might cause JSON parsing errors properly escaped.
</user_prompt>
"""

# Verbatim from prompt/score_prompt_zh.py:generate_merged_score_prompt.
RACE_SCORE_PROMPT_ZH = """
<system_role>你是一名严格、细致、客观的调研文章评估专家。你擅长根据具体的评估标准，深入比较两篇针对同一任务的文章，并给出精确的评分和清晰的理由。</system_role>

<user_prompt>
**任务背景**
有一个深度调研任务，你需要评估针对该任务撰写的两篇调研文章。我们会从以下四个维度评估文章：全面性、洞察力、指令遵循能力和可读性。内容如下：
<task>
"{task_prompt}"
</task>

**待评估文章**
<article_1>
"{article_1}"
</article_1>

<article_2>
"{article_2}"
</article_2>

**评估标准**
现在，你需要根据以下**评判标准列表**，逐条评估并比较这两篇文章的表现，输出对比分析，然后给出0-10的分数。每个标准都附有其解释，请仔细理解。

<criteria_list>
{criteria_list}
</criteria_list>

<Instruction>
**你的任务**
请严格按照 `<criteria_list>` 中的**每一条标准**，对比评估 `<article_1>` 和 `<article_2>` 在该标准上的具体表现。你需要：
1.  **逐条分析**：针对列表中的每一条标准，分别思考两篇文章是如何满足该标准要求的。
2.  **对比评估**：结合文章内容与标准解释，对比分析两篇文章在每一条标准上的表现。
3.  **分别打分**：基于你的对比分析，为两篇文章在该条标准上的表现分别打分（0-10分）。

**打分规则**
对每一条标准，分别为两篇文章打分，打分范围为 0-10 分（连续的数值）。分数高低应体现文章在该标准上表现的好坏：
*   0-2分：表现很差。几乎完全不符合标准要求。
*   2-4分：表现较差。少量符合标准要求，但有明显不足。
*   4-6分：表现中等。基本符合标准要求，不好不坏。
*   6-8分：表现较好。大部分符合标准要求，有可取之处。
*   8-10分：表现出色/极好。完全或超预期符合标准要求。

**输出格式要求**
请**严格**按照下列`<output_format>`格式输出每一条标准的评估结果，**不要包含任何其他无关内容、引言或总结**。从"标准1"开始，按顺序输出所有标准的评估：
</Instruction>

<output_format>
{{
    "comprehensiveness": [
        {{
            "criterion": [全面性维度的第一条评判标准文本内容],
            "analysis": [对比分析],
            "article_1_score": [0-10连续分数],
            "article_2_score": [0-10连续分数]
        }},
        {{
            "criterion": [全面性维度的第二条评判标准文本内容],
            "analysis": [对比分析],
            "article_1_score": [0-10连续分数],
            "article_2_score": [0-10连续分数]
        }},
        ...
    ],
    "insight": [
        {{
            "criterion": [洞察力维度的第一条评判标准文本内容],
            "analysis": [对比分析],
            "article_1_score": [0-10连续分数],
            "article_2_score": [0-10连续分数]
        }},
        ...
    ],
    ...
}}
</output_format>

现在，请根据调研任务和标准，对两篇文章进行评估，并按照上述要求给出详细的对比分析和评分，请确保输出格式遵守上述`<output_format>`，而且保证其中的json格式可以解析，注意所有可能导致json解析错误的要转义的符号。
</user_prompt>
"""

# Verbatim from utils/extract.py:prompt_template and prompt_template_en.
FACT_EXTRACTION_PROMPT_ZH = """你会看到一篇研究报告，研究报告正文中会有一些对参考文献的引用。
正文中的引用可能以如下形式出现：
1. 一段文字+空格+数字，例如："李强基于收入、教育和职业构造了一个社会经济地位指数（SES），将社会划分为7个等级 15"
2. 一段文字+[（一个或多个)数字]，例如："李强基于收入、教育和职业构造了一个社会经济地位指数（SES），将社会划分为7个等级[15]"
3. 一段文字+[（一个或多个)数字†(一些行号等内容)]，例如："李强基于收入、教育和职业构造了一个社会经济地位指数（SES），将社会划分为7个等级[15†L10][5L23][7†summary][9summary]"
4. [引用来源](引用链接)，例如："根据[ChinaFile: A Guide to Social Class in Modern China](https://www.chinafile.com/reporting-opinion/media/guide-social-class-modern-china)'s分类，中国社会可分为九个阶层"

请从正文中找出**所有**引用了参考文献的地方，提取出(fact, ref_idx, url)三元组，提取的时候，注意以下事项：
1. 由于后续需要检验这些facts是否正确，你可能需要在引用的前后寻找一些上下文，以确保fact是完整可理解的，而不是简单的词组或短语
2. 如果一个fact引用了多个文献，那么它应该对应多个三元组，例如如果引用了2个文献，则应该是(fact, ref_idx_1, url_1)和(fact, ref_idx_2, url_2)
3. 对于第三种形式的引用，ref_idx仅考虑第一个数字部分，不考虑其他指示具体位置的内容；对于第四种形式的引用（即引用来源和链接直接出现在正文中）的情况，ref_idx统一设置为0
4. 如果正文中没有标出引用的具体位置（比如仅在文章结尾列出了参考文献列表，而没有在正文中标出），请返回空列表

你应该返回json列表格式，列表中的每一项是一个三元组，例如：
[
    {{
        "fact": "原文中的文本片段，注意中文引号要用全角, 英文引号前加单个反斜杠转义",
        "ref_idx": "该段文字引用的参考文献在参考文献列表中的索引",
        "url": "该段文字引用的参考文献链接（从研究报告结尾的参考文献列表或引用处的括号中提取）"
    }}
]

下面是研究报告的正文：
{report_text}

下面开始提取，直接输出json列表，不要输出任何闲聊或解释。"""

FACT_EXTRACTION_PROMPT_EN = """You will be provided with a research report. The body of the report will contain some citations to references.

Citations in the main text may appear in the following forms:
1. A segment of text + space + number, for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels 15"
2. A segment of text + [number], for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels[15]"
3. A segment of text + [number†(some line numbers, etc.)], for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels[15†L10][5L23][7†summary]"
4. [Citation Source](Citation Link), for example: "According to [ChinaFile: A Guide to Social Class in Modern China](https://www.chinafile.com/reporting-opinion/media/guide-social-class-modern-china)'s classification, Chinese society can be divided into nine strata"

Please identify **all** instances where references are cited in the main text, and extract (fact, ref_idx, url) triplets. When extracting, pay attention to the following:
1. Since these facts will need to be verified later, you may need to look for some context before and after the citation to ensure that the fact is complete and understandable, rather than just a simple phrase or short expression.
2. If a fact cites multiple references, then it should correspond to two triplets: (fact, ref_idx_1, url_1) and (fact, ref_idx_2, url_2).
3. For the third form of citation (i.e., where the citation source and link appear directly in the text), the ref_idx should be uniformly set to 0.
4. If the main text does not specify the exact location of the citation (for example, only the reference list is listed at the end of the article, without specifying the citation point in the text), please return an empty list.

You should return a JSON list format, where each item in the list is a triplet, for example:
[
    {{
        "fact": "Text segment from the original document. Note that Chinese quotation marks should use full-width marks. And add a single backslash before the English quotation mark to make it a readable for python json module.",
        "ref_idx": "The index of the cited reference in the reference list for this text segment.",
        "url": "The URL of the cited reference for this text segment (extracted from the reference list at the end of the research report or from the parentheses at the citation point)."
    }}
]

Here is the main text of the research report:
{report_text}

Please begin the extraction now. Output only the JSON list directly, without any chitchat or explanations."""

# Verbatim from utils/deduplicate.py:prompt_template and prompt_template_en.
FACT_DEDUPLICATION_PROMPT_ZH = """你会看到一个statement列表，你需要对其去重，并返回去重后的statement序号列表，注意：只有表达完全一致的事情时，两个statement才被认为是重复的，如果列表中没有重复的statement，则返回完整的列表。

你应该返回一个List(int)，列表中的每一项是去重后留下的，不重复的statement的序号，例如：
[1, 3, 5]

下面是你需要去重的statement列表
{statements}

下面开始提取，直接输出整数列表，不要输出任何闲聊或解释。"""

FACT_DEDUPLICATION_PROMPT_EN = """You will be given a list of statements. You need to de-duplicate them and return a list of indices of the unique statements. Note: Two statements are considered duplicates only if they express *exactly the same thing*. If there are no duplicate statements in the list, return the complete list of indices.

You should return a List(int), where each item in the list is the index of a unique, non-duplicated statement that has been retained. For example:
[1, 3, 5]

Below is the list of statements you need to de-duplicate:
{statements}

Please begin the extraction now. Output only the integer list, without any conversational text or explanations."""

# Verbatim from utils/validate.py:prompt_template and prompt_template_en.
FACT_VALIDATION_PROMPT_ZH = """你会看到一个参考资料和一些statement，请你判断对于参考资料来说statement是supported、unsupported、或者unknown，注意：
首先判断参考资料是否存在有效内容，如果参考资料中没有任何有效信息，如"page not found"页面，则认为所有statement的状态都是unknown。
除此之外，参考资料有效的情况下，对于一个statement来说，如果它包含的事实或数据在参考资料中可以全部或部分找到，就认为它是supported的（数据接受四舍五入）；如果statement中所有的事实和数据在参考资料中都找不到，认为它是unsupported的。

你应该返回json列表格式，列表中的每一项包含statement的序号和判断结果，例如：
[
    {{
        "idx": 1,
        "result": "supported"
    }},
    {{
        "idx": 2,
        "result": "unsupported"
    }}
]

下面是参考资料和statements：
<reference>
{reference}
</reference>

<statements>
{statements}
</statements>

下面开始判断，直接输出json列表，不要输出任何闲聊或解释。"""

FACT_VALIDATION_PROMPT_EN = """You will be provided with a reference and some statements. Please determine whether each statement is 'supported', 'unsupported', or 'unknown' with respect to the reference. Please note:
First, assess whether the reference contains any valid content. If the reference contains no valid information, such as a 'page not found' message, then all statements should be considered 'unknown'.
If the reference is valid, for a given statement: if the facts or data it contains can be found entirely or partially within the reference, it is considered 'supported' (data accepts rounding); if all facts and data in the statement cannot be found in the reference, it is considered 'unsupported'.

You should return the result in a JSON list format, where each item in the list contains the statement's index and the judgment result, for example:
[
    {{
        "idx": 1,
        "result": "supported"
    }},
    {{
        "idx": 2,
        "result": "unsupported"
    }}
]

Below are the reference and statements:
<reference>
{reference}
</reference>

<statements>
{statements}
</statements>

Begin the assessment now. Output only the JSON list, without any conversational text or explanations."""


def format_criteria_list(criteria_data: Mapping[str, Any]) -> str:
    """Format criteria for the judge without revealing official weights."""
    criteria_for_prompt: dict[str, list[dict[str, Any]]] = {}
    criterions_dict = criteria_data.get("criterions", {})
    if not isinstance(criterions_dict, Mapping):
        criterions_dict = {}

    for dimension, criterions_list in criterions_dict.items():
        if not isinstance(dimension, str) or not isinstance(criterions_list, list):
            logger.warning("Invalid criteria list for dimension %r; skipping", dimension)
            continue
        criteria_for_prompt[dimension] = []
        for criterion_item in criterions_list:
            if (
                isinstance(criterion_item, Mapping)
                and "criterion" in criterion_item
                and "explanation" in criterion_item
            ):
                criteria_for_prompt[dimension].append(
                    {
                        "criterion": criterion_item["criterion"],
                        "explanation": criterion_item["explanation"],
                    }
                )
            else:
                logger.warning("Invalid criterion in dimension %r; skipping", dimension)

    try:
        return json.dumps(criteria_for_prompt, ensure_ascii=False, indent=2)
    except TypeError as exc:
        raise ValueError(f"Failed to serialize criteria to JSON: {exc}") from exc


def extract_json_from_markdown(text: Any) -> str | None:
    """Port the official layered JSON extraction and score-specific fallback."""
    if not isinstance(text, str):
        return None

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

    marker_start = text.find("```json")
    if marker_start >= 0:
        start = marker_start + len("```json")
        end = text.find("```", start)
        if end > start:
            candidate = text[start:end].strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start != -1:
        level = 0
        for offset, character in enumerate(text[start:]):
            if character == "{":
                level += 1
            elif character == "}":
                level -= 1
                if level == 0:
                    candidate = text[start : start + offset + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        pass
                    break

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    if "comprehensiveness" in text and "article_1_score" in text and "article_2_score" in text:
        dimensions = list(DEEPRESEARCH_DIMENSIONS)
        result: dict[str, list[dict[str, Any]]] = {}
        for dimension in dimensions:
            if dimension not in text:
                continue
            result[dimension] = []
            dimension_start = text.find(f'"{dimension}"')
            if dimension_start == -1:
                dimension_start = text.find(f"'{dimension}'")
            if dimension_start == -1:
                dimension_start = text.find(dimension)
            if dimension_start == -1:
                continue

            next_dimension_start = len(text)
            for next_dimension in dimensions:
                if next_dimension == dimension:
                    continue
                position = text.find(f'"{next_dimension}"', dimension_start)
                if position == -1:
                    position = text.find(f"'{next_dimension}'", dimension_start)
                if position == -1:
                    position = text.find(next_dimension, dimension_start + len(dimension))
                if position != -1:
                    next_dimension_start = min(next_dimension_start, position)

            dimension_content = text[dimension_start:next_dimension_start]
            criteria = re.findall(r'"criterion"\s*:\s*"([^"]+)"', dimension_content)
            scores_1 = re.findall(r'"article_1_score"\s*:\s*(\d+\.?\d*)', dimension_content)
            scores_2 = re.findall(r'"article_2_score"\s*:\s*(\d+\.?\d*)', dimension_content)
            for index in range(min(len(criteria), len(scores_1), len(scores_2))):
                result[dimension].append(
                    {
                        "criterion": criteria[index],
                        "article_1_score": float(scores_1[index]),
                        "article_2_score": float(scores_2[index]),
                    }
                )
        if any(result.values()):
            return json.dumps(result)
    return None


def calculate_weighted_scores(
    llm_output_json: Mapping[str, Any],
    criteria_data: Mapping[str, Any],
    language: str = "en",
) -> dict[str, dict[str, Any]]:
    """Faithfully port the official criteria and dimension weighting math."""
    del language  # Kept for API parity with the official implementation.
    results: dict[str, dict[str, Any]] = {
        "target": {"dims": {}, "total": 0.0},
        "reference": {"dims": {}, "total": 0.0},
    }
    total_target_score = 0.0
    total_reference_score = 0.0
    dimension_weights = criteria_data.get("dimension_weight", {})
    task_id = criteria_data.get("id", "Unknown")
    if not isinstance(dimension_weights, Mapping):
        dimension_weights = {}

    criterions_data = criteria_data.get("criterions")
    if not isinstance(criterions_data, Mapping) or not criterions_data:
        raise ValueError(
            f"ID: {task_id} - Missing required criterions data, cannot calculate weighted scores"
        )

    criterion_weights: dict[str, dict[str, Any]] = {}
    for dimension, criterions in criterions_data.items():
        if not isinstance(dimension, str) or not isinstance(criterions, list):
            continue
        criterion_weights[dimension] = {
            criterion["criterion"]: criterion["weight"]
            for criterion in criterions
            if isinstance(criterion, Mapping)
            and isinstance(criterion.get("criterion"), str)
            and "weight" in criterion
        }

    unmatched_criteria: set[str] = set()
    for dimension, scores_list in llm_output_json.items():
        if not isinstance(dimension, str) or not isinstance(scores_list, list):
            logger.warning(
                "ID: %s - Dimension %r in LLM output is not a list; skipping",
                task_id,
                dimension,
            )
            continue
        if dimension not in dimension_weights or dimension not in criterion_weights:
            logger.warning(
                "ID: %s - Dimension %r is absent from criteria weights/details; skipping",
                task_id,
                dimension,
            )
            continue

        dimension_map = criterion_weights[dimension]
        if not dimension_map:
            logger.warning("ID: %s - No criteria mapping for %r; skipping", task_id, dimension)
            continue

        target_weighted_sum = 0.0
        reference_weighted_sum = 0.0
        total_weight = 0.0
        article_2_score: float | None = None

        for score_item in scores_list:
            if not isinstance(score_item, Mapping):
                logger.warning(
                    "ID: %s - Non-mapping score item in %r; skipping", task_id, dimension
                )
                continue
            raw_criterion = score_item.get("criterion")
            criterion_text = raw_criterion.strip() if isinstance(raw_criterion, str) else None
            article_1_raw = score_item.get("article_1_score")
            article_2_raw = score_item.get("article_2_score")
            target_raw = score_item.get("target_score")
            if target_raw is not None and article_1_raw is None:
                article_1_raw = target_raw

            try:
                article_1_score = float(article_1_raw) if article_1_raw is not None else None
                article_2_score = float(article_2_raw) if article_2_raw is not None else None
            except (TypeError, ValueError):
                logger.warning(
                    "ID: %s - Invalid scores for criterion %r in %r; skipping",
                    task_id,
                    criterion_text,
                    dimension,
                )
                continue

            if not criterion_text or article_1_score is None:
                logger.warning(
                    "ID: %s - Missing criterion text or target score in %r; skipping",
                    task_id,
                    dimension,
                )
                continue

            weight = dimension_map.get(criterion_text)
            criterion_lower = criterion_text.lower()
            if weight is None:
                for key, value in dimension_map.items():
                    if key.lower() == criterion_lower:
                        weight = value
                        break
            if weight is None:
                for key, value in dimension_map.items():
                    if criterion_lower in key.lower() or key.lower() in criterion_lower:
                        weight = value
                        break
            if weight is None:
                unmatched_criteria.add(f"{dimension}:{criterion_text}")
                weight = sum(dimension_map.values()) / len(dimension_map)

            target_weighted_sum += article_1_score * weight
            total_weight += weight
            if article_2_score is not None:
                reference_weighted_sum += article_2_score * weight

        if total_weight > 0:
            target_average = target_weighted_sum / total_weight
            # Official code keys this on the last parsed item. Preserve it: mixed
            # single/comparative output can therefore change the reference result.
            reference_average = (
                reference_weighted_sum / total_weight if article_2_score is not None else 0.0
            )
        else:
            target_average = 0.0
            reference_average = 0.0

        dimension_key = f"{dimension}_weighted_avg"
        results["target"]["dims"][dimension_key] = target_average
        results["reference"]["dims"][dimension_key] = reference_average
        dimension_weight = dimension_weights.get(dimension, 0)
        total_target_score += target_average * dimension_weight
        total_reference_score += reference_average * dimension_weight

    if unmatched_criteria:
        logger.warning(
            "ID: %s - %d criteria used average-weight fallback: %s",
            task_id,
            len(unmatched_criteria),
            unmatched_criteria,
        )
    results["target"]["total"] = total_target_score
    results["reference"]["total"] = total_reference_score
    return results


def normalize_comparative_score(target: float, reference: float) -> float:
    """Normalize one target/reference score pair with the official zero guard."""
    denominator = target + reference
    return target / denominator if denominator > 0 else 0.0


def normalize_weighted_scores(scores: Mapping[str, Any]) -> dict[str, float]:
    """Return official overall and per-dimension target shares."""
    target = scores["target"]
    reference = scores["reference"]
    normalized = {
        "race_overall": normalize_comparative_score(
            float(target["total"]), float(reference["total"])
        )
    }
    for dimension in DEEPRESEARCH_DIMENSIONS:
        key = f"{dimension}_weighted_avg"
        target_score = float(target["dims"].get(key, 0.0))
        reference_score = float(reference["dims"].get(key, 0.0))
        normalized[f"race_{dimension}"] = normalize_comparative_score(target_score, reference_score)
    return normalized


def clean_urls(input_text: str) -> str:
    """Strip text-fragment suffixes from inline Markdown citation URLs."""
    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    def replace(match: re.Match[str]) -> str:
        title, url = match.groups()
        return f"[{title}]({clean_citation_url(url)})"

    return pattern.sub(replace, input_text)


def clean_citation_url(url: str) -> str:
    """Apply the official ``#:~:text=`` URL cleanup to a raw URL."""
    return url.strip().strip("<>").split("#:~:text=", maxsplit=1)[0]


def remove_urls(input_text: str) -> str:
    """Remove URL targets from Markdown citations while retaining their titles."""
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"[\1]", input_text)


def clean_escape(input_text: str) -> str:
    """Remove the four illegal JSON escapes handled by the official extractor."""
    for escaped, replacement in ((r"\>", ">"), (r"\<", "<"), (r"\+", "+"), (r"\~", "~")):
        input_text = input_text.replace(escaped, replacement)
    return input_text


def group_citations_by_url(
    citations: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group extracted citation triplets by their cleaned URL."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for citation in citations:
        url = citation.get("url")
        fact = citation.get("fact")
        if not isinstance(url, str) or not url or not isinstance(fact, str) or not fact:
            continue
        cleaned_url = clean_citation_url(url)
        if not cleaned_url:
            continue
        groups.setdefault(cleaned_url, []).append(dict(citation, url=cleaned_url))
    return groups


def _json_without_fences(raw: str) -> Any:
    return json.loads(raw.replace("```json", "").replace("```", ""))


def _try_parse_dedup_indices(raw: str, group_size: int) -> list[int] | None:
    try:
        parsed = cast(list[int], _json_without_fences(raw))
    except json.JSONDecodeError:
        return None
    if not parsed or 0 in parsed or len(parsed) > group_size:
        return list(range(1, group_size + 1))
    if any(index > group_size or index <= -group_size for index in parsed):
        # The official list access raises IndexError here; retain every fact instead.
        return list(range(1, group_size + 1))
    return parsed


def parse_dedup_indices(raw: str, group_size: int) -> list[int]:
    """Parse official one-based dedup indices, falling back to every statement."""
    parsed = _try_parse_dedup_indices(raw, group_size)
    return parsed if parsed is not None else list(range(1, group_size + 1))


def calculate_fact_statistics(results: Sequence[str], n_tasks: int) -> dict[str, float]:
    """Calculate official FACT statistics for citation-bearing tasks."""
    supported = sum(result == "supported" for result in results)
    evaluated = sum(result != "unknown" for result in results)
    return {
        "fact_citation_accuracy": supported / evaluated if evaluated else 0.0,
        "fact_avg_effective_citations": supported / n_tasks if n_tasks else 0.0,
        "fact_avg_citations": evaluated / n_tasks if n_tasks else 0.0,
    }


def _unknown_results(fact_count: int) -> list[dict[str, Any]]:
    return [{"idx": index, "result": "unknown"} for index in range(fact_count)]


def is_obvious_scrape_failure(page_text: str) -> bool:
    """Identify fetch-layer failures that cannot support any statement."""
    normalized = page_text.strip().lower()
    return not normalized or normalized.startswith(
        (
            "error:",
            "error fetching webpage:",
            "scrape failed:",
            "no content extracted from webpage.",
        )
    )


def scrape_failure_unknown_results(page_text: str, fact_count: int) -> list[dict[str, Any]] | None:
    """Short-circuit obvious fetch errors; valid pages still go to the judge."""
    return _unknown_results(fact_count) if is_obvious_scrape_failure(page_text) else None


async def fetch_crawl4ai_page(url: str) -> str:
    """Call crawl4ai's underlying fetch logic without the registered-tool truncation."""
    sanitized_url = url.strip()
    if not sanitized_url:
        return "Error: Empty URL."
    if urlsplit(sanitized_url).scheme.lower() not in {"http", "https"}:
        return "Error: Only http(s) URLs are supported."

    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError:
        raise RuntimeError(
            "DeepResearch Bench FACT scoring requires crawl4ai. "
            "Install it with `pip install 'olmo-eval[crawl4ai]'`."
        ) from None

    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(sanitized_url)
    except Exception as exc:
        logger.exception("crawl4ai FACT fetch failed for %r", sanitized_url)
        return f"Error fetching webpage: {exc}"

    if not getattr(result, "success", False):
        error_message = getattr(result, "error_message", None)
        if error_message:
            return f"Error fetching webpage: {error_message}"
        status_code = getattr(result, "status_code", None)
        if status_code is not None:
            return f"Error fetching webpage: HTTP {status_code}"
        return "Error fetching webpage: unknown error"

    markdown = getattr(result, "markdown", None)
    text = getattr(markdown, "raw_markdown", None) or str(markdown or "")
    if not text.strip():
        return "No content extracted from webpage."
    if len(text) > DEEPRESEARCH_PAGE_CONTENT_LIMIT:
        text = text[:DEEPRESEARCH_PAGE_CONTENT_LIMIT]
    return text


def _build_judge_fn(default_spec: str, scorer_name: str, default_effort: str) -> JudgeFn:
    spec = os.getenv("OLMO_EVAL_JUDGE", default_spec)
    model, separator, effort = spec.partition(":")
    return build_openai_judge_fn(
        model=model,
        temperature=0.0,
        max_tokens=DEEPRESEARCH_JUDGE_MAX_TOKENS,
        scorer_name=scorer_name,
        reasoning_effort=(effort if separator else default_effort),
    )


def _fact_scoring_disabled() -> bool:
    """Whether FACT scoring is switched off for this run.

    FACT crawls every cited URL and judges each extracted statement, so it dominates the cost of
    scoring this benchmark. RACE alone is often what is wanted, and there was previously no way
    to ask for it.
    """
    value = os.environ.get("DEEPRESEARCH_SKIP_FACT", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_deepresearch_race_judge_fn() -> JudgeFn:
    """Build the RACE judge with medium effort unless a spec suffix overrides it."""
    return _build_judge_fn(DEEPRESEARCH_RACE_DEFAULT_JUDGE_SPEC, "DeepResearchBenchRACE", "medium")


def build_deepresearch_fact_judge_fn() -> JudgeFn:
    """Build the FACT judge with low effort unless a spec suffix overrides it."""
    return _build_judge_fn(DEEPRESEARCH_FACT_DEFAULT_JUDGE_SPEC, "DeepResearchBenchFACT", "low")


@dataclass(frozen=True)
class DeepResearchBenchScorer(Scorer):
    """Placeholder scorer; the task precomputes all RACE and FACT channels."""

    name: str = "deepresearch_bench"
    score_key: str = "race_overall"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return float((output.metadata or {}).get(self.score_key, 0.0))


class DeepResearchMetricBase(Metric, ABC):
    scorer: type[Scorer] = DeepResearchBenchScorer

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class DeepResearchMeanMetric(DeepResearchMetricBase):
    """Arithmetic mean of a precomputed per-task score."""

    name: str = "race_overall"
    scorer: type[Scorer] = DeepResearchBenchScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(response.scores.get(self.name, 0.0) for response in responses) / len(responses)

    def pairwise_display_format(self) -> str:
        return "percentage" if self.name.startswith("race_") else "raw"

    def pairwise_unit(self) -> str:
        return "proportion" if self.name.startswith("race_") else self.name


@dataclass(frozen=True)
class FactCitationAccuracyMetric(DeepResearchMetricBase):
    """Corpus-level supported / non-unknown FACT accuracy."""

    name: str = "fact_citation_accuracy"
    scorer: type[Scorer] = DeepResearchBenchScorer

    def compute(self, responses: Sequence[Response]) -> float:
        supported = sum(
            response.scores.get("fact_avg_effective_citations", 0.0) for response in responses
        )
        evaluated = sum(response.scores.get("fact_avg_citations", 0.0) for response in responses)
        return supported / evaluated if evaluated else 0.0

    def pairwise_display_format(self) -> str:
        return "percentage"

    def pairwise_unit(self) -> str:
        return "proportion"


@dataclass(frozen=True)
class FactAverageCitationMetric(DeepResearchMetricBase):
    """Average a FACT count over responses with at least one extracted citation."""

    name: str = "fact_avg_citations"
    scorer: type[Scorer] = DeepResearchBenchScorer

    def compute(self, responses: Sequence[Response]) -> float:
        numerator = sum(response.scores.get(self.name, 0.0) for response in responses)
        citation_bearing = sum(
            response.scores.get("fact_has_citations", 0.0) for response in responses
        )
        return numerator / citation_bearing if citation_bearing else 0.0

    def pairwise_display_format(self) -> str:
        return "raw"

    def pairwise_unit(self) -> str:
        return self.name


RACE_OVERALL_METRIC = DeepResearchMeanMetric(name="race_overall")
DEEPRESEARCH_METRICS = (
    RACE_OVERALL_METRIC,
    *(DeepResearchMeanMetric(name=f"race_{dimension}") for dimension in DEEPRESEARCH_DIMENSIONS),
    FactCitationAccuracyMetric(),
    FactAverageCitationMetric(name="fact_avg_effective_citations"),
    FactAverageCitationMetric(name="fact_avg_citations"),
)


def strip_think_block(text: str) -> str:
    """Strip a reasoning prefix using the intentionally lossy ResearchQA pattern.

    Only text after the first ``</think>`` is retained, so a mid-report closing tag
    discards the report prefix as well. This pre-existing behavior matches ResearchQA.
    """
    think_end = text.find("</think>")
    if think_end >= 0:
        text = text[think_end + len("</think>") :]
    return text.strip()


_SCAFFOLD_IDENTIFIER = r"[A-Za-z_][\w.:-]*"
_SCAFFOLD_QUOTED_VALUE = r"""(?:"[^"<>\r\n]*"|'[^'<>\r\n]*')"""
_SCAFFOLD_VALUE = rf"(?:{_SCAFFOLD_IDENTIFIER}|{_SCAFFOLD_QUOTED_VALUE})"
_SCAFFOLD_OPEN_WITH_ATTRIBUTES_RE = re.compile(
    rf"<(?:tool_call|tool_response|function|parameter)(?:"
    rf"[ \t]*=[ \t]*{_SCAFFOLD_VALUE}|"
    rf"(?:[ \t]+{_SCAFFOLD_IDENTIFIER}[ \t]*=[ \t]*{_SCAFFOLD_VALUE})+"
    rf")[ \t]*>",
    flags=re.IGNORECASE,
)
_SCAFFOLD_TAG_RE = re.compile(
    r"</?(?:tool_call|tool_response)[ \t]*>|</(?:function|parameter)[ \t]*>",
    flags=re.IGNORECASE,
)
_SPECIAL_TOKEN_PATTERN = r"<(?:\|[A-Za-z_][\w.:-]*\|?|[A-Za-z_][\w.:-]*\|)>"
_SPECIAL_TOKEN_RE = re.compile(_SPECIAL_TOKEN_PATTERN, flags=re.IGNORECASE)
_CHANNEL_HEADER_RE = re.compile(
    rf"<\|channel\|?>[ \t]*(?P<channel>analysis|commentary|final|thought|tool)"
    rf"[ \t\r\n]*(?={_SPECIAL_TOKEN_PATTERN})",
    flags=re.IGNORECASE,
)
_ASSISTANT_TURN_PREFIX_RE = re.compile(
    r"(?:<\|start\|>|<start\|>)[ \t]*assistant\b", flags=re.IGNORECASE
)
_MARKDOWN_CODE_SPAN_RE = re.compile(r"```[\s\S]*?```|`[^`\n]*`")
_PAIRED_BARE_SCAFFOLD_TAGS = ("function", "parameter")
_BARE_SCAFFOLD_OPEN_RES = {
    tag: re.compile(rf"<{tag}[ \t]*>", flags=re.IGNORECASE) for tag in _PAIRED_BARE_SCAFFOLD_TAGS
}
_BARE_SCAFFOLD_CLOSE_RES = {
    tag: re.compile(rf"</{tag}[ \t]*>", flags=re.IGNORECASE) for tag in _PAIRED_BARE_SCAFFOLD_TAGS
}
_REASONING_CHANNELS = frozenset({"analysis", "commentary", "thought"})


def _partition_markdown_code(text: str) -> list[tuple[bool, str]]:
    """Return ``(is_code, text)`` parts without modifying protected Markdown code."""
    parts: list[tuple[bool, str]] = []
    cursor = 0
    for match in _MARKDOWN_CODE_SPAN_RE.finditer(text):
        if cursor < match.start():
            parts.append((False, text[cursor : match.start()]))
        parts.append((True, match.group()))
        cursor = match.end()
    if cursor < len(text):
        parts.append((False, text[cursor:]))
    return parts


def _retain_final_channel(parts: Sequence[tuple[bool, str]]) -> list[tuple[bool, str]]:
    """Keep pre-header and final prose when reasoning and final channels coexist."""
    channel_names = {
        match.group("channel").lower()
        for is_code, part in parts
        if not is_code
        for match in _CHANNEL_HEADER_RE.finditer(part)
    }
    if "final" not in channel_names or channel_names.isdisjoint(_REASONING_CHANNELS):
        return list(parts)

    retained: list[tuple[bool, str]] = []
    active_channel: str | None = None
    for is_code, part in parts:
        if is_code:
            retained.append((True, part))
            continue

        cursor = 0
        retained_chunks: list[str] = []
        for match in _CHANNEL_HEADER_RE.finditer(part):
            if active_channel in (None, "final"):
                retained_chunks.append(part[cursor : match.start()])
            active_channel = match.group("channel").lower()
            cursor = match.end()
        if active_channel in (None, "final"):
            retained_chunks.append(part[cursor:])
        retained.append((False, "".join(retained_chunks)))
    return retained


def _clean_report_segment(text: str, paired_bare_tags: frozenset[str]) -> str:
    """Remove scaffold tokens from a non-code Markdown segment in linear passes."""
    text = _ASSISTANT_TURN_PREFIX_RE.sub(" ", text)
    text = _CHANNEL_HEADER_RE.sub(" ", text)
    text = _SPECIAL_TOKEN_RE.sub(" ", text)
    for tag in paired_bare_tags:
        text = _BARE_SCAFFOLD_OPEN_RES[tag].sub("", text)
    text = _SCAFFOLD_OPEN_WITH_ATTRIBUTES_RE.sub("", text)
    return _SCAFFOLD_TAG_RE.sub("", text)


def _clean_generated_report(text: str) -> str:
    """Strip reasoning and tool-call scaffold while preserving report prose and code.

    Explicit analysis/commentary/thought content is dropped only when an explicit final
    channel is also present; ordinary report content before the first channel header and
    final-channel content are retained. This intentionally limited parser infers
    boundaries from channel headers rather than assigning semantics to every possible
    special token; protected Markdown code spans are always retained verbatim.
    """
    text = strip_think_block(text)
    parts = _retain_final_channel(_partition_markdown_code(text))
    paired_bare_tags = frozenset(
        tag
        for tag, close_re in _BARE_SCAFFOLD_CLOSE_RES.items()
        if any(not is_code and close_re.search(part) for is_code, part in parts)
    )
    return "".join(
        part if is_code else _clean_report_segment(part, paired_bare_tags)
        for is_code, part in parts
    ).strip()


def _validated_criteria(doc: Mapping[str, Any]) -> dict[str, Any] | None:
    criteria = doc.get("criteria")
    if not isinstance(criteria, Mapping):
        return None
    dimension_weight = criteria.get("dimension_weight")
    criterions = criteria.get("criterions")
    if not isinstance(dimension_weight, Mapping) or not isinstance(criterions, Mapping):
        return None
    return {
        "dimension_weight": dict(dimension_weight),
        "criterions": {key: value for key, value in criterions.items()},
    }


@register("deepresearch_bench")
class DeepResearchBench(Task):
    """DeepResearch Bench RACE + FACT evaluation."""

    split = Split.TEST
    data_source = DataSource(path=DEEPRESEARCH_BENCH_REPO, subset="default", split="test")
    metrics = DEEPRESEARCH_METRICS
    primary_metric = RACE_OVERALL_METRIC
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16_384)
    required_secrets = ("OPENAI_API_KEY",)

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._scrape_cache: dict[str, str] = {}

    @property
    def instances(self) -> Iterator[Instance]:
        split = (
            self.config.data_source.split
            if isinstance(self.config.data_source, DataSource)
            else None
        )
        yield from self._load_instances_cached(split=split)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        language = doc.get("language")
        if language not in {"en", "zh"}:
            return None
        prompt = doc.get("prompt")
        reference_article = doc.get("reference_article")
        criteria = _validated_criteria(doc)
        if (
            not isinstance(prompt, str)
            or not prompt
            or not isinstance(reference_article, str)
            or not reference_article
            or criteria is None
        ):
            return None

        identifier = doc.get("id", index + 1)
        generation_instruction = DEEPRESEARCH_GENERATION_INSTRUCTIONS[language]
        return Instance(
            question=f"{prompt}\n\n{generation_instruction}",
            metadata={
                "id": identifier,
                "case_id": identifier,
                "language": language,
                "topic": doc.get("topic", ""),
                "prompt": prompt,
                "criteria": criteria,
                "reference_article": reference_article,
                "index": index,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )

    def extract_answer(self, output: LMOutput) -> str:
        return _clean_generated_report(output.text)

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Run RACE and FACT, retaining every instance in aggregate denominators."""
        del context
        self._extract_answers(responses)
        race_judge = build_deepresearch_race_judge_fn()
        fact_judge = build_deepresearch_fact_judge_fn()

        race_failures = 0
        fact_skipped = 0
        fact_failures = 0
        for response in responses:
            output = response.outputs[0] if response.outputs else None
            answer = ""
            if output is not None:
                answer = (
                    output.extracted_answer
                    if isinstance(output.extracted_answer, str)
                    else output.text
                )

            if answer:
                race_scores, race_details, race_failed = await self._score_race(
                    response.instance, answer, race_judge
                )
                if _fact_scoring_disabled():
                    # Skipped, not zero. Leaving silent zeros here would be indistinguishable
                    # from "ran and scored nothing", which is exactly the confusion that made an
                    # earlier all-zero metrics column unreadable.
                    fact_scores = self._zero_fact_scores()
                    fact_details = [{"fact_scoring": "skipped"}]
                    fact_failed = False
                    fact_skipped += 1
                else:
                    try:
                        fact_scores, fact_details = await self._score_fact(
                            response.instance, answer, fact_judge
                        )
                        fact_failed = False
                    except Exception as exc:
                        logger.warning(
                            "DeepResearch Bench FACT scoring failed for id %r: %s",
                            response.instance.metadata.get("id"),
                            exc,
                        )
                        fact_scores = self._zero_fact_scores()
                        fact_details = []
                        fact_failed = True
            else:
                race_scores = self._zero_race_scores()
                race_details = None
                race_failed = False
                fact_scores = self._zero_fact_scores()
                fact_details = []
                fact_failed = False

            race_failures += int(race_failed)
            fact_failures += int(fact_failed)
            response.scores.update(race_scores)
            response.scores.update(fact_scores)
            if output is not None:
                if output.metadata is None:
                    output.metadata = {}
                output.metadata["deepresearch_race"] = race_details
                output.metadata["deepresearch_fact"] = fact_details
                for metric_name, score in response.scores.items():
                    output.metadata[f"score:{metric_name}"] = score

        if fact_skipped:
            logger.warning(
                "DeepResearch Bench FACT scoring skipped for %d instance(s) because "
                "DEEPRESEARCH_SKIP_FACT is set; every fact_* metric below is a placeholder "
                "and must not be read as a score of zero.",
                fact_skipped,
            )

        if race_failures:
            logger.warning(
                "DeepResearch Bench RACE judge failed for %d instance(s) after %d "
                "attempts; assigned zero and retained them in the denominator.",
                race_failures,
                DEEPRESEARCH_JUDGE_ATTEMPTS,
            )
        if fact_failures:
            logger.warning(
                "DeepResearch Bench FACT scoring failed for %d instance(s); assigned zero "
                "and retained them in the denominator.",
                fact_failures,
            )
        return responses

    async def _score_race(
        self,
        instance: Instance,
        answer: str,
        judge_fn: JudgeFn,
    ) -> tuple[dict[str, float], dict[str, Any] | None, bool]:
        metadata = instance.metadata
        criteria = metadata["criteria"]
        language = metadata["language"]
        prompt_template = RACE_SCORE_PROMPT_ZH if language == "zh" else RACE_SCORE_PROMPT_EN
        judge_prompt = prompt_template.format(
            task_prompt=metadata["prompt"],
            article_1=answer,
            article_2=metadata["reference_article"],
            criteria_list=format_criteria_list(criteria),
        )

        for attempt in range(DEEPRESEARCH_JUDGE_ATTEMPTS):
            try:
                raw = await judge_fn(judge_prompt)
                extracted_json = extract_json_from_markdown(raw)
                if extracted_json is None:
                    raise ValueError("Failed to extract JSON from RACE judge response")
                parsed = json.loads(extracted_json)
                if not isinstance(parsed, dict):
                    raise TypeError("RACE judge response must be a JSON object")
                missing = [
                    dimension for dimension in DEEPRESEARCH_DIMENSIONS if dimension not in parsed
                ]
                if missing:
                    raise ValueError(f"RACE judge response is missing dimensions: {missing}")
                weighted = calculate_weighted_scores(parsed, criteria, language)
                normalized = normalize_weighted_scores(weighted)
                return normalized, {"judge_scores": parsed, "weighted_scores": weighted}, False
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                logger.warning(
                    "DeepResearch Bench RACE attempt %d/%d failed for id %r: %s",
                    attempt + 1,
                    DEEPRESEARCH_JUDGE_ATTEMPTS,
                    metadata.get("id"),
                    exc,
                )
            except Exception as exc:
                logger.warning(
                    "DeepResearch Bench RACE judge call %d/%d failed for id %r: %s",
                    attempt + 1,
                    DEEPRESEARCH_JUDGE_ATTEMPTS,
                    metadata.get("id"),
                    exc,
                )
            if attempt < DEEPRESEARCH_JUDGE_ATTEMPTS - 1:
                await asyncio.sleep(2**attempt)
        return self._zero_race_scores(), None, True

    async def _score_fact(
        self,
        instance: Instance,
        answer: str,
        judge_fn: JudgeFn,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        language = instance.metadata["language"]
        citations = await self._extract_citations(answer, language, judge_fn)
        fact_has_citations = float(bool(citations))
        groups = group_citations_by_url(citations)
        validation_details: list[dict[str, Any]] = []

        for url, group in groups.items():
            facts = [str(citation["fact"]) for citation in group]
            if len(facts) > 1:
                facts = await self._deduplicate_facts(facts, language, judge_fn)

            page_text = await self._fetch_page(url)
            validation_results = scrape_failure_unknown_results(page_text, len(facts))
            if validation_results is None:
                validation_results = await self._validate_facts(
                    facts, page_text, language, judge_fn
                )
            for position, validation in enumerate(validation_results):
                returned_index = validation["idx"]
                fact_index = (
                    returned_index
                    if isinstance(returned_index, int) and 0 <= returned_index < len(facts)
                    else position
                )
                validation_details.append(
                    {
                        "url": url,
                        "fact": facts[fact_index],
                        "result": validation["result"],
                    }
                )

        statuses = [detail["result"] for detail in validation_details]
        statistics = calculate_fact_statistics(statuses, n_tasks=1)
        statistics["fact_has_citations"] = fact_has_citations
        return statistics, validation_details

    async def _extract_citations(
        self, answer: str, language: str, judge_fn: JudgeFn
    ) -> list[dict[str, Any]]:
        template = FACT_EXTRACTION_PROMPT_ZH if language == "zh" else FACT_EXTRACTION_PROMPT_EN
        prompt = template.format(report_text=clean_urls(answer))
        for attempt in range(DEEPRESEARCH_JUDGE_ATTEMPTS):
            try:
                raw = clean_escape(await judge_fn(prompt))
                parsed = _json_without_fences(raw)
                if not isinstance(parsed, list):
                    raise TypeError("FACT extraction result must be a list")
                citations: list[dict[str, Any]] = []
                for item in parsed:
                    if not isinstance(item, Mapping):
                        raise TypeError("FACT extraction item must be an object")
                    fact = item.get("fact")
                    url = item.get("url")
                    if not isinstance(fact, str) or not isinstance(url, str):
                        raise TypeError("FACT extraction fact and url must be strings")
                    citations.append(
                        {
                            "fact": remove_urls(fact),
                            "ref_idx": item.get("ref_idx"),
                            "url": clean_citation_url(url),
                        }
                    )
                return citations
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    "FACT extraction attempt %d/%d failed: %s",
                    attempt + 1,
                    DEEPRESEARCH_JUDGE_ATTEMPTS,
                    exc,
                )
            except Exception as exc:
                logger.warning(
                    "FACT extraction judge call %d/%d failed: %s",
                    attempt + 1,
                    DEEPRESEARCH_JUDGE_ATTEMPTS,
                    exc,
                )
            if attempt < DEEPRESEARCH_JUDGE_ATTEMPTS - 1:
                await asyncio.sleep(2**attempt)
        return []

    async def _deduplicate_facts(
        self, facts: list[str], language: str, judge_fn: JudgeFn
    ) -> list[str]:
        statements = "\n".join(f"{index + 1}. {fact}" for index, fact in enumerate(facts))
        template = (
            FACT_DEDUPLICATION_PROMPT_ZH if language == "zh" else FACT_DEDUPLICATION_PROMPT_EN
        )
        prompt = template.format(statements=statements)
        indices: list[int] | None = None
        for attempt in range(DEEPRESEARCH_JUDGE_ATTEMPTS):
            try:
                indices = _try_parse_dedup_indices(await judge_fn(prompt), len(facts))
                if indices is not None:
                    break
                raise ValueError("FACT deduplication result must be valid JSON")
            except Exception as exc:
                logger.warning(
                    "FACT deduplication attempt %d/%d failed: %s",
                    attempt + 1,
                    DEEPRESEARCH_JUDGE_ATTEMPTS,
                    exc,
                )
            if attempt < DEEPRESEARCH_JUDGE_ATTEMPTS - 1:
                await asyncio.sleep(2**attempt)
        indices = indices or list(range(1, len(facts) + 1))
        return [facts[index - 1] for index in indices]

    async def _fetch_page(self, url: str) -> str:
        if url not in self._scrape_cache:
            page_text = ""
            for attempt in range(DEEPRESEARCH_SCRAPE_ATTEMPTS):
                page_text = await fetch_crawl4ai_page(url)
                if not is_obvious_scrape_failure(page_text):
                    break
                if attempt < DEEPRESEARCH_SCRAPE_ATTEMPTS - 1:
                    await asyncio.sleep(1)
            self._scrape_cache[url] = page_text
        return self._scrape_cache[url]

    async def _validate_facts(
        self,
        facts: list[str],
        page_text: str,
        language: str,
        judge_fn: JudgeFn,
    ) -> list[dict[str, Any]]:
        statements = "\n".join(f"{index + 1}. {fact}" for index, fact in enumerate(facts))
        template = FACT_VALIDATION_PROMPT_ZH if language == "zh" else FACT_VALIDATION_PROMPT_EN
        prompt = template.format(reference=page_text, statements=statements)
        for attempt in range(DEEPRESEARCH_JUDGE_ATTEMPTS):
            try:
                raw = await judge_fn(prompt)
                parsed = cast(list[dict[str, Any]], _json_without_fences(raw))
                for item in parsed:
                    if not isinstance(item, Mapping) or "idx" not in item or "result" not in item:
                        raise ValueError(
                            "FACT validation item must be an object containing idx and result"
                        )
                    item["idx"] -= 1
                assert len(parsed) == len(facts), "FACT validation result length mismatch"
                return parsed
            except (AssertionError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "FACT validation attempt %d/%d failed: %s",
                    attempt + 1,
                    DEEPRESEARCH_JUDGE_ATTEMPTS,
                    exc,
                )
            except Exception as exc:
                logger.warning("FACT validation judge call failed: %s", exc)
            if attempt < DEEPRESEARCH_JUDGE_ATTEMPTS - 1:
                await asyncio.sleep(2**attempt)
        return _unknown_results(len(facts))

    @staticmethod
    def _zero_race_scores() -> dict[str, float]:
        return {
            "race_overall": 0.0,
            **{f"race_{dimension}": 0.0 for dimension in DEEPRESEARCH_DIMENSIONS},
        }

    @staticmethod
    def _zero_fact_scores() -> dict[str, float]:
        return {
            "fact_citation_accuracy": 0.0,
            "fact_avg_effective_citations": 0.0,
            "fact_avg_citations": 0.0,
            "fact_has_citations": 0.0,
        }


register_variant(
    "deepresearch_bench",
    "en",
    data_source=DataSource(path=DEEPRESEARCH_BENCH_REPO, subset="en", split="test"),
)
