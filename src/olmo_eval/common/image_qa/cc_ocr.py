"""Per-sample scoring for CC-OCR, vendored from the official evaluator.

Source: ``Benchmarks/CC-OCR/evaluation/evaluator`` of
https://github.com/AlibabaResearch/AdvancedLiterateMachinery (MIT; the TEDS classes are
IBM's, Apache-2.0, and the KIE tree metric is Donut's, MIT). Behavior is preserved exactly,
with two mechanical changes:

* every function scores one sample and returns its sufficient statistics, so the harness can
  score responses as they arrive; the dataset-level reductions live in
  :mod:`olmo_eval.common.scorers.cc_ocr`;
* Levenshtein distance comes from ``rapidfuzz`` instead of ``nltk.edit_distance`` — the same
  integer, but ``nltk``'s pure-Python loop takes minutes on a full page and recent releases
  refuse inputs over 2,000 characters, which most document references exceed.

The three tracks' rules:

* **OCR** (multi-scene and multilingual): multiset overlap of basic units between the
  prediction and the reference — characters for Chinese, Japanese, Korean and Arabic sets,
  lower-cased words otherwise, and alphanumeric-only words on the multi-scene track.
* **Document parsing**: ``1 - normalized edit distance`` on whitespace-stripped text for
  documents, formulas and molecules; TEDS on the extracted ``<table>`` for tables.
* **KIE**: the response is parsed as JSON, values are width/space-normalized, and
  flattened ``(key path, value)`` fields are matched exactly (field-level F1); ``acc`` is
  Donut's normalized tree-edit-distance accuracy.
"""

from __future__ import annotations

import json
import re
from collections import Counter, deque
from typing import Any

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def edit_distance(a: Any, b: Any) -> int:
    """Levenshtein distance between two sequences (strings or token lists)."""
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(a, b)


def convert_to_halfwidth(text: str) -> str:
    halfwidth_chars = str.maketrans(
        "！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～",
        "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~",
    )
    return text.translate(halfwidth_chars)


# ---------------------------------------------------------------------------
# OCR tracks (``ocr_evaluator.py``)
# ---------------------------------------------------------------------------

#: Datasets scored on characters rather than words, besides any whose name contains "zh".
_CHARACTER_LEVEL_DATASETS = ("Arabic", "Japanese", "Korean")


def ocr_eval_config(track: str, dataset: str) -> dict[str, bool]:
    """The official per-dataset tokenization switches (``OcrEvaluator.evaluate``)."""
    is_word_level = not (dataset in _CHARACTER_LEVEL_DATASETS or "zh" in dataset)
    is_alphanum_only = "multi_scene_ocr" in track and is_word_level
    return {"word_level": is_word_level, "alphanum_only": is_alphanum_only, "lowercase": True}


def _token_normalize(token_text: str, is_lower: bool, is_alphanum_only: bool) -> str:
    if is_lower:
        token_text = token_text.lower()
    if is_alphanum_only:
        token_text = re.sub("[^A-Za-z0-9]+", "", token_text)
    return token_text


def text_normalize_and_tokenize(
    text: str, is_keep_blank: bool = True, is_lower: bool = True, is_alphanum_only: bool = False
) -> list[str]:
    text = text.replace("\t", " ").replace("\n", " ").replace("###", "").replace("***", "")
    text = re.sub(r"\s+", " ", text)
    if not is_keep_blank:
        text = text.replace(" ", "")
    text_tokens = text.split(" ") if is_keep_blank else list(text)
    normalized = [_token_normalize(t, is_lower, is_alphanum_only) for t in text_tokens]
    return [x for x in normalized if len(x) > 0]


def score_ocr_sample(prediction: str, answer: str, track: str, dataset: str) -> dict[str, float]:
    """Multiset unit overlap for one image, plus its per-image recall/precision/f1."""
    config = ocr_eval_config(track, dataset)
    args = (config["word_level"], config["lowercase"], config["alphanum_only"])
    preds = text_normalize_and_tokenize(str(prediction).strip(), *args)
    gts = text_normalize_and_tokenize(str(answer).strip(), *args)

    pred_counter = Counter(preds)
    right_num = sum(min(count, pred_counter.get(token, 0)) for token, count in Counter(gts).items())
    recall = right_num / (len(gts) + 1e-9)
    precision = right_num / (len(preds) + 1e-9)
    return {
        "right_num": right_num,
        "gt_num": len(gts),
        "pred_num": len(preds),
        "recall": recall,
        "precision": precision,
        "f1": 2 * recall * precision / (recall + precision + 1e-9),
    }


# ---------------------------------------------------------------------------
# Document parsing (``doc_parsing_evaluator.py``)
# ---------------------------------------------------------------------------

_LATEX_PREAMBLE_PATTERNS = [
    r"\\documentclass\{.*?\}",
    r"\\usepackage\[.*?\]\{.*?\}",
    r"\\usepackage\{.*?\}",
    r"\\geometry\{.*?\}",
    r"\\begin\{document\}",
    r"\\end\{document\}",
    r"\\noindent",
]


def _strip_code_fence(pred: str, language: str) -> str:
    match = re.search(rf"```{language}(.+?)```", pred, re.DOTALL)
    if match is not None:
        return match.group(1)
    if f"```{language}" in pred:
        return pred.split(f"```{language}")[1]
    return pred


def _similarity(pred: str, gt: str) -> float:
    return 1 - edit_distance(pred, gt) / max(len(pred), len(gt))


def score_doc_sample(prediction: str, answer: str) -> float:
    pred = prediction
    for pattern in _LATEX_PREAMBLE_PATTERNS:
        pred = re.sub(pattern, "", pred)
    pred = _strip_code_fence(pred, "latex")
    pred = pred.replace(" ", "").replace("\n", "")
    gt = answer.replace(" ", "").replace("\n", "")
    return _similarity(pred, gt)


def score_formula_sample(prediction: str, answer: str, op: str) -> float:
    if op == "formula":
        pred = prediction.replace("\n", " ").replace("```latex", "").replace("```", "")
        pred = pred.replace("\t", " ").replace(" ", "")
    elif op == "molecular":
        pred = prediction.replace("\n", "").replace(" ", "")
        pred = pred.replace("<smiles>", "").replace("</smiles>", "")
    else:
        raise ValueError(f"doc parsing unsupported op: {op}")
    return _similarity(pred, answer.replace(" ", ""))


def extract_and_clean_tables(text: str) -> str:
    if "</table>" not in text:
        text += "</table>"
    tables = re.findall(r"<table.*?>.*?</table>", text, re.DOTALL)

    clean_tables = []
    for table in tables:
        table_content = re.sub(r"<table.*?>", "<table>", table)
        table_content = re.sub(r">\s+<", "><", table_content)
        table_content = re.sub(
            r">(.*?)<",
            lambda m: ">" + m.group(1).replace("\n", "").replace(" ", "") + "<",
            table_content,
            flags=re.DOTALL,
        )
        clean_tables.append(table_content.replace("\n", "").strip())
    return "".join(clean_tables)


def _teds(pred_html: str, true_html: str) -> float:
    """Tree-edit-distance based similarity (IBM PubTabNet ``TEDS``, content included)."""
    from apted import APTED, Config
    from apted.helpers import Tree
    from lxml import html

    class TableTree(Tree):
        def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
            self.tag = tag
            self.colspan = colspan
            self.rowspan = rowspan
            self.content = content
            self.children = list(children)

        def bracket(self):
            if self.tag == "td":
                result = (
                    f'"tag": {self.tag}, "colspan": {self.colspan:d}, '
                    f'"rowspan": {self.rowspan:d}, "text": {self.content}'
                )
            else:
                result = f'"tag": {self.tag}'
            for child in self.children:
                result += child.bracket()
            return f"{{{result}}}"

    class CustomConfig(Config):
        def rename(self, node1, node2):
            if (
                (node1.tag != node2.tag)
                or (node1.colspan != node2.colspan)
                or (node1.rowspan != node2.rowspan)
            ):
                return 1.0
            if node1.tag == "td" and (node1.content or node2.content):
                return edit_distance(node1.content, node2.content) / max(
                    len(node1.content), len(node2.content)
                )
            return 0.0

    def tokenize(node, tokens: list[str]) -> None:
        tokens.append(f"<{node.tag}>")
        if node.text is not None:
            tokens.extend(node.text)
        for child in node.getchildren():
            tokenize(child, tokens)
        if node.tag != "unk":
            tokens.append(f"</{node.tag}>")
        if node.tag != "td" and node.tail is not None:
            tokens.extend(node.tail)

    def load_html_tree(node, parent=None):
        if node.tag == "td":
            tokens: list[str] = []
            tokenize(node, tokens)
            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                tokens[1:-1],
                *deque(),
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for child in node.getchildren():
                load_html_tree(child, new_node)
        return new_node if parent is None else None

    if (not pred_html) or (not true_html):
        return 0.0
    parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
    pred = html.fromstring(pred_html, parser=parser)
    true = html.fromstring(true_html, parser=parser)
    if not (pred.xpath("body/table") and true.xpath("body/table")):
        return 0.0
    pred = pred.xpath("body/table")[0]
    true = true.xpath("body/table")[0]
    n_nodes = max(len(pred.xpath(".//*")), len(true.xpath(".//*")))
    distance = APTED(
        load_html_tree(pred), load_html_tree(true), CustomConfig()
    ).compute_edit_distance()
    return 1.0 - (float(distance) / n_nodes)


def score_table_sample(prediction: str, answer: str) -> float:
    pred = _strip_code_fence(prediction, "html")
    pred = convert_to_halfwidth(extract_and_clean_tables(pred))
    gt = convert_to_halfwidth(extract_and_clean_tables(answer))
    return _teds(f"<html><body>{pred}</body></html>", f"<html><body>{gt}</body></html>")


def score_doc_parsing_sample(prediction: str, answer: str, op: str) -> float:
    """One document-parsing sample; ``op`` is the sub-track (doc/table/formula/molecular)."""
    if op == "doc":
        return score_doc_sample(prediction, answer)
    if op == "table":
        return score_table_sample(prediction, answer)
    return score_formula_sample(prediction, answer, op)


# ---------------------------------------------------------------------------
# KIE (``kie_evaluator.py``; Donut's JSON metrics)
# ---------------------------------------------------------------------------


def post_process_to_json(response_text: str) -> Any:
    """The JSON payload of a response (fenced or bare), or ``None`` if it does not parse."""
    if "```json" in response_text:
        match = re.search(r"```json(.*?)```", response_text, re.DOTALL)
        if match is None:  # an unclosed fence is not repaired
            return None
        json_str = match.group(1).strip().replace("\n", "")
    else:
        json_str = response_text.strip().replace("\n", "")
    try:
        return json.loads(json_str)
    except Exception:
        return None


def fullwidth_to_halfwidth(text: str) -> str:
    result = ""
    for char in text:
        code_point = ord(char)
        if code_point == 0x3000:
            code_point = 0x0020
        elif 0xFF01 <= code_point <= 0xFF5E:
            code_point -= 0xFEE0
        result += chr(code_point)
    return result.replace("、", ",")


def remove_unnecessary_spaces(text: str) -> str:
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[a-zA-Z0-9])", "", text)
    text = re.sub(r"(?<=[a-zA-Z0-9])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<![0-9])\s*([,.!?:;])\s*", r"\1 ", text)
    text = re.sub(r"(?<=[0-9])(?=[a-zA-Z])", " ", text)
    text = re.sub(r"(?<=[a-zA-Z])(?=[0-9])", " ", text)
    return re.sub(r"\s+", " ", text)


def _normalize_value(text: str) -> str:
    return remove_unnecessary_spaces(fullwidth_to_halfwidth(str(text)))


def normalize_values_of_nested_dict(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: normalize_values_of_nested_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [normalize_values_of_nested_dict(x) if isinstance(x, dict) else x for x in d]
    if isinstance(d, str):
        return _normalize_value(d)
    return d


def normalize_dict(data: Any) -> Any:
    """Sort keys and wrap scalar values in lists, dropping empty ones (Donut)."""
    if isinstance(data, dict):
        new_data: Any = {}
        for key in sorted(data.keys(), key=lambda k: (len(k), k)):
            value = normalize_dict(data[key])
            if value:
                if not isinstance(value, list):
                    value = [value]
                new_data[key] = value
    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            new_data = []
            for item in data:
                item = normalize_dict(item)
                if item:
                    new_data.append(item)
        else:
            new_data = [
                str(item).strip()
                for item in data
                if type(item) in {str, int, float} and str(item).strip()
            ]
    else:
        new_data = [str(data).strip()]
    return new_data


def flatten(data: Any) -> list[tuple[str, Any]]:
    """Nested dict -> list of ``("a.b", value)`` fields."""
    flat: list[tuple[str, Any]] = []

    def _flatten(value: Any, key: str = "") -> None:
        if type(value) is dict:
            for child_key, child_value in value.items():
                _flatten(child_value, f"{key}.{child_key}" if key else child_key)
        elif type(value) is list:
            for value_item in value:
                _flatten(value_item, key)
        else:
            flat.append((key, value))

    _flatten(data)
    return flat


def _field_counts(pred: Any, answer: Any) -> tuple[int, int]:
    """``(true positives, false negatives + false positives)`` over flattened fields."""
    pred_fields, answer_fields = flatten(normalize_dict(pred)), flatten(normalize_dict(answer))
    tp = fn_or_fp = 0
    for field in pred_fields:
        if field in answer_fields:
            tp += 1
            answer_fields.remove(field)
        else:
            fn_or_fp += 1
    return tp, fn_or_fp + len(answer_fields)


def _tree_accuracy(pred: Any, answer: Any) -> float:
    """Donut's normalized tree-edit-distance accuracy, ``max(0, 1 - nTED)``."""
    import zss
    from zss import Node

    def construct_tree(data: Any, node_name: str | None = None) -> Node:
        node = Node("<root>" if node_name is None else node_name)
        if isinstance(data, dict):
            for key, value in data.items():
                node.addkid(construct_tree(value, key))
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                for item in data:
                    node.addkid(construct_tree(item, "<subtree>"))
            else:
                for item in data:
                    node.addkid(Node(f"<leaf>{item}"))
        else:
            raise ValueError(f"cannot build a tree from {data!r} under {node_name!r}")
        return node

    def update_cost(node1: Node, node2: Node) -> int:
        label1, label2 = node1.label, node2.label
        leaf1, leaf2 = "<leaf>" in label1, "<leaf>" in label2
        if leaf1 and leaf2:
            return edit_distance(label1.replace("<leaf>", ""), label2.replace("<leaf>", ""))
        if not leaf1 and leaf2:
            return 1 + len(label2.replace("<leaf>", ""))
        if leaf1 and not leaf2:
            return 1 + len(label1.replace("<leaf>", ""))
        return int(label1 != label2)

    def insert_and_remove_cost(node: Node) -> int:
        label = node.label
        return len(label.replace("<leaf>", "")) if "<leaf>" in label else 1

    def distance(a: Node, b: Node) -> float:
        return zss.distance(
            a,
            b,
            get_children=zss.Node.get_children,
            insert_cost=insert_and_remove_cost,
            remove_cost=insert_and_remove_cost,
            update_cost=update_cost,
            return_operations=False,
        )

    answer_tree = construct_tree(normalize_dict(answer))
    pred_tree = construct_tree(normalize_dict(pred))
    empty_tree = construct_tree(normalize_dict({}))
    return max(0, 1 - distance(pred_tree, answer_tree) / distance(empty_tree, answer_tree))


def score_kie_sample(prediction: str, answer: str | dict) -> dict[str, float]:
    """Field counts and tree accuracy for one KIE response.

    A response that does not parse as JSON is scored as an empty extraction, which is what
    the official evaluator's bookkeeping amounts to.
    """
    parsed = post_process_to_json(prediction)
    pred = normalize_values_of_nested_dict({} if parsed is None else parsed)
    gt = normalize_values_of_nested_dict(json.loads(answer) if isinstance(answer, str) else answer)
    tp, fn_or_fp = _field_counts(pred, gt)
    return {
        "tp": tp,
        "fn_or_fp": fn_or_fp,
        "acc": _tree_accuracy(pred, gt),
        "parsed": float(parsed is not None),
        "f1": tp / (tp + fn_or_fp / 2 + 1e-6),
    }
