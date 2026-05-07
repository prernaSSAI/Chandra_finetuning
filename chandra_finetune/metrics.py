from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from statistics import mean
from typing import Any


METRIC_NAMES = ("cer", "wer", "teds", "table_teds")


def clean_html(html: str) -> str:
    """Clean raw HTML: collapse whitespace, strip inter-tag spaces, trim cell padding.

    This normalises harmless formatting differences (literal newlines,
    redundant whitespace around/between tags) that would otherwise inflate
    CER/WER and distort TEDS scores.
    """
    # Remove literal newlines
    html = html.replace("\n", "")
    # Collapse runs of spaces/tabs into a single space
    html = re.sub(r"[ \t]+", " ", html)
    # Remove whitespace between consecutive tags:  > <  →  ><
    html = re.sub(r">\s+<", "><", html)
    # Strip leading whitespace inside common block/cell tags
    html = re.sub(r"(<(?:td|th|p|div|li|caption)[^>]*>)\s+", r"\1", html)
    # Strip trailing whitespace before closing block/cell tags
    html = re.sub(r"\s+(</(?:td|th|p|div|li|caption)>)", r"\1", html)
    return html.strip()



@dataclass
class HtmlNode:
    label: str
    children: list["HtmlNode"] = field(default_factory=list)


def compute_metrics(
    prediction: str,
    reference: str | None,
    *,
    metric_names: list[str] | tuple[str, ...] = METRIC_NAMES,
) -> dict[str, float | None]:
    if reference is None:
        return {name: None for name in metric_names}

    # Clean both prediction and reference HTML before computing any metric
    prediction = clean_html(prediction)
    reference = clean_html(reference)

    requested = set(metric_names)
    scores: dict[str, float | None] = {}
    if "cer" in requested:
        scores["cer"] = character_error_rate(prediction, reference)
    if "wer" in requested:
        scores["wer"] = word_error_rate(prediction, reference)
    if "teds" in requested:
        scores["teds"] = teds_score(prediction, reference)
    if "table_teds" in requested:
        scores["table_teds"] = table_teds_score(prediction, reference)
    return scores


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    aggregate: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        values: list[float] = []
        for row in rows:
            metrics = row.get("metrics") or {}
            value = metrics.get(name)
            if value is not None:
                values.append(float(value))
        aggregate[name] = mean(values) if values else None
    return aggregate


def character_error_rate(prediction: str, reference: str) -> float:
    pred_chars = list(_normalize_text(prediction))
    ref_chars = list(_normalize_text(reference))
    return _normalized_edit_distance(pred_chars, ref_chars)


def word_error_rate(prediction: str, reference: str) -> float:
    pred_words = _normalize_text(prediction).split()
    ref_words = _normalize_text(reference).split()
    return _normalized_edit_distance(pred_words, ref_words)


def teds_score(prediction_html: str, reference_html: str) -> float:
    # Score HTML structure with table-aware normalization.
    # This avoids over-penalizing visually/structurally correct tables for
    # harmless differences such as attributes, whitespace, <tbody>, or cell text.
    pred_tree = html_to_tree(
        prediction_html,
        ignore_attrs=True,
        ignore_table_text=True,
        normalize_table_sections=True,
    )
    ref_tree = html_to_tree(
        reference_html,
        ignore_attrs=True,
        ignore_table_text=True,
        normalize_table_sections=True,
    )
    return _tree_similarity(pred_tree, ref_tree)


def table_teds_score(prediction_html: str, reference_html: str) -> float | None:
    # Table TEDS should measure table structure, not exact text/formatting.
    pred_tree = html_to_tree(
        prediction_html,
        tables_only=True,
        ignore_attrs=False,
        keep_table_attrs=("rowspan", "colspan"),
        ignore_table_text=True,
        normalize_table_sections=True,
    )
    ref_tree = html_to_tree(
        reference_html,
        tables_only=True,
        ignore_attrs=False,
        keep_table_attrs=("rowspan", "colspan"),
        ignore_table_text=True,
        normalize_table_sections=True,
    )
    if ref_tree is None:
        return None
    if pred_tree is None:
        return 0.0
    return _tree_similarity(pred_tree, ref_tree)


def html_to_tree(
    html: str,
    *,
    tables_only: bool = False,
    ignore_attrs: bool = False,
    keep_table_attrs: tuple[str, ...] = ("rowspan", "colspan"),
    ignore_table_text: bool = False,
    normalize_table_sections: bool = False,
) -> HtmlNode | None:
    try:
        from bs4 import BeautifulSoup
        from bs4.element import NavigableString, Tag
    except ImportError as exc:
        raise RuntimeError("beautifulsoup4 is required for TEDS metrics. Install it with: pip install beautifulsoup4") from exc

    soup = BeautifulSoup(html or "", "html.parser")
    roots = soup.find_all("table") if tables_only else list(soup.contents)
    if tables_only and not roots:
        return None

    table_section_tags = {"thead", "tbody", "tfoot"}
    table_tags = {"table", "tr", "td", "th", "caption", "colgroup", "col"} | table_section_tags

    def is_inside_table(node: Any) -> bool:
        parent = getattr(node, "parent", None)
        while parent is not None:
            if getattr(parent, "name", None) == "table":
                return True
            parent = getattr(parent, "parent", None)
        return False

    def convert_many(node: Any) -> list[HtmlNode]:
        if isinstance(node, NavigableString):
            if ignore_table_text and is_inside_table(node):
                return []
            text = _normalize_text(str(node))
            if not text:
                return []
            return [HtmlNode(f"#text:{text}")]

        if not isinstance(node, Tag):
            return []

        name = (node.name or "").lower()
        children = [child for child_node in node.children for child in convert_many(child_node)]

        # BeautifulSoup/input HTML may include or omit these wrappers. Flatten them
        # so equivalent row/cell structures do not get penalized.
        if normalize_table_sections and name in table_section_tags:
            return children

        attrs = ""
        if not ignore_attrs:
            attrs_to_score = node.attrs
            if name in table_tags and keep_table_attrs:
                keep = set(keep_table_attrs)
                attrs_to_score = {key: value for key, value in node.attrs.items() if key in keep}
            attrs = _format_attrs(attrs_to_score)

        label = name if not attrs else f"{name}[{attrs}]"
        return [HtmlNode(label, children)]

    children = [child for node in roots for child in convert_many(node)]
    if not children and not tables_only:
        text = _normalize_text(soup.get_text(" "))
        if text:
            children = [HtmlNode(f"#text:{text}")]
    return HtmlNode("document", children)


def parse_metric_names(metrics: str) -> list[str]:
    names = [name.strip().lower() for name in metrics.split(",") if name.strip()]
    unknown = sorted(set(names) - set(METRIC_NAMES))
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Valid metrics: {', '.join(METRIC_NAMES)}")
    return names or list(METRIC_NAMES)


def _normalized_edit_distance(prediction: list[str], reference: list[str]) -> float:
    if not reference:
        return 0.0 if not prediction else 1.0
    return _edit_distance(prediction, reference) / len(reference)


def _edit_distance(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            substitution = previous[right_index - 1] + (left_item != right_item)
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def _tree_similarity(prediction: HtmlNode, reference: HtmlNode) -> float:
    max_size = max(_tree_size(prediction), _tree_size(reference), 1)
    distance = _tree_distance(prediction, reference)
    return max(0.0, 1.0 - (distance / max_size))


def _tree_size(node: HtmlNode) -> int:
    return 1 + sum(_tree_size(child) for child in node.children)


def _tree_distance(left: HtmlNode, right: HtmlNode) -> int:
    @lru_cache(maxsize=None)
    def distance(left_id: int, right_id: int) -> int:
        left_node = nodes[left_id]
        right_node = nodes[right_id]
        label_cost = 0 if left_node.label == right_node.label else 1
        return label_cost + sequence_distance(
            tuple(ids_by_parent[left_id]),
            tuple(ids_by_parent[right_id]),
        )

    @lru_cache(maxsize=None)
    def subtree_size(node_id: int) -> int:
        return 1 + sum(subtree_size(child_id) for child_id in ids_by_parent[node_id])

    @lru_cache(maxsize=None)
    def sequence_distance(left_ids: tuple[int, ...], right_ids: tuple[int, ...]) -> int:
        rows = len(left_ids) + 1
        cols = len(right_ids) + 1
        dp = [[0] * cols for _ in range(rows)]

        for i in range(1, rows):
            dp[i][0] = dp[i - 1][0] + subtree_size(left_ids[i - 1])
        for j in range(1, cols):
            dp[0][j] = dp[0][j - 1] + subtree_size(right_ids[j - 1])

        for i in range(1, rows):
            for j in range(1, cols):
                delete_cost = dp[i - 1][j] + subtree_size(left_ids[i - 1])
                insert_cost = dp[i][j - 1] + subtree_size(right_ids[j - 1])
                replace_cost = dp[i - 1][j - 1] + distance(left_ids[i - 1], right_ids[j - 1])
                dp[i][j] = min(delete_cost, insert_cost, replace_cost)
        return dp[-1][-1]

    nodes: list[HtmlNode] = []
    ids_by_parent: dict[int, list[int]] = {}

    def register(node: HtmlNode) -> int:
        node_id = len(nodes)
        nodes.append(node)
        ids_by_parent[node_id] = [register(child) for child in node.children]
        return node_id

    left_root = register(left)
    right_root = register(right)
    return distance(left_root, right_root)


def _format_attrs(attrs: dict[str, Any]) -> str:
    if not attrs:
        return ""

    parts: list[str] = []
    for key in sorted(attrs):
        value = attrs[key]
        if isinstance(value, list):
            value = " ".join(str(item) for item in value)
        parts.append(f"{key}={_normalize_text(str(value))}")
    return ";".join(parts)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

