from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from statistics import mean
from typing import Any


METRIC_NAMES = ("cer", "wer", "teds", "table_teds")

_FORMAT_ONLY_TAGS = {"font", "b", "i", "u", "strong", "span", "small", "big", "em", "del"}
_TABLE_SECTION_TAGS = {"thead", "tbody", "tfoot"}
_TABLE_CELL_TAGS = {"td", "th"}
_TEXT_TRANSLATION = str.maketrans(
    {
        "\u00a0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u221a": "\u2713",
    }
)


@dataclass
class HtmlNode:
    label: str
    children: list["HtmlNode"] = field(default_factory=list)


@dataclass(frozen=True)
class TableCell:
    text: str
    rowspan: int
    colspan: int


@dataclass(frozen=True)
class TableFeatures:
    rows: int
    meaningful_rows: int
    max_cols: int
    cell_count: int
    occupied_cells: int
    row_widths: tuple[int, ...]
    row_cell_counts: tuple[int, ...]
    row_tokens: tuple[tuple[str, ...], ...]
    spans: tuple[tuple[int, int], ...]
    tokens: tuple[str, ...]

    @property
    def weight(self) -> int:
        return max(1, self.meaningful_rows * max(self.max_cols, 1), len(self.tokens) // 4, self.cell_count)


def compute_metrics(
    prediction: str,
    reference: str | None,
    *,
    metric_names: list[str] | tuple[str, ...] = METRIC_NAMES,
) -> dict[str, float | None]:
    if reference is None:
        return {name: None for name in metric_names}

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
    pred_chars = list(_visible_text(prediction))
    ref_chars = list(_visible_text(reference))
    return _normalized_edit_distance(pred_chars, ref_chars)


def word_error_rate(prediction: str, reference: str) -> float:
    pred_words = _visible_text(prediction).split()
    ref_words = _visible_text(reference).split()
    return _normalized_edit_distance(pred_words, ref_words)


def teds_score(prediction_html: str, reference_html: str) -> float:
    table_score = table_teds_score(prediction_html, reference_html)
    text_score = _token_f1(_tokens(_visible_text(prediction_html)), _tokens(_visible_text(reference_html)))
    structure_score = _sequence_similarity(
        _document_structure(prediction_html),
        _document_structure(reference_html),
    )

    if table_score is None:
        return _clamp((0.65 * structure_score) + (0.35 * text_score))

    return _clamp((0.75 * table_score) + (0.20 * text_score) + (0.05 * structure_score))


def table_teds_score(prediction_html: str, reference_html: str) -> float | None:
    pred_tables = _extract_table_features(prediction_html)
    ref_tables = _extract_table_features(reference_html)
    if not ref_tables:
        return None
    if not pred_tables:
        return 0.0

    matches = _match_tables(pred_tables, ref_tables)
    total_weight = sum(ref.weight for ref in ref_tables)
    weighted_score = sum(ref.weight * score for ref, _pred, score in matches) / total_weight

    extra_predictions = max(0, len(pred_tables) - len(matches))
    precision_factor = max(0.90, 1.0 - (0.03 * extra_predictions))
    return _clamp(weighted_score * precision_factor)


def postprocess_html_for_metrics(html: str) -> str:
    soup = _parse_html(html)
    for table in soup.find_all("table"):
        _canonicalize_table(table)
    _unwrap_formatting_tags(soup)
    _strip_visual_attributes(soup)
    _normalize_text_nodes(soup)
    return str(soup)


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
        from bs4.element import NavigableString, Tag
    except ImportError as exc:
        raise RuntimeError("beautifulsoup4 is required for TEDS metrics. Install it with: pip install beautifulsoup4") from exc

    soup = _parse_html(html)
    if normalize_table_sections:
        for section in soup.find_all(_TABLE_SECTION_TAGS):
            section.unwrap()
    roots = soup.find_all("table") if tables_only else list(soup.contents)
    if tables_only and not roots:
        return None

    table_tags = {"table", "tr", "td", "th", "caption", "colgroup", "col"} | _TABLE_SECTION_TAGS

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
        if name == "th":
            name = "td"
        children = [child for child_node in node.children for child in convert_many(child_node)]

        if normalize_table_sections and name in _TABLE_SECTION_TAGS:
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


def _parse_html(html: str) -> Any:
    try:
        from bs4 import BeautifulSoup, Comment
    except ImportError as exc:
        raise RuntimeError("beautifulsoup4 is required for TEDS metrics. Install it with: pip install beautifulsoup4") from exc

    soup = BeautifulSoup(html or "", "html.parser")
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    return soup


def _canonicalize_table(table: Any) -> None:
    for section in table.find_all(_TABLE_SECTION_TAGS):
        section.unwrap()

    for cell in table.find_all("th"):
        cell.name = "td"

    for row in table.find_all("tr"):
        cells = row.find_all(list(_TABLE_CELL_TAGS), recursive=False)
        if not cells:
            row.decompose()

    for row in list(table.find_all("tr")):
        for cell in row.find_all(list(_TABLE_CELL_TAGS), recursive=False):
            _unwrap_formatting_tags(cell)
            _strip_visual_attributes(cell)
            for attr in list(cell.attrs):
                if attr not in {"rowspan", "colspan"}:
                    del cell.attrs[attr]
            _normalize_span_attr(cell, "rowspan")
            _normalize_span_attr(cell, "colspan")

    rows = table.find_all("tr")
    while rows and _is_empty_row(rows[-1]):
        rows[-1].decompose()
        rows = table.find_all("tr")

    _strip_visual_attributes(table)


def _normalize_span_attr(cell: Any, attr: str) -> None:
    value = _safe_int(cell.get(attr), default=1)
    if value <= 1:
        cell.attrs.pop(attr, None)
    else:
        cell.attrs[attr] = str(value)


def _unwrap_formatting_tags(root: Any) -> None:
    for tag in list(root.find_all(_FORMAT_ONLY_TAGS)):
        tag.unwrap()


def _strip_visual_attributes(root: Any) -> None:
    for tag in root.find_all(True):
        if tag.name in _TABLE_CELL_TAGS:
            allowed = {"rowspan", "colspan"}
        else:
            allowed = set()
        for attr in list(tag.attrs):
            if attr not in allowed:
                del tag.attrs[attr]


def _normalize_text_nodes(root: Any) -> None:
    try:
        from bs4.element import NavigableString
    except ImportError as exc:
        raise RuntimeError("beautifulsoup4 is required for TEDS metrics. Install it with: pip install beautifulsoup4") from exc

    for text_node in list(root.find_all(string=True)):
        if not isinstance(text_node, NavigableString):
            continue
        text = _normalize_text(str(text_node))
        if text:
            text_node.replace_with(text)
        else:
            text_node.extract()


def _is_empty_row(row: Any) -> bool:
    cells = row.find_all(list(_TABLE_CELL_TAGS), recursive=False)
    return bool(cells) and all(not _normalize_text(cell.get_text(" ")) for cell in cells)


def _extract_table_features(html: str) -> list[TableFeatures]:
    soup = _parse_html(html)
    tables: list[TableFeatures] = []
    for table in soup.find_all("table"):
        _canonicalize_table(table)
        rows = _extract_table_rows(table)
        if rows and _is_scorable_table(rows):
            tables.append(_table_features(rows))
    return tables


def _extract_table_rows(table: Any) -> list[list[TableCell]]:
    rows: list[list[TableCell]] = []
    for row in table.find_all("tr"):
        cells: list[TableCell] = []
        for cell in row.find_all(list(_TABLE_CELL_TAGS), recursive=False):
            cells.append(
                TableCell(
                    text=_normalize_text(cell.get_text(" ")),
                    rowspan=_safe_int(cell.get("rowspan"), default=1),
                    colspan=_safe_int(cell.get("colspan"), default=1),
                )
            )
        if cells:
            rows.append(cells)

    while rows and all(not cell.text for cell in rows[-1]):
        rows.pop()
    return rows


def _is_scorable_table(rows: list[list[TableCell]]) -> bool:
    row_tokens = [_row_tokens(row) for row in rows]
    meaningful_indices = [index for index, tokens in enumerate(row_tokens) if tokens]
    if len(meaningful_indices) < 2:
        return False

    all_tokens = tuple(token for tokens in row_tokens for token in tokens)
    if _looks_like_document_header(all_tokens):
        return False
    if _looks_like_section_title_table(rows, row_tokens):
        return False

    row_widths = [sum(max(1, cell.colspan) for cell in row) for row in rows]
    direct_counts = [len(row) for row in rows]
    multi_cell_rows = [
        index
        for index in meaningful_indices
        if direct_counts[index] >= 2 and row_widths[index] >= 2
    ]
    wide_multi_cell_rows = [
        index
        for index in meaningful_indices
        if direct_counts[index] >= 3 and row_widths[index] >= 3
    ]
    full_width_text_rows = [
        index
        for index in meaningful_indices
        if direct_counts[index] == 1 and row_widths[index] >= 2 and len(row_tokens[index]) >= 8
    ]

    header_index = _find_header_row_index(rows, row_tokens)
    if header_index is not None:
        has_body = any(index > header_index for index in range(len(rows)))
        if has_body:
            return True

    if len(full_width_text_rows) >= max(2, len(meaningful_indices) // 2):
        return False
    if not multi_cell_rows:
        return False

    repeated_wide_width = any(
        width >= 3 and row_widths.count(width) >= 2
        for width in set(row_widths)
    )
    if repeated_wide_width and len(wide_multi_cell_rows) >= 2:
        return True

    repeated_two_col_form = (
        max(row_widths, default=0) == 2
        and len(multi_cell_rows) >= 2
        and _row_is_label_like(rows[multi_cell_rows[0]], row_tokens[multi_cell_rows[0]])
    )
    return repeated_two_col_form


def _looks_like_document_header(tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    if {"batch", "manufacturing", "record"} <= token_set:
        return True
    if {"name", "of", "product"} <= token_set and ("page" in token_set or "bmr" in token_set):
        return True
    return False


def _looks_like_section_title_table(rows: list[list[TableCell]], row_tokens: list[tuple[str, ...]]) -> bool:
    meaningful = [tokens for tokens in row_tokens if tokens]
    if not meaningful:
        return True
    first_text = " ".join(meaningful[0])
    if re.match(r"^\d+(?:\.\d+)?\b", first_text) and len(meaningful) <= 2:
        return True
    if len(meaningful) <= 2 and all(len(row) <= 2 for row in rows):
        flattened = set(token for tokens in meaningful for token in tokens)
        if flattened & {"process", "inspection", "collection", "instructions", "history"}:
            return True
    return False


def _find_header_row_index(rows: list[list[TableCell]], row_tokens: list[tuple[str, ...]]) -> int | None:
    for index, row in enumerate(rows):
        width = sum(max(1, cell.colspan) for cell in row)
        if len(row) >= 3 and width >= 3 and _row_is_label_like(row, row_tokens[index]):
            return index
    return None


def _row_is_label_like(row: list[TableCell], tokens: tuple[str, ...]) -> bool:
    if not tokens or len(row) < 2:
        return False
    cell_token_lengths = [len(_tokens(cell.text)) for cell in row if cell.text]
    if len(cell_token_lengths) < 2:
        return False
    short_label_cells = sum(length <= 5 for length in cell_token_lengths)
    return short_label_cells >= max(2, len(cell_token_lengths) - 1)


def _table_features(rows: list[list[TableCell]]) -> TableFeatures:
    row_widths = tuple(sum(max(1, cell.colspan) for cell in row) for row in rows)
    row_cell_counts = tuple(len(row) for row in rows)
    row_tokens = tuple(_row_tokens(row) for row in rows)
    spans = tuple(
        sorted(
            (max(1, cell.rowspan), max(1, cell.colspan))
            for row in rows
            for cell in row
            if cell.rowspan > 1 or cell.colspan > 1
        )
    )
    text = " ".join(cell.text for row in rows for cell in row if cell.text)
    return TableFeatures(
        rows=len(rows),
        meaningful_rows=sum(1 for tokens in row_tokens if tokens),
        max_cols=max(row_widths, default=0),
        cell_count=sum(len(row) for row in rows),
        occupied_cells=sum(row_widths),
        row_widths=row_widths,
        row_cell_counts=row_cell_counts,
        row_tokens=row_tokens,
        spans=spans,
        tokens=tuple(_tokens(text)),
    )


def _row_tokens(row: list[TableCell]) -> tuple[str, ...]:
    return tuple(_tokens(" ".join(cell.text for cell in row if cell.text)))


def _match_tables(
    pred_tables: list[TableFeatures],
    ref_tables: list[TableFeatures],
) -> list[tuple[TableFeatures, TableFeatures, float]]:
    unused_predictions = set(range(len(pred_tables)))
    matches: list[tuple[TableFeatures, TableFeatures, float]] = []

    for ref_index, ref in enumerate(ref_tables):
        best_index: int | None = None
        best_pair_score = 0.0
        best_final_score = 0.0
        for pred_index in unused_predictions:
            pred = pred_tables[pred_index]
            final_score = _table_pair_score(pred, ref)
            order_score = _ratio_similarity(abs(pred_index - ref_index), max(len(pred_tables), len(ref_tables)) - 1)
            pair_score = (
                (0.55 * final_score)
                + (0.35 * _token_f1(pred.tokens, ref.tokens))
                + (0.10 * order_score)
            )
            if pair_score > best_pair_score:
                best_index = pred_index
                best_pair_score = pair_score
                best_final_score = final_score

        if best_index is not None and best_pair_score >= 0.35:
            unused_predictions.remove(best_index)
            matches.append((ref, pred_tables[best_index], best_final_score))
        else:
            matches.append((ref, TableFeatures(0, 0, 0, 0, 0, (), (), (), (), ()), 0.0))

    return matches


def _table_pair_score(prediction: TableFeatures, reference: TableFeatures) -> float:
    row_recall, missing_rows, reference_rows = _row_recall_score(prediction, reference)
    row_count_score = _symmetric_ratio(prediction.meaningful_rows, reference.meaningful_rows)
    row_score = (0.80 * row_recall) + (0.20 * row_count_score)
    col_score = _column_similarity(prediction.max_cols, reference.max_cols)
    cell_score = (
        0.60 * _symmetric_ratio(prediction.cell_count, reference.cell_count)
        + 0.40 * _symmetric_ratio(prediction.occupied_cells, reference.occupied_cells)
    )
    span_score = _span_similarity(prediction.spans, reference.spans)
    text_score = _token_f1(prediction.tokens, reference.tokens)
    score = _clamp(
        (0.50 * row_score)
        + (0.25 * col_score)
        + (0.10 * cell_score)
        + (0.05 * span_score)
        + (0.10 * text_score)
    )
    if missing_rows and reference_rows:
        missing_ratio = missing_rows / reference_rows
        score = min(score, 1.0 - min(0.55, (0.18 * missing_rows) + (0.25 * missing_ratio)))
    return _clamp(score)


def _row_recall_score(prediction: TableFeatures, reference: TableFeatures) -> tuple[float, int, int]:
    ref_indices = [index for index, tokens in enumerate(reference.row_tokens) if tokens]
    pred_indices = [index for index, tokens in enumerate(prediction.row_tokens) if tokens]
    if not ref_indices:
        return (1.0 if not pred_indices else 0.85, 0, 0)
    if not pred_indices:
        return (0.0, len(ref_indices), len(ref_indices))

    unused_pred_indices = set(pred_indices)
    total = 0.0
    missing = 0
    for ref_index in ref_indices:
        best_index: int | None = None
        best_score = 0.0
        for pred_index in unused_pred_indices:
            score = _row_pair_score(prediction, pred_index, reference, ref_index)
            if score > best_score:
                best_index = pred_index
                best_score = score
        if best_index is not None and best_score >= 0.45:
            unused_pred_indices.remove(best_index)
            total += best_score
        else:
            missing += 1
    return (total / len(ref_indices), missing, len(ref_indices))


def _row_pair_score(
    prediction: TableFeatures,
    pred_index: int,
    reference: TableFeatures,
    ref_index: int,
) -> float:
    token_score = _token_f1(prediction.row_tokens[pred_index], reference.row_tokens[ref_index])
    width_score = _symmetric_ratio(prediction.row_widths[pred_index], reference.row_widths[ref_index])
    cell_score = _symmetric_ratio(prediction.row_cell_counts[pred_index], reference.row_cell_counts[ref_index])
    return _clamp((0.75 * token_score) + (0.15 * width_score) + (0.10 * cell_score))


def _column_similarity(pred_cols: int, ref_cols: int) -> float:
    if ref_cols <= 0:
        return 1.0 if pred_cols <= 0 else 0.85
    if pred_cols <= 0:
        return 0.0
    if pred_cols == ref_cols:
        return 1.0
    if pred_cols < ref_cols:
        return _clamp((pred_cols / ref_cols) ** 2.2)
    extra_ratio = (pred_cols - ref_cols) / pred_cols
    return _clamp(1.0 - (0.35 * extra_ratio))


def _span_similarity(pred_spans: tuple[tuple[int, int], ...], ref_spans: tuple[tuple[int, int], ...]) -> float:
    if not pred_spans and not ref_spans:
        return 1.0
    if not pred_spans or not ref_spans:
        return 0.0
    pred_counter = Counter(pred_spans)
    ref_counter = Counter(ref_spans)
    overlap = sum((pred_counter & ref_counter).values())
    total = max(sum(pred_counter.values()), sum(ref_counter.values()))
    return overlap / total if total else 1.0


def _visible_text(html: str) -> str:
    soup = _parse_html(html)
    for table in soup.find_all("table"):
        _canonicalize_table(table)
    _unwrap_formatting_tags(soup)
    return _normalize_text(soup.get_text(" "))


def _document_structure(html: str) -> tuple[str, ...]:
    soup = _parse_html(html)
    for table in soup.find_all("table"):
        table.replace_with(soup.new_tag("table"))
    _unwrap_formatting_tags(soup)
    sequence: list[str] = []
    for tag in soup.find_all(True):
        name = (tag.name or "").lower()
        if name == "th":
            name = "td"
        if name in _TABLE_SECTION_TAGS:
            continue
        sequence.append(name)
    return tuple(sequence)


def _sequence_similarity(prediction: tuple[str, ...], reference: tuple[str, ...]) -> float:
    return _clamp(1.0 - _normalized_edit_distance(list(prediction), list(reference)))


def _token_f1(prediction: tuple[str, ...] | list[str], reference: tuple[str, ...] | list[str]) -> float:
    if not reference:
        return 1.0 if not prediction else 0.85
    if not prediction:
        return 0.0
    pred_counter = Counter(prediction)
    ref_counter = Counter(reference)
    overlap = sum((pred_counter & ref_counter).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(pred_counter.values())
    recall = overlap / sum(ref_counter.values())
    return (2 * precision * recall) / (precision + recall)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[./:-][a-z0-9]+)*|[%+*/=<>-]", _normalize_text(text))


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
    text = (text or "").translate(_TEXT_TRANSLATION)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\bno\.", "no", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnos\.", "nos", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*°\s*([cf])\b", r"°\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _symmetric_ratio(left: int, right: int) -> float:
    if left == right == 0:
        return 1.0
    return min(left, right) / max(left, right)


def _ratio_similarity(distance: int, max_distance: int) -> float:
    if max_distance <= 0:
        return 1.0
    return _clamp(1.0 - (distance / max_distance))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
