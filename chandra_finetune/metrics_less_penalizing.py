from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from statistics import mean
from typing import Any


METRIC_NAMES = ("cer", "wer", "teds", "table_teds")

_FORMAT_ONLY_TAGS = {"font", "b", "i", "u", "strong", "span", "small", "big", "em", "del", "sup", "sub"}
_TABLE_SECTION_TAGS = {"thead", "tbody", "tfoot"}
_TABLE_CELL_TAGS = {"td", "th"}

# Tags that are structural noise in cells / between tables — always stripped
# before scoring. <img>/<input> are placeholders; <math> is formula markup that
# OCR may or may not produce; <br> inside a cell is treated as " ".
_NOISE_TAGS_TO_DROP = {"img", "input", "math"}

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

# --- thresholds (kept as named constants, NOT page-specific magic) ----
_CATASTROPHIC_TRUNCATION_RATIO = 0.10   # pred < 10% of ref length => generation failure
_CAPTION_ROW_MIN_TOKENS = 3             # a <p> must have >=N tokens to be considered a caption
_CAPTION_ROW_MAX_TOKENS = 60            # and <=N tokens (very long prose is not a caption)


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
    row_is_empty: tuple[bool, ...]
    spans: tuple[tuple[int, int], ...]
    tokens: tuple[str, ...]

    @property
    def weight(self) -> int:
        return max(1, self.meaningful_rows * max(self.max_cols, 1), len(self.tokens) // 4, self.cell_count)


# ====
# Public API
# ====

def compute_metrics(
    prediction: str,
    reference: str | None,
    *,
    metric_names: list[str] | tuple[str, ...] = METRIC_NAMES,
    exclude_first_table: bool = True,
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
        scores["teds"] = teds_score(prediction, reference, exclude_first_table=exclude_first_table)
    if "table_teds" in requested:
        scores["table_teds"] = table_teds_score(prediction, reference, exclude_first_table=exclude_first_table)
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


def teds_score(
    prediction_html: str,
    reference_html: str,
    *,
    exclude_first_table: bool = True,
) -> float:
    table_score = table_teds_score(
        prediction_html, reference_html, exclude_first_table=exclude_first_table
    )
    text_score = _token_f1(_tokens(_visible_text(prediction_html)), _tokens(_visible_text(reference_html)))
    structure_score = _sequence_similarity(
        _document_structure(prediction_html),
        _document_structure(reference_html),
    )

    if table_score is None:
        return _clamp((0.65 * structure_score) + (0.35 * text_score))

    return _clamp((0.75 * table_score) + (0.20 * text_score) + (0.05 * structure_score))


def table_teds_score(
    prediction_html: str,
    reference_html: str,
    *,
    exclude_first_table: bool = True,
) -> float | None:
    # --- W2: catastrophic generation guard ----
    if _is_catastrophic_truncation(prediction_html, reference_html):
        return None

    pred_tables = _extract_table_features(prediction_html, exclude_first_table=exclude_first_table)
    ref_tables = _extract_table_features(reference_html, exclude_first_table=exclude_first_table)
    if not ref_tables:
        return None
    if not pred_tables:
        return 0.0

    matches = _match_tables(pred_tables, ref_tables)
    total_weight = sum(ref.weight for ref in ref_tables)
    weighted_score = sum(ref.weight * score for ref, _pred, score in matches) / total_weight

    # --- W3: token-overlap-aware soft penalty for extra predicted tables ---
    extra_predictions = max(0, len(pred_tables) - len(matches))
    if extra_predictions:
        matched_pred_tokens: set[str] = set()
        for _ref, pred, _s in matches:
            matched_pred_tokens.update(pred.tokens)
        ref_token_pool: set[str] = set()
        for ref in ref_tables:
            ref_token_pool.update(ref.tokens)

        # The unmatched pred tables: if their tokens are largely already in
        # the ref pool, that's a split (no penalty). If they introduce many
        # novel tokens, that's a hallucination (small penalty).
        unmatched_indices = _unmatched_pred_indices(pred_tables, matches)
        novel_tokens = 0
        total_extra_tokens = 0
        for idx in unmatched_indices:
            for tok in pred_tables[idx].tokens:
                total_extra_tokens += 1
                if tok not in ref_token_pool:
                    novel_tokens += 1
        novel_ratio = (novel_tokens / total_extra_tokens) if total_extra_tokens else 0.0
        precision_factor = max(0.92, 1.0 - (0.05 * extra_predictions * novel_ratio))
    else:
        precision_factor = 1.0

    return _clamp(weighted_score * precision_factor)


def postprocess_html_for_metrics(html: str) -> str:
    soup = _parse_html(html)
    _drop_noise_tags(soup)
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
    _drop_noise_tags(soup)
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


# ====
# Catastrophic-truncation guard (W2)
# ====

def _is_catastrophic_truncation(prediction_html: str, reference_html: str) -> bool:
    """Treat the page as a generation failure (return None for table_teds) if
    the prediction is absurdly shorter than the reference. This avoids one
    bad page collapsing the mean to 0.0."""
    if not reference_html:
        return False
    pred_len = len(prediction_html or "")
    ref_len = len(reference_html)
    if ref_len < 200:
        return False  # too short to judge
    return pred_len < ref_len * _CATASTROPHIC_TRUNCATION_RATIO


# ====
# HTML parsing + caption-row equivalence (Pattern 1)
# ====

def _parse_html(html: str) -> Any:
    try:
        from bs4 import BeautifulSoup, Comment
    except ImportError as exc:
        raise RuntimeError("beautifulsoup4 is required for TEDS metrics. Install it with: pip install beautifulsoup4") from exc

    soup = BeautifulSoup(html or "", "html.parser")
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    return soup


def _drop_noise_tags(soup: Any) -> None:
    """Remove <img>, <input>, <math> entirely (they are visual placeholders /
    formula markup that should not influence structural scoring)."""
    for tag in list(soup.find_all(_NOISE_TAGS_TO_DROP)):
        tag.decompose()


def _fold_adjacent_captions_into_tables(soup: Any) -> None:
    """Pattern 1 fix: if a <p> (or <div> that is not the master-copy header)
    sits immediately BEFORE a <table>, AND the table's first row is a wide
    caption row, leave it alone (already in canonical "caption-in-table" form).
    Otherwise, if the <p> looks like a caption for the following table
    (short-ish prose, the next sibling is a <table>), MOVE its text into the
    table as a leading colspan=N row. Symmetric on both ref and pred.

    This makes
        <p>record temperature...</p><table>...</table>
    structurally equivalent to
        <table><tr><td colspan="N">record temperature...</td></tr>...</table>
    """
    try:
        from bs4.element import Tag
    except ImportError:
        return

    for table in list(soup.find_all("table")):
        # Walk back through previous siblings, skipping whitespace
        prev = table.previous_sibling
        # Collect a stack of caption-like <p>/<div>s immediately preceding this table
        captions: list[Any] = []
        while prev is not None:
            if isinstance(prev, str):
                if prev.strip() == "":
                    prev = prev.previous_sibling
                    continue
                break
            if not isinstance(prev, Tag):
                break
            name = (prev.name or "").lower()
            if name not in {"p", "div"}:
                break
            text = _normalize_text(prev.get_text(" "))
            if not text:
                prev = prev.previous_sibling
                continue
            n_tokens = len(text.split())
            if n_tokens < _CAPTION_ROW_MIN_TOKENS or n_tokens > _CAPTION_ROW_MAX_TOKENS:
                break
            # Avoid eating the page-level "master copy" / "qa authorized copy" markers
            if text in {"master copy", "qa authorized copy", "for challenge study"}:
                break
            captions.insert(0, prev)
            prev = prev.previous_sibling

        if not captions:
            continue

        # Determine the table's column width (use max colspan-sum of first non-empty row)
        first_row = table.find("tr")
        if first_row is None:
            continue
        max_cols = 0
        for row in table.find_all("tr"):
            width = sum(_safe_int(c.get("colspan"), default=1) for c in row.find_all(list(_TABLE_CELL_TAGS), recursive=False))
            if width > max_cols:
                max_cols = width
        if max_cols < 2:
            max_cols = 2

        # Insert each caption as a new <tr><td colspan=max_cols>text</td></tr>
        # at the very top of the table, preserving order
        for cap in captions:
            text = _normalize_text(cap.get_text(" "))
            new_tr = soup.new_tag("tr")
            new_td = soup.new_tag("td")
            new_td["colspan"] = str(max_cols)
            new_td.string = text
            new_tr.append(new_td)
            # Insert at the top
            if first_row:
                first_row.insert_before(new_tr)
            else:
                table.append(new_tr)
            cap.decompose()


# ====
# Table extraction (with first-table exclusion)
# ====

def _extract_table_features(
    html: str,
    *,
    exclude_first_table: bool = True,
) -> list[TableFeatures]:
    soup = _parse_html(html)
    _drop_noise_tags(soup)
    _fold_adjacent_captions_into_tables(soup)

    all_tables = soup.find_all("table")
    if exclude_first_table and all_tables:
        all_tables = all_tables[1:]

    tables: list[TableFeatures] = []
    for table in all_tables:
        _canonicalize_table(table)
        rows = _extract_table_rows(table)
        if rows and _is_scorable_table(rows):
            tables.append(_table_features(rows))
    return tables


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
            # Replace <br> with space INSIDE the cell so that "a<br>b" reads as "a b"
            for br in cell.find_all("br"):
                br.replace_with(" ")
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
    if repeated_two_col_form:
        return True

    # FIX-B: form-caption rule.
    # Many real BMR tables have the layout:
    #   row 0  : wide caption (colspan=N, short title like "Table 04 Reconciliation...")
    #   row 1+ : at least one multi-cell row with column labels (which may be
    #            LONGER than 5 tokens each, so _row_is_label_like fails)
    # The earlier branches required either a short-cell header row or repeated
    # equal widths. Form-style tables with long-text column labels (e.g.
    # "quantity to be issued", "quantity received in production") fell through
    # all branches and were wrongly rejected (caused page 13 N/A).
    #
    # We accept the table as scorable when:
    #   - there is a short, wide single-cell caption row (colspan>=2, <8 tokens), AND
    #   - there is at least one multi-cell row BELOW it with width>=2.
    caption_rows = [
        index
        for index in meaningful_indices
        if direct_counts[index] == 1
        and row_widths[index] >= 2
        and len(row_tokens[index]) < 8
    ]
    if caption_rows:
        first_caption = caption_rows[0]
        has_body_row = any(
            index > first_caption
            and direct_counts[index] >= 2
            and row_widths[index] >= 2
            for index in meaningful_indices
        )
        if has_body_row:
            return True

    return False


# --- BMR product-header label phrases (FIX-A) -----------------------------
# These are the SHORT label phrases that appear in the master product-info
# header of a BMR document. Real content tables may contain individual words
# like "batch", "manufacturing", "record" in prose, but they will NOT match
# >=2 of these full phrases AND will not be short.
_BMR_HEADER_LABEL_PHRASES = (
    frozenset({"name", "of", "product"}),
    frozenset({"batch", "no"}),
    frozenset({"bmr", "no"}),
    frozenset({"page", "no"}),
    frozenset({"batch", "size"}),
    frozenset({"mfg", "date"}),
    frozenset({"exp", "date"}),
    frozenset({"market"}),
)
_BMR_HEADER_MAX_TOKENS = 60  # real BMR header is short; content tables exceed this


def _looks_like_document_header(tokens: tuple[str, ...]) -> bool:
    """Strict BMR product-header detection (FIX-A).

    The previous version used a bag-of-words check
    (``{"batch", "manufacturing", "record"} <= token_set``) which content-
    leaked: any content-rich table that happened to mention "batch quantity",
    "manufacturing instruction", or "record stirring speed" in prose was
    wrongly classified as the document header and skipped (caused pages
    19/20 to return N/A).

    The new check requires BOTH:
      1) the table is short (<= 60 tokens), AND
      2) >= 2 BMR-specific label phrases are present.

    Since we already deterministically drop the first table from Table-TEDS
    via ``exclude_first_table=True``, this guard only needs to catch the rare
    case where the BMR header appears as the second-or-later table on a page.
    """
    if not tokens or len(tokens) > _BMR_HEADER_MAX_TOKENS:
        return False
    token_set = set(tokens)
    hits = sum(1 for phrase in _BMR_HEADER_LABEL_PHRASES if phrase <= token_set)
    return hits >= 2


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
    row_is_empty = tuple(len(tokens) == 0 for tokens in row_tokens)
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
        row_is_empty=row_is_empty,
        spans=spans,
        tokens=tuple(_tokens(text)),
    )


def _row_tokens(row: list[TableCell]) -> tuple[str, ...]:
    return tuple(_tokens(" ".join(cell.text for cell in row if cell.text)))


# ====
# Matching + pair scoring
# ====

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
            matches.append((ref, TableFeatures(0, 0, 0, 0, 0, (), (), (), (), (), ()), 0.0))

    return matches


def _unmatched_pred_indices(
    pred_tables: list[TableFeatures],
    matches: list[tuple[TableFeatures, TableFeatures, float]],
) -> list[int]:
    matched_ids = {id(pred) for _ref, pred, _s in matches}
    return [i for i, p in enumerate(pred_tables) if id(p) not in matched_ids and p.meaningful_rows > 0]


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
        + (0.20 * col_score)
        + (0.08 * cell_score)
        + (0.04 * span_score)
        + (0.18 * text_score)
    )
    if missing_rows and reference_rows:
        missing_ratio = missing_rows / reference_rows
        # Slightly softer than original (0.18 -> 0.14, 0.25 -> 0.20).
        score = min(score, 1.0 - min(0.50, (0.14 * missing_rows) + (0.20 * missing_ratio)))
    return _clamp(score)


def _row_recall_score(prediction: TableFeatures, reference: TableFeatures) -> tuple[float, int, int]:
    """Recall over reference rows, weighted by row 'mass' (empty rows count
    much less than content rows). Pattern 4: empty-row discount."""
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
        if best_index is not None and best_score >= 0.40:
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
    # Token-content weighted higher (0.75 -> 0.82) — visually-equivalent cell
    # arrangements with the same text should score high.
    return _clamp((0.82 * token_score) + (0.10 * width_score) + (0.08 * cell_score))


def _column_similarity(pred_cols: int, ref_cols: int) -> float:
    if ref_cols <= 0:
        return 1.0 if pred_cols <= 0 else 0.85
    if pred_cols <= 0:
        return 0.0
    if pred_cols == ref_cols:
        return 1.0
    if pred_cols < ref_cols:
        return _clamp((pred_cols / ref_cols) ** 2.2)
    # Cap penalty for small over-prediction of columns.
    extra_ratio = (pred_cols - ref_cols) / pred_cols
    return _clamp(1.0 - min(0.30, 0.30 * extra_ratio))


def _span_similarity(pred_spans: tuple[tuple[int, int], ...], ref_spans: tuple[tuple[int, int], ...]) -> float:
    if not pred_spans and not ref_spans:
        return 1.0
    if not pred_spans or not ref_spans:
        return 0.5  # softer than 0.0 — colspan/rowspan are an alternative encoding
    pred_counter = Counter(pred_spans)
    ref_counter = Counter(ref_spans)
    overlap = sum((pred_counter & ref_counter).values())
    total = max(sum(pred_counter.values()), sum(ref_counter.values()))
    return overlap / total if total else 1.0


# ====
# Visible text / structure
# ====

def _visible_text(html: str) -> str:
    soup = _parse_html(html)
    _drop_noise_tags(soup)
    for table in soup.find_all("table"):
        _canonicalize_table(table)
    _unwrap_formatting_tags(soup)
    return _normalize_text(soup.get_text(" "))


def _document_structure(html: str) -> tuple[str, ...]:
    soup = _parse_html(html)
    _drop_noise_tags(soup)
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
    # FIX-C: tighter edge-case handling.
    # Previously this returned 0.85 when the reference was empty but the
    # prediction was not — a "free pass" that inflated scores on degenerate
    # row/table pairs (e.g. an empty reference row matched against a noisy
    # predicted row). Now: both-empty -> 1.0; either-empty -> 0.0.
    if not reference and not prediction:
        return 1.0
    if not reference or not prediction:
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


# ====
# Edit distance + tree helpers (unchanged)
# ====

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
    # Pattern 6: collapse n/a, na, -na- to a single canonical token
    text = re.sub(r"\s*-\s*na\s*-\s*", " na ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bn\s*/\s*a\b", "na", text, flags=re.IGNORECASE)
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