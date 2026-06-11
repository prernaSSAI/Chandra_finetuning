from __future__ import annotations

from chandra_finetune.metrics_relaxed import (
    METRIC_NAMES,
    HtmlNode,
    aggregate_metrics,
    character_error_rate,
    compute_metrics,
    html_to_tree,
    parse_metric_names,
    postprocess_html_for_metrics,
    table_teds_score,
    teds_score,
    word_error_rate,
)

clean_html = postprocess_html_for_metrics

__all__ = [
    "METRIC_NAMES",
    "HtmlNode",
    "aggregate_metrics",
    "character_error_rate",
    "clean_html",
    "compute_metrics",
    "html_to_tree",
    "parse_metric_names",
    "postprocess_html_for_metrics",
    "table_teds_score",
    "teds_score",
    "word_error_rate",
]
