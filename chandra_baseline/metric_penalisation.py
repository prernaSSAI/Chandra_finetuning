
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---- optional deps (same as your pipeline) ----
try:
    from apted import APTED, Config
    from apted.helpers import Tree
    APTED_AVAILABLE = True
except ImportError:
    APTED_AVAILABLE = False
    Config = object
    Tree = object

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


def _lxml_available() -> bool:
    try:
        import lxml  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------- HTML → text ----------------
def html_to_plain_text(html: str) -> str:
    if BS4_AVAILABLE:
        parser = "lxml" if _lxml_available() else "html.parser"
        return BeautifulSoup(html, parser).get_text(separator=" ").strip()
    return re.sub(r"<[^>]+>", " ", html).strip()


# ---------------- TEDS ----------------
class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content="", *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)


class TEDSConfig(Config):
    def rename(self, n1, n2):
        if n1.tag != n2.tag:
            return 1.0
        if n1.tag == "td":
            if n1.colspan != n2.colspan or n1.rowspan != n2.rowspan:
                return 1.0
            return 1.0 - difflib.SequenceMatcher(None, n1.content or "", n2.content or "").ratio()
        return 0.0

    def children(self, node):
        return node.children


def _normalize_html_table(html: str):
    if not BS4_AVAILABLE:
        raise ImportError("Install: pip install beautifulsoup4 lxml apted")

    parser = "lxml" if _lxml_available() else "html.parser"
    soup = BeautifulSoup(html or "", parser)

    tables = soup.find_all("table")
    if not tables:
        return None

    root = soup.new_tag("document")
    for table in tables:
        root.append(table)

    return root


def _cell_text(cell) -> str:
    return re.sub(r"\s+", " ", cell.get_text(separator=" ")).strip()


def html_to_tree(html: str) -> Optional[TableTree]:
    table = _normalize_html_table(html)
    if table is None:
        return None

    def convert(node):
        if getattr(node, "name", None) is None:
            return None
        children = []
        for c in node.children:
            cc = convert(c)
            if cc is not None:
                children.append(cc)
        tag = node.name.lower()
        if tag in ("td", "th"):
            return TableTree("td", str(node.get("colspan", "1")),
                             str(node.get("rowspan", "1")), _cell_text(node), *children)
        return TableTree(tag, None, None, "", *children)

    return convert(table)


def count_nodes(n: TableTree) -> int:
    return 1 + sum(count_nodes(c) for c in n.children)


def compute_teds(pred: str, gt: str) -> float:
    if not APTED_AVAILABLE:
        raise ImportError("pip install apted")
    if "<table" not in (pred or "").lower() or "<table" not in (gt or "").lower():
        return 0.0
    pt, gtt = html_to_tree(pred), html_to_tree(gt)
    if pt is None or gtt is None:
        return 0.0
    max_nodes = max(count_nodes(pt), count_nodes(gtt))
    if max_nodes == 0:
        return 0.0
    dist = APTED(pt, gtt, TEDSConfig()).compute_edit_distance()
    return round(max(0.0, min(1.0, 1.0 - dist / max_nodes)), 6)


# ---------------- CER / WER ----------------
def _edit_distance_rate(ref, hyp) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            tmp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[m] / n


def compute_cer(pred: str, gt: str) -> float:
    if JIWER_AVAILABLE:
        try:
            t = jiwer.Compose([jiwer.ReduceToListOfListOfChars()])
            return jiwer.wer([gt], [pred], truth_transform=t, hypothesis_transform=t)
        except Exception:
            pass
    return _edit_distance_rate(list(gt), list(pred))


def compute_wer(pred: str, gt: str) -> float:
    if JIWER_AVAILABLE:
        try:
            return jiwer.wer([gt], [pred])
        except Exception:
            pass
    return _edit_distance_rate(gt.split(), pred.split())


def compute_metrics(pred: str, gt: str) -> Dict[str, float]:
    pp, gp = html_to_plain_text(pred), html_to_plain_text(gt)
    return {
        "cer":  round(min(compute_cer(pp, gp), 999.0), 6),
        "wer":  round(min(compute_wer(pp, gp), 999.0), 6),
        "teds": round(compute_teds(pred, gt), 6),
    }


# ---------------- Driver ----------------
def evaluate(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for e in entries:
        m = compute_metrics(e["prediction"], e["ground_truth"])
        out.append({
            "file": e.get("file"),
            "doc_id": e.get("doc_id"),
            "page_idx": e.get("page_idx"),
            **m,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSON file with prediction & ground_truth")
    ap.add_argument("--output", default=None, help="Optional output JSON path")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]

    results = evaluate(data)

    n = len(results)
    summary = {
        "sample_count": n,
        "avg_cer":  round(sum(r["cer"]  for r in results) / n, 6),
        "avg_wer":  round(sum(r["wer"]  for r in results) / n, 6),
        "avg_teds": round(sum(r["teds"] for r in results) / n, 6),
    }

    print(json.dumps({"per_sample": results, "summary": summary}, indent=2))

    if args.output:
        Path(args.output).write_text(
            json.dumps({"per_sample": results, "summary": summary}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()