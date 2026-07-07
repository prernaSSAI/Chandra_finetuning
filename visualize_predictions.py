import json
import sys
import os
import base64
import html as _html

try:
    import fitz  # PyMuPDF — used to render a PDF page when reference is empty.
except Exception:  # noqa: BLE001
    fitz = None

try:
    from bs4 import BeautifulSoup
except Exception:  # noqa: BLE001
    BeautifulSoup = None


def _balance_html(fragment):
    """Auto-close any unclosed tags in a raw HTML fragment.

    prediction_raw is unprocessed model output and may contain unbalanced tags
    (e.g. an unclosed <table>). Injected directly into the page, one bad page
    swallows every page after it. Re-serializing through a lenient parser adds
    the missing closing tags without removing content. No-op if bs4 is missing.
    """
    if not fragment or BeautifulSoup is None:
        return fragment
    try:
        return str(BeautifulSoup(fragment, 'html.parser'))
    except Exception:  # noqa: BLE001
        return fragment

# DPI for the on-screen PDF preview (not the OCR dpi). Modest to keep the HTML small.
PREVIEW_DPI = 130

# Cache of opened PDF documents, keyed by path.
_PDF_DOCS = {}


def _render_pdf_page(pdf_path, page_number_1based):
    """Return an <img> tag (base64 JPEG) for a PDF page, or an error div."""
    if fitz is None:
        return '<div class="img-err">PyMuPDF (fitz) not installed — cannot render PDF page.</div>'
    if not pdf_path:
        return '<div class="img-err">No pdf_path in metadata; cannot render page.</div>'
    try:
        if pdf_path not in _PDF_DOCS:
            _PDF_DOCS[pdf_path] = fitz.open(pdf_path)
        doc = _PDF_DOCS[pdf_path]
        pix = doc.load_page(int(page_number_1based) - 1).get_pixmap(dpi=PREVIEW_DPI)
        b64 = base64.b64encode(pix.tobytes("jpeg", jpg_quality=80)).decode("ascii")
        return f'<img src="data:image/jpeg;base64,{b64}" alt="page {page_number_1based}"/>'
    except Exception as exc:  # noqa: BLE001
        return f'<div class="img-err">Could not render page {page_number_1based}: {_html.escape(str(exc))}</div>'


def _fmt_metric(v):
    if v is None:
        return '—'
    if isinstance(v, float):
        return f'{v:.4f}'
    return str(v)


def generate_html(json_path, output_path=None, pdf_override=None, title='OCR Reference vs Prediction'):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    # Order by index (fall back to position).
    items = sorted(
        enumerate(data),
        key=lambda iv: iv[1].get('index', iv[0]),
    )
    items = [it for _, it in items]

    if output_path is None:
        base = os.path.splitext(json_path)[0]
        output_path = base + '_side_by_side.html'

    option_rows = []
    page_divs = []
    page_ids = []

    for i, item in enumerate(items):
        idx = item.get('index', i)
        meta = item.get('metadata', {}) or {}
        doc_id = meta.get('doc_id', '')
        page_idx = meta.get('page_idx', '')
        metrics = item.get('metrics', {}) or {}

        reference = item.get('reference', '') or ''
        prediction = _balance_html(item.get('prediction_raw', '') or '')

        # Left column: use the JSON reference if present; otherwise fall back to
        # rendering the original PDF page image (for runs with no ground truth).
        # A pdf_override forces the PDF-page view regardless of reference.
        if reference.strip() and not pdf_override:
            left_head = 'REFERENCE (ground truth)'
            left_head_cls = 'ref-head'
            left_body = reference
        else:
            left_head = 'PDF PAGE (source)'
            left_head_cls = 'img-head'
            left_body = _render_pdf_page(pdf_override or meta.get('pdf_path'), meta.get('page_number', idx))

        pid = f'page-{i}'
        page_ids.append(pid)

        label = f'#{idx}'
        if doc_id != '':
            label += f' · {doc_id}'
        if page_idx != '':
            label += f' (p{page_idx})'
        option_rows.append(f'    <option value="{i}">{_html.escape(label)}</option>')

        metrics_html = ' &nbsp;|&nbsp; '.join(
            f'<b>{_html.escape(str(k))}</b>: {_html.escape(_fmt_metric(v))}'
            for k, v in metrics.items()
        )

        meta_line = _html.escape(
            f"doc_id={doc_id}  page_idx={page_idx}  source={meta.get('source_file', '')}"
        )

        display = '' if i == 0 else ' style="display:none"'
        page_divs.append(f"""<div class="page-block" id="{pid}"{display}>
  <div class="meta-bar">{meta_line}</div>
  <div class="metrics-bar">{metrics_html}</div>
  <div class="columns">
    <div class="col">
      <div class="col-head {left_head_cls}">{left_head}</div>
      <div class="rendered">{left_body}</div>
    </div>
    <div class="col">
      <div class="col-head pred-head">PREDICTION (model)</div>
      <div class="rendered">{prediction}</div>
    </div>
  </div>
</div>""")

    options_html = '\n'.join(option_rows)
    pages_html = '\n'.join(page_divs)
    page_ids_js = json.dumps(page_ids)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{_html.escape(title)}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Arial, sans-serif; font-size: 14px; color: #222; background: #f0f2f5; }}

  #toolbar {{
    position: sticky; top: 0; z-index: 100;
    background: #1e3a5f; color: white;
    padding: 10px 20px; display: flex; align-items: center; gap: 14px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.4); flex-wrap: wrap;
  }}
  #toolbar h1 {{ font-size: 15px; white-space: nowrap; margin-right: 6px; }}
  #toolbar label {{ font-size: 13px; white-space: nowrap; }}
  #page-select {{
    padding: 5px 8px; border-radius: 4px; border: none;
    font-size: 13px; cursor: pointer; min-width: 220px;
  }}
  .nav-btn {{
    background: #3a7bd5; color: white; border: none;
    padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 13px;
  }}
  .nav-btn:hover {{ background: #2e62ad; }}
  .nav-btn:disabled {{ background: #555; cursor: default; }}
  #page-counter {{ font-size: 13px; color: #bcd; }}

  #content-wrapper {{ margin: 18px auto; padding: 0 16px 60px; max-width: 1700px; }}

  .meta-bar {{
    font-size: 12px; color: #555; background: #dde3ec;
    padding: 6px 12px; border-radius: 4px; margin-bottom: 8px;
    font-family: monospace;
  }}
  .metrics-bar {{
    font-size: 13px; color: #222; background: #fff4d6;
    padding: 6px 12px; border-radius: 4px; margin-bottom: 12px;
  }}

  .columns {{ display: flex; gap: 16px; align-items: flex-start; }}
  .col {{ flex: 1 1 0; min-width: 0; }}
  .col-head {{
    font-size: 12px; font-weight: bold; color: white;
    padding: 6px 12px; border-radius: 4px 4px 0 0;
  }}
  .ref-head {{ background: #2e7d32; }}
  .img-head {{ background: #6a1b9a; }}
  .pred-head {{ background: #1565c0; }}
  .img-err {{ color: #b00; font-family: monospace; }}
  .rendered {{
    background: white; padding: 24px 28px;
    border-radius: 0 0 6px 6px; box-shadow: 0 1px 5px rgba(0,0,0,0.12);
    min-height: 300px; overflow-x: auto;
  }}

  /* --- rendered OCR content styling --- */
  .rendered h1 {{ font-size: 18px; margin: 16px 0 10px; color: #1e3a5f; }}
  .rendered h2 {{ font-size: 16px; margin: 14px 0 8px; color: #1e3a5f; }}
  .rendered h3 {{ font-size: 14px; margin: 12px 0 6px; color: #1e3a5f; }}
  .rendered h4, .rendered h5 {{ font-size: 13px; margin: 10px 0 5px; color: #1e3a5f; }}
  .rendered p  {{ margin: 6px 0; line-height: 1.6; }}
  .rendered div {{ margin: 2px 0; line-height: 1.55; }}
  .rendered ul, .rendered ol {{ margin: 6px 0 6px 22px; line-height: 1.6; }}
  .rendered strong, .rendered b {{ font-weight: bold; }}
  .rendered em, .rendered i {{ font-style: italic; }}
  .rendered pre {{
    background: #f4f4f4; padding: 12px; border-radius: 4px;
    overflow-x: auto; font-family: monospace; font-size: 12px;
    white-space: pre-wrap; margin: 10px 0;
  }}
  .rendered img {{ max-width: 180px; height: auto; }}
  /* The source PDF page image is a direct child — show it full width. */
  .rendered > img {{ max-width: 100%; border: 1px solid #ddd; }}
  .rendered table {{
    border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px;
  }}
  .rendered table th, .rendered table td {{
    border: 1px solid #bbb; padding: 6px 9px; text-align: left; vertical-align: top;
  }}
  .rendered table th {{ background: #eef2f7; font-weight: bold; }}
  .rendered input {{ border: 1px solid #aaa; padding: 1px 4px; margin: 0 2px; }}
</style>
</head>
<body>

<div id="toolbar">
  <h1>{_html.escape(title)}</h1>
  <label for="page-select">Page:</label>
  <select id="page-select" onchange="goToPage(parseInt(this.value))">
{options_html}
  </select>
  <button class="nav-btn" id="btn-prev" onclick="navigate(-1)">&#8592; Prev</button>
  <button class="nav-btn" id="btn-next" onclick="navigate(1)">Next &#8594;</button>
  <span id="page-counter"></span>
</div>

<div id="content-wrapper">
{pages_html}
</div>

<script>
const pageIds = {page_ids_js};
let cur = 0;

function showPage(idx) {{
  document.getElementById(pageIds[cur]).style.display = 'none';
  cur = idx;
  document.getElementById(pageIds[cur]).style.display = '';
  document.getElementById('page-select').value = cur;
  document.getElementById('page-counter').textContent = (cur + 1) + ' / ' + pageIds.length;
  document.getElementById('btn-prev').disabled = cur === 0;
  document.getElementById('btn-next').disabled = cur === pageIds.length - 1;
  window.scrollTo({{top: 0, behavior: 'smooth'}});
}}

function navigate(d) {{ if (cur + d >= 0 && cur + d < pageIds.length) showPage(cur + d); }}
function goToPage(n) {{ if (n >= 0 && n < pageIds.length) showPage(n); }}

showPage(0);
</script>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_doc)

    print(f"Written: {output_path}  ({len(items)} pages, {os.path.getsize(output_path):,} bytes)")
    return output_path


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Render predictions JSON to a side-by-side HTML.')
    ap.add_argument('json_file', nargs='?', default='eval_vllm_predictions.json')
    ap.add_argument('output', nargs='?', default=None, help='output HTML path')
    ap.add_argument('--pdf', default=None,
                    help='override the source PDF path; forces the PDF-page view on the left')
    ap.add_argument('--title', default='OCR Reference vs Prediction', help='page title / heading')
    a = ap.parse_args()
    generate_html(a.json_file, a.output, pdf_override=a.pdf, title=a.title)
