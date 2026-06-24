import json
import os
import io
from pathlib import Path
import fitz
from prompts import OCR_PROMPT


PDF_PATH    = "/mnt/disk/ml_data/prerna/chandra_pdf/chandra_AH25020.pdf"
OCR_PATH    = "/mnt/disk/ml_data/prerna/annotated_json/AH250020_corrected.json"
OUTPUT_DIR  = "/mnt/disk/ml_data/prerna/final_arrow/AH25020_dpi300"
DPI         = 600
INSTRUCTION = OCR_PROMPT



def pdf_page_to_png_bytes(doc: fitz.Document, page_index: int, dpi: int = 600) -> bytes:
    """Render a single PDF page directly to PNG bytes for Arrow storage."""
    page = doc[page_index]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return pix.tobytes("png")


def build_dataset_arrow():
    from datasets import Dataset, Features, Image as DatasetImage, Value

    # Load OCR JSON
    with open(OCR_PATH, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    # Build lookup: page_number → markdown
    page_to_markdown = {}
    for entry in ocr_data:
        page_num = entry.get("page")
        markdown  = entry.get("markdown", "")
        if page_num is not None:
            page_to_markdown[page_num] = markdown

    print(f"[INFO] Loaded OCR for {len(page_to_markdown)} pages: {sorted(page_to_markdown.keys())}")

    doc = fitz.open(PDF_PATH)
    print(f"[INFO] PDF has {doc.page_count} pages")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    arrow_rows   = []   # Arrow dataset rows (images as PNG bytes)
    skipped_pages = []

    for page_num, markdown_text in sorted(page_to_markdown.items()):
        page_index = page_num - 1  # 0-indexed

        if page_index < 0 or page_index >= doc.page_count:
            print(f"[WARN] Page {page_num} out of PDF range. Skipping.")
            skipped_pages.append(page_num)
            continue

        if not markdown_text.strip():
            print(f"[WARN] Page {page_num} has empty markdown. Skipping.")
            skipped_pages.append(page_num)
            continue

        # Render PDF page → PIL image → PNG bytes (no file written to disk)
        png_bytes = pdf_page_to_png_bytes(doc, page_index, dpi=DPI)

        arrow_rows.append({
            "image":     {"bytes": png_bytes, "path": None},
            "prompt":    INSTRUCTION,
            "reference": markdown_text,
            "metadata":  json.dumps({
                "pdf_path":    PDF_PATH,
                "ocr_path":    OCR_PATH,
                "page_number": page_num,
            }, ensure_ascii=False),
        })

        print(f"[OK]  Page {page_num:>3}  →  encoded to PNG bytes ({len(png_bytes):,} bytes)")

    doc.close()

    if not arrow_rows:
        print("[ERROR] No samples were built. Check your PDF/OCR paths.")
        return

    # Save Arrow dataset directly 
    features = Features({
        "image":     DatasetImage(),
        "prompt":    Value("string"),
        "reference": Value("string"),
        "metadata":  Value("string"),
    })
    dataset = Dataset.from_list(arrow_rows, features=features)
    dataset.save_to_disk(OUTPUT_DIR)

    print(f"\n{'='*50}")
    print(f"  Dataset Summary")
    print(f"{'='*50}")
    print(f"  Total samples saved     : {len(dataset)}")
    print(f"  Skipped pages           : {skipped_pages if skipped_pages else 'None'}")
    print(f"  Arrow output            : {OUTPUT_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    build_dataset_arrow()
    print("\n[DONE] Arrow dataset ready.")
    print("To load it later:")
    print(f"  from datasets import load_from_disk")
    print(f"  ds = load_from_disk('{OUTPUT_DIR}')")