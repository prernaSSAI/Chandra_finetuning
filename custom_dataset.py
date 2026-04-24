import json
import pickle
import os
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF
from prompts import OCR_PROMPT

PDF_PATH   = "/home/prerna/ssx-amneal-fresh/chandra_train_pdf/chandra-bmr-pa-166-00.pdf"
OCR_PATH   = "/home/prerna/ssx-amneal-fresh/json/annotated_BMR-PA-166-00.json"
OUTPUT_DIR = "/home/prerna/ssx-amneal-fresh/data_pkl"
DPI        = 600 #600
INSTRUCTION = OCR_PROMPT


def pdf_page_to_pil(doc: fitz.Document, page_index: int, dpi: int = 400) -> Image.Image:
    """Convert a single PDF page (0-indexed) to a PIL Image."""
    page = doc[page_index]
    mat  = fitz.Matrix(dpi / 72, dpi / 72)   # 72 is the base DPI for PDF
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def build_dataset():
    # Load OCR JSON 
    with open(OCR_PATH, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    # Build a lookup: page_number → markdown_page
    page_to_markdown = {}
    for entry in ocr_data:
        page_num = entry.get("page")
        markdown  = entry.get("markdown", "")
        if page_num is not None:
            page_to_markdown[page_num] = markdown

    print(f"[INFO] Loaded OCR for {len(page_to_markdown)} pages: {sorted(page_to_markdown.keys())}")

    # Open PDF ────────────────────────────────────────────────────────────
    doc = fitz.open(PDF_PATH)
    print(f"[INFO] PDF has {doc.page_count} pages")

    #  Prepare output directory for saved images 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images_dir = os.path.join(OUTPUT_DIR, "page_images")
    os.makedirs(images_dir, exist_ok=True)

    # Build dataset
    dataset          = []   # list of dicts with PIL images  (for training)
    dataset_json     = []   # list of dicts with image paths (for inspection)
    skipped_pages    = []

    # OCR pages are 1-indexed; PDF pages are 0-indexed
    for page_num, markdown_text in sorted(page_to_markdown.items()):
        page_index = page_num - 1   # convert to 0-indexed

        if page_index < 0 or page_index >= doc.page_count:
            print(f"[WARN] Page {page_num} out of PDF range (PDF has {doc.page_count} pages). Skipping.")
            skipped_pages.append(page_num)
            continue

        if not markdown_text.strip():
            print(f"[WARN] Page {page_num} has empty markdown. Skipping.")
            skipped_pages.append(page_num)
            continue

        # Convert PDF page → PIL image
        pil_img = pdf_page_to_pil(doc, page_index, dpi=DPI)

        # Save image to disk (for JSON-serializable version)
        img_filename = f"page_{page_num:03d}.png"
        img_path     = os.path.join(images_dir, img_filename)
        pil_img.save(img_path)

        # Load the saved PNG back so type is PIL.PngImagePlugin.PngImageFile
        
        png_img = Image.open(img_path)
        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",  "text": INSTRUCTION},
                        {"type": "image", "image": png_img}     # PIL.PngImagePlugin.PngImageFile
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": markdown_text}
                    ]
                }
            ]
        }
        dataset.append(sample)

        # JSON-serializable version (image stored as path string)
        sample_json = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",  "text": INSTRUCTION},
                        {"type": "image", "image_path": img_path,
                         "image_size": {"width": pil_img.width, "height": pil_img.height}}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": markdown_text}
                    ]
                }
            ],
            "metadata": {
                "pdf_path":   PDF_PATH,
                "ocr_path":   OCR_PATH,
                "page_number": page_num,
                "image_path": img_path
            }
        }
        dataset_json.append(sample_json)
        print(f"[OK]  Page {page_num:>3}  →  image saved to {img_path}")

    doc.close()

    # Save outputs

    # 1. JSON file (human readable / inspectable)
    json_out = os.path.join(OUTPUT_DIR, "vlm_dataset.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] JSON dataset  → {json_out}")

    # 2. Pickle file (contains actual PIL Images — ready for training)
    pkl_out = os.path.join(OUTPUT_DIR, "vlm_dataset.pkl")
    with open(pkl_out, "wb") as f:
        pickle.dump(dataset, f)
    print(f"[SAVED] Pickle dataset → {pkl_out}")

    # Summary 
    print(f"\n{'='*50}")
    print(f"  Dataset Summary")
    print(f"{'='*50}")
    print(f"  Total samples generated : {len(dataset)}")
    print(f"  Skipped pages           : {skipped_pages if skipped_pages else 'None'}")
    print(f"  Page images saved to    : {images_dir}")
    print(f"  JSON output             : {json_out}")
    print(f"  Pickle output           : {pkl_out}")
    print(f"{'='*50}")
    return dataset


if __name__ == "__main__":
    dataset = build_dataset() or []
    print(f"\n[DONE] Dataset ready with {len(dataset)} samples.")
    print("Load the pickle file in your training notebook like:")
    print("  import pickle")
    print("  dataset = pickle.load(open('vlm_dataset_output/vlm_dataset.pkl', 'rb'))")
