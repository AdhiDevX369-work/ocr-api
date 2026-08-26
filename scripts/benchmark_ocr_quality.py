#!/usr/bin/env python3
"""
OCR Quality & Benchmarking Suite for Open Models
Computes Character Error Rate (CER), Word Error Rate (WER), Latency, and Throughput.
"""
import os
import sys
import time
import json
import logging
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
from app.services.image_processor import ImageProcessor
from app.services.llm_client import llm_client
from app.schemas.ocr import OCRRequest, OCRFormat, OCRTaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark-suite")

def compute_cer(reference: str, hypothesis: str) -> float:
    """Computes Character Error Rate (CER) using Levenshtein distance."""
    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref:
        return 0.0 if not hyp else 1.0

    r_len = len(ref)
    h_len = len(hyp)
    d = [[0] * (h_len + 1) for _ in range(r_len + 1)]

    for i in range(r_len + 1):
        d[i][0] = i
    for j in range(h_len + 1):
        d[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1   # substitution
                )

    return d[r_len][h_len] / float(r_len)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Computes Word Error Rate (WER)."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    r_len = len(ref_words)
    h_len = len(hyp_words)
    d = [[0] * (h_len + 1) for _ in range(r_len + 1)]

    for i in range(r_len + 1):
        d[i][0] = i
    for j in range(h_len + 1):
        d[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + 1
                )

    return d[r_len][h_len] / float(r_len)


async def run_benchmark_on_pdf(pdf_path: str, model: str = "qwen3-vl:4b", backend: str = "ollama"):
    logger.info(f"=== Starting Benchmark on {os.path.basename(pdf_path)} with [{model}] ===")
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    doc_info = ImageProcessor.process_document(pdf_bytes)
    page_count = doc_info["page_count"]
    digital_ref_text = doc_info.get("digital_text") or ""
    page_uris = doc_info["page_data_uris"]

    logger.info(f"Rendered {page_count} high-DPI page(s). Digital reference length: {len(digital_ref_text)} chars")

    prompt = (
        "Transcribe all text from this document naturally in exact reading order preserving structural hierarchy.\n"
        "Formatting Rules:\n"
        "- Represent tables using clean HTML (<table>...</table>) or Markdown tables.\n"
        "- Format mathematical expressions and chemical formulas in LaTeX ($...$ or $$...$$).\n"
        "- Maintain original headings, bullet lists, and paragraphs faithfully without summarizing."
    )

    user_content = [{"type": "text", "text": prompt}]
    for uri in page_uris:
        user_content.append({"type": "image_url", "image_url": {"url": uri}})

    messages = [
        {"role": "system", "content": "You are a cutting-edge Vision-Language OCR model."},
        {"role": "user", "content": user_content}
    ]

    start_time = time.monotonic()
    try:
        raw_res = await llm_client.chat_completion(
            messages=messages,
            model=model,
            backend=backend,
            temperature=0.0,
            max_tokens=8192,
            stream=False,
            json_mode=False
        )
        elapsed = round(time.monotonic() - start_time, 2)
        choices = raw_res.get("choices", [])
        output_text = choices[0].get("message", {}).get("content", "") if choices else ""

        logger.info(f"Completed inference in {elapsed}s. Generated {len(output_text)} characters.")
        
        if digital_ref_text and len(digital_ref_text) > 100:
            cer = compute_cer(digital_ref_text, output_text)
            wer = compute_wer(digital_ref_text, output_text)
            logger.info(f"Accuracy Metrics vs Digital Ground Truth -> CER: {cer:.4f} | WER: {wer:.4f}")
        else:
            logger.info("Scanned document without embedded digital reference; visual OCR transcription complete.")

        print("\n--- Model Output Preview (First 500 chars) ---")
        print(output_text[:500])
        print("----------------------------------------------\n")

        return {
            "file": os.path.basename(pdf_path),
            "model": model,
            "backend": backend,
            "pages": page_count,
            "elapsed_seconds": elapsed,
            "output_length": len(output_text)
        }

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {"file": os.path.basename(pdf_path), "error": str(e)}


if __name__ == "__main__":
    import asyncio
    pdf_sample = os.path.join(os.path.dirname(__file__), "..", "pdf", "FBC.pdf")
    if os.path.exists(pdf_sample):
        asyncio.run(run_benchmark_on_pdf(pdf_sample, model=settings.default_model, backend=settings.default_backend))
    else:
        logger.info("No sample PDF found at pdf/FBC.pdf")
