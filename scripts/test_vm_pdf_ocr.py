import os
import sys
import time
import json
import uuid
import urllib.request
import urllib.error
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pdf-test-runner")

API_BASE = os.getenv("OCR_API_URL", "http://aiagent.monoroc.com")

def http_get_json(url: str, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "OCR-Test-Runner/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def encode_multipart_formdata(fields: dict, files: list) -> tuple:
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    body = bytearray()

    for k, v in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{v}\r\n".encode("utf-8"))

    for field_name, filename, file_bytes, content_type in files:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8"))
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(file_bytes)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    content_type_header = f"multipart/form-data; boundary={boundary}"
    return bytes(body), content_type_header

def test_health():
    logger.info(f"Connecting to OCR Health endpoint at {API_BASE}/ocr/health...")
    data = http_get_json(f"{API_BASE}/ocr/health", timeout=10.0)
    logger.info(f"Health Status: {data.get('status')} | Service: {data.get('service')}")
    backend_info = data.get("direct_backends", {})
    logger.info(f"Default Backend: {backend_info.get('default_backend')} | Model: {backend_info.get('default_model')}")
    logger.info(f"Database: {data.get('database', {}).get('status')}")

def test_single_pdf_upload(pdf_path: str):
    filename = os.path.basename(pdf_path)
    logger.info(f"Testing direct single PDF upload OCR for: {filename}...")
    start_t = time.monotonic()
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    fields = {
        "format": "json",
        "prompt": "Extract all medical report data into structured JSON with 100% precision.",
        "temperature": "0.0",
        "max_tokens": "4096"
    }
    files = [("file", filename, pdf_bytes, "application/pdf")]
    body, content_type = encode_multipart_formdata(fields, files)

    req = urllib.request.Request(
        f"{API_BASE}/ocr/api/ocr/upload",
        data=body,
        headers={"Content-Type": content_type, "User-Agent": "OCR-Test-Runner/1.0"}
    )

    with urllib.request.urlopen(req, timeout=180.0) as resp:
        elapsed = round(time.monotonic() - start_t, 2)
        res_json = json.loads(resp.read().decode("utf-8"))
        
        logger.info(f"Successfully processed '{filename}' in {elapsed}s (Server Duration: {res_json.get('duration_seconds')}s)")
        logger.info(f"Model used: {res_json.get('model')} on [{res_json.get('backend')}]")
        
        extracted = res_json.get("data", {})
        if isinstance(extracted, dict):
            logger.info(f"Report Title: {extracted.get('report_title')}")
            patient = extracted.get("patient_info", {})
            logger.info(f"Patient Name: {patient.get('patient_name')} | Age: {patient.get('age')} | Sex: {patient.get('sex')}")
            investigations = extracted.get("investigations", [])
            if isinstance(investigations, list):
                logger.info(f"Extracted {len(investigations)} investigation parameter(s):")
                for item in investigations[:4]:
                    logger.info(f"  * {item.get('investigation')}: {item.get('observed_value')} {item.get('unit')} (Ref: {item.get('reference_interval')})")
            elif isinstance(investigations, dict):
                logger.info(f"Extracted {len(investigations)} investigation parameter group(s):")
                for k, v in list(investigations.items())[:4]:
                    logger.info(f"  * {k}: {v}")
        return res_json

def test_multi_pdf_batch(pdf_paths: list):
    logger.info(f"Testing multi-document batch processing for {len(pdf_paths)} PDF(s)...")
    start_t = time.monotonic()
    
    files = []
    for p in pdf_paths:
        with open(p, "rb") as f:
            files.append(("files", os.path.basename(p), f.read(), "application/pdf"))

    batch_name = f"Direct_PDF_Suite_Batch_{int(time.time())}"
    fields = {
        "name": batch_name,
        "prompt": "Extract all medical report data into structured JSON with 100% precision.",
        "temperature": "0.0",
        "max_tokens": "2048"
    }

    body, content_type = encode_multipart_formdata(fields, files)
    req = urllib.request.Request(
        f"{API_BASE}/ocr/api/batch/upload",
        data=body,
        headers={"Content-Type": content_type, "User-Agent": "OCR-Test-Runner/1.0"}
    )

    with urllib.request.urlopen(req, timeout=30.0) as resp:
        batch_info = json.loads(resp.read().decode("utf-8"))
        batch_id = batch_info["batch_id"]
        logger.info(f"Batch submitted successfully! ID: {batch_id} (Total files: {batch_info['total_files']})")

    # Poll batch progress
    for i in range(60):
        time.sleep(3)
        b_stat = http_get_json(f"{API_BASE}/ocr/api/batch/{batch_id}", timeout=10.0)
        logger.info(f"Batch {batch_id} Progress: {b_stat['progress_percentage']}% | Processed: {b_stat['processed_files']}/{b_stat['total_files']} | Status: {b_stat['status']}")
        
        if b_stat["status"] in ("completed", "partial_failed", "failed"):
            break

    # Fetch detailed job records
    det_data = http_get_json(f"{API_BASE}/ocr/api/batch/{batch_id}/jobs", timeout=10.0)
    total_elapsed = round(time.monotonic() - start_t, 2)
    logger.info(f"Batch {batch_id} completed in {total_elapsed}s with final status: '{det_data['status']}'")
    for job in det_data.get("jobs", []):
        logger.info(f"  -> Doc: {job.get('document_name')} | Status: {job.get('status')} | Duration: {job.get('duration_seconds')}s")

    # Download combined batch result
    dl_req = urllib.request.Request(
        f"{API_BASE}/ocr/api/batch/{batch_id}/download?format=json",
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    with urllib.request.urlopen(dl_req, timeout=10.0) as dl_resp:
        dl_bytes = dl_resp.read()
        logger.info(f"Successfully downloaded consolidated batch JSON export ({len(dl_bytes)} bytes)")

def main():
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pdf_dir = os.path.join(workspace_root, "pdf")
    
    pdf_files = [
        os.path.join(pdf_dir, "EGFR.pdf"),
        os.path.join(pdf_dir, "FBC.pdf"),
        os.path.join(pdf_dir, "LIPID PROFILE.pdf")
    ]
    
    existing_pdfs = [p for p in pdf_files if os.path.exists(p)]
    if not existing_pdfs:
        logger.error(f"No PDF files found in '{pdf_dir}'.")
        sys.exit(1)

    logger.info(f"Starting Direct PDF OCR Test Suite against {API_BASE}")
    logger.info(f"Target PDFs: {[os.path.basename(p) for p in existing_pdfs]}")

    # 1. Health Verification
    test_health()

    # 2. Single Document Upload Tests
    for pdf in existing_pdfs:
        test_single_pdf_upload(pdf)

    # 3. Multi-Document Batch Pipeline Test
    test_multi_pdf_batch(existing_pdfs)

    logger.info("Direct PDF OCR Test Suite completed all test cases successfully!")

if __name__ == "__main__":
    main()
