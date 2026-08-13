import os
import sys
import json
import glob
import asyncio
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.image_processor import ImageProcessor
from app.services.llm_client import llm_client
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("batch-extractor")

BATCH_SYSTEM_PROMPT = (
    "You are an expert Medical Lab Report Data Extractor and Visual Auditor. "
    "Your objective is to extract 100% of all data from the document scan with absolute precision.\n"
    "Capture:\n"
    "1. All Patient & Report Metadata Header details (Name, Age, Sex, PID, Tel No, Reference Dr, Sample Collected At, Registered/Collected/Reported timestamps).\n"
    "2. Exact Report Title.\n"
    "3. ALL Investigation Parameters: Investigation name, Observed Value (if blank or missing, record as '' or 'N/A'), Flag ('H'/'L' if present), Unit, and full Biological Reference Intervals.\n"
    "4. Any special age tables or risk criteria grids (e.g. AVERAGE ESTIMATED GFR by Age, Lipid Risk Factors).\n"
    "5. Footnotes, instrument details, and all Doctor/MLT signatures printed at the bottom.\n"
    "Never skip any field or number."
)

BATCH_USER_PROMPT = (
    "Extract all details from this medical lab report into a structured JSON object with the following schema:\n"
    "{\n"
    '  "patient_info": {\n'
    '    "patient_name": "",\n'
    '    "tel_no": "",\n'
    '    "pid_no": "",\n'
    '    "age": "",\n'
    '    "sex": "",\n'
    '    "reference_dr": "",\n'
    '    "sample_collected_at": "",\n'
    '    "collecting_center": "",\n'
    '    "registered_on": "",\n'
    '    "collected_on": "",\n'
    '    "reported_on": ""\n'
    '  },\n'
    '  "report_title": "",\n'
    '  "investigations": [\n'
    '    {\n'
    '      "section": "",\n'
    '      "investigation": "",\n'
    '      "observed_value": "",\n'
    '      "flag": "",\n'
    '      "unit": "",\n'
    '      "reference_interval": ""\n'
    '    }\n'
    '  ],\n'
    '  "additional_tables": [],\n'
    '  "footnotes": "",\n'
    '  "signatures": []\n'
    "}\n"
    "Ensure 100% accuracy. Output ONLY valid JSON."
)

async def process_pdf_file(pdf_path: str, output_dir: str):
    basename = os.path.basename(pdf_path)
    logger.info(f"📄 Processing PDF report: {basename}...")

    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # 1. Render PDF pages into combined Data URI
        doc_uri = ImageProcessor.process_image_bytes(pdf_bytes)

        # 2. Prepare LLM Vision Payload
        messages = [
            {"role": "system", "content": BATCH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": BATCH_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": doc_uri}}
                ]
            }
        ]

        # 3. Request LLM Vision Extraction
        raw_res = await llm_client.chat_completion(
            messages=messages,
            model=settings.default_model,
            backend=settings.default_backend,
            temperature=0.0,
            max_tokens=2048,
            stream=False
        )

        choices = raw_res.get("choices", [])
        content = ""
        if choices:
            content = choices[0].get("message", {}).get("content", "")

        # Clean markdown formatting if present
        json_str = content.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        # Save result JSON
        out_filename = os.path.splitext(basename)[0].replace(" ", "_") + "_extracted.json"
        out_path = os.path.join(output_dir, out_filename)
        
        try:
            parsed_data = json.loads(json_str)
            with open(out_path, "w", encoding="utf-8") as f_out:
                json.dump(parsed_data, f_out, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            with open(out_path, "w", encoding="utf-8") as f_out:
                f_out.write(content)

        logger.info(f"✅ Successfully extracted {basename} -> Saved to '{out_path}'")

    except Exception as e:
        logger.error(f"❌ Failed to process {basename}: {e}")

async def main():
    pdf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pdf"))
    output_dir = os.path.join(pdf_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in '{pdf_dir}'.")
        return

    logger.info(f"🚀 Found {len(pdf_files)} PDF sample(s) in '{pdf_dir}'. Starting batch extraction...")

    for pdf_file in sorted(pdf_files):
        await process_pdf_file(pdf_file, output_dir)

    await llm_client.close()
    logger.info("🎉 All PDF samples processed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
