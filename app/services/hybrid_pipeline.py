import time
import json
import logging
from typing import Dict, Any, List, Tuple
from app.services.native_ocr_service import native_ocr_service
from app.services.llm_client import llm_client
from app.services.schema_validator import SchemaValidator
from app.schemas.medical import MedicalReportExtraction
from app.config import settings

logger = logging.getLogger("hybrid-pipeline")

CLINICAL_CLASSIFIER_PROMPT = """You are an expert clinical diagnostic report parser and classifier.
Convert the following raw OCR text lines from a medical laboratory report into a strict, clean JSON object matching this schema:
{
  "report_title": "Clean Title (e.g. Urine Protein - Creatinine Ratio (UPCR), Full Blood Count, Lipid Profile, Biochemistry)",
  "patient_info": {
    "patient_name": "Patient Full Name",
    "pid_no": "Patient ID or Registration Number",
    "tel_no": "",
    "age": "Age with Years/Months",
    "sex": "Male or Female",
    "reference_dr": "Doctor Name",
    "sample_collected_at": null,
    "collecting_center": null,
    "registered_on": "Registered timestamp",
    "collected_on": "Collected timestamp",
    "reported_on": "Reported timestamp"
  },
  "results": [
    {
      "type": "snake_case_parameter_slug",
      "name": "Original Parameter Name",
      "value": "Observed numeric or textual value",
      "unit": "Unit of measurement (e.g. mg/dL, %, cells/mm³)"
    }
  ],
  "raw_text": null
}

CRITICAL RULES:
1. Output ONLY the JSON object. Do not wrap in markdown quotes if possible.
2. Extract actual patient test observations ONLY from the 'Result' or 'Observed Value' column.
3. NEVER extract numbers from the 'Reference Value', 'Biological Reference Interval', or age normal charts as test values!
4. DO NOT include reference guideline charts, interpretation intervals (e.g. '< 0.2 Normal', '0.2 - 1.0 Low grade proteinuria'), or doctor signature labels inside 'results'.
5. Separate combined values and units cleanly (e.g. '11.00 mg/dL' -> value: '11.00', unit: 'mg/dL').
6. For percentages and counts, preserve exact numbers (e.g. '80 %' -> value: '80', unit: '%').

Raw OCR Text Rows:
"""

class HybridPipelineService:
    @classmethod
    async def process_image_bytes(
        cls,
        image_bytes: bytes,
        classifier_model: str = "ministral-3:latest",
        classifier_backend: str = "ollama"
    ) -> Tuple[Dict[str, Any], List[str], float, float]:
        """
        Executes Two-Stage Hybrid OCR:
        Stage 1: Native High-Speed Optical Extraction (~2-4s)
        Stage 2: Mistral Pure-Text Clinical Structurer & Classifier (~8-12s)

        Returns:
            (structured_medical_json, raw_text_lines, stage1_duration, total_duration)
        """
        t0 = time.monotonic()

        # STAGE 1: Native Fast Optical Extraction
        _, raw_lines, stage1_dur = native_ocr_service.process_image(image_bytes)
        logger.info(f"⚡ [Hybrid Stage 1] Extracted {len(raw_lines)} raw text lines in {stage1_dur}s")

        # Combine text lines
        raw_text_payload = "\n".join(raw_lines)

        # STAGE 2: Mistral Clinical Classification & JSON Structuring
        t_stage2_start = time.monotonic()
        user_prompt = CLINICAL_CLASSIFIER_PROMPT + "\n" + raw_text_payload

        messages = [
            {"role": "user", "content": user_prompt}
        ]

        llm_res = await llm_client.chat_completion(
            messages=messages,
            model=classifier_model or "ministral-3:latest",
            backend=classifier_backend or "ollama",
            temperature=0.0,
            max_tokens=2048,
            json_mode=True
        )

        choices = llm_res.get("choices", [])
        raw_output = choices[0].get("message", {}).get("content", "") if choices else ""
        stage2_dur = round(time.monotonic() - t_stage2_start, 3)
        total_dur = round(time.monotonic() - t0, 3)

        logger.info(f"🧠 [Hybrid Stage 2] Mistral structured {len(raw_lines)} lines in {stage2_dur}s (Total: {total_dur}s)")

        # Validate with strict schema
        validated_data, err = SchemaValidator.parse_and_validate(raw_output, MedicalReportExtraction)

        return validated_data if validated_data else {"raw": raw_output}, raw_lines, stage1_dur, total_dur


hybrid_pipeline_service = HybridPipelineService()
