import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from app.config import settings
from app.schemas.ocr import OCRTaskType
from app.schemas.medical import MedicalReportExtraction
from app.services.llm_client import llm_client, LLMClientError
from app.services.schema_validator import SchemaValidator

logger = logging.getLogger("dual-pipeline")

STAGE2_MEDICAL_SYSTEM_PROMPT = (
    "You are an expert Clinical Diagnostic Report Vision & Medical Structuring AI. "
    "Your objective is to accurately transcribe patient demographics and lab test observations "
    "from OCR document text into strict, structured JSON.\n"
    "CRITICAL RULES:\n"
    "1. In 'results', include ONLY actual patient test observations. Never extract reference range guideline charts (e.g. '< 0.2 Normal' or 'Man: 0.7 - 1.2'), interpretation grids, or doctor signatures as results.\n"
    "2. If the value and unit are together (e.g. '104 mg/dl'), separate them cleanly into value ('104') and unit ('mg/dl').\n"
    "3. Set 'type' as a lowercase snake_case identifier (e.g. 'fasting_blood_sugar', 'total_cholesterol', 'protein_total', 'creatinine', 'protein_creatinine_ratio').\n"
    "4. ANTI-HALLUCINATION: Extract only visually confirmed data from the text. Never invent or calculate unprinted values.\n"
    "5. Output valid, raw JSON directly matching the target schema without preamble or thinking tags."
)

STAGE2_MEDICAL_USER_TEMPLATE = (
    "Analyze this transcribed laboratory report text and extract all patient demographics and test observations into a JSON object matching this exact structure:\n"
    "{{\n"
    '  "report_title": "Full Blood Count / Urine UPCR / Biochemistry",\n'
    '  "patient_info": {{\n'
    '    "patient_name": "...",\n'
    '    "pid_no": "...",\n'
    '    "age": "...",\n'
    '    "sex": "...",\n'
    '    "tel_no": "",\n'
    '    "reference_dr": "",\n'
    '    "registered_on": "",\n'
    '    "collected_on": "",\n'
    '    "reported_on": ""\n'
    "  }},\n"
    '  "results": [\n'
    "    {{\n"
    '      "type": "fasting_blood_sugar",\n'
    '      "name": "Fasting Blood Sugar",\n'
    '      "value": "104",\n'
    '      "unit": "mg/dl"\n'
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "Transcribed Document OCR Text:\n"
    "\"\"\"\n"
    "{ocr_text}\n"
    "\"\"\"\n\n"
    "Output strictly valid raw JSON without preamble."
)


class DualLayerPipeline:
    def __init__(self):
        self.default_ocr_model = "deepseek-ocr:3b"
        self.default_ocr_backend = "ollama"
        self.default_structurer_model = "Qwen3.5-4B-BF16.gguf"
        self.default_structurer_backend = "llm-server"  # Port 8100

    async def run(
        self,
        page_uris: List[str],
        ocr_model: Optional[str] = None,
        ocr_backend: Optional[str] = None,
        structurer_model: Optional[str] = None,
        structurer_backend: Optional[str] = None,
        backend: Optional[str] = None,
        task_type: OCRTaskType = OCRTaskType.MEDICAL_EXTRACTION,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        strict_schema: bool = True
    ) -> Tuple[Dict[str, Any], str, Dict[str, float], Dict[str, str]]:
        """
        Executes 2-Stage Cascaded OCR Pipeline:
        Stage 1: High-Speed Document & Table Vision OCR (DeepSeek-OCR via Ollama) -> Raw HTML/Markdown layout
        Stage 2: Semantic Clinical Structurer (Port 8100 LLM Server) -> Strict Structured JSON
        
        Returns:
            (parsed_json_dict, raw_ocr_text, stage_timings, used_models)
        """
        target_ocr_backend = ocr_backend or self.default_ocr_backend
        target_ocr_model = ocr_model or self.default_ocr_model
        
        target_structurer_backend = structurer_backend or (backend if backend in ("llm-server", "llama-cpp") else self.default_structurer_backend)
        target_structurer_model = structurer_model or self.default_structurer_model

        # ---------------------------------------------------------
        # STAGE 1: Vision OCR (Image -> Markdown/HTML Layout)
        # ---------------------------------------------------------
        logger.info(f"DualLayerPipeline Stage 1: Running Vision OCR with [{target_ocr_model}] on [{target_ocr_backend}]")
        t0 = time.monotonic()

        stage1_user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": "Transcribe all patient demographics, report title, and clinical lab test observations and tables on this page into clean markdown/HTML tables."
            }
        ]
        for uri in page_uris:
            stage1_user_content.append({"type": "image_url", "image_url": {"url": uri}})

        stage1_messages = [
            {
                "role": "system",
                "content": "You are an expert Document Layout and Vision OCR AI. Transcribe all text, patient demographics, and tables accurately."
            },
            {"role": "user", "content": stage1_user_content}
        ]

        stage1_res = await llm_client.chat_completion(
            messages=stage1_messages,
            model=target_ocr_model,
            backend=target_ocr_backend,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            json_mode=False  # Layout models output HTML/Markdown tables
        )

        t1 = time.monotonic()
        stage1_duration = round(t1 - t0, 2)
        s1_choices = stage1_res.get("choices", [])
        raw_ocr_text = s1_choices[0].get("message", {}).get("content", "") if s1_choices else ""
        used_ocr_model = stage1_res.get("model", target_ocr_model)
        logger.info(f"DualLayerPipeline Stage 1 completed in {stage1_duration}s ({len(raw_ocr_text)} chars)")

        if not raw_ocr_text.strip():
            logger.warning("DualLayerPipeline Stage 1 produced empty OCR text. Falling back to empty structure.")
            return {"report_title": "Laboratory Report", "patient_info": {}, "results": []}, "", {"stage1_ocr_seconds": stage1_duration, "stage2_structurer_seconds": 0.0}, {"ocr_model": used_ocr_model, "structurer_model": target_structurer_model}

        # ---------------------------------------------------------
        # STAGE 2: Clinical Structuring (Text -> Clean JSON via Port 8100)
        # ---------------------------------------------------------
        logger.info(f"DualLayerPipeline Stage 2: Running Clinical Structuring with [{target_structurer_model}] on [{target_structurer_backend}]")
        t2 = time.monotonic()

        stage2_system_prompt = STAGE2_MEDICAL_SYSTEM_PROMPT
        stage2_user_prompt = STAGE2_MEDICAL_USER_TEMPLATE.format(ocr_text=raw_ocr_text)

        stage2_messages = [
            {"role": "system", "content": stage2_system_prompt},
            {"role": "user", "content": stage2_user_prompt}
        ]

        stage2_res = await llm_client.chat_completion(
            messages=stage2_messages,
            model=target_structurer_model,
            backend=target_structurer_backend,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            json_mode=True  # Force grammar-constrained JSON from LLM
        )

        t3 = time.monotonic()
        stage2_duration = round(t3 - t2, 2)
        s2_choices = stage2_res.get("choices", [])
        raw_json_output = s2_choices[0].get("message", {}).get("content", "") if s2_choices else ""
        used_structurer_model = stage2_res.get("model", target_structurer_model)
        logger.info(f"DualLayerPipeline Stage 2 completed in {stage2_duration}s")

        # ---------------------------------------------------------
        # STAGE 3: Validation & Normalization
        # ---------------------------------------------------------
        target_schema = MedicalReportExtraction if strict_schema else None
        parsed_data, err = SchemaValidator.parse_and_validate(raw_json_output, target_schema)

        # Safety Fallback: If Stage 2 JSON decode fails, parse directly from Stage 1 OCR table text
        if not parsed_data or not isinstance(parsed_data, dict) or not parsed_data.get("results"):
            logger.warning("Stage 2 JSON had no results; falling back to Stage 1 table parser")
            table_parsed = SchemaValidator.parse_table_or_markdown_text(raw_ocr_text)
            if table_parsed.get("results"):
                parsed_data = SchemaValidator.normalize_keys(table_parsed)

        stage_timings = {
            "stage1_ocr_seconds": stage1_duration,
            "stage2_structurer_seconds": stage2_duration
        }
        used_models = {
            "ocr_model": used_ocr_model,
            "structurer_model": used_structurer_model
        }

        return parsed_data if parsed_data else {"raw": raw_json_output or raw_ocr_text}, raw_ocr_text, stage_timings, used_models


dual_pipeline = DualLayerPipeline()
