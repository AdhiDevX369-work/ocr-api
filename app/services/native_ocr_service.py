import io
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
from app.services.clinical_rule_parser import clinical_rule_parser
from app.services.schema_validator import SchemaValidator
from app.schemas.medical import MedicalReportExtraction

logger = logging.getLogger("native-ocr-service")

class NativeOCRService:
    _instance: Optional["NativeOCRService"] = None

    def __init__(self):
        self._reader = None
        self._initialized = False

    def _get_reader(self):
        if not self._initialized:
            try:
                import easyocr
                logger.info("Initializing Ultra-Fast Native OCR Engine (EasyOCR PyTorch)...")
                self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                self._initialized = True
                logger.info("Native OCR Engine initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Native OCR Engine: {e}")
                raise RuntimeError(f"Native OCR initialization failed: {e}")
        return self._reader

    def process_image(self, image_input: Any) -> Tuple[Dict[str, Any], List[str], float]:
        """
        Executes Ultra-Fast Native OCR on an image input (PIL Image, bytes, or numpy array).
        
        Returns:
            (structured_medical_json, raw_text_lines, execution_time_seconds)
        """
        t0 = time.monotonic()
        reader = self._get_reader()

        # Convert input to numpy array RGB
        if isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
            img_np = np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            img_np = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            img_np = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Run inference
        results = reader.readtext(img_np)

        # Group extracted bounding boxes into aligned 2D horizontal rows
        merged_rows = clinical_rule_parser.group_spatial_rows(results)
        extracted_lines: List[str] = [" | ".join(r) for r in merged_rows]

        t1 = time.monotonic()
        duration = round(t1 - t0, 3)
        logger.info(f"⚡ [Native OCR] Line extraction finished in {duration}s. Extracted {len(extracted_lines)} lines.")
        logger.debug(f"📄 [Native OCR] Raw extracted lines:\n" + "\n".join(f"   [{i+1}] {l}" for i, l in enumerate(extracted_lines)))

        # Parse with deterministic 2D spatial clinical rule parser
        parsed_data = clinical_rule_parser.parse_from_spatial_boxes(results)
        logger.info(f"🧪 [Clinical Rule Parser] Parsed Title: '{parsed_data.get('report_title')}', Patient: {parsed_data.get('patient_info')}, Results count: {len(parsed_data.get('results', []))}")

        # Validate with strict schema
        validated_data, err = SchemaValidator.parse_and_validate(parsed_data, MedicalReportExtraction)
        if err:
            logger.warning(f"⚠️ [Schema Validation] Soft validation warning: {err}")
        else:
            logger.info("✅ [Schema Validation] MedicalReportExtraction validation successful.")

        return validated_data if validated_data else parsed_data, extracted_lines, duration


native_ocr_service = NativeOCRService()
