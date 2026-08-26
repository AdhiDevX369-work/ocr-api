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

        # Sort extracted bounding boxes top-to-bottom, left-to-right
        # bbox structure: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        # sort by y1 first (with 10px bucket threshold), then x1
        sorted_results = sorted(results, key=lambda item: (round(item[0][0][1] / 12) * 12, item[0][0][0]))

        extracted_lines: List[str] = []
        for bbox, text, prob in sorted_results:
            clean_t = text.strip()
            if clean_t and prob > 0.2:
                extracted_lines.append(clean_t)

        t1 = time.monotonic()
        duration = round(t1 - t0, 3)
        logger.info(f"Native OCR inference finished in {duration}s ({len(extracted_lines)} lines extracted)")

        # Parse with deterministic clinical rule parser
        parsed_data = clinical_rule_parser.parse(extracted_lines)

        # Validate with strict schema
        validated_data, err = SchemaValidator.parse_and_validate(parsed_data, MedicalReportExtraction)

        return validated_data if validated_data else parsed_data, extracted_lines, duration


native_ocr_service = NativeOCRService()
