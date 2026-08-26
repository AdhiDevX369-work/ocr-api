import io
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
from app.services.clinical_rule_parser import clinical_rule_parser
from app.services.schema_validator import SchemaValidator
from app.schemas.medical import MedicalReportExtraction

logger = logging.getLogger("paddle-ocr-service")

class PaddleOCRService:
    _instance: Optional["PaddleOCRService"] = None

    def __init__(self):
        self._ocr = None
        self._initialized = False

    def _get_ocr(self):
        if not self._initialized:
            try:
                from paddleocr import PaddleOCR
                # use_gpu can be dynamically enabled if paddle is compiled with CUDA
                logger.info("Initializing Native PaddleOCR Engine (PP-OCRv4)...")
                self._ocr = PaddleOCR(use_angle_cls=True, lang="en")
                self._initialized = True
                logger.info("Native PaddleOCR Engine initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Native PaddleOCR: {e}")
                raise RuntimeError(f"PaddleOCR initialization failed: {e}")
        return self._ocr

    def process_image(self, image_input: Any) -> Tuple[Dict[str, Any], List[str], float]:
        """
        Executes Ultra-Fast Native PaddleOCR on an image input (PIL Image, bytes, or numpy array).
        
        Returns:
            (structured_medical_json, raw_text_lines, execution_time_seconds)
        """
        t0 = time.monotonic()
        ocr = self._get_ocr()

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

        # Run PaddleOCR inference
        result = ocr.ocr(img_np, cls=True)

        extracted_lines: List[str] = []
        if result and len(result) > 0 and result[0]:
            for line in result[0]:
                # line structure: [ [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ], (text, confidence) ]
                text = line[1][0]
                if text.strip():
                    extracted_lines.append(text.strip())

        t1 = time.monotonic()
        duration = round(t1 - t0, 3)
        logger.info(f"PaddleOCR inference finished in {duration}s ({len(extracted_lines)} lines extracted)")

        # Parse with deterministic clinical rule parser
        parsed_data = clinical_rule_parser.parse(extracted_lines)

        # Validate with strict schema
        validated_data, err = SchemaValidator.parse_and_validate(parsed_data, MedicalReportExtraction)

        return validated_data if validated_data else parsed_data, extracted_lines, duration


paddle_ocr_service = PaddleOCRService()
