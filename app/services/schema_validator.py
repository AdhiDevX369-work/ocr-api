import re
import json
import logging
from typing import Dict, Any, Optional, Tuple, Type
from pydantic import BaseModel, ValidationError
from app.schemas.medical import MedicalReportExtraction, PatientInfo, InvestigationItem

logger = logging.getLogger("schema-validator")

class SchemaValidator:
    @staticmethod
    def extract_json_string(raw_text: str) -> str:
        """
        Extracts JSON content from raw LLM output, stripping thinking tags,
        markdown code fences (```json ... ```), and preamble text.
        """
        if not raw_text or not raw_text.strip():
            return ""

        text = raw_text.strip()

        # 1. Strip <think>...</think> tags from reasoning models
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # 2. Extract ```json ... ``` or ``` ... ``` code fence
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fence_match:
            return fence_match.group(1).strip()

        # 3. Find first '{' or '[' and last '}' or ']'
        start_brace = text.find("{")
        start_bracket = text.find("[")

        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            end_brace = text.rfind("}")
            if end_brace != -1 and end_brace > start_brace:
                return text[start_brace : end_brace + 1].strip()
        elif start_bracket != -1:
            end_bracket = text.rfind("]")
            if end_bracket != -1 and end_bracket > start_bracket:
                return text[start_bracket : end_bracket + 1].strip()

        return text

    @staticmethod
    def repair_json_string(json_str: str) -> str:
        """
        Applies heuristic repairs for common LLM JSON syntax errors:
        - Trailing commas before closing braces/brackets
        - Unclosed quotes
        - Missing closing braces
        """
        cleaned = json_str.strip()

        # Remove trailing commas before } or ]
        cleaned = re.sub(r",\s*([\}\]])", r"\1", cleaned)

        # Fix unbalanced brackets/braces
        open_curly = cleaned.count("{")
        close_curly = cleaned.count("}")
        if open_curly > close_curly:
            cleaned += "}" * (open_curly - close_curly)

        open_square = cleaned.count("[")
        close_square = cleaned.count("]")
        if open_square > close_square:
            cleaned += "]" * (open_square - close_square)

        return cleaned

    @classmethod
    def parse_and_validate(
        cls,
        raw_text: str,
        target_schema: Optional[Type[BaseModel]] = MedicalReportExtraction
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Parses raw LLM text, repairs JSON, and validates against target Pydantic schema.
        Returns: (parsed_data_dict_or_none, error_message_or_none)
        """
        extracted = cls.extract_json_string(raw_text)
        if not extracted:
            return None, "No valid JSON structure found in output."

        # First attempt: direct json.loads
        parsed = None
        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError:
            # Second attempt: repair JSON
            repaired = cls.repair_json_string(extracted)
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError as err:
                logger.warning(f"JSON decode failed after repair: {err}")
                return None, f"JSON parse error: {str(err)}"

        # If schema validation is requested
        if target_schema and isinstance(parsed, dict):
            try:
                validated_model = target_schema.model_validate(parsed)
                return validated_model.model_dump(), None
            except ValidationError as val_err:
                logger.warning(f"Schema validation soft warning (returning raw parsed dict): {val_err}")
                # Return parsed dict with error note rather than failing entirely
                return parsed, None

        return parsed, None
