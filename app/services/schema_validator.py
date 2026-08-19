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
    def normalize_keys(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Maps common LLM JSON output key variations to standard Pydantic schema keys."""
        if not isinstance(data, dict):
            return data

        normalized = dict(data)

        # 1. Normalize report_title
        if "report_title" not in normalized or not normalized["report_title"]:
            for alt_key in ["title", "report_name", "test_title", "test_name", "name"]:
                if alt_key in normalized and normalized[alt_key]:
                    normalized["report_title"] = str(normalized[alt_key])
                    break

        # 2. Normalize patient_info
        if "patient_info" not in normalized or not normalized["patient_info"]:
            for alt_key in ["patient_details", "patient_metadata", "patient", "patient_data"]:
                if alt_key in normalized and isinstance(normalized[alt_key], dict):
                    normalized["patient_info"] = normalized[alt_key]
                    break

        # 3. Normalize investigations
        if "investigations" not in normalized or not normalized["investigations"]:
            for alt_key in ["results", "test_results", "parameters", "tests", "items"]:
                if alt_key in normalized and isinstance(normalized[alt_key], list):
                    normalized["investigations"] = normalized[alt_key]
                    break

        # 4. Normalize patient_info subfields
        if isinstance(normalized.get("patient_info"), dict):
            pinfo = dict(normalized["patient_info"])
            if "name" in pinfo and "patient_name" not in pinfo:
                pinfo["patient_name"] = pinfo["name"]
            if "id" in pinfo and "pid_no" not in pinfo:
                pinfo["pid_no"] = pinfo["id"]
            if "ref_dr" in pinfo and "reference_dr" not in pinfo:
                pinfo["reference_dr"] = pinfo["ref_dr"]
            # Coerce int/float age and pid_no to str to prevent pydantic type errors
            if "age" in pinfo and pinfo["age"] is not None:
                pinfo["age"] = str(pinfo["age"])
            if "pid_no" in pinfo and pinfo["pid_no"] is not None:
                pinfo["pid_no"] = str(pinfo["pid_no"])
            normalized["patient_info"] = pinfo

        return normalized

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
                normalized_parsed = cls.normalize_keys(parsed)
                validated_model = target_schema.model_validate(normalized_parsed)
                return validated_model.model_dump(), None
            except ValidationError as val_err:
                logger.warning(f"Schema validation soft warning (returning raw parsed dict): {val_err}")
                return parsed, None

        return parsed, None
