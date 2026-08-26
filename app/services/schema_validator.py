import re
import json
import logging
from typing import Dict, Any, Optional, Tuple, Type
from pydantic import BaseModel, ValidationError
from app.schemas.medical import MedicalReportExtraction

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
        - Unclosed brackets/braces
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

    @staticmethod
    def slugify_test_name(name: str) -> str:
        """
        Algorithmic slugifier: converts any clinical test name dynamically
        into a clean snake_case identifier without rigid hardcoded dictionaries.
        """
        if not name:
            return "test_parameter"
        # Strip non-alphanumeric punctuation, lowercase, collapse spaces to underscores
        clean = re.sub(r"[^\w\s-]", "", str(name).strip().lower())
        clean = re.sub(r"[\s_-]+", "_", clean).strip("_")
        return clean or "test_parameter"

    @classmethod
    def normalize_results_list(cls, raw_list: Any) -> list[Dict[str, Any]]:
        """
        Generically normalizes any list or dictionary of extracted tests
        into a clean, standardized [ResultItem] array.
        """
        standardized_items = []

        def process_raw_item(item: Any):
            if not isinstance(item, dict):
                return

            # Extract test name
            name = (
                item.get("name")
                or item.get("parameter")
                or item.get("investigation")
                or item.get("test_name")
                or item.get("test")
                or ""
            )
            name_str = str(name).strip()

            # Extract value
            val = (
                item.get("value")
                if item.get("value") is not None
                else item.get("observed_value")
                if item.get("observed_value") is not None
                else item.get("result")
                if item.get("result") is not None
                else item.get("patient_value")
                if item.get("patient_value") is not None
                else ""
            )
            val_str = str(val).strip() if val is not None else ""

            # Extract unit
            unit = item.get("unit") or item.get("units") or item.get("measurement_unit") or ""
            unit_str = str(unit).strip() if unit is not None else ""

            # Extract or algorithmically derive slug type
            slug_type = item.get("type") or ""
            if not slug_type or not isinstance(slug_type, str):
                slug_type = cls.slugify_test_name(name_str)
            else:
                slug_type = cls.slugify_test_name(slug_type)

            if name_str or val_str:
                standardized_items.append({
                    "type": slug_type,
                    "name": name_str or slug_type.replace("_", " ").title(),
                    "value": val_str,
                    "unit": unit_str
                })

        if isinstance(raw_list, list):
            for elem in raw_list:
                if isinstance(elem, dict):
                    has_nested = False
                    for k, v in elem.items():
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            for sub in v:
                                process_raw_item(sub)
                            has_nested = True
                    if not has_nested:
                        process_raw_item(elem)
        elif isinstance(raw_list, dict):
            for k, v in raw_list.items():
                if isinstance(v, list):
                    for sub in v:
                        process_raw_item(sub)
                elif isinstance(v, dict):
                    process_raw_item(v)

        return standardized_items

    @classmethod
    def normalize_keys(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maps raw LLM JSON structure to standardized schema with patient_info and results array.
        """
        if not isinstance(data, dict):
            return data

        normalized = dict(data)

        # 1. Normalize report_title
        report_title = "Medical Report"
        for alt_key in ["report_title", "title", "report_name", "test_title", "test_name", "name"]:
            if alt_key in normalized and normalized[alt_key]:
                report_title = str(normalized[alt_key])
                break

        # 2. Normalize patient_info
        patient_info = {}
        for alt_key in ["patient_info", "patient_details", "patient_metadata", "patient", "patient_data"]:
            if alt_key in normalized and isinstance(normalized[alt_key], dict):
                patient_info = dict(normalized[alt_key])
                break

        if patient_info:
            if "name" in patient_info and "patient_name" not in patient_info:
                patient_info["patient_name"] = patient_info["name"]
            if "patient" in patient_info and "patient_name" not in patient_info:
                patient_info["patient_name"] = patient_info["patient"]

            # PID / Identification mapping
            for pid_key in ["pid", "id", "test_no", "ref_no", "reg_no", "registration_no", "patient_id", "sample_id", "barcode"]:
                if pid_key in patient_info and "pid_no" not in patient_info and patient_info[pid_key]:
                    patient_info["pid_no"] = str(patient_info[pid_key])
                    break

            # Ref Dr mapping
            for dr_key in ["ref_by", "ref_dr", "referring_doctor", "doctor", "referred_by", "consultant"]:
                if dr_key in patient_info and "reference_dr" not in patient_info and patient_info[dr_key]:
                    patient_info["reference_dr"] = str(patient_info[dr_key])
                    break

            # Gender / Sex mapping
            for sex_key in ["gender", "sex"]:
                if sex_key in patient_info and patient_info[sex_key]:
                    patient_info["sex"] = str(patient_info[sex_key])
                    break

            # Age normalization
            if "age" in patient_info and patient_info["age"] is not None:
                patient_info["age"] = str(patient_info["age"])
            if "pid_no" in patient_info and patient_info["pid_no"] is not None:
                patient_info["pid_no"] = str(patient_info["pid_no"])

        # 3. Extract and normalize all test results into single `results` list
        raw_test_source = None
        for key in ["results", "investigations", "parameters", "test_results", "observed_values", "tests", "items"]:
            if key in normalized and normalized[key]:
                raw_test_source = normalized[key]
                break

        if raw_test_source is None:
            nested_tests = []
            for k, v in normalized.items():
                if k not in ("report_title", "title", "patient_info", "patient_details", "patient", "raw_text"):
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        nested_tests.extend(v)
                    elif isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, list) and sub_v and isinstance(sub_v[0], dict):
                                nested_tests.extend(sub_v)
            if nested_tests:
                raw_test_source = nested_tests

        standardized_results = cls.normalize_results_list(raw_test_source or [])

        return {
            "report_title": report_title,
            "patient_info": patient_info,
            "results": standardized_results
        }

    @classmethod
    def parse_table_or_markdown_text(cls, raw_text: str) -> Dict[str, Any]:
        """
        Extracts structured patient information and lab test parameters
        from HTML tables, Markdown tables, and Markdown key-value lines
        (such as those generated by specialized Vision OCR models like deepseek-ocr:3b).
        """
        if not raw_text or not raw_text.strip():
            return {"report_title": "Laboratory Report", "patient_info": {}, "results": []}

        text = raw_text.strip()
        report_title = "Laboratory Report"
        patient_info: Dict[str, Any] = {}
        results: list[Dict[str, Any]] = []

        def split_val_unit(val_str: str) -> Tuple[str, str]:
            v_clean = val_str.strip()
            m = re.match(r"^([\d\.,\+\-<>]+)\s*([a-zA-Z%/µu°][\w/µu°\^\-\.\s]*)$", v_clean)
            if m:
                return m.group(1).replace(",", "").strip(), m.group(2).strip()
            return v_clean.replace(",", "").strip(), ""

        # 1. Parse Patient Info from Markdown bold keys or HTML cells
        name_m = re.search(r"\*\*(?:Name|Patient(?:\s+Name)?)[^:]*:\*\*\s*([^\n\r]+)", text, re.IGNORECASE)
        if not name_m:
            name_m = re.search(r"(?:Name|Patient(?:\s+Name)?)\s*[:\*\-]?\s*([A-Za-z\.\s]{3,35})(?:\n|\r|<|\s{2,})", text, re.IGNORECASE)
        if name_m:
            val = re.sub(r"[*_]", "", name_m.group(1)).strip()
            if not any(k in val.lower() for k in ["test", "gender", "female", "male", "date", "ref", "source", "investigation"]):
                patient_info["patient_name"] = val

        age_m = re.search(r"\*\*(?:Age)[^:]*:\*\*\s*([^\n\r]+)", text, re.IGNORECASE)
        if not age_m:
            age_m = re.search(r"(?:Age)\s*[:\*\-]?\s*(\d{1,3}\s*(?:Y(?:ears?)?|M(?:onths?)?)?)", text, re.IGNORECASE)
        if age_m:
            patient_info["age"] = re.sub(r"[*_]", "", age_m.group(1)).strip()

        sex_m = re.search(r"\*\*(?:Sex|Gender)[^:]*:\*\*\s*([^\n\r]+)", text, re.IGNORECASE)
        if not sex_m:
            sex_m = re.search(r"(?:Sex|Gender)\s*[:\*\-]?\s*(Male|Female|M|F)\b", text, re.IGNORECASE)
        if sex_m:
            patient_info["sex"] = re.sub(r"[*_]", "", sex_m.group(1)).strip()

        for dr_pattern in [r"\*\*(?:Ref(?:erred)?(?:\s*by|\s*Dr)?)[^:]*:\*\*\s*([^\n\r]+)", r"(?:Ref\.?\s*Dr\.?|Referred\s*by)\s*[:\*\-]?\s*([A-Za-z\.\s]{3,35})"]:
            dr_m = re.search(dr_pattern, text, re.IGNORECASE)
            if dr_m:
                dr_val = re.sub(r"[*_]", "", dr_m.group(1)).strip()
                if dr_val and not any(k in dr_val.lower() for k in ["date", "time", "sample", "test"]):
                    patient_info["reference_dr"] = dr_val
                    break

        for pid_pattern in [r"\*\*(?:PID|ID|Ref\.?\s*No\.?|Test\s*No\.?)[^:]*:\*\*\s*([^\n\r]+)", r"(?:PID|Test\s*No\.?|Ref\.?\s*No\.?)\s*[:\*\-]?\s*([A-Za-z0-9\/\-\_]{2,20})"]:
            pid_m = re.search(pid_pattern, text, re.IGNORECASE)
            if pid_m:
                pid_val = re.sub(r"[*_]", "", pid_m.group(1)).strip()
                if pid_val:
                    patient_info["pid_no"] = pid_val
                    break

        # 2. Extract Title and Tabular / List Results
        lines = text.strip().split("\n")
        for line in lines:
            # HTML Table Row
            if "<td" in line.lower() or "<th" in line.lower():
                cells = re.findall(r"<(?:td|th)[^>]*>(.*?)(?=<(?:td|th)|$)", line, flags=re.IGNORECASE)
                c = [re.sub(r"<[^>]+>", "", cell).strip() for cell in cells if cell.strip()]
                if not c:
                    continue
                if len(c) == 1 and len(c[0]) > 4 and not any(char.isdigit() for char in c[0]):
                    if report_title == "Laboratory Report":
                        report_title = c[0]
                elif len(c) >= 2:
                    col0 = c[0].rstrip(" :")
                    col1 = c[1].strip()
                    if any(hdr in col0.lower() for hdr in ["investigation", "test name", "parameter", "name", "test no", "ref. no", "date", "gender", "source", "sample", "leucocytes", "erythrocytes", "platelets"]):
                        continue
                    if re.search(r"[\d\.]+", col1):
                        val, unit = split_val_unit(col1)
                        if len(c) >= 4 and c[3] and not unit:
                            unit = c[3].strip()
                        elif len(c) >= 3 and c[2] and not unit and not re.search(r"[\d\-]+", c[2]):
                            unit = c[2].strip()
                        results.append({
                            "type": cls.slugify_test_name(col0),
                            "name": col0,
                            "value": val,
                            "unit": unit
                        })
            # Markdown list item (- **Protein Total:** 11.00 mg/dL)
            elif "**" in line:
                m = re.search(r"[\-\*]*\s*\*\*([^\*:]+)(?::\*\*|\*\*:?)\s*([^\n\r\|]+)", line)
                if m:
                    k = m.group(1).strip()
                    v_full = m.group(2).strip()
                    if any(hdr in k.lower() for hdr in ["name", "patient", "age", "sex", "gender", "location", "date", "doctor", "investigation", "comments", "primary sample", "test method", "reference", "remark"]):
                        continue
                    if re.search(r"[\d\.]+", v_full):
                        val, unit = split_val_unit(v_full)
                        results.append({
                            "type": cls.slugify_test_name(k),
                            "name": k,
                            "value": val,
                            "unit": unit
                        })
            # Markdown table row (| WBC Count | 3000 | cells/mm3 |)
            elif "|" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2 and not all(set(p) <= {"-", ":", " "} for p in parts):
                    col0 = parts[0].rstrip(" :")
                    col1 = parts[1].strip()
                    if any(hdr in col0.lower() for hdr in ["investigation", "test", "parameter", "name", "protein creatinine ratio"]):
                        continue
                    if re.search(r"[\d\.]+", col1):
                        val, unit = split_val_unit(col1)
                        if len(parts) >= 3 and not unit:
                            unit = parts[2].strip()
                        results.append({
                            "type": cls.slugify_test_name(col0),
                            "name": col0,
                            "value": val,
                            "unit": unit
                        })

        return {
            "report_title": report_title,
            "patient_info": patient_info,
            "results": results
        }

    @classmethod
    def parse_and_validate(
        cls,
        raw_text: str,
        target_schema: Optional[Type[BaseModel]] = MedicalReportExtraction
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Parses raw LLM text, repairs JSON, or extracts table/markdown layouts,
        and validates against target Pydantic schema.
        Returns: (parsed_data_dict_or_none, error_message_or_none)
        """
        extracted = cls.extract_json_string(raw_text)
        parsed = None

        if extracted:
            try:
                parsed = json.loads(extracted)
            except json.JSONDecodeError:
                repaired = cls.repair_json_string(extracted)
                try:
                    parsed = json.loads(repaired)
                except json.JSONDecodeError as err:
                    logger.debug(f"Direct JSON decode failed: {err}")

        # If direct JSON was not found or failed, use universal Table/Markdown layout parser
        if parsed is None or not isinstance(parsed, (dict, list)):
            parsed = cls.parse_table_or_markdown_text(raw_text)

        if target_schema and isinstance(parsed, dict):
            try:
                normalized_parsed = cls.normalize_keys(parsed)
                validated_model = target_schema.model_validate(normalized_parsed)
                return validated_model.model_dump(), None
            except ValidationError as val_err:
                logger.warning(f"Schema validation soft warning: {val_err}")
                normalized_dict = cls.normalize_keys(parsed)
                return normalized_dict, None

        return parsed, None

