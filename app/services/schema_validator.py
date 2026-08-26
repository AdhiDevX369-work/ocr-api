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

    @staticmethod
    def split_value_and_unit(val_str: str, unit_str: str) -> Tuple[str, str]:
        """
        Cleans and separates combined value and unit strings (e.g. '104 mg/dl' -> '104', 'mg/dl').
        Strips trailing status flags like 'Low', 'High', '*', '(L)', '(H)'.
        """
        v = (val_str or "").strip()
        u = (unit_str or "").strip()

        if not v:
            return "", u

        # 1. Strip trailing status flags from value (e.g. '11.00 Low' -> '11.00', '1.00 High' -> '1.00')
        v = re.sub(r"\s+(?:Low|High|Normal|Critical|Abnormal|\(L\)|\(H\)|\*)\b", "", v, flags=re.IGNORECASE).strip()

        # 2. Known common medical units to match at the end of value string
        known_units_pattern = r"(?:mg\/dL|mg\/dl|g\/dL|g\/dl|mEq\/L|meq\/l|mmol\/L|mmol\/l|µmol\/L|umol\/L|cells\/mm³|cells\/mm3|mill\/mm³|mill\/mm3|\/cumm|\/uL|\/µl|%|fL|fl|pg|g\/L|U\/L|IU\/L|IU\/mL|ng\/mL|ng\/dl|µg\/dL|ug\/dl|mL\/min\/1\.73m2|mL\/min|mm\/hr|Ratio)\b"

        # Check if unit is already present in value string
        unit_match = re.search(rf"\s*({known_units_pattern})\s*$", v, re.IGNORECASE)
        if unit_match:
            detected_unit = unit_match.group(1)
            v = v[:unit_match.start()].strip()
            if not u:
                u = detected_unit

        # Check for comparator operator attached to numbers (e.g., '< 14.00' or '<14.00' or '>= 60')
        comp_match = re.match(r"^([<>]=?|=)\s*([0-9]+(?:\.[0-9]+)?)\s*(.*)$", v)
        if comp_match:
            operator, num_part, remainder = comp_match.groups()
            v = f"{operator} {num_part}".strip()
            if remainder.strip() and not u:
                u = remainder.strip()

        return v, u

    @staticmethod
    def is_guideline_or_noise_item(name: str, val: str) -> bool:
        """
        Detects if an extracted row is actually an interpretation remark, age reference table row,
        sample description, or guideline chart rather than a patient test observation.
        """
        n = (name or "").strip().lower()
        v = (val or "").strip().lower()

        # Ignore empty garbage
        if not n and not v:
            return True

        # Header / Section / Non-test rows
        ignored_name_keywords = [
            "interpretation", "remark", "protein creatinine ratio remark",
            "low grade proteinuria", "moderate proteinuria", "nephrosis",
            "primary sample type", "sample type", "test method", "method", "methodology",
            "average estimated gfr by age", "estimated gfr by age", "age related normal",
            "upcr high levels cause", "upcr low levels cause", "comments", "comment",
            "thanks for reference", "end of report", "notes", "note", "instrument",
            "department", "biochemistry", "hematology", "clinical pathology", "parameter"
        ]
        for kw in ignored_name_keywords:
            if kw in n:
                return True

        # Non-test values (e.g. value is a medical diagnosis description rather than a measurement)
        if v in ["remark", "low grade proteinuria", "moderate proteinuria", "nephrosis"]:
            return True

        # Age group reference rows (e.g. name is "18-29", "30-39", "40-49", "50-59", "60-69", "70+")
        if re.match(r"^(?:18\s*-\s*29|20\s*-\s*29|30\s*-\s*39|40\s*-\s*49|50\s*-\s*59|60\s*-\s*69|70\s*\+|80\s*\+)(?:\s*years)?$", n):
            return True

        # Pure ranges as name (e.g. "< 0.2", "0.2 - 1.0", "1.0 - 5.0", "> 5.0")
        if re.match(r"^[<>]?\s*[0-9]+(?:\.[0-9]+)?(?:\s*-\s*[0-9]+(?:\.[0-9]+)?)?$", n):
            return True

        return False

    @staticmethod
    def slugify_test_name(name: str) -> str:
        """Converts any test name to standardized lowercase canonical slug identifier."""
        if not name:
            return "unknown_test"
        clean = name.strip()
        lower = clean.lower()

        alias_map = {
            # Glucose & Glycated
            "hba1c": "hba1c",
            "glycated hemoglobin": "hba1c",
            "glycosylated hemoglobin": "hba1c",
            "fbs": "fbs",
            "fasting blood sugar": "fbs",
            "fasting blood glucose": "fbs",
            "fasting plasma glucose": "fbs",
            "glucose fasting": "fbs",
            "glucose - fasting": "fbs",
            "glucose, fasting": "fbs",
            "ppbs": "ppbs",
            "post prandial blood sugar": "ppbs",
            "post prandial blood glucose": "ppbs",
            "glucose post prandial": "ppbs",
            "rbs": "rbs",
            "random blood sugar": "rbs",
            "random blood glucose": "rbs",
            "glucose random": "rbs",

            # Urine & Renal Proteins
            "protein total": "protein_total",
            "total protein": "protein_total",
            "urine protein total": "urine_protein_total",
            "urine protein": "urine_protein",
            "urine creatinine": "urine_creatinine",
            "creatinin": "creatinine",
            "creatinine": "creatinine",
            "serum creatinine": "serum_creatinine",
            "protein creatinine ratio": "protein_creatinine_ratio",
            "protein - creatinine ratio": "protein_creatinine_ratio",
            "protein / creatinine ratio": "protein_creatinine_ratio",
            "upcr": "protein_creatinine_ratio",
            "microalbumin": "microalbumin",
            "urine microalbumin": "urine_microalbumin",
            "albumin creatinine ratio": "albumin_creatinine_ratio",
            "acr": "albumin_creatinine_ratio",

            # Renal Function
            "blood urea": "blood_urea",
            "urea": "blood_urea",
            "bun": "blood_urea_nitrogen",
            "blood urea nitrogen": "blood_urea_nitrogen",
            "egfr": "egfr",
            "estimated gfr": "egfr",
            "estimated glomerular filtration rate": "egfr",
            "serum uric acid": "uric_acid",
            "uric acid": "uric_acid",

            # Lipid Profile
            "total cholesterol": "total_cholesterol",
            "cholesterol - total": "total_cholesterol",
            "cholesterol total": "total_cholesterol",
            "cholesterol": "total_cholesterol",
            "serum cholesterol": "total_cholesterol",
            "triglycerides": "triglycerides",
            "serum triglycerides": "triglycerides",
            "hdl": "hdl_cholesterol",
            "hdl cholesterol": "hdl_cholesterol",
            "hdl - cholesterol": "hdl_cholesterol",
            "direct hdl": "hdl_cholesterol",
            "ldl": "ldl_cholesterol",
            "ldl cholesterol": "ldl_cholesterol",
            "ldl - cholesterol": "ldl_cholesterol",
            "calculated ldl": "ldl_cholesterol",
            "vldl": "vldl_cholesterol",
            "vldl cholesterol": "vldl_cholesterol",
            "vldl - cholesterol": "vldl_cholesterol",
            "chol/hdl ratio": "chol_hdl_ratio",
            "cholesterol / hdl ratio": "chol_hdl_ratio",
            "tc/hdl ratio": "chol_hdl_ratio",
            "ldl/hdl ratio": "ldl_hdl_ratio",
            "non-hdl cholesterol": "non_hdl_cholesterol",

            # Hematology & CBC
            "wbc count": "wbc_count",
            "total wbc": "wbc_count",
            "total leucocyte count": "wbc_count",
            "total leukocyte count": "wbc_count",
            "w.b.c": "wbc_count",
            "white blood cells": "wbc_count",
            "wbc": "wbc_count",
            "rbc count": "rbc_count",
            "total rbc": "rbc_count",
            "total erythrocyte count": "rbc_count",
            "r.b.c": "rbc_count",
            "red blood cells": "rbc_count",
            "rbc": "rbc_count",
            "hemoglobin": "hemoglobin",
            "haemoglobin": "hemoglobin",
            "hb": "hemoglobin",
            "hb (hemoglobin)": "hemoglobin",
            "platelet count": "platelet_count",
            "platelets": "platelet_count",
            "total platelets": "platelet_count",
            "pcv": "pcv",
            "pcv (packed cell volume)": "pcv",
            "packed cell volume": "pcv",
            "hematocrit": "hematocrit",
            "mcv": "mcv",
            "mcv (mean corpuscular volume)": "mcv",
            "mean corpuscular volume": "mcv",
            "mch": "mch",
            "mch (mean corpuscular hemoglobin)": "mch",
            "mean corpuscular hemoglobin": "mch",
            "mchc": "mchc",
            "mchc (mean corpuscular hemoglobin concentration)": "mchc",
            "mean corpuscular hemoglobin concentration": "mchc",
            "rdw": "rdw",
            "rdw-cv": "rdw_cv",
            "rdw-sd": "rdw_sd",
            "neutrophils": "neutrophils",
            "segmented neutrophils": "neutrophils",
            "lymphocytes": "lymphocytes",
            "monocytes": "monocytes",
            "eosinophils": "eosinophils",
            "basophils": "basophils",
            "esr": "esr",
            "erythrocyte sedimentation rate": "esr",

            # Liver Function (LFT)
            "sgpt": "sgpt_alt",
            "sgpt / alt": "sgpt_alt",
            "alt": "sgpt_alt",
            "alanine aminotransferase": "sgpt_alt",
            "sgot": "sgot_ast",
            "sgot / ast": "sgot_ast",
            "ast": "sgot_ast",
            "aspartate aminotransferase": "sgot_ast",
            "alkaline phosphatase": "alkaline_phosphatase",
            "alp": "alkaline_phosphatase",
            "bilirubin total": "bilirubin_total",
            "total bilirubin": "bilirubin_total",
            "bilirubin direct": "bilirubin_direct",
            "direct bilirubin": "bilirubin_direct",
            "bilirubin indirect": "bilirubin_indirect",
            "gamma gt": "ggt",
            "ggt": "ggt",
            "serum albumin": "serum_albumin",
            "albumin": "albumin",
            "serum globulin": "serum_globulin",
            "globulin": "globulin",
            "a:g ratio": "ag_ratio",
            "a/g ratio": "ag_ratio",

            # Thyroid & Hormones
            "tsh": "tsh",
            "thyroid stimulating hormone": "tsh",
            "t3": "total_t3",
            "t4": "total_t4",
            "free t3": "free_t3",
            "ft3": "free_t3",
            "free t4": "free_t4",
            "ft4": "free_t4",

            # Electrolytes & Minerals
            "serum sodium": "serum_sodium",
            "sodium": "serum_sodium",
            "na+": "serum_sodium",
            "serum potassium": "serum_potassium",
            "potassium": "serum_potassium",
            "k+": "serum_potassium",
            "serum chloride": "serum_chloride",
            "chloride": "serum_chloride",
            "cl-": "serum_chloride",
            "calcium": "calcium",
            "serum calcium": "calcium",
            "serum phosphorus": "phosphorus",
            "magnesium": "magnesium",
            "crp": "crp",
            "c-reactive protein": "crp"
        }

        if lower in alias_map:
            return alias_map[lower]

        # Clean punctuation and slugify
        s = re.sub(r"[^\w\s-]", "", lower)
        s = re.sub(r"[\s_-]+", "_", s).strip("_")
        return s or "test_parameter"

    @classmethod
    def normalize_results_list(cls, raw_list: Any) -> list[Dict[str, Any]]:
        """Normalizes any array or dictionary of test items into standardized [ResultItem] format."""
        standardized_items = []

        def process_raw_item(item: Any):
            if not isinstance(item, dict):
                return

            # 1. Extract test name
            name = (
                item.get("name")
                or item.get("parameter")
                or item.get("investigation")
                or item.get("test_name")
                or item.get("test")
                or ""
            )
            name_str = str(name).strip()

            # 2. Extract value
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

            # 3. Extract unit
            unit = item.get("unit") or item.get("units") or item.get("measurement_unit") or ""
            unit_str = str(unit).strip() if unit is not None else ""

            # 4. Filter out guideline / interpretation chart noise
            if cls.is_guideline_or_noise_item(name_str, val_str):
                return

            # 5. Clean & Split combined Value and Unit (e.g. '104 mg/dl' -> value: '104', unit: 'mg/dl')
            clean_val, clean_unit = cls.split_value_and_unit(val_str, unit_str)

            # 6. Extract or generate canonical slug type
            generic_categories = {
                "leucocytes", "erythrocytes", "platelets", "investigations", "general",
                "parameters", "tests", "test", "results", "blood", "urine", "biochemistry",
                "lipid_profile", "renal_profile", "liver_function"
            }
            slug_type = item.get("type") or ""
            if not slug_type or not isinstance(slug_type, str) or slug_type.strip().lower() in generic_categories:
                slug_type = cls.slugify_test_name(name_str)
            else:
                slug_type = cls.slugify_test_name(slug_type)

            if name_str or clean_val:
                standardized_items.append({
                    "type": slug_type,
                    "name": name_str or slug_type.replace("_", " ").title(),
                    "value": clean_val,
                    "unit": clean_unit
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
        """Maps LLM JSON output into strict standardized structure with patient_info and results array."""
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

            # PID / Test No / Sample ID mapping
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

        # If not found in primary keys, check nested sections (e.g. full_blood_count, lipid_profile, renal_profile)
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

        # Return strictly clean structured dictionary
        return {
            "report_title": report_title,
            "patient_info": patient_info,
            "results": standardized_results
        }

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
                normalized_dict = cls.normalize_keys(parsed)
                return normalized_dict, None

        return parsed, None
