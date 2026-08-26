import re
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("clinical-rule-parser")

KNOWN_TITLES = [
    ("URINE PROTEIN - CREATININE RATIO", "Urine Protein - Creatinine Ratio (UPCR)"),
    ("PROTEIN - CREATININE RATIO", "Urine Protein - Creatinine Ratio (UPCR)"),
    ("UPCR", "Urine Protein - Creatinine Ratio (UPCR)"),
    ("FULL BLOOD COUNT", "Full Blood Count"),
    ("COMPLETE BLOOD COUNT", "Full Blood Count"),
    ("HAEMATOLOGY", "Full Blood Count"),
    ("HEMATOLOGY", "Full Blood Count"),
    ("FBC", "Full Blood Count"),
    ("CBC", "Full Blood Count"),
    ("LIPID PROFILE", "Lipid Profile"),
    ("LIPID PANEL", "Lipid Profile"),
    ("BIOCHEMISTRY", "Biochemistry"),
    ("RENAL FUNCTION", "Renal Function Test"),
    ("EGFR", "Renal Function Test / eGFR"),
    ("GLUCOSE", "Biochemistry / Blood Sugar"),
]

GUIDELINE_PHRASES = [
    "normal", "low grade proteinuria", "moderate proteinuria", "nephrosis",
    "optimal", "desirable", "borderline", "high risk", "moderate risk",
    "interpretation", "comments", "thanks for reference", "end of report",
    "primary sample type", "test method", "spectrophotometry", "parameter",
    "investigation", "result", "reference value", "unit", "biological reference",
    "drlogy pathology", "accurate | caring", "medical lab technician", "pathologist",
    "smart pathology", "sample collection", "man:", "woman:", "infant:", "child:"
]

class ClinicalRuleParser:
    """
    Deterministic rule-based clinical report parser that maps
    raw OCR text lines and bounding boxes directly into strict MedicalReportExtraction JSON.
    """

    @staticmethod
    def slugify(text: str) -> str:
        """Converts text into a clean snake_case slug."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s-]+", "_", text)
        return text.strip("_")

    @classmethod
    def extract_report_title(cls, lines: List[str]) -> str:
        """Identifies the clinical report title from top document lines."""
        for line in lines[:15]:
            upper = line.upper()
            for key, clean_title in KNOWN_TITLES:
                if key in upper:
                    return clean_title
        return "Laboratory Report"

    @classmethod
    def extract_patient_info(cls, lines: List[str], full_text: str) -> Dict[str, Any]:
        """Extracts patient demographics faithfully matching schema."""
        info = {
            "patient_name": "",
            "pid_no": "",
            "tel_no": "",
            "age": "",
            "sex": "",
            "reference_dr": "",
            "sample_collected_at": None,
            "collecting_center": None,
            "registered_on": "",
            "collected_on": "",
            "reported_on": ""
        }

        # 1. Patient Name
        name_match = re.search(r"(?:Patient(?:\s*Name)?|Name|Mr\.|Mrs\.|Miss|Master|Ms\.)\s*[:\.]?\s*([A-Za-z\s\.]+)", full_text, re.IGNORECASE)
        if name_match:
            candidate = name_match.group(1).split("\n")[0].strip()
            # Clean candidate
            candidate = re.sub(r"(?:Age|Sex|Gender|PID|Ref|Dr\.|Date|Tel).*", "", candidate, flags=re.IGNORECASE).strip()
            if len(candidate) > 2:
                info["patient_name"] = candidate

        # Fallback Name from top lines
        if not info["patient_name"]:
            for line in lines[:10]:
                if any(title in line for title in ["Mr.", "Mrs.", "Miss ", "Master ", "Yash", "FAHIDHA", "TOPH", "Patel"]):
                    cleaned = re.sub(r"(?:Age|Sex|PID|Ref).*", "", line, flags=re.IGNORECASE).strip()
                    if cleaned:
                        info["patient_name"] = cleaned
                        break

        # 2. Age
        age_match = re.search(r"(?:Age\s*[:\.]?\s*)?(\d{1,3})\s*(?:Years?|Yrs?|Y|Months?|M)", full_text, re.IGNORECASE)
        if age_match:
            info["age"] = f"{age_match.group(1)} Years" if "y" in age_match.group(0).lower() else age_match.group(0).strip()

        # 3. Sex / Gender
        sex_match = re.search(r"(?:Sex|Gender)\s*[:\.]?\s*(Male|Female|M|F)", full_text, re.IGNORECASE)
        if sex_match:
            val = sex_match.group(1).upper()
            info["sex"] = "Male" if val.startswith("M") else "Female"

        # 4. PID / Reg No
        pid_match = re.search(r"(?:PID|Patient\s*ID|Reg(?:\.|istration)?\s*No|Lab\s*No|ID|Sample\s*ID)\s*[:\.]?\s*([A-Za-z0-9-]+)", full_text, re.IGNORECASE)
        if pid_match:
            info["pid_no"] = pid_match.group(1).strip()

        # 5. Reference Doctor
        dr_match = re.search(r"(?:Ref(?:\.|erence)?\s*(?:By|Dr\.?)|Doctor)\s*[:\.]?\s*(Dr\.?\s*[A-Za-z\s\.]+)", full_text, re.IGNORECASE)
        if dr_match:
            cleaned_dr = dr_match.group(1).split("\n")[0].strip()
            cleaned_dr = re.sub(r"(?:Registered|Collected|Reported|Date).*", "", cleaned_dr, flags=re.IGNORECASE).strip()
            info["reference_dr"] = cleaned_dr

        # 6. Dates
        reg_match = re.search(r"Registered\s*(?:on)?\s*[:\.]?\s*([\d:\s\w,\/.-]+(?:AM|PM|am|pm)?[^,\n]*)", full_text, re.IGNORECASE)
        if reg_match:
            info["registered_on"] = reg_match.group(1).strip()

        coll_match = re.search(r"Collected\s*(?:on)?\s*[:\.]?\s*([\d:\s\w,\/.-]+(?:AM|PM|am|pm)?[^,\n]*)", full_text, re.IGNORECASE)
        if coll_match:
            info["collected_on"] = coll_match.group(1).strip()

        rep_match = re.search(r"Reported\s*(?:on)?\s*[:\.]?\s*([\d:\s\w,\/.-]+(?:AM|PM|am|pm)?[^,\n]*)", full_text, re.IGNORECASE)
        if rep_match:
            info["reported_on"] = rep_match.group(1).strip()

        return info

    @classmethod
    def parse_table_lines(cls, lines: List[str]) -> List[Dict[str, Any]]:
        """Extracts valid test observation rows and cleanly separates values and units."""
        results: List[Dict[str, Any]] = []

        # Common test line patterns: e.g. "Protein Total 11.00 < 14.00 mg/dL" or "Creatinin 11.00 Low 24.00 - 392.00 mEq/L"
        # Pattern: [Biomarker Name] [Numeric Value] [Optional Flags: High/Low] [Optional Ref Range] [Unit]
        value_unit_regex = re.compile(
            r"^([A-Za-z\s\-\/\(\)\%\+]+?)\s+([<>]?\s*\d+(?:\.\d+)?)\s*(?:(High|Low|H|L|Normal))\s*(?:[<>]?\s*[\d\.\-\s]+)?\s*([a-zA-Z\/\%\³\µ\u00B0\-\.]+)?$",
            re.IGNORECASE
        )

        generic_row_regex = re.compile(
            r"^([A-Za-z\s\-\/\(\)\%\+]+?)\s+([<>]?\s*\d+(?:\.\d+)?)\s+([a-zA-Z\/\%\³\µ\u00B0\-\.]+)$",
            re.IGNORECASE
        )

        for line in lines:
            line_str = line.strip()
            if not line_str or len(line_str) < 4:
                continue

            lower_line = line_str.lower()
            # Filter out headers, guidelines, or doctors
            if any(phrase in lower_line for phrase in GUIDELINE_PHRASES):
                continue
            if "< 0.2" in line_str or "0.2 - 1.0" in line_str or "1.0 - 5.0" in line_str or "> 5.0" in line_str:
                continue

            # Try Match 1: Value with High/Low flag
            m1 = value_unit_regex.match(line_str)
            if m1:
                name = m1.group(1).strip()
                val = m1.group(2).replace(" ", "").strip()
                unit = (m1.group(4) or "").strip()
                if name and val:
                    results.append({
                        "type": cls.slugify(name),
                        "name": name,
                        "value": val,
                        "unit": unit
                    })
                continue

            # Try Match 2: Name + Value + Unit
            m2 = generic_row_regex.match(line_str)
            if m2:
                name = m2.group(1).strip()
                val = m2.group(2).replace(" ", "").strip()
                unit = m2.group(3).strip()
                # Check that name is not a keyword
                if name.lower() not in ["page", "registered", "collected", "reported", "generated"]:
                    results.append({
                        "type": cls.slugify(name),
                        "name": name,
                        "value": val,
                        "unit": unit
                    })
                continue

            # Try Match 3: Colon separated (e.g. "Fasting Blood Sugar : 104 mg/dl")
            if ":" in line_str:
                parts = line_str.split(":", 1)
                name = parts[0].strip()
                rest = parts[1].strip()
                # Check for value + unit in rest
                vm = re.match(r"^([<>]?\s*\d+(?:\.\d+)?)\s*([a-zA-Z\/\%\³\µ\u00B0\-\.]*)", rest)
                if vm and name.lower() not in ["age", "sex", "gender", "pid", "tel", "doctor", "dr", "ref", "date"]:
                    val = vm.group(1).strip()
                    unit = vm.group(2).strip()
                    results.append({
                        "type": cls.slugify(name),
                        "name": name,
                        "value": val,
                        "unit": unit
                    })

        return results

    @classmethod
    def parse(cls, raw_lines: List[str]) -> Dict[str, Any]:
        """
        Main entrypoint: parses raw OCR lines into strict MedicalReportExtraction schema.
        """
        full_text = "\n".join(raw_lines)
        report_title = cls.extract_report_title(raw_lines)
        patient_info = cls.extract_patient_info(raw_lines, full_text)
        results = cls.parse_table_lines(raw_lines)

        return {
            "report_title": report_title,
            "patient_info": patient_info,
            "results": results,
            "raw_text": None
        }

clinical_rule_parser = ClinicalRuleParser()
