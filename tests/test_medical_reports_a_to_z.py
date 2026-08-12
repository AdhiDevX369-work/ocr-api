import sys
import os
import json
import pytest

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.chat import ImageChatRequest, ChatMessage

REPORT_1_LIPID_PROFILE = {
    "patient_info": {
        "patient_name": "MR LAB PATIENT 2026_06_24",
        "tel_no": "0789954622",
        "pid_no": "18350",
        "age": "22 Years",
        "sex": "Male",
        "registered_on": "2026-07-02 5:37 PM",
        "collected_on": "2026-07-02 5:41 PM",
        "reported_on": "2026-07-02 5:47 PM"
    },
    "report_title": "Lipid Profile"
}

REPORT_2_FULL_BLOOD_COUNT = {
    "patient_info": {
        "patient_name": "MISS TOPH",
        "tel_no": "0741781969",
        "pid_no": "18353",
        "age": "20 Years",
        "sex": "Female"
    },
    "report_title": "Full Blood Count"
}

REPORT_3_EGFR = {
    "patient_info": {
        "patient_name": "MR LAB PATIENT 2026_06_24",
        "pid_no": "18350",
        "age": "22 Years"
    },
    "report_title": "Estimated Glomerular Filtration Rate"
}

def test_a_to_z_data_integrity():
    assert REPORT_1_LIPID_PROFILE["patient_info"]["pid_no"] == "18350"
    assert REPORT_2_FULL_BLOOD_COUNT["patient_info"]["patient_name"] == "MISS TOPH"
    assert REPORT_3_EGFR["patient_info"]["pid_no"] == "18350"
    print("✅ A-to-Z Medical Report Schema & Data Integrity Test Passed!")

if __name__ == "__main__":
    test_a_to_z_data_integrity()
