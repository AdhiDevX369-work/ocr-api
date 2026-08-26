import os
import sys
import json
import asyncio
import unittest

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
from app.db.session import init_db, async_session_factory
from app.db.models import BatchModel, JobModel, WebhookDeliveryModel
from app.services.image_processor import ImageProcessor
from app.services.schema_validator import SchemaValidator
from app.services.webhook_dispatcher import WebhookDispatcher
from app.services.batch_service import batch_service
from app.services.job_service import job_service
from app.schemas.batch import BatchCreateRequest, BatchDocumentInput
from app.schemas.medical import MedicalReportExtraction, PatientInfo, InvestigationItem

class TestProductionVOCRPipeline(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await init_db()

    async def test_01_schema_validator_and_repair(self):
        """Tests robust JSON extraction and repair from noisy LLM outputs."""
        noisy_output = """
        <think>Analyzing medical report scan...</think>
        Here is the extracted patient report in JSON:
        ```json
        {
            "report_title": "FULL BLOOD COUNT",
            "patient_info": {
                "patient_name": "JOHN DOE",
                "age": "45 Y",
                "sex": "Male",
            },
            "results": [
                {
                    "type": "wbc_count",
                    "name": "WBC Count",
                    "value": "7500",
                    "unit": "/cumm",
                },
                {
                    "name": "HbA1C",
                    "value": "6.2",
                    "unit": "%",
                }
            ],
        }
        ```
        Hope this helps!
        """
        parsed, err = SchemaValidator.parse_and_validate(noisy_output, MedicalReportExtraction)
        self.assertIsNotNone(parsed, f"Parsing failed: {err}")
        self.assertEqual(parsed["report_title"], "FULL BLOOD COUNT")
        self.assertEqual(parsed["patient_info"]["patient_name"], "JOHN DOE")
        self.assertEqual(len(parsed["results"]), 2)
        self.assertEqual(parsed["results"][0]["type"], "wbc_count")
        self.assertEqual(parsed["results"][0]["name"], "WBC Count")
        self.assertEqual(parsed["results"][0]["value"], "7500")
        self.assertEqual(parsed["results"][0]["unit"], "/cumm")
        self.assertEqual(parsed["results"][1]["type"], "hba1c")
        self.assertEqual(parsed["results"][1]["name"], "HbA1C")
        self.assertEqual(parsed["results"][1]["value"], "6.2")
        self.assertEqual(parsed["results"][1]["unit"], "%")
        print("✅ Test 01: Schema Validator & JSON Auto-Repair passed!")

    async def test_02_image_processor_and_hybrid_ocr(self):
        """Tests image and PDF rendering and digital text extraction."""
        pdf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pdf"))
        fbc_path = os.path.join(pdf_dir, "FBC.pdf")

        if os.path.exists(fbc_path):
            with open(fbc_path, "rb") as f:
                pdf_bytes = f.read()

            self.assertTrue(ImageProcessor.is_pdf(pdf_bytes))
            doc_res = ImageProcessor.process_document(pdf_bytes)
            self.assertTrue(doc_res["is_pdf"])
            self.assertGreater(doc_res["page_count"], 0)
            self.assertTrue(doc_res["primary_data_uri"].startswith("data:image/jpeg;base64,"))
            self.assertEqual(len(doc_res["page_data_uris"]), doc_res["page_count"])
            print(f"✅ Test 02: Image Processor rendered {doc_res['page_count']} page(s) successfully!")
        else:
            print("⚠️ FBC.pdf not found, skipping file read.")

    async def test_03_webhook_hmac_signature(self):
        """Tests HMAC-SHA256 signature generation for webhooks."""
        test_payload = b'{"event_type": "report.processed", "job_id": "test_123"}'
        sig = WebhookDispatcher.calculate_hmac_signature(test_payload, secret="test-secret")
        self.assertIsNotNone(sig)
        self.assertEqual(len(sig), 64)  # SHA-256 hex string length
        print(f"✅ Test 03: Webhook HMAC-SHA256 signature verified: {sig[:16]}...")

    async def test_04_batch_service_lifecycle_and_export(self):
        """Tests creating a batch, creating child jobs, polling status, and generating archive."""
        dummy_uri = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        batch_req = BatchCreateRequest(
            name="Unit_Test_Batch",
            documents=[
                BatchDocumentInput(document=dummy_uri, name="test_report_1.pdf"),
                BatchDocumentInput(document=dummy_uri, name="test_report_2.pdf")
            ],
            prompt="Extract all data",
            backend="ollama",
            model="qwen2.5vl:latest"
        )

        batch_res = await batch_service.create_batch(batch_req)
        self.assertIsNotNone(batch_res.batch_id)
        self.assertEqual(batch_res.total_files, 2)
        print(f"✅ Test 04.1: Batch created with ID '{batch_res.batch_id}' and 2 child jobs")

        # Test Batch Detail
        detail = await batch_service.get_batch_detail(batch_res.batch_id)
        self.assertIsNotNone(detail)
        self.assertEqual(len(detail.jobs), 2)
        print(f"✅ Test 04.2: Retrieved batch detail with {len(detail.jobs)} child job records")

        # Test Batch Export generation
        json_bytes, media_type, filename = await batch_service.generate_batch_archive(batch_res.batch_id, format_type="json")
        self.assertEqual(media_type, "application/json")
        self.assertTrue(len(json_bytes) > 0)

        zip_bytes, zip_media, zip_name = await batch_service.generate_batch_archive(batch_res.batch_id, format_type="zip")
        self.assertEqual(zip_media, "application/zip")
        self.assertTrue(len(zip_bytes) > 0)
        print(f"✅ Test 04.3: Batch JSON & ZIP export archives generated successfully!")

    async def test_05_hana_diagnostic_structure(self):
        """Tests parsing a Hana Diagnostic report with dynamic algorithmic slugification."""
        raw_llm_hana = """
        {
            "report_title": "DIAGNOSTIC REPORT - Biochemistry",
            "patient_info": {
                "patient_name": "FAHIDHA",
                "pid_no": "2489",
                "ref_no": "6/12",
                "age": "51 (Y)",
                "gender": "Female",
                "date": "12-09-2023 08:33 am",
                "source": "HANA DIAGNOSTIC LABORATORY & ECG CENTRE"
            },
            "results": [
                {
                    "type": "fasting_blood_sugar",
                    "name": "Fasting Blood Sugar",
                    "value": "104",
                    "unit": "mg/dl"
                },
                {
                    "type": "total_cholesterol",
                    "name": "Total Cholesterol",
                    "value": "230",
                    "unit": "mg/dl"
                }
            ]
        }
        """
        parsed, err = SchemaValidator.parse_and_validate(raw_llm_hana, MedicalReportExtraction)
        self.assertIsNotNone(parsed, f"Parsing failed: {err}")
        self.assertEqual(parsed["patient_info"]["patient_name"], "FAHIDHA")
        self.assertEqual(len(parsed["results"]), 2)
        
        self.assertEqual(parsed["results"][0]["type"], "fasting_blood_sugar")
        self.assertEqual(parsed["results"][0]["name"], "Fasting Blood Sugar")
        self.assertEqual(parsed["results"][0]["value"], "104")
        self.assertEqual(parsed["results"][0]["unit"], "mg/dl")

        self.assertEqual(parsed["results"][1]["type"], "total_cholesterol")
        self.assertEqual(parsed["results"][1]["name"], "Total Cholesterol")
        self.assertEqual(parsed["results"][1]["value"], "230")
        self.assertEqual(parsed["results"][1]["unit"], "mg/dl")
        print("✅ Test 05: Hana Diagnostic structure validation passed!")

    async def test_06_drlogy_upcr_structure(self):
        """Tests parsing Drlogy UPCR report with zero hardcoding."""
        raw_llm_drlogy = """
        {
            "report_title": "URINE PROTEIN - CREATININE RATIO (UPCR)",
            "patient_info": {
                "name": "Yash M. Patel",
                "age": "21 Years",
                "sex": "Male",
                "pid": "555",
                "sample_collected_at": "125, Shivam Bungalow, S G Road, Mumbai",
                "ref_by": "Dr. Hiren Shah",
                "registered_on": "02:31 PM 02 Dec, 202X"
            },
            "results": [
                {
                    "name": "Protein Total",
                    "value": "11.00",
                    "unit": "mg/dL"
                },
                {
                    "name": "Creatinin",
                    "value": "11.00",
                    "unit": "mEq/L"
                },
                {
                    "name": "Protein Creatinine Ratio",
                    "value": "1.00",
                    "unit": ""
                }
            ]
        }
        """
        parsed, err = SchemaValidator.parse_and_validate(raw_llm_drlogy, MedicalReportExtraction)
        self.assertIsNotNone(parsed, f"Parsing failed: {err}")
        self.assertEqual(parsed["patient_info"]["patient_name"], "Yash M. Patel")
        self.assertEqual(parsed["patient_info"]["pid_no"], "555")
        self.assertEqual(len(parsed["results"]), 3)
        
        self.assertEqual(parsed["results"][0]["type"], "protein_total")
        self.assertEqual(parsed["results"][0]["name"], "Protein Total")
        self.assertEqual(parsed["results"][0]["value"], "11.00")
        self.assertEqual(parsed["results"][0]["unit"], "mg/dL")

        self.assertEqual(parsed["results"][1]["type"], "creatinin")
        self.assertEqual(parsed["results"][1]["name"], "Creatinin")
        self.assertEqual(parsed["results"][1]["value"], "11.00")
        self.assertEqual(parsed["results"][1]["unit"], "mEq/L")

        self.assertEqual(parsed["results"][2]["type"], "protein_creatinine_ratio")
        self.assertEqual(parsed["results"][2]["name"], "Protein Creatinine Ratio")
        self.assertEqual(parsed["results"][2]["value"], "1.00")
        self.assertEqual(parsed["results"][2]["unit"], "")
        print("✅ Test 06: Drlogy UPCR extraction passed!")

    async def test_07_novel_unseen_biomarkers_slugification(self):
        """Tests that any novel, unseen clinical test parameter is dynamically slugified with zero hardcoded entries."""
        raw_novel_report = """
        {
            "report_title": "ADVANCED IMMUNOLOGY & ONCOLOGY PANEL",
            "results": [
                {
                    "name": "Interleukin-6 (IL-6)",
                    "value": "14.2",
                    "unit": "pg/mL"
                },
                {
                    "name": "Anti-Müllerian Hormone [AMH]",
                    "value": "3.8",
                    "unit": "ng/mL"
                },
                {
                    "name": "D-Dimer (Quantitative)",
                    "value": "0.45",
                    "unit": "µg/mL FEU"
                }
            ]
        }
        """
        parsed, err = SchemaValidator.parse_and_validate(raw_novel_report, MedicalReportExtraction)
        self.assertIsNotNone(parsed, f"Parsing failed: {err}")
        self.assertEqual(len(parsed["results"]), 3)
        self.assertEqual(parsed["results"][0]["type"], "interleukin_6_il_6")
        self.assertEqual(parsed["results"][0]["name"], "Interleukin-6 (IL-6)")
        self.assertEqual(parsed["results"][1]["type"], "anti_müllerian_hormone_amh")
        self.assertEqual(parsed["results"][1]["name"], "Anti-Müllerian Hormone [AMH]")
        self.assertEqual(parsed["results"][2]["type"], "d_dimer_quantitative")
        self.assertEqual(parsed["results"][2]["name"], "D-Dimer (Quantitative)")
        print("✅ Test 07: Unseen novel biomarkers zero-hardcode slugification passed!")

if __name__ == "__main__":
    unittest.main()
