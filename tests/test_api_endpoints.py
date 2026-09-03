import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app

class TestAPIEndpoints(unittest.TestCase):
    def test_root_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "online")
            self.assertIn("ocr_endpoints", data)

            response_ocr = client.get("/ocr")
            self.assertEqual(response_ocr.status_code, 200)

    def test_health_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn(data["status"], ["healthy", "degraded"])
            self.assertIn("database", data)
            self.assertEqual(data["database"]["status"], "healthy")

            response_ocr_health = client.get("/ocr/health")
            self.assertEqual(response_ocr_health.status_code, 200)

    def test_batch_list_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/ocr/api/batches?page=1&page_size=10")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("batches", data)
            self.assertIn("total", data)

    def test_clean_api_routes(self):
        with TestClient(app) as client:
            # Test /ocr/api/batches and /ocr/api/batch
            res_batch = client.get("/ocr/api/batches?page=1&page_size=10")
            self.assertEqual(res_batch.status_code, 200)

            res_batch_alias = client.get("/ocr/api/batch?page=1&page_size=10")
            self.assertEqual(res_batch_alias.status_code, 200)

            # Check openapi json contains all wrapped /ocr endpoints
            openapi_res = client.get("/openapi.json")
            self.assertEqual(openapi_res.status_code, 200)
            paths = openapi_res.json().get("paths", {})
            self.assertIn("/ocr/api/chat", paths)
            self.assertIn("/ocr/api/ocr", paths)
            self.assertIn("/ocr/api/ocr/sync", paths)
            self.assertIn("/ocr/api/ocr/upload", paths)
            self.assertIn("/ocr/api/batch", paths)
            self.assertIn("/ocr/api/jobs", paths)
            self.assertIn("/ocr/health", paths)

if __name__ == "__main__":
    unittest.main()
