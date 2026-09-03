.PHONY: help run run-ui test benchmark clean lint

help:
	@echo "========================================================================"
	@echo " 🚀 Vision OCR & Medical Document Processing Platform"
	@echo "========================================================================"
	@echo " Commands:"
	@echo "   make run         : Start the FastAPI Backend Server on port 8200"
	@echo "   make run-ui      : Start the Streamlit Dashboard on port 8501"
	@echo "   make test        : Run all unit and integration tests"
	@echo "   make benchmark   : Run canonical Ground Truth accuracy benchmark suite"
	@echo "   make clean       : Clean temporary cache, logs, and build artifacts"
	@echo "========================================================================"

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8200 --reload

run-ui:
	streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

test:
	pytest tests/ -v

benchmark:
	python tests/test_ground_truth_accuracy.py --engine native

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	rm -rf .pytest_cache .coverage htmlcov
