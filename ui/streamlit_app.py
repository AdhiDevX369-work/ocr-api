import sys
import os

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import time
import json
import base64
import glob
import urllib.request
import requests
import streamlit as st
from PIL import Image

try:
    import httpx
except ImportError:
    httpx = None

try:
    import pandas as pd
except ImportError:
    pd = None

from app.config import settings

# Page Configuration
st.set_page_config(
    page_title="Medical Report OCR & Clinical AI Studio",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling & Glassmorphism Theme
st.markdown("""
<style>
    /* Dark Slate Theme */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    .main-title {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.0rem;
    }
    .badge-healthy {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-offline {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .patient-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 15px;
    }
    .patient-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 8px;
    }
    .patient-item {
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-bottom: 4px;
    }
    .patient-item strong {
        color: #f8fafc;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_image_b64" not in st.session_state:
    st.session_state.current_image_b64 = None
if "current_image_display" not in st.session_state:
    st.session_state.current_image_display = None
if "last_extracted_data" not in st.session_state:
    st.session_state.last_extracted_data = None
if "last_extraction_duration" not in st.session_state:
    st.session_state.last_extraction_duration = None
if "active_batch_id" not in st.session_state:
    st.session_state.active_batch_id = None

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-history.png", width=56)
    st.markdown("### 🩺 OCR Engine & Model Settings")

    api_base_url = st.text_input("API Base URL", value="http://localhost:8200")

    # Engine Selection
    engine_choice = st.selectbox(
        "🚀 OCR Engine Pipeline",
        options=[
            "🏎️ Hybrid Pipeline (Fast OCR + Mistral Structurer)",
            "⚡ Vision LLM (Direct Multimodal - Ministral-3)",
            "🚀 Native Sub-Second OCR (Non-LLM Optical)"
        ],
        index=0,
        help="Hybrid Pipeline extracts raw text in 2s with Native OCR and structures clinical data with Mistral in text-only mode."
    )

    if "Hybrid" in engine_choice:
        engine_param = "hybrid"
    elif "Vision" in engine_choice:
        engine_param = "vocr"
    else:
        engine_param = "native"

    # Backend & Model Selection
    if engine_param in ("vocr", "hybrid"):
        backend_options = ["ollama", "llm-server", "llama-cpp"]
        default_backend_idx = backend_options.index(settings.default_backend) if settings.default_backend in backend_options else 0
        backend = st.selectbox("LLM Backend", options=backend_options, index=default_backend_idx)
        
        env_model = settings.default_model or "ministral-3:latest"
        known_models = ["ministral-3:latest", "medgemma:latest", "qwen3-vl:4b-fast", "qwen2.5vl:latest", "deepseek-ocr:3b", "qwen2.5:7b"]
        if env_model and env_model not in known_models:
            known_models.insert(0, env_model)

        default_model_idx = known_models.index(env_model) if env_model in known_models else 0

        selected_model_option = st.selectbox(
            "Classifier / Structurer Model" if engine_param == "hybrid" else "Vision LLM Model",
            options=known_models + ["Custom..."],
            index=default_model_idx,
            help=f"Active default model in .env: '{env_model}'"
        )
        if selected_model_option == "Custom...":
            model_name = st.text_input("Custom Model Name", value=env_model)
        else:
            model_name = selected_model_option
    else:
        backend = "native-engine"
        model_name = "Native-OCR-PP"
        st.info("🏎️ Running Native PyTorch OCR Engine (No LLM required, sub-second latency).")

    st.divider()

    st.markdown("### 🎯 System Preset Mode")
    preset_mode = st.selectbox(
        "Task Mode",
        options=[
            "🩺 Medical Lab Report Extractor",
            "🔍 Print Verification & Line Check",
            "📋 General Document OCR & Tables",
            "🧾 Invoice / Receipt Parser",
            "🤖 Custom Prompt"
        ]
    )

    default_system_prompts = {
        "🩺 Medical Lab Report Extractor": (
            "You are an expert Medical Report OCR and Clinical Data Extraction AI. "
            "Analyze the medical report image with high precision. Transcribe exact Patient details (Name, Age, Sex, PID, Dates) "
            "and all Investigation parameters (Observed Values, Units, Biological Reference Intervals). "
            "Output valid JSON matching schema only. Filter out interpretation guideline tables."
        ),
        "🔍 Print Verification & Line Check": (
            "You are a High-Precision Medical Report Verification and OCR Auditor. "
            "Transcribe EVERY SINGLE printed line from the document scan in 1-to-1 fidelity."
        ),
        "📋 General Document OCR & Tables": (
            "You are an expert Document OCR system. Extract all visible text, headings, and tabular data accurately."
        ),
        "🧾 Invoice / Receipt Parser": (
            "You are a financial document parsing AI. Extract vendor info, invoice number, date, line items, and totals."
        ),
        "🤖 Custom Prompt": "You are a helpful AI vision assistant."
    }

    system_prompt = st.text_area(
        "System Instruction",
        value=default_system_prompts[preset_mode],
        height=90
    )

    st.divider()

    st.markdown("### 🎛️ Parameters")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    max_tokens = st.slider("Max Output Tokens", min_value=1024, max_value=16384, value=8192, step=512)
    stream_response = st.checkbox("Enable SSE Streaming", value=False)

    st.divider()

    if st.button("🗑️ Reset All Sessions", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_image_b64 = None
        st.session_state.current_image_display = None
        st.session_state.last_extracted_data = None
        st.session_state.last_extraction_duration = None
        st.session_state.active_batch_id = None
        st.rerun()

# Health Check
def check_health(base_url):
    try:
        r = requests.get(f"{base_url}/ocr/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    try:
        r = requests.get(f"{base_url}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

health_status = check_health(api_base_url)

# Header Section
col_head, col_stat = st.columns([3, 1])
with col_head:
    st.markdown("<div class='main-title'>Medical Report OCR & Vision AI Platform</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>High-Accuracy Single & Batch Medical Document Extraction (Ministral-3, MedGemma & Native High-Speed OCR)</div>", unsafe_allow_html=True)

with col_stat:
    if health_status and health_status.get("status") in ("healthy", "degraded", "ok"):
        db_stat = health_status.get("database", {}).get("status", "ok")
        st.markdown(f"<div class='badge-healthy'>🟢 API Online (DB: {db_stat})</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='badge-offline'>🔴 API Offline</div>", unsafe_allow_html=True)

st.divider()

# Navigation Tabs
tab_single, tab_batch, tab_history = st.tabs([
    "📄 Single Document OCR & Clinical Studio",
    "📦 Multi-Document Batch Processing",
    "📊 Batch & Job History Inspector"
])

# ---------------------------------------------------------
# TAB 1: SINGLE DOCUMENT OCR & CLINICAL STUDIO
# ---------------------------------------------------------
with tab_single:
    st.markdown("### 📄 Single Report Analysis & Extraction")
    col_upload, col_preview = st.columns([1, 1])

    uploaded_b64 = None
    pil_image = None

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Report Scan (PDF, WEBP, PNG, JPG)",
            type=["pdf", "webp", "jpg", "jpeg", "png"],
            key="single_uploader"
        )
        if uploaded_file:
            bytes_data = uploaded_file.read()
            is_pdf = uploaded_file.name.lower().endswith(".pdf") or bytes_data.startswith(b"%PDF")
            if is_pdf:
                try:
                    import pypdfium2 as pdfium
                    pdf = pdfium.PdfDocument(bytes_data)
                    page = pdf[0]
                    pil_image = page.render(scale=2.0).to_pil_image()
                    buf = io.BytesIO()
                    pil_image.save(buf, format="JPEG")
                    uploaded_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
                    st.success("📄 PDF Document rendered at 150 DPI!")
                except Exception as pdf_err:
                    st.error(f"Error processing PDF: {pdf_err}")
            else:
                pil_image = Image.open(io.BytesIO(bytes_data))
                uploaded_b64 = f"data:image/jpeg;base64,{base64.b64encode(bytes_data).decode('utf-8')}"

        pdf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pdf"))
        if os.path.exists(pdf_dir):
            sample_files = sorted(
                glob.glob(os.path.join(pdf_dir, "*.pdf")) +
                glob.glob(os.path.join(pdf_dir, "*.webp")) +
                glob.glob(os.path.join(pdf_dir, "*.png")) +
                glob.glob(os.path.join(pdf_dir, "*.jpg"))
            )
            if sample_files:
                options = ["-- Select Sample File from pdf/ --"] + [os.path.basename(p) for p in sample_files]
                selected_sample = st.selectbox("📁 Or pick a Sample Report from pdf/ folder", options=options, key="sample_sel")
                if selected_sample != "-- Select Sample File from pdf/ --":
                    sample_path = os.path.join(pdf_dir, selected_sample)
                    with open(sample_path, "rb") as f_sample:
                        s_bytes = f_sample.read()
                        if selected_sample.lower().endswith(".pdf") or s_bytes.startswith(b"%PDF"):
                            import pypdfium2 as pdfium
                            pdf = pdfium.PdfDocument(s_bytes)
                            page = pdf[0]
                            pil_image = page.render(scale=2.0).to_pil_image()
                            buf = io.BytesIO()
                            pil_image.save(buf, format="JPEG")
                            uploaded_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
                        else:
                            pil_image = Image.open(io.BytesIO(s_bytes))
                            uploaded_b64 = f"data:image/jpeg;base64,{base64.b64encode(s_bytes).decode('utf-8')}"
                        st.success(f"📄 Loaded sample: `{selected_sample}`")

    if uploaded_b64:
        st.session_state.current_image_b64 = uploaded_b64
        st.session_state.current_image_display = pil_image

    with col_preview:
        if st.session_state.current_image_display:
            st.image(st.session_state.current_image_display, caption="Loaded Document View", use_container_width=True)
        else:
            st.info("👆 Upload a report scan or select a sample above.")

    st.divider()

    # Extraction Trigger
    col_btn, col_metric = st.columns([2, 2])
    with col_btn:
        extract_btn = st.button("🚀 Extract Structured Medical Report (JSON)", use_container_width=True)

    if extract_btn:
        if not st.session_state.current_image_b64:
            st.warning("Please upload a document first.")
        else:
            with st.spinner(f"Extracting with {engine_choice} ({model_name})..."):
                t0 = time.monotonic()
                payload = {
                    "document": st.session_state.current_image_b64,
                    "format": "json",
                    "task_type": "medical_extraction",
                    "engine": engine_param,
                    "backend": backend,
                    "model": model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                try:
                    # Direct Sync API call
                    resp = requests.post(f"{api_base_url}/ocr/api/ocr/sync", json=payload, timeout=120)
                    t1 = time.monotonic()
                    if resp.status_code == 200:
                        res_json = resp.json()
                        st.session_state.last_extracted_data = res_json.get("data", {})
                        st.session_state.last_extraction_duration = res_json.get("duration_seconds", round(t1 - t0, 2))
                        st.success(f"✅ Extracted in {st.session_state.last_extraction_duration}s (Engine: {engine_param})")
                    else:
                        st.error(f"API Error ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

    # Structured Results Presentation
    if st.session_state.last_extracted_data and isinstance(st.session_state.last_extracted_data, dict):
        ext_data = st.session_state.last_extracted_data
        patient = ext_data.get("patient_info", {})
        results = ext_data.get("results", [])

        st.markdown(f"### 📋 Report: `{ext_data.get('report_title', 'Laboratory Report')}`")

        # Patient Info Card
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            st.markdown(f"**👤 Patient Name:** `{patient.get('patient_name') or 'N/A'}`")
            st.markdown(f"**🎂 Age / Sex:** `{patient.get('age') or 'N/A'}` / `{patient.get('sex') or 'N/A'}`")
            st.markdown(f"**🆔 PID No:** `{patient.get('pid_no') or 'N/A'}`")
        with c_p2:
            st.markdown(f"**👨‍⚕️ Ref. Doctor:** `{patient.get('reference_dr') or 'N/A'}`")
            st.markdown(f"**📞 Tel No:** `{patient.get('tel_no') or 'N/A'}`")
            st.markdown(f"**🏥 Center:** `{patient.get('collecting_center') or 'N/A'}`")
        with c_p3:
            st.markdown(f"**📝 Registered:** `{patient.get('registered_on') or 'N/A'}`")
            st.markdown(f"**🧪 Collected:** `{patient.get('collected_on') or 'N/A'}`")
            st.markdown(f"**📄 Reported:** `{patient.get('reported_on') or 'N/A'}`")

        st.divider()

        # Results Table
        st.markdown(f"#### 🧪 Extracted Observations ({len(results)} items)")
        if results:
            if pd is not None:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
            else:
                st.table(results)
        else:
            st.info("No test observation rows parsed.")

        # Raw JSON & Download
        with st.expander("📦 View Full Structured JSON Response"):
            json_str = json.dumps(ext_data, indent=2)
            st.code(json_str, language="json")
            st.download_button(
                "⬇️ Download Structured Medical JSON",
                data=json_str,
                file_name=f"medical_report_{patient.get('pid_no', 'extract')}.json",
                mime="application/json",
                use_container_width=True
            )

# ---------------------------------------------------------
# TAB 2: MULTI-DOCUMENT BATCH PROCESSING
# ---------------------------------------------------------
with tab_batch:
    st.markdown("### 📦 Multi-Document Batch Processing Studio")
    st.markdown("Submit up to **100 PDF documents or scans** simultaneously. The worker pool processes them asynchronously in background with database persistence.")

    col_b_up, col_b_params = st.columns([1, 1])

    with col_b_up:
        batch_files = st.file_uploader(
            "Upload Batch of PDF/Image Lab Reports",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="batch_multi_uploader"
        )
        batch_title = st.text_input("Batch Title / Name", value=f"Lab_Batch_{time.strftime('%Y%m%d_%H%M')}")
        batch_webhook = st.text_input("Webhook Callback URL (Optional)", placeholder="https://webhook.site/your-id or http://your-server:8000/webhook")

    with col_b_params:
        batch_engine = st.selectbox("Batch OCR Engine", options=["Vision LLM (ministral-3:latest)", "Native Sub-Second OCR"], key="batch_eng")
        batch_engine_val = "vocr" if "Vision" in batch_engine else "native"
        submit_batch_btn = st.button("Submit Multi-Document Batch Job", use_container_width=True)

    if submit_batch_btn:
        if not batch_files:
            st.warning("Please upload at least one PDF or image file for batch processing.")
        else:
            with st.spinner(f"Uploading and submitting {len(batch_files)} document(s) as batch..."):
                files_payload = [("files", (f.name, f.read(), f.type)) for f in batch_files]
                data_payload = {
                    "name": batch_title,
                    "engine": batch_engine_val,
                    "task_type": "medical_extraction",
                    "backend": backend,
                    "model": model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "webhook_url": batch_webhook.strip() if batch_webhook.strip() else ""
                }
                try:
                    r = requests.post(f"{api_base_url}/ocr/api/batch/upload", files=files_payload, data=data_payload, timeout=60)
                    if r.status_code in (200, 201):
                        res_data = r.json()
                        st.session_state.active_batch_id = res_data.get("batch_id")
                        st.success(f"🎉 Batch Created Successfully! Batch ID: `{st.session_state.active_batch_id}`")
                    else:
                        st.error(f"Batch Creation Failed ({r.status_code}): {r.text}")
                except Exception as e:
                    st.error(f"Failed to submit batch: {e}")

# ---------------------------------------------------------
# TAB 3: BATCH & JOB HISTORY INSPECTOR
# ---------------------------------------------------------
with tab_history:
    st.markdown("### 📊 Batch Jobs History & Export Studio")
    if st.button("🔄 Refresh Batch List", use_container_width=True):
        st.rerun()

    try:
        r_batches = requests.get(f"{api_base_url}/ocr/api/batch", timeout=5)
        if r_batches.status_code == 200:
            batches = r_batches.json().get("batches", [])
            if batches:
                df_batches = pd.DataFrame(batches)
                st.dataframe(df_batches[["batch_id", "name", "status", "total_jobs", "completed_jobs", "failed_jobs", "created_at"]], use_container_width=True)
            else:
                st.info("No batches found in database.")
    except Exception as e:
        st.warning(f"Could not load batch history: {e}")
