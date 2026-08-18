import sys
import os

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import time
import json
import base64
import glob
import httpx
import requests
import streamlit as st
from PIL import Image
from app.config import settings

# Page Configuration
st.set_page_config(
    page_title="Medical Report OCR & Batch AI Studio",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling & Glassmorphism Theme
st.markdown("""
<style>
    /* Dark Slate Background */
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
    .stat-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .stat-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
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
if "active_batch_id" not in st.session_state:
    st.session_state.active_batch_id = None

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-history.png", width=56)
    st.markdown("### 🩺 Production vOCR Settings")

    api_base_url = st.text_input("API Base URL (Port 8200)", value=f"http://localhost:{settings.port}")
    backend_options = ["ollama", "llm-server", "llama-cpp"]
    default_backend_idx = backend_options.index(settings.default_backend) if settings.default_backend in backend_options else 0
    backend = st.selectbox("LLM Backend", options=backend_options, index=default_backend_idx)
    
    known_models = ["qwen2.5vl:latest", "qwen3-vl:4b", "qwen2.5:7b", "llama3.1:latest", "mistral:latest"]
    default_model_val = settings.default_model if settings.default_model in known_models else "qwen2.5vl:latest"
    default_model_idx = known_models.index(default_model_val) if default_model_val in known_models else 0

    selected_model_option = st.selectbox(
        "Vision Model",
        options=known_models + ["Custom..."],
        index=default_model_idx,
        help="qwen2.5vl:latest is fast with zero thinking delay. qwen3-vl:4b is reasoning model."
    )
    if selected_model_option == "Custom...":
        model_name = st.text_input("Custom Model Name", value=settings.default_model)
    else:
        model_name = selected_model_option

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
            "and all Investigation parameters (Observed Values, Units, Biological Reference Intervals, and Abnormal High/Low flags 'H'/'L'). "
            "Never guess numbers or decimal points. Be 100% accurate."
        ),
        "🔍 Print Verification & Line Check": (
            "You are a High-Precision Medical Report Verification and OCR Auditor. "
            "Your task is to transcribe and audit EVERY SINGLE printed value from the lab report scan, organized section-by-section. "
            "Perform a strict 1-to-1 verification against the printed document. "
            "Extract exact numbers, units, flags ('H'/'L'), and reference intervals without omitting any detail."
        ),
        "📋 General Document OCR & Tables": (
            "You are an expert Document OCR system. Extract all visible text, headings, tabular data, and key-value pairs accurately into clean Markdown format."
        ),
        "🧾 Invoice / Receipt Parser": (
            "You are a financial document parsing AI. Extract vendor info, invoice number, date, line items, prices, tax, and total amount into structured JSON."
        ),
        "🤖 Custom Prompt": "You are a helpful AI vision assistant."
    }

    system_prompt = st.text_area(
        "System Instruction",
        value=default_system_prompts[preset_mode],
        height=100
    )

    st.divider()

    st.markdown("### 🎛️ Parameters")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    max_tokens = st.slider("Max Output Tokens", min_value=1024, max_value=16384, value=4096, step=512)
    stream_response = st.checkbox("Enable SSE Streaming", value=True)

    st.divider()

    if st.button("🗑️ Reset All Sessions", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_image_b64 = None
        st.session_state.current_image_display = None
        st.session_state.active_batch_id = None
        st.rerun()

# Health Check
def check_health(base_url):
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
    st.markdown("<div class='main-title'>🩺 Medical Report OCR & Vision AI Platform</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Enterprise-Grade Single & Batch Medical Document Processing (FastAPI, Redis, PostgreSQL, Ollama & llama-server)</div>", unsafe_allow_html=True)

with col_stat:
    if health_status and health_status.get("status") in ("healthy", "degraded"):
        db_stat = health_status.get("database", {}).get("status", "ok")
        st.markdown(f"<div class='badge-healthy'>🟢 API Online (DB: {db_stat})</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='badge-offline'>🔴 API Offline</div>", unsafe_allow_html=True)

st.divider()

# Navigation Tabs
tab_single, tab_batch, tab_history = st.tabs([
    "📄 Single Document OCR & Chat",
    "📦 Multi-Document Batch Processing",
    "📊 Batch & Job History Inspector"
])

# ---------------------------------------------------------
# TAB 1: SINGLE DOCUMENT OCR & CHAT
# ---------------------------------------------------------
with tab_single:
    st.markdown("### 📄 Single Report Analysis & Chat")
    col_upload, col_preview = st.columns([1, 1])

    uploaded_b64 = None
    pil_image = None

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Report Scan (PDF, PNG, JPG, WEBP)",
            type=["pdf", "jpg", "jpeg", "png", "webp"],
            key="single_uploader"
        )
        if uploaded_file:
            bytes_data = uploaded_file.read()
            is_pdf = uploaded_file.name.lower().endswith(".pdf") or bytes_data.startswith(b"%PDF")
            if is_pdf:
                try:
                    from app.services.image_processor import ImageProcessor
                    processed_uri = ImageProcessor.process_image_bytes(bytes_data)
                    uploaded_b64 = processed_uri
                    header, b64_str = processed_uri.split(",", 1)
                    img_bytes = base64.b64decode(b64_str)
                    pil_image = Image.open(io.BytesIO(img_bytes))
                    st.success("📄 PDF Document converted and rendered!")
                except Exception as pdf_err:
                    st.error(f"Error processing PDF: {pdf_err}")
            else:
                pil_image = Image.open(io.BytesIO(bytes_data))
                uploaded_b64 = f"data:image/jpeg;base64,{base64.b64encode(bytes_data).decode('utf-8')}"

        pdf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pdf"))
        if os.path.exists(pdf_dir):
            sample_pdfs = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
            if sample_pdfs:
                options = ["-- Select Sample PDF from pdf/ --"] + [os.path.basename(p) for p in sample_pdfs]
                selected_sample = st.selectbox("📁 Or pick a Sample PDF report from pdf/ folder", options=options, key="sample_sel")
                if selected_sample != "-- Select Sample PDF from pdf/ --":
                    sample_path = os.path.join(pdf_dir, selected_sample)
                    with open(sample_path, "rb") as f_sample:
                        s_bytes = f_sample.read()
                        from app.services.image_processor import ImageProcessor
                        processed_uri = ImageProcessor.process_image_bytes(s_bytes)
                        uploaded_b64 = processed_uri
                        header, b64_str = processed_uri.split(",", 1)
                        img_bytes = base64.b64decode(b64_str)
                        pil_image = Image.open(io.BytesIO(img_bytes))
                        st.success(f"📄 Loaded sample report: `{selected_sample}`")

    if uploaded_b64:
        st.session_state.current_image_b64 = uploaded_b64
        st.session_state.current_image_display = pil_image

    with col_preview:
        if st.session_state.current_image_display:
            st.image(st.session_state.current_image_display, caption="Loaded Document View", use_container_width=True)
        else:
            st.info("👆 Upload a report scan or select a sample above.")

    # Preset Actions
    preset_prompt = None
    if st.session_state.current_image_display:
        st.markdown("#### ⚡ Quick Actions")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("📊 Extract Structured JSON", key="btn_json"):
                preset_prompt = (
                    "Extract all data from this lab report into a structured JSON object containing:\n"
                    "1. patient_info (patient_name, pid_no, age, sex, tel_no, reference_dr, registered_on, collected_on, reported_on)\n"
                    "2. report_title\n"
                    "3. investigations: array of [{section, investigation, observed_value, flag, unit, reference_interval}]\n"
                    "Ensure exact numbers and output ONLY JSON."
                )
        with c2:
            if st.button("📋 Table & Line Check", key="btn_table"):
                preset_prompt = "Perform an exact line-by-line verification check of all values in this lab report and format as a Markdown table."
        with c3:
            if st.button("⚠️ Out-of-Range Flags", key="btn_flags"):
                preset_prompt = "Identify all abnormal or out-of-range parameters (flagged 'H'/'L') and summarize them."
        with c4:
            if st.button("📝 Full Transcription", key="btn_text"):
                preset_prompt = "Transcribe all visible text from top to bottom accurately."

    st.divider()

    # Chat Feed
    st.markdown("### 💬 Interactive Q&A & Extraction")
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            st.markdown(content)

    user_prompt = st.chat_input("Ask any question or query about the document...")
    active_prompt = preset_prompt if preset_prompt else user_prompt

    if active_prompt:
        user_turn = {"role": "user", "content": active_prompt}
        st.session_state.messages.append(user_turn)

        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            payload = {
                "document": st.session_state.current_image_b64,
                "format": "json" if "JSON" in active_prompt else "markdown",
                "prompt": active_prompt,
                "system_prompt": system_prompt,
                "backend": backend,
                "model": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            if not st.session_state.current_image_b64:
                st.warning("Please upload a document first.")
            else:
                try:
                    if stream_response:
                        with httpx.stream(
                            "POST",
                            f"{api_base_url}/api/v1/ocr/stream",
                            json=payload,
                            timeout=httpx.Timeout(300.0, connect=30.0, read=180.0)
                        ) as resp:
                            if resp.status_code != 200:
                                st.error(f"API Error ({resp.status_code}): {resp.text}")
                            else:
                                for line in resp.iter_lines():
                                    if not line:
                                        continue
                                    if line.startswith("data: "):
                                        data_str = line[6:].strip()
                                        try:
                                            chunk = json.loads(data_str)
                                            if "error" in chunk:
                                                st.error(f"Error: {chunk['error']}")
                                                break
                                            c_text = chunk.get("content", "")
                                            if c_text:
                                                full_response += c_text
                                                message_placeholder.markdown(full_response + "▌")
                                            if chunk.get("done"):
                                                break
                                        except Exception:
                                            continue
                                message_placeholder.markdown(full_response)
                    else:
                        resp = requests.post(f"{api_base_url}/api/v1/ocr/sync", json=payload, timeout=120)
                        if resp.status_code == 200:
                            data_json = resp.json()
                            full_response = json.dumps(data_json.get("data"), indent=2) if isinstance(data_json.get("data"), dict) else str(data_json.get("data"))
                            message_placeholder.markdown(full_response)
                        else:
                            st.error(f"API Error: {resp.text}")

                except Exception as e:
                    st.error(f"Request failed: {e}")

            if full_response:
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.download_button(
                    "📥 Download Output (JSON/Text)",
                    data=full_response,
                    file_name="ocr_extracted_result.json",
                    mime="application/json"
                )

# ---------------------------------------------------------
# TAB 2: MULTI-DOCUMENT BATCH PROCESSING
# ---------------------------------------------------------
with tab_batch:
    st.markdown("### 📦 Enterprise Multi-Document Batch Processing Studio")
    st.markdown("Submit up to **100 PDF documents or scans** simultaneously. The worker pool processes them asynchronously in background with database persistence.")

    col_b_up, col_b_params = st.columns([1, 1])

    with col_b_up:
        batch_files = st.file_uploader(
            "Upload Batch of PDF/Image Lab Reports",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="batch_multi_uploader"
        )
        batch_title = st.text_input("Batch Title / Name", value=f"Lab_Batch_{time.strftime('%Y%m%d_%H%M')}")
        batch_webhook = st.text_input("Webhook Callback URL (Optional)", placeholder="https://webhook.site/your-id or http://your-server:8000/webhook")

    with col_b_params:
        batch_prompt = st.text_area(
            "Batch Extraction Instruction Prompt",
            value="Perform an exact line-by-line verification check and extract all values into structured JSON.",
            height=110
        )
        submit_batch_btn = st.button("🚀 Submit Multi-Document Batch Job", use_container_width=True)

    if submit_batch_btn:
        if not batch_files:
            st.warning("Please upload at least one PDF or image file for batch processing.")
        else:
            with st.spinner(f"Uploading and submitting {len(batch_files)} document(s) as batch..."):
                files_payload = [("files", (f.name, f.read(), f.type)) for f in batch_files]
                data_payload = {
                    "name": batch_title,
                    "prompt": batch_prompt,
                    "system_prompt": system_prompt,
                    "backend": backend,
                    "model": model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "webhook_url": batch_webhook.strip() if batch_webhook.strip() else ""
                }
                try:
                    res = requests.post(f"{api_base_url}/api/v1/batches/upload", files=files_payload, data=data_payload, timeout=30)
                    if res.status_code == 202:
                        batch_info = res.json()
                        st.session_state.active_batch_id = batch_info["batch_id"]
                        st.success(f"🎉 Batch Created! ID: `{batch_info['batch_id']}` with {batch_info['total_files']} files.")
                    else:
                        st.error(f"Failed to submit batch ({res.status_code}): {res.text}")
                except Exception as b_err:
                    st.error(f"Batch submission error: {b_err}")

    # Live Batch Progress Viewer
    if st.session_state.active_batch_id:
        st.divider()
        b_id = st.session_state.active_batch_id
        st.markdown(f"#### 🔄 Live Batch Progress: `{b_id}`")

        try:
            b_res = requests.get(f"{api_base_url}/api/v1/batches/{b_id}/jobs", timeout=5)
            if b_res.status_code == 200:
                b_data = b_res.json()
                total = b_data.get("total_files", 0)
                processed = b_data.get("processed_files", 0)
                failed = b_data.get("failed_files", 0)
                pct = b_data.get("progress_percentage", 0.0)
                status_str = b_data.get("status", "pending")

                # Metrics display
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"<div class='stat-card'><div class='stat-val'>{status_str.upper()}</div><div class='stat-lbl'>Status</div></div>", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"<div class='stat-card'><div class='stat-val'>{total}</div><div class='stat-lbl'>Total Documents</div></div>", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"<div class='stat-card'><div class='stat-val'>{processed}</div><div class='stat-lbl'>Processed</div></div>", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"<div class='stat-card'><div class='stat-val'>{failed}</div><div class='stat-lbl'>Failed</div></div>", unsafe_allow_html=True)

                st.progress(pct / 100.0)

                # Action buttons
                col_dl1, col_dl2, col_ref = st.columns(3)
                with col_dl1:
                    dl_json_url = f"{api_base_url}/api/v1/batches/{b_id}/download?format=json"
                    st.markdown(f"📥 [**Download Merged JSON**]({dl_json_url})")
                with col_dl2:
                    dl_zip_url = f"{api_base_url}/api/v1/batches/{b_id}/download?format=zip"
                    st.markdown(f"📦 [**Download ZIP Archive**]({dl_zip_url})")
                with col_ref:
                    if st.button("🔄 Refresh Status", key="btn_refresh_batch"):
                        st.rerun()

                # Table of individual jobs
                st.markdown("##### 📄 Documents in Batch")
                jobs_list = b_data.get("jobs", [])
                table_data = []
                for j in jobs_list:
                    table_data.append({
                        "Document Name": j.get("document_name", j.get("job_id")),
                        "Status": j.get("status"),
                        "Duration (s)": j.get("duration_seconds", 0.0),
                        "Job ID": j.get("job_id"),
                        "Completed At": j.get("completed_at", "-")
                    })
                st.dataframe(table_data, use_container_width=True)

        except Exception as poll_err:
            st.warning(f"Could not poll batch status: {poll_err}")

# ---------------------------------------------------------
# TAB 3: BATCH & JOB HISTORY INSPECTOR
# ---------------------------------------------------------
with tab_history:
    st.markdown("### 📊 Historical Batches & Jobs Explorer")
    try:
        b_list_res = requests.get(f"{api_base_url}/api/v1/batches?page=1&page_size=20", timeout=5)
        if b_list_res.status_code == 200:
            batches_payload = b_list_res.json()
            batches = batches_payload.get("batches", [])
            if not batches:
                st.info("No batches found in database. Create a batch in the Batch Processing tab.")
            else:
                for b in batches:
                    with st.expander(f"📦 {b['name']} ({b['batch_id']}) — Status: {b['status'].upper()} ({b['processed_files']}/{b['total_files']} files)"):
                        st.write(f"**Created At:** `{b['created_at']}` | **Completed At:** `{b['completed_at'] or 'In Progress'}`")
                        st.write(f"**Progress:** `{b['progress_percentage']}%` | **Failed:** `{b['failed_files']}`")
                        st.markdown(f"[📥 Download Merged JSON]({api_base_url}/api/v1/batches/{b['batch_id']}/download?format=json) | [📦 Download ZIP]({api_base_url}/api/v1/batches/{b['batch_id']}/download?format=zip)")
                        if st.button(f"🔍 Load into Live Progress View", key=f"btn_load_{b['batch_id']}"):
                            st.session_state.active_batch_id = b["batch_id"]
                            st.rerun()
    except Exception as hist_err:
        st.error(f"Failed to fetch batch history: {hist_err}")
