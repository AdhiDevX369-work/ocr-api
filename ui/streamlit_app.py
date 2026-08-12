import io
import json
import base64
import httpx
import requests
import streamlit as st
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Medical Report OCR & Vision AI Studio",
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
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
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
    .preset-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
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

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-history.png", width=56)
    st.markdown("### 🩺 Medical OCR Settings")

    api_base_url = st.text_input("API Base URL (Port 8200)", value="http://localhost:8200")
    backend = st.selectbox("LLM Backend", options=["llama-cpp", "ollama"], index=0)
    
    model_name = st.text_input(
        "Model Name",
        value="qwen2.5-vl:7b",
        help="Recommended: qwen2.5-vl:7b or qwen2.5-vl:3b for document/report OCR precision."
    )

    st.divider()

    st.markdown("### 🎯 System Preset Mode")
    preset_mode = st.selectbox(
        "Task Mode",
        options=[
            "🩺 Medical Lab Report Extractor",
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
        height=110
    )

    st.divider()

    st.markdown("### 🎛️ Parameters")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05, help="0.0 for exact deterministic extraction & caching")
    max_tokens = st.slider("Max Output Tokens", min_value=512, max_value=4096, value=2048, step=128)
    stream_response = st.checkbox("Enable SSE Streaming", value=True)

    st.divider()

    if st.button("🗑️ Reset Chat & Report", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_image_b64 = None
        st.session_state.current_image_display = None
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
    st.markdown("<div class='main-title'>🩺 Medical Report OCR & Vision AI Studio</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>High-Precision Laboratory & Document Data Extraction (Port 8200 & 8100)</div>", unsafe_allow_html=True)

with col_stat:
    if health_status and health_status.get("status") == "healthy":
        st.markdown("<div class='badge-healthy'>🟢 API Online (Port 8200)</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='badge-offline'>🔴 API Offline</div>", unsafe_allow_html=True)

st.divider()

# Document / Report Upload Section
st.markdown("### 📄 Upload Report Image")
col_upload, col_preview = st.columns([1, 1])

uploaded_b64 = None
pil_image = None

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Medical Report / Document Scan (PNG, JPG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key="report_uploader"
    )
    if uploaded_file:
        bytes_data = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(bytes_data))
        uploaded_b64 = f"data:image/jpeg;base64,{base64.b64encode(bytes_data).decode('utf-8')}"

    url_input = st.text_input("Or enter Report Image URL", placeholder="https://example.com/medical-report.jpg")
    if url_input.strip():
        uploaded_b64 = url_input.strip()
        try:
            r = requests.get(uploaded_b64, timeout=5)
            if r.status_code == 200:
                pil_image = Image.open(io.BytesIO(r.content))
        except Exception:
            st.error("Could not fetch image from URL.")

if uploaded_b64:
    st.session_state.current_image_b64 = uploaded_b64
    st.session_state.current_image_display = pil_image

with col_preview:
    if st.session_state.current_image_display:
        st.image(st.session_state.current_image_display, caption="Loaded Document Scan", use_column_width=True)
    else:
        st.info("👆 Upload a medical report image above to start extracting data.")

# Presets for Medical Extraction
preset_prompt = None
if st.session_state.current_image_display:
    st.markdown("#### ⚡ Quick Medical Extraction Actions")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("📊 Full JSON Extraction"):
            preset_prompt = (
                "Extract all data from this lab report into a structured JSON object containing:\n"
                "1. patient_info (name, age, sex, pid_no, reference_dr, registered_on, reported_on)\n"
                "2. report_title (e.g. Full Blood Count, Lipid Profile, eGFR)\n"
                "3. test_results: array of objects [{investigation, observed_value, unit, flag, reference_interval}]\n"
                "Ensure exact numeric precision."
            )
    with c2:
        if st.button("📋 Table & Values"):
            preset_prompt = (
                "Extract all patient metadata and format all lab test investigations into a clean Markdown table "
                "with columns: | Investigation | Observed Value | Flag (H/L) | Unit | Biological Reference Interval |."
            )
    with c3:
        if st.button("⚠️ Out-of-Range Flags"):
            preset_prompt = (
                "Identify and list all test parameters that are Abnormal or Out-of-Range "
                "(e.g. flagged with 'H', 'L', or falling outside the Biological Reference Interval). Explain their clinical context briefly."
            )
    with c4:
        if st.button("📝 Full OCR Text"):
            preset_prompt = "Transcribe all visible text from top to bottom of this medical report document accurately."

st.divider()

# Chat History Feed
st.markdown("### 💬 Report Interactive Q&A")

for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    img_data = msg.get("image")

    with st.chat_message(role):
        if img_data:
            st.image(img_data, width=220, caption="Attached Report")
        st.markdown(content)

# Chat Input Box
user_prompt = st.chat_input("Ask any question about the medical report (e.g., 'What is the WBC count?', 'Is Creatinine elevated?')...")

active_prompt = preset_prompt if preset_prompt else user_prompt

if active_prompt:
    # Append User Turn
    user_turn = {"role": "user", "content": active_prompt}
    if st.session_state.current_image_b64:
        user_turn["image"] = st.session_state.current_image_b64

    st.session_state.messages.append(user_turn)

    with st.chat_message("user"):
        if st.session_state.current_image_b64:
            st.image(st.session_state.current_image_b64, width=220)
        st.markdown(active_prompt)

    # Call API for Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        payload = {
            "image": st.session_state.current_image_b64,
            "prompt": active_prompt,
            "system_prompt": system_prompt,
            "backend": backend,
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream_response,
            "history": [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
        }

        try:
            if stream_response:
                with httpx.stream(
                    "POST",
                    f"{api_base_url}/api/v1/image-chat",
                    json=payload,
                    timeout=120.0
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
                                    if "content" in chunk and chunk["content"]:
                                        full_response += chunk["content"]
                                        message_placeholder.markdown(full_response + "▌")
                                    if chunk.get("done"):
                                        break
                                except json.JSONDecodeError:
                                    continue
                        message_placeholder.markdown(full_response)
            else:
                resp = requests.post(f"{api_base_url}/api/v1/image-chat", json=payload, timeout=120)
                if resp.status_code == 200:
                    res_data = resp.json()
                    full_response = res_data.get("message", {}).get("content", "")
                    message_placeholder.markdown(full_response)
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")

        except Exception as e:
            st.error(f"Request failed: {str(e)}")

        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            # Offer download option if response looks like JSON or Markdown report
            if "{" in full_response and "}" in full_response:
                st.download_button(
                    "📥 Download Extracted Data (JSON/Text)",
                    data=full_response,
                    file_name="extracted_report_data.json",
                    mime="application/json"
                )
