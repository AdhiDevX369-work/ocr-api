#!/usr/bin/env python3
"""
Streaming Benchmark & Diagnostics CLI Utility
Measures Time-To-First-Token (TTFT), throughput (Tokens/sec), payload size, and end-to-end latency.
"""

import sys
import os
import io
import time
import json
import base64
import argparse
import asyncio
import httpx
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.image_processor import ImageProcessor
from app.config import settings

def create_synthetic_lab_report(width=1000, height=1400) -> bytes:
    """Generates a synthetic high-density document image buffer for testing."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

async def run_benchmark(
    api_url: str,
    pdf_path: str = None,
    backend: str = "llm-server",
    model: str = "qwen2.5vl:latest",
    prompt: str = "Extract all investigations from this report."
):
    print("=" * 65)
    print("🔬 OCR & Vision AI Streaming Benchmark Utility")
    print("=" * 65)
    print(f"📡 API Endpoint : {api_url}")
    print(f"🤖 LLM Backend  : {backend}")
    print(f"📦 Model Target : {model}")

    # 1. Prepare Document Input
    t0 = time.monotonic()
    if pdf_path and os.path.exists(pdf_path):
        print(f"📄 Loading PDF Document: {pdf_path}")
        with open(pdf_path, "rb") as f:
            raw_bytes = f.read()
        doc_uri = ImageProcessor.process_image_bytes(raw_bytes)
    else:
        print("🖼️ Generating Synthetic High-Res Document Scan (1000x1400)...")
        raw_bytes = create_synthetic_lab_report()
        doc_uri = ImageProcessor.process_image_bytes(raw_bytes)

    prep_time = time.monotonic() - t0
    payload_kb = len(doc_uri) / 1024.0
    print(f"⏱️ Preprocessing Time : {prep_time * 1000:.1f} ms")
    print(f"📊 Base64 Payload Size: {payload_kb:.1f} KB")
    print("-" * 65)

    payload = {
        "image": doc_uri,
        "prompt": prompt,
        "system_prompt": "You are a medical OCR extraction system.",
        "backend": backend,
        "model": model,
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": True,
        "history": []
    }

    print("🚀 Connecting and initiating SSE stream...")
    start_req_t = time.monotonic()
    first_token_t = None
    token_count = 0
    full_text = ""

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            async with client.stream("POST", f"{api_url}/api/v1/image-chat", json=payload) as resp:
                if resp.status_code != 200:
                    err_text = await resp.aread()
                    print(f"❌ Server returned HTTP {resp.status_code}: {err_text.decode()}")
                    return

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    try:
                        chunk = json.loads(data_str)
                        content = chunk.get("content", "")
                        if content:
                            if first_token_t is None:
                                first_token_t = time.monotonic()
                                ttft_ms = (first_token_t - start_req_t) * 1000.0
                                print(f"⚡ First Token Received (TTFT): {ttft_ms:.1f} ms ({ttft_ms / 1000.0:.2f}s)")
                            token_count += 1
                            full_text += content
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

            end_t = time.monotonic()
            total_duration = end_t - start_req_t

            if first_token_t:
                ttft = first_token_t - start_req_t
                gen_duration = end_t - first_token_t
                tps = token_count / gen_duration if gen_duration > 0 else 0.0
            else:
                ttft = total_duration
                tps = 0.0

            print("=" * 65)
            print("📈 BENCHMARK RESULTS")
            print("=" * 65)
            print(f"  • TTFT (Time To First Token) : {ttft:.3f} s ({ttft * 1000:.1f} ms)")
            print(f"  • Generation Throughput      : {tps:.2f} tokens/sec")
            print(f"  • Total Tokens Emitted       : {token_count}")
            print(f"  • Total Request Latency      : {total_duration:.3f} s")
            print(f"  • Base64 Transfer Payload    : {payload_kb:.1f} KB")
            print("=" * 65)
            print("📝 Sample Generated Content (First 150 chars):")
            print(f"\"{full_text[:150]}...\"")
            print("=" * 65)

        except Exception as e:
            print(f"❌ Benchmark execution failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Benchmark Tool")
    parser.add_argument("--url", default="http://localhost:8200", help="API URL (default: http://localhost:8200)")
    parser.add_argument("--pdf", default=None, help="Path to PDF report file")
    parser.add_argument("--backend", default="llm-server", help="LLM backend (llm-server, ollama, llama-cpp)")
    parser.add_argument("--model", default="qwen2.5vl:latest", help="Model name")
    parser.add_argument("--prompt", default="Extract all medical test names and observed values.", help="Test prompt")

    args = parser.parse_args()
    asyncio.run(run_benchmark(
        api_url=args.url,
        pdf_path=args.pdf,
        backend=args.backend,
        model=args.model,
        prompt=args.prompt
    ))
