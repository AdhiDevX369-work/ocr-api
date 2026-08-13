import io
import base64
import asyncio
import httpx
from PIL import Image

def create_sample_image_b64() -> str:
    # Create a simple 200x200 image with text or shapes
    img = Image.new("RGB", (200, 200), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

async def test_api_image_chat():
    b64_img = create_sample_image_b64()
    headers = {"Content-Type": "application/json"}
    
    print("--- 1. Testing /api/v1/image-chat on Port 8200 ---")
    payload = {
        "image": b64_img,
        "prompt": "What color is this image?",
        "backend": "ollama",
        "model": "gemma4:latest",
        "temperature": 0.0,
        "stream": False
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            r = await client.post("http://localhost:8200/api/v1/image-chat", json=payload)
            print(f"Status Code: {r.status_code}")
            print(f"Response: {r.text[:500]}")
        except Exception as e:
            print(f"Request failed: {e}")

    print("\n--- 2. Testing /v1/chat/completions on Port 8200 ---")
    payload_openai = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": b64_img}}
                ]
            }
        ],
        "temperature": 0.0,
        "stream": False
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            r = await client.post("http://localhost:8200/v1/chat/completions", json=payload_openai)
            print(f"Status Code: {r.status_code}")
            print(f"Response: {r.text[:500]}")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_api_image_chat())
