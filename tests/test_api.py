import sys
import os
import io
import base64
import asyncio
from PIL import Image

# Ensure app package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.image_processor import ImageProcessor
from app.schemas.chat import ImageChatRequest, ChatMessage

def create_dummy_image_b64(width=100, height=100, color="red") -> str:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

def test_image_processor_resizing():
    # Create large 2000x2000 image
    large_b64 = create_dummy_image_b64(2000, 2000, "blue")
    processed_b64 = asyncio.run(ImageProcessor.process_image_input(large_b64))

    assert processed_b64.startswith("data:image/jpeg;base64,")

    # Decode and verify dimensions downscaled to max 1280
    header, data = processed_b64.split(",", 1)
    img_bytes = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_bytes))

    assert max(img.size) == 1280
    print("✅ Image processor downscaling test passed!")

def test_schemas():
    req = ImageChatRequest(
        image=create_dummy_image_b64(),
        prompt="What color is this square?",
        stream=False,
        temperature=0.0
    )
    assert req.prompt == "What color is this square?"
    assert req.temperature == 0.0
    print("✅ Schema validation test passed!")

if __name__ == "__main__":
    test_image_processor_resizing()
    test_schemas()
    print("🎉 All unit tests passed!")
