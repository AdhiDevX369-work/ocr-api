import io
import base64
import logging
import httpx
from PIL import Image, ImageOps
from app.config import settings

logger = logging.getLogger(__name__)

class ImageProcessingError(ValueError):
    """Custom exception raised when image processing fails."""
    pass

class ImageProcessor:
    @staticmethod
    def process_image_bytes(
        image_bytes: bytes,
        max_dim: int = settings.max_image_size_px,
        quality: int = settings.image_jpeg_quality
    ) -> str:
        """
        Processes raw image bytes:
        - Parses image with PIL
        - Fixes EXIF rotation
        - Downscales image if larger than max_dim
        - Converts RGB if RGBA/Palette
        - Encodes to Base64 JPEG Data URL
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            raise ImageProcessingError(f"Failed to decode image data: {str(e)}")

        # Convert to RGB mode for JPEG encoding
        if image.mode in ("RGBA", "P", "LA"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "RGBA":
                background.paste(image, mask=image.split()[3])
            else:
                background.paste(image.convert("RGBA"))
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # Downscale if exceeding max dimension
        w, h = image.size
        if max(w, h) > max_dim:
            if w > h:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            else:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h}")

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        b64_encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_encoded}"

    @classmethod
    async def process_image_input(
        cls,
        image_input: str,
        http_client: httpx.AsyncClient = None
    ) -> str:
        """
        Normalizes any input (Base64 string, Data URI, or HTTP URL) into a clean, optimized Data URI.
        """
        if not image_input or not image_input.strip():
            raise ImageProcessingError("Empty image input provided.")

        image_input = image_input.strip()

        # Handle HTTP(S) URL
        if image_input.startswith(("http://", "https://")):
            try:
                client = http_client or httpx.AsyncClient(timeout=15.0)
                should_close = http_client is None
                try:
                    resp = await client.get(image_input)
                    resp.raise_for_status()
                    image_bytes = resp.content
                finally:
                    if should_close:
                        await client.aclose()
                return cls.process_image_bytes(image_bytes)
            except Exception as e:
                raise ImageProcessingError(f"Failed to download image from URL '{image_input}': {str(e)}")

        # Handle Base64 (Data URI or raw base64)
        if "," in image_input:
            header, b64_str = image_input.split(",", 1)
        else:
            b64_str = image_input

        try:
            image_bytes = base64.b64decode(b64_str)
            return cls.process_image_bytes(image_bytes)
        except Exception as e:
            raise ImageProcessingError(f"Failed to decode base64 image input: {str(e)}")
