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
    def is_pdf(data: bytes) -> bool:
        """Checks if byte buffer starts with %PDF magic header."""
        return data.strip().startswith(b"%PDF")

    @staticmethod
    def render_pdf_to_pil_images(pdf_bytes: bytes) -> list[Image.Image]:
        """
        Renders PDF pages into PIL images using available PDF rendering libraries.
        Tries pypdfium2 first, then PyMuPDF (fitz), then pdf2image.
        """
        images = []

        # Try pypdfium2
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(pdf_bytes)
            for page in pdf:
                # Render at 2x scale (~144 dpi) for high visual clarity
                image = page.render(scale=2).to_pil()
                images.append(image)
            if images:
                logger.info(f"Rendered {len(images)} PDF page(s) using pypdfium2")
                return images
        except Exception as e:
            logger.debug(f"pypdfium2 rendering failed/unavailable: {e}")

        # Try PyMuPDF (pymupdf)
        try:
            import pymupdf as fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(image)
            if images:
                logger.info(f"Rendered {len(images)} PDF page(s) using PyMuPDF (fitz)")
                return images
        except Exception as e:
            logger.debug(f"PyMuPDF rendering failed/unavailable: {e}")

        # Try pdf2image
        try:
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(pdf_bytes, dpi=150)
            if images:
                logger.info(f"Rendered {len(images)} PDF page(s) using pdf2image")
                return images
        except Exception as e:
            logger.debug(f"pdf2image rendering failed/unavailable: {e}")

        raise ImageProcessingError(
            "Could not render PDF document. Please install 'pypdfium2' or 'PyMuPDF' ('pip install pypdfium2 pypdf')."
        )

    @staticmethod
    def stitch_images_vertically(images: list[Image.Image]) -> Image.Image:
        """Stitches multiple page PIL images into a single combined vertical document image."""
        if not images:
            raise ImageProcessingError("No images to stitch.")
        if len(images) == 1:
            return images[0]

        max_w = max(img.width for img in images)
        total_h = sum(img.height for img in images) + (len(images) - 1) * 20

        combined = Image.new("RGB", (max_w, total_h), (240, 240, 240))
        y_offset = 0
        for img in images:
            # Ensure RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            x_offset = (max_w - img.width) // 2
            combined.paste(img, (x_offset, y_offset))
            y_offset += img.height + 20

        logger.info(f"Stitching {len(images)} pages into combined document image ({max_w}x{total_h})")
        return combined

    @classmethod
    def process_image_bytes(
        cls,
        data_bytes: bytes,
        max_dim: int = settings.max_image_size_px,
        quality: int = settings.image_jpeg_quality
    ) -> str:
        """
        Processes raw bytes (Images or PDF documents):
        - If PDF: renders pages to PIL images & stitches multi-page documents
        - Fixes EXIF rotation & color modes
        - Downscales image if larger than max_dim
        - Encodes to Base64 JPEG Data URL
        """
        try:
            if cls.is_pdf(data_bytes):
                logger.info("PDF document format detected. Converting pages to image...")
                page_images = cls.render_pdf_to_pil_images(data_bytes)
                image = cls.stitch_images_vertically(page_images)
            else:
                image = Image.open(io.BytesIO(data_bytes))
                image = ImageOps.exif_transpose(image)
        except ImageProcessingError:
            raise
        except Exception as e:
            raise ImageProcessingError(f"Failed to decode document/image data: {str(e)}")

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

        # Downscale if exceeding max dimension or max stitched height
        w, h = image.size
        scale = 1.0
        if w > max_dim:
            scale = min(scale, max_dim / w)
        if h > getattr(settings, "max_stitched_height_px", 1536):
            scale = min(scale, getattr(settings, "max_stitched_height_px", 1536) / h)

        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            logger.info(f"Resized document image from {w}x{h} to {new_w}x{new_h} (scale={scale:.2f})")

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True, subsampling=0)
        b64_encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_encoded}"

    @classmethod
    async def process_image_input(
        cls,
        image_input: str,
        http_client: httpx.AsyncClient = None
    ) -> str:
        """
        Normalizes any input (Base64 string, Data URI, Image URL, or PDF URL) into a clean Data URI.
        """
        if not image_input or not image_input.strip():
            raise ImageProcessingError("Empty image or document input provided.")

        image_input = image_input.strip()

        # Handle HTTP(S) URL (PDF or Image)
        if image_input.startswith(("http://", "https://")):
            try:
                client = http_client or httpx.AsyncClient(timeout=30.0)
                should_close = http_client is None
                try:
                    resp = await client.get(image_input)
                    resp.raise_for_status()
                    data_bytes = resp.content
                finally:
                    if should_close:
                        await client.aclose()
                return cls.process_image_bytes(data_bytes)
            except ImageProcessingError:
                raise
            except Exception as e:
                raise ImageProcessingError(f"Failed to download document from URL '{image_input}': {str(e)}")

        # Handle Base64 (Data URI or raw base64)
        if "," in image_input:
            header, b64_str = image_input.split(",", 1)
        else:
            b64_str = image_input

        try:
            data_bytes = base64.b64decode(b64_str)
            return cls.process_image_bytes(data_bytes)
        except ImageProcessingError:
            raise
        except Exception as e:
            raise ImageProcessingError(f"Failed to decode base64 input data: {str(e)}")

