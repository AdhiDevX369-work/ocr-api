import io
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple
import httpx
from PIL import Image, ImageOps
from app.config import settings

logger = logging.getLogger("image-processor")

class ImageProcessingError(ValueError):
    """Custom exception raised when image processing fails."""
    pass

class ImageProcessor:
    @staticmethod
    def is_pdf(data: bytes) -> bool:
        """Checks if byte buffer starts with %PDF magic header."""
        return data.strip().startswith(b"%PDF")

    @staticmethod
    def extract_digital_text(pdf_bytes: bytes) -> Optional[str]:
        """
        Extracts digital text directly from PDF if available (Hybrid OCR).
        Returns clean string or None if scanned/empty.
        """
        text_parts = []
        # Try PyMuPDF
        try:
            import pymupdf as fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                t = page.get_text()
                if t and t.strip():
                    text_parts.append(t.strip())
            if text_parts:
                combined_text = "\n\n--- Page Break ---\n\n".join(text_parts)
                if len(combined_text.strip()) > 50:  # Meaningful text length
                    logger.info(f"Extracted {len(combined_text)} chars of digital text from PDF via PyMuPDF")
                    return combined_text
        except Exception as e:
            logger.debug(f"PyMuPDF text extraction failed/unavailable: {e}")

        # Try pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                t = page.extract_text()
                if t and t.strip():
                    text_parts.append(t.strip())
            if text_parts:
                combined_text = "\n\n--- Page Break ---\n\n".join(text_parts)
                if len(combined_text.strip()) > 50:
                    logger.info(f"Extracted {len(combined_text)} chars of digital text from PDF via pypdf")
                    return combined_text
        except Exception as e:
            logger.debug(f"pypdf text extraction failed/unavailable: {e}")

        return None

    @staticmethod
    def render_pdf_to_pil_images(pdf_bytes: bytes, dpi: int = 110) -> list[Image.Image]:
        """
        Renders PDF pages into individual PIL images at optimal clarity (100-120 DPI).
        Tries pypdfium2 first, then PyMuPDF (fitz), then pdf2image.
        """
        images = []

        # 1. Try pypdfium2
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(pdf_bytes)
            # scale=1.5 gives ~108-110 dpi (optimal vision patch token ratio)
            scale = max(1.2, dpi / 72.0)
            for page in pdf:
                image = page.render(scale=scale).to_pil()
                images.append(image)
            if images:
                logger.info(f"Rendered {len(images)} PDF page(s) using pypdfium2 (scale={scale:.1f})")
                return images
        except Exception as e:
            logger.debug(f"pypdfium2 rendering failed/unavailable: {e}")

        # 2. Try PyMuPDF (fitz)
        try:
            import pymupdf as fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(image)
            if images:
                logger.info(f"Rendered {len(images)} PDF page(s) using PyMuPDF (fitz)")
                return images
        except Exception as e:
            logger.debug(f"PyMuPDF rendering failed/unavailable: {e}")

        # 3. Try pdf2image
        try:
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(pdf_bytes, dpi=dpi)
            if images:
                logger.info(f"Rendered {len(images)} PDF page(s) using pdf2image")
                return images
        except Exception as e:
            logger.debug(f"pdf2image rendering failed/unavailable: {e}")

        raise ImageProcessingError(
            "Could not render PDF document. Please ensure 'pypdfium2' or 'PyMuPDF' is installed."
        )

    @classmethod
    def pil_to_base64_data_uri(
        cls,
        image: Image.Image,
        max_dim: int = settings.max_image_size_px,
        quality: int = settings.image_jpeg_quality
    ) -> str:
        """Converts PIL Image to optimized base64 JPEG data URI."""
        # Convert modes
        if image.mode in ("RGBA", "P", "LA"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "RGBA":
                background.paste(image, mask=image.split()[3])
            else:
                background.paste(image.convert("RGBA"))
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # Downscale proportionally if exceeding max_dim
        w, h = image.size
        scale = 1.0
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))

        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=False, subsampling=0)
        b64_encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_encoded}"

    @staticmethod
    def stitch_images_vertically(images: list[Image.Image]) -> Image.Image:
        """Stitches multiple page PIL images into a single combined vertical document image."""
        if not images:
            raise ImageProcessingError("No images to stitch.")
        if len(images) == 1:
            return images[0]

        max_w = max(img.width for img in images)
        total_h = sum(img.height for img in images) + (len(images) - 1) * 16

        combined = Image.new("RGB", (max_w, total_h), (240, 240, 240))
        y_offset = 0
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            x_offset = (max_w - img.width) // 2
            combined.paste(img, (x_offset, y_offset))
            y_offset += img.height + 16

        logger.info(f"Stitched {len(images)} pages into combined document image ({max_w}x{total_h})")
        return combined

    @classmethod
    def process_document(
        cls,
        data_bytes: bytes,
        max_dim: int = settings.max_image_size_px,
        quality: int = settings.image_jpeg_quality
    ) -> Dict[str, Any]:
        """
        Comprehensive document processing engine:
        - Detects PDF vs Image
        - Extracts digital text if present (Hybrid OCR)
        - Renders high-DPI page images
        - Returns list of individual page Data URIs and primary stitched Data URI
        """
        is_pdf_doc = cls.is_pdf(data_bytes)
        digital_text = None
        page_images: List[Image.Image] = []

        if is_pdf_doc:
            digital_text = cls.extract_digital_text(data_bytes)
            page_images = cls.render_pdf_to_pil_images(data_bytes)
        else:
            try:
                img = Image.open(io.BytesIO(data_bytes))
                img = ImageOps.exif_transpose(img)
                page_images = [img]
            except Exception as e:
                raise ImageProcessingError(f"Failed to decode image scan: {str(e)}")

        if not page_images:
            raise ImageProcessingError("Document contained zero renderable pages.")

        page_data_uris = [cls.pil_to_base64_data_uri(p, max_dim=max_dim, quality=quality) for p in page_images]

        # Stitched image for single-view
        if len(page_images) == 1:
            primary_uri = page_data_uris[0]
        else:
            stitched = cls.stitch_images_vertically(page_images)
            primary_uri = cls.pil_to_base64_data_uri(stitched, max_dim=max_dim, quality=quality)

        return {
            "is_pdf": is_pdf_doc,
            "page_count": len(page_images),
            "digital_text": digital_text,
            "page_data_uris": page_data_uris,
            "primary_data_uri": primary_uri
        }

    @classmethod
    def process_image_bytes(
        cls,
        data_bytes: bytes,
        max_dim: int = settings.max_image_size_px,
        quality: int = settings.image_jpeg_quality
    ) -> str:
        """Processes raw bytes and returns primary Base64 JPEG Data URL."""
        res = cls.process_document(data_bytes, max_dim=max_dim, quality=quality)
        return res["primary_data_uri"]

    @classmethod
    async def process_image_input(
        cls,
        image_input: str,
        http_client: httpx.AsyncClient = None
    ) -> str:
        """Normalizes Base64 string, Data URI, or URL into a clean primary Data URI."""
        if not image_input or not image_input.strip():
            raise ImageProcessingError("Empty image or document input provided.")

        image_input = image_input.strip()

        # Handle HTTP(S) URL
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

        # Handle Base64 Data URI or raw base64
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
