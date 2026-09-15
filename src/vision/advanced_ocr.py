
#!/usr/bin/env python3
"""
PolicyGuard AI - Advanced OCR Engine
====================================
Production OCR engine for scanned documents with:
- Tesseract OCR engine
- Lightweight subprocess-based OCR with no ML model resident in process
- Image preprocessing
- PDF page-to-image conversion
- Batch processing
- Thread-safe lazy loading
- Memory-conscious temporary-file handling
- Cross-platform path handling

Author: PolicyGuard AI Team
Version: 2.0.0
Last Updated: 2026-09-13
"""

from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# =============================================================================
# PROJECT PATH
# =============================================================================

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from config.settings import settings
from src.core.exceptions import ProcessingError, RAGException

logger = logging.getLogger(__name__)


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

class ImagePreprocessor:
    """Utilities for improving OCR input quality."""

    @staticmethod
    def enhance_contrast(
        image: Any,
        factor: float = 1.5,
    ) -> Any:
        """Enhance image contrast."""
        try:
            from PIL import ImageEnhance

            factor = max(0.1, float(factor))

            return ImageEnhance.Contrast(
                image
            ).enhance(factor)

        except ImportError:
            logger.warning(
                "PIL ImageEnhance unavailable; "
                "skipping contrast enhancement"
            )
            return image

        except Exception as exc:
            logger.warning(
                "Contrast enhancement failed: %s",
                exc,
            )
            return image

    @staticmethod
    def reduce_noise(
        image: Any,
        level: str = "medium",
    ) -> Any:
        """Reduce image noise."""
        try:
            from PIL import ImageFilter

            filters = {
                "low": ImageFilter.MedianFilter(
                    size=3
                ),
                "medium": ImageFilter.MedianFilter(
                    size=3
                ),
                "high": ImageFilter.MedianFilter(
                    size=5
                ),
            }

            selected = filters.get(
                str(level).lower(),
                filters["medium"],
            )

            return image.filter(
                selected
            )

        except ImportError:
            logger.warning(
                "PIL ImageFilter unavailable; "
                "skipping noise reduction"
            )
            return image

        except Exception as exc:
            logger.warning(
                "Noise reduction failed: %s",
                exc,
            )
            return image

    @staticmethod
    def binarize(
        image: Any,
        threshold: int = 128,
    ) -> Any:
        """Convert image to black/white."""
        try:
            threshold = max(
                0,
                min(
                    255,
                    int(threshold),
                ),
            )

            gray = image.convert(
                "L"
            )

            return gray.point(
                lambda pixel: (
                    255
                    if pixel > threshold
                    else 0
                ),
                mode="1",
            )

        except Exception as exc:
            logger.warning(
                "Binarization failed: %s",
                exc,
            )
            return image

    @classmethod
    def prepare_for_ocr(
        cls,
        image: Any,
        enhance: bool = True,
        denoise: bool = True,
        binarize: bool = False,
    ) -> Any:
        """Apply configured OCR preprocessing."""
        result = image

        if enhance:
            result = cls.enhance_contrast(
                result
            )

        if denoise:
            result = cls.reduce_noise(
                result
            )

        if binarize:
            result = cls.binarize(
                result
            )

        return result



def _max_upload_size_bytes() -> int:
    """Read the configured upload size with a safe OCR fallback."""
    raw = getattr(settings, "MAX_UPLOAD_SIZE_MB", None)
    if raw is None:
        raw = getattr(settings, "MAX_FILE_SIZE_MB", None)
    try:
        mb = int(raw)
        if mb > 0:
            return mb * 1024 * 1024
    except (TypeError, ValueError):
        pass
    return AdvancedOCR.DEFAULT_MAX_FILE_SIZE_BYTES if "AdvancedOCR" in globals() else 50 * 1024 * 1024


def _validate_existing_file(
    file_path: Union[str, Path],
    allowed_extensions: set[str],
) -> Optional[Path]:
    """Validate an OCR source before handing it to image/PDF libraries."""
    try:
        raw = str(file_path)
        if not raw or "\x00" in raw:
            return None
        path = Path(raw).expanduser()
        if path.is_symlink():
            return None
        path = path.resolve()
        if not path.exists() or not path.is_file():
            return None
        if path.suffix.lower() not in allowed_extensions:
            return None
        size = path.stat().st_size
        if size <= 0 or size > _max_upload_size_bytes():
            return None
        return path
    except (OSError, RuntimeError, ValueError):
        return None

# =============================================================================
# ADVANCED OCR
# =============================================================================

class AdvancedOCR:
    """Production OCR engine using Tesseract without an in-process ML OCR model."""

    SUPPORTED_LANGUAGES = [
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "ru",
        "zh",
        "ja",
        "ko",
        "ar",
        "hi",
        "th",
        "vi",
        "tr",
        "pl",
        "nl",
        "sv",
        "da",
        "no",
    ]

    SUPPORTED_IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
        ".bmp",
        ".webp",
    }

    DEFAULT_DPI = 300
    MIN_DPI = 72
    MAX_DPI = 600
    DEFAULT_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
    MAX_IMAGE_PIXELS = 40_000_000
    MAX_PDF_PAGES = 500
    MAX_OCR_TEXT_LENGTH = 10_000_000

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: bool = False,
        enable_preprocessing: bool = True,
        confidence_threshold: float = 0.5,
    ):
        """Initialize OCR configuration without loading models."""
        requested_languages = (
            languages
            if languages is not None
            else ["en"]
        )

        if not requested_languages:
            requested_languages = ["en"]

        valid_languages = []

        for language in requested_languages:
            language = str(
                language
            ).strip().lower()

            if (
                language
                in self.SUPPORTED_LANGUAGES
            ):
                if language not in valid_languages:
                    valid_languages.append(
                        language
                    )
            else:
                logger.warning(
                    "Unsupported OCR language '%s'; ignoring",
                    language,
                )

        self.languages = (
            valid_languages
            or ["en"]
        )

        self.use_gpu = bool(
            use_gpu
        )

        self.enable_preprocessing = bool(
            enable_preprocessing
        )

        try:
            self.confidence_threshold = float(
                confidence_threshold
            )
        except (
            TypeError,
            ValueError,
        ):
            self.confidence_threshold = 0.5

        self.confidence_threshold = max(
            0.0,
            min(
                1.0,
                self.confidence_threshold,
            ),
        )

        self._tesseract_available = False
        self._tesseract_checked = False
        self._loaded_device: Optional[
            str
        ] = None

        self._load_lock = threading.RLock()

        self._stats = {
            "total_ocr_calls": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "avg_confidence": 0.0,
            "total_processing_time_ms": 0.0,
        }

        logger.info(
            "AdvancedOCR initialized: "
            "languages=%s, gpu=%s, preprocessing=%s, threshold=%.2f",
            self.languages,
            self.use_gpu,
            self.enable_preprocessing,
            self.confidence_threshold,
        )

    # -------------------------------------------------------------------------
    # DEVICE
    # -------------------------------------------------------------------------

    def _detect_device(self) -> str:
        """Return the OCR execution device.

        Tesseract runs as an external process, so GPU/CUDA/MPS detection is
        intentionally avoided. Keeping this method preserves the existing
        status/API contract without importing PyTorch.
        """
        return "cpu"

    # -------------------------------------------------------------------------
    # MODEL LOADING
    # -------------------------------------------------------------------------

    def _check_tesseract(self) -> bool:
        """Detect whether Tesseract is usable."""
        with self._load_lock:
            if self._tesseract_checked:
                return self._tesseract_available

            self._tesseract_checked = True

            try:
                import pytesseract
                from PIL import Image

                test_image = Image.new(
                    "L",
                    (32, 32),
                    color=255,
                )

                pytesseract.image_to_string(
                    test_image,
                    config="--psm 10",
                )

                self._tesseract_available = True

                logger.info(
                    "Tesseract OCR available"
                )

                return True

            except ImportError:
                logger.debug(
                    "pytesseract is not installed"
                )

            except Exception as exc:
                logger.debug(
                    "Tesseract is unavailable: %s",
                    exc,
                )

            self._tesseract_available = False
            return False

    def _ensure_engine(self) -> bool:
        """Ensure Tesseract is available."""
        return self._check_tesseract()

    # -------------------------------------------------------------------------
    # IMAGE VALIDATION
    # -------------------------------------------------------------------------

    def _validate_image_path(
        self,
        image_path: Union[str, Path],
    ) -> Optional[Path]:
        """Validate and normalize an image path."""
        path = _validate_existing_file(
            image_path,
            self.SUPPORTED_IMAGE_EXTENSIONS,
        )

        if path is None:
            logger.error(
                "Invalid, unsupported, empty, oversized, or symlink image"
            )
            return None

        return path

    # -------------------------------------------------------------------------
    # IMAGE OCR
    # -------------------------------------------------------------------------

    def extract_text_from_image(
        self,
        image_path: Union[str, Path],
        preprocess: Optional[bool] = None,
        paragraph: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract text from an image.

        Returns:
            Standardized OCR result or None on failure.
        """
        start_time = time.perf_counter()

        path = self._validate_image_path(
            image_path
        )

        if path is None:
            return None

        with self._load_lock:
            self._stats[
                "total_ocr_calls"
            ] += 1

        try:
            from PIL import Image

            # Reject decompression-bomb style images before expensive OCR.
            Image.MAX_IMAGE_PIXELS = self.MAX_IMAGE_PIXELS
            Image.DecompressionBombError = getattr(
                Image,
                "DecompressionBombError",
                Image.DecompressionBombError,
            )

            if not self._ensure_engine():
                logger.error(
                    "No OCR backend is available"
                )
                self._record_failure(
                    start_time
                )
                return None

            do_preprocess = (
                self.enable_preprocessing
                if preprocess is None
                else bool(preprocess)
            )

            with Image.open(path) as source_image:
                source_image.load()

                width, height = source_image.size
                if width <= 0 or height <= 0:
                    raise ProcessingError("Invalid image dimensions")

                if width * height > self.MAX_IMAGE_PIXELS:
                    raise ProcessingError(
                        "Image exceeds the maximum supported pixel count"
                    )

                image = source_image.convert(
                    "RGB"
                )

            if do_preprocess:
                image = (
                    ImagePreprocessor.prepare_for_ocr(
                        image,
                        enhance=True,
                        denoise=True,
                        binarize=False,
                    )
                )

            # Use a system temporary directory. Never create OCR scratch files
            # next to user documents.
            with tempfile.TemporaryDirectory(
                prefix="policyguard_ocr_"
            ) as temp_dir:

                ocr_input = path

                if do_preprocess:
                    temp_path = (
                        Path(temp_dir)
                        / "preprocessed.png"
                    )

                    image.save(
                        temp_path,
                        "PNG",
                    )

                    ocr_input = temp_path

                result = None

                # ---------------------------------------------------------
                # Tesseract OCR
                # ---------------------------------------------------------

                if self._check_tesseract():
                    result = self._extract_with_tesseract(
                        image,
                        path.name,
                    )

                if result is None:
                    self._record_failure(
                        start_time
                    )

                    logger.warning(
                        "OCR failed for %s",
                        path.name,
                    )

                    return None

                elapsed_ms = (
                    time.perf_counter()
                    - start_time
                ) * 1000

                result[
                    "processing_time_ms"
                ] = round(
                    elapsed_ms,
                    2,
                )

                self._record_success(
                    result.get(
                        "confidence",
                        0.0,
                    ),
                    elapsed_ms,
                )

                logger.info(
                    "OCR complete: %s, chars=%s, "
                    "confidence=%.2f, %.1fms",
                    path.name,
                    len(
                        result.get(
                            "text",
                            "",
                        )
                    ),
                    result.get(
                        "confidence",
                        0.0,
                    ),
                    elapsed_ms,
                )

                return result

        except Exception as exc:
            logger.error(
                "OCR error for %s: %s",
                path.name,
                exc,
                exc_info=True,
            )

            self._record_failure(
                start_time
            )

            return None

    def _extract_with_tesseract(
        self,
        image: Any,
        source_filename: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract text using Tesseract."""
        try:
            import pytesseract

            data = pytesseract.image_to_data(
                image,
                config=(
                    "--psm 6 "
                    "-c preserve_interword_spaces=1"
                ),
                output_type=pytesseract.Output.DICT,
            )

            regions = []
            confidences = []
            boxes = []

            for index, raw_text in enumerate(
                data.get(
                    "text",
                    [],
                )
            ):
                text = str(
                    raw_text
                ).strip()

                if not text:
                    continue

                try:
                    confidence = float(
                        data[
                            "conf"
                        ][index]
                    )
                except (
                    KeyError,
                    IndexError,
                    TypeError,
                    ValueError,
                ):
                    confidence = -1.0

                if confidence < 0:
                    continue

                normalized_confidence = (
                    confidence / 100.0
                )

                if (
                    normalized_confidence
                    < self.confidence_threshold
                ):
                    continue

                try:
                    left = int(
                        data["left"][index]
                    )
                    top = int(
                        data["top"][index]
                    )
                    width = int(
                        data["width"][index]
                    )
                    height = int(
                        data["height"][index]
                    )

                    box = [
                        [
                            left,
                            top,
                        ],
                        [
                            left + width,
                            top,
                        ],
                        [
                            left + width,
                            top + height,
                        ],
                        [
                            left,
                            top + height,
                        ],
                    ]

                    boxes.append(
                        box
                    )

                except (
                    KeyError,
                    IndexError,
                    TypeError,
                    ValueError,
                ):
                    boxes.append(
                        []
                    )

                regions.append(
                    text
                )
                confidences.append(
                    normalized_confidence
                )

            if not regions:
                # A second call is useful for very sparse/low-confidence
                # documents where image_to_data returns no accepted words.
                fallback_text = (
                    pytesseract.image_to_string(
                        image,
                        config="--psm 6",
                    ).strip()
                )

                if not fallback_text:
                    return None

                return {
                    "text": fallback_text,
                    "confidence": 0.0,
                    "boxes": [],
                    "text_regions": [
                        fallback_text
                    ],
                    "confidences": [],
                    "image_path": source_filename,
                    "engine": "tesseract",
                    "region_count": 1,
                }

            average_confidence = (
                sum(confidences)
                / len(confidences)
            )

            return {
                "text": " ".join(
                    regions
                ),
                "confidence": average_confidence,
                "boxes": boxes,
                "text_regions": regions,
                "confidences": confidences,
                "image_path": source_filename,
                "engine": "tesseract",
                "region_count": len(
                    regions
                ),
            }

        except Exception as exc:
            logger.warning(
                "Tesseract extraction failed for %s: %s",
                source_filename,
                exc,
            )
            return None

    # -------------------------------------------------------------------------
    # PDF OCR
    # -------------------------------------------------------------------------

    def _validate_pdf_pages(
        self,
        pages: Optional[List[int]],
    ) -> Optional[List[int]]:
        """Validate 1-based requested PDF pages."""
        if pages is None:
            return None

        if not isinstance(
            pages,
            list,
        ):
            raise ProcessingError(
                "pages must be a list of 1-based page numbers"
            )

        normalized = []

        for page in pages:
            try:
                page_number = int(
                    page
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ProcessingError(
                    f"Invalid PDF page number: {page}"
                ) from exc

            if page_number < 1:
                raise ProcessingError(
                    f"PDF page numbers are 1-based: {page_number}"
                )

            if page_number not in normalized:
                normalized.append(
                    page_number
                )

        return sorted(
            normalized
        )

    def extract_text_from_pdf(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[List[int]] = None,
        dpi: int = DEFAULT_DPI,
        preprocess: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract OCR text from selected/all PDF pages.

        Requested page numbers are 1-based.
        """
        start_time = time.perf_counter()

        pdf_path = _validate_existing_file(
            pdf_path,
            {".pdf"},
        )
        if pdf_path is None:
            logger.error(
                "Invalid, empty, oversized, or symlink PDF"
            )
            return None

        try:
            pdf_path = (
                Path(
                    pdf_path
                )
                .expanduser()
                .resolve()
            )
        except Exception:
            return None

        if (
            not pdf_path.exists()
            or not pdf_path.is_file()
        ):
            logger.error(
                "PDF not found: %s",
                pdf_path,
            )
            return None

        if pdf_path.suffix.lower() != ".pdf":
            logger.error(
                "Not a PDF file: %s",
                pdf_path,
            )
            return None

        try:
            dpi = int(
                dpi
            )
        except (
            TypeError,
            ValueError,
        ):
            dpi = self.DEFAULT_DPI

        dpi = max(
            self.MIN_DPI,
            min(
                self.MAX_DPI,
                dpi,
            ),
        )

        try:
            requested_pages = (
                self._validate_pdf_pages(
                    pages
                )
            )
        except ProcessingError as exc:
            logger.error(
                "Invalid PDF pages: %s",
                exc,
            )
            return None

        with self._load_lock:
            self._stats[
                "total_ocr_calls"
            ] += 1

        try:
            from pdf2image import (
                convert_from_path,
            )

        except ImportError:
            logger.error(
                "pdf2image is not installed; "
                "PDF OCR unavailable"
            )
            self._record_failure(
                start_time
            )
            return None

        try:
            if not self._ensure_engine():
                logger.error(
                    "No OCR backend available for PDF OCR"
                )
                self._record_failure(
                    start_time
                )
                return None

            logger.info(
                "Starting PDF OCR: %s, dpi=%s",
                pdf_path.name,
                dpi,
            )

            # Process pages individually rather than loading an entire large
            # PDF into RAM.
            if requested_pages:
                page_numbers = requested_pages
            else:
                page_numbers = None

            page_texts: List[
                Dict[str, Any]
            ] = []

            total_chars = 0
            total_pages = 0

            # If no page list is supplied, first obtain the page count with
            # PDF library/PyPDF where available. Otherwise convert in batches.
            if page_numbers is None:
                total_pages = (
                    self._get_pdf_page_count(
                        pdf_path
                    )
                )

                if total_pages <= 0:
                    logger.warning(
                        "Could not determine PDF page count: %s",
                        pdf_path,
                    )
                    self._record_failure(
                        start_time
                    )
                    return None

                if total_pages > self.MAX_PDF_PAGES:
                    raise ProcessingError(
                        f"PDF exceeds the maximum supported page count of "
                        f"{self.MAX_PDF_PAGES}"
                    )

                page_numbers = list(
                    range(
                        1,
                        total_pages + 1,
                    )
                )
            else:
                if max(page_numbers) > self.MAX_PDF_PAGES:
                    raise ProcessingError(
                        f"Requested page exceeds the maximum supported page "
                        f"count of {self.MAX_PDF_PAGES}"
                    )
                total_pages = max(page_numbers)

            # Process one page at a time. This avoids retaining all rendered
            # PDF images in memory.
            for page_number in page_numbers:
                try:
                    images = convert_from_path(
                        str(pdf_path),
                        dpi=dpi,
                        first_page=page_number,
                        last_page=page_number,
                        thread_count=1,
                    )

                    if not images:
                        logger.warning(
                            "No image rendered for PDF page %s",
                            page_number,
                        )
                        continue

                    image = images[0]

                    temp_path: Optional[Path] = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=".png",
                            prefix="policyguard_ocr_",
                            delete=False,
                        ) as tmp:
                            temp_path = Path(
                                tmp.name
                            )

                        image.save(
                            temp_path,
                            "PNG",
                        )

                        result = (
                            self.extract_text_from_image(
                                temp_path,
                                preprocess=preprocess,
                                paragraph=True,
                            )
                        )

                    finally:
                        try:
                            image.close()
                        except Exception:
                            pass

                        try:
                            if temp_path is not None and temp_path.exists():
                                temp_path.unlink()
                        except Exception:
                            logger.debug(
                                "Failed to remove OCR temp file: %s",
                                temp_path,
                            )

                    if (
                        result
                        and result.get("text")
                    ):
                        text = str(
                            result["text"]
                        )

                        page_result = {
                            "page": page_number,
                            "text": text,
                            "confidence": float(
                                result.get(
                                    "confidence",
                                    0.0,
                                )
                            ),
                            "char_count": len(
                                text
                            ),
                            "engine": result.get(
                                "engine",
                                "unknown",
                            ),
                        }

                        page_texts.append(
                            page_result
                        )

                        total_chars += len(
                            text
                        )

                        if total_chars > self.MAX_OCR_TEXT_LENGTH:
                            raise ProcessingError(
                                "OCR output exceeds the maximum supported text size"
                            )

                except ProcessingError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "PDF OCR failed on page %s: %s",
                        page_number,
                        exc,
                    )

            if not page_texts:
                elapsed_ms = (
                    time.perf_counter()
                    - start_time
                ) * 1000

                self._record_failure(
                    start_time
                )

                return {
                    "text": "",
                    "confidence": 0.0,
                    "page_texts": [],
                    "total_pages": total_pages,
                    "pages_with_text": 0,
                    "total_chars": 0,
                    "pdf_path": str(
                        pdf_path
                    ),
                    "dpi": dpi,
                    "processing_time_ms": round(
                        elapsed_ms,
                        2,
                    ),
                    "error": "No text extracted",
                }

            confidences = [
                float(
                    page["confidence"]
                )
                for page in page_texts
            ]

            average_confidence = (
                sum(confidences)
                / len(confidences)
            )

            full_text = "\n\n".join(
                (
                    f"[Page {page['page']}]\n"
                    f"{page['text']}"
                )
                for page in page_texts
            )

            elapsed_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

            # PDF extraction is a single public operation, so record one
            # success for the PDF call. Individual image OCR calls also
            # maintain their own counters for backward compatibility.
            self._record_success(
                average_confidence,
                elapsed_ms,
            )

            return {
                "text": full_text,
                "confidence": average_confidence,
                "page_texts": page_texts,
                "total_pages": total_pages,
                "pages_with_text": len(
                    page_texts
                ),
                "total_chars": total_chars,
                "pdf_path": str(
                    pdf_path
                ),
                "dpi": dpi,
                "processing_time_ms": round(
                    elapsed_ms,
                    2,
                ),
            }

        except Exception as exc:
            logger.error(
                "PDF OCR error for %s: %s",
                pdf_path.name,
                exc,
                exc_info=True,
            )

            self._record_failure(
                start_time
            )

            return None

    def _get_pdf_page_count(
        self,
        pdf_path: Path,
    ) -> int:
        """Get PDF page count without rendering the document, using pypdf."""
        try:
            from pypdf import (
                PdfReader,
            )

            reader = PdfReader(
                str(pdf_path)
            )

            return len(
                reader.pages
            )

        except ImportError:
            pass

        except Exception as exc:
            logger.debug(
                "pypdf page count failed: %s",
                exc,
            )

        # Last-resort count through pdf2image metadata.
        try:
            from pdf2image import (
                pdfinfo_from_path,
            )

            info = pdfinfo_from_path(
                str(pdf_path)
            )

            return int(
                info.get(
                    "Pages",
                    0,
                )
            )

        except Exception:
            return 0

    # -------------------------------------------------------------------------
    # BYTES OCR
    # -------------------------------------------------------------------------

    def extract_text_from_bytes(
        self,
        image_bytes: bytes,
        filename: str = "image.png",
        preprocess: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extract OCR text from in-memory image bytes."""
        if not isinstance(
            image_bytes,
            (
                bytes,
                bytearray,
            ),
        ):
            logger.error(
                "image_bytes must be bytes or bytearray"
            )
            return None

        if not image_bytes:
            logger.error(
                "image_bytes is empty"
            )
            return None

        if len(image_bytes) > _max_upload_size_bytes():
            logger.error(
                "image_bytes exceeds the configured maximum size"
            )
            return None

        try:
            from PIL import Image

            with Image.open(
                io.BytesIO(
                    image_bytes
                )
            ) as image:
                image.load()

                # Validate the supplied content as an actual image before
                # creating any temporary file.
                image_format = (
                    image.format
                    or "PNG"
                )

                if image.width <= 0 or image.height <= 0:
                    logger.error("Invalid image dimensions")
                    return None

                if image.width * image.height > self.MAX_IMAGE_PIXELS:
                    logger.error(
                        "Image exceeds the maximum supported pixel count"
                    )
                    return None

                safe_filename = Path(
                    str(filename)
                ).name
                if (
                    not safe_filename
                    or safe_filename in {".", ".."}
                    or len(safe_filename) > 255
                ):
                    safe_filename = "image.png"

                suffix = Path(
                    safe_filename
                ).suffix.lower()

                if (
                    suffix
                    not in self.SUPPORTED_IMAGE_EXTENSIONS
                ):
                    suffix = (
                        ".png"
                        if image_format.upper()
                        == "PNG"
                        else ".jpg"
                    )

                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    prefix="policyguard_ocr_",
                    delete=False,
                ) as tmp:
                    temp_path = Path(
                        tmp.name
                    )

                try:
                    save_format = (
                        image_format
                        if image_format.upper()
                        in {
                            "PNG",
                            "JPEG",
                            "TIFF",
                            "BMP",
                            "WEBP",
                        }
                        else "PNG"
                    )

                    image.save(
                        temp_path,
                        format=save_format,
                    )

                    result = (
                        self.extract_text_from_image(
                            temp_path,
                            preprocess=preprocess,
                            paragraph=True,
                        )
                    )

                    if result:
                        result[
                            "original_filename"
                        ] = safe_filename

                    return result

                finally:
                    try:
                        if temp_path.exists():
                            temp_path.unlink()
                    except Exception:
                        logger.debug(
                            "Failed to remove bytes OCR temp file"
                        )

        except Exception as exc:
            logger.error(
                "Bytes OCR error: %s",
                exc,
                exc_info=True,
            )
            return None

    # -------------------------------------------------------------------------
    # BATCH OCR
    # -------------------------------------------------------------------------

    def batch_extract_from_images(
        self,
        image_paths: List[
            Union[
                str,
                Path,
            ]
        ],
        max_workers: int = 2,
        on_progress: Optional[
            Callable[
                [int, int],
                None,
            ]
        ] = None,
    ) -> List[
        Optional[
            Dict[str, Any]
        ]
    ]:
        """OCR multiple images while preserving input order."""
        if not isinstance(
            image_paths,
            list,
        ):
            raise ProcessingError(
                "image_paths must be a list"
            )

        if not image_paths:
            return []

        try:
            max_workers = int(
                max_workers
            )
        except (
            TypeError,
            ValueError,
        ):
            max_workers = 2

        max_workers = max(
            1,
            min(
                max_workers,
                8,
            ),
        )

        results: List[
            Optional[
                Dict[str, Any]
            ]
        ] = [
            None
            for _ in image_paths
        ]

        # OCR model loading is expensive and many OCR backends are not
        # genuinely parallel. Two workers remains a conservative default.
        with ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:

            futures = {
                executor.submit(
                    self.extract_text_from_image,
                    path,
                ): index
                for index, path in enumerate(
                    image_paths
                )
            }

            completed = 0

            for future in as_completed(
                futures
            ):
                index = futures[
                    future
                ]

                try:
                    results[
                        index
                    ] = future.result()

                except Exception as exc:
                    logger.error(
                        "Batch OCR failed for %s: %s",
                        image_paths[index],
                        exc,
                    )

                completed += 1

                if on_progress:
                    try:
                        on_progress(
                            completed,
                            len(image_paths),
                        )
                    except Exception:
                        logger.debug(
                            "OCR progress callback failed",
                            exc_info=True,
                        )

        successful = sum(
            1
            for result in results
            if (
                result
                and result.get("text")
            )
        )

        logger.info(
            "Batch OCR complete: %s/%s successful",
            successful,
            len(image_paths),
        )

        return results

    # -------------------------------------------------------------------------
    # STATUS / STATS
    # -------------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Return whether an OCR backend is currently available."""
        return self._tesseract_available

    @property
    def active_engine(self) -> Optional[str]:
        """Return currently loaded primary backend."""
        if self._tesseract_available:
            return "tesseract"
        return None

    def get_stats(
        self,
    ) -> Dict[str, Any]:
        """Return OCR runtime statistics."""
        with self._load_lock:
            total_calls = int(
                self._stats[
                    "total_ocr_calls"
                ]
            )

            successful = int(
                self._stats[
                    "successful_extractions"
                ]
            )

            total_time = float(
                self._stats[
                    "total_processing_time_ms"
                ]
            )

            avg_time = (
                total_time / successful
                if successful
                else 0.0
            )

            success_rate = (
                successful
                / total_calls
                * 100
                if total_calls
                else 0.0
            )

            return {
                "is_available": self.is_available,
                "active_engine": self.active_engine,
                "languages": list(
                    self.languages
                ),
                "use_gpu": self.use_gpu,
                "loaded_device": self._loaded_device,
                "enable_preprocessing": (
                    self.enable_preprocessing
                ),
                "confidence_threshold": (
                    self.confidence_threshold
                ),
                "total_ocr_calls": total_calls,
                "successful_extractions": successful,
                "failed_extractions": int(
                    self._stats[
                        "failed_extractions"
                    ]
                ),
                "success_rate": round(
                    success_rate,
                    2,
                ),
                "avg_confidence": round(
                    float(
                        self._stats[
                            "avg_confidence"
                        ]
                    ),
                    3,
                ),
                "avg_processing_time_ms": round(
                    avg_time,
                    1,
                ),
                "total_processing_time_ms": round(
                    total_time,
                    1,
                ),
            }

    # -------------------------------------------------------------------------
    # STAT HELPERS
    # -------------------------------------------------------------------------

    def _record_success(
        self,
        confidence: float,
        elapsed_ms: float,
    ) -> None:
        """Update successful OCR statistics."""
        try:
            confidence = float(
                confidence
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        with self._load_lock:
            self._stats[
                "successful_extractions"
            ] += 1

            self._stats[
                "total_processing_time_ms"
            ] += max(
                0.0,
                float(
                    elapsed_ms
                ),
            )

            count = self._stats[
                "successful_extractions"
            ]

            old_average = self._stats[
                "avg_confidence"
            ]

            self._stats[
                "avg_confidence"
            ] = (
                (
                    old_average
                    * (count - 1)
                )
                + confidence
            ) / count

    def _record_failure(
        self,
        start_time: float,
    ) -> None:
        """Update failed OCR statistics."""
        elapsed_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        with self._load_lock:
            self._stats[
                "failed_extractions"
            ] += 1

            self._stats[
                "total_processing_time_ms"
            ] += max(
                0.0,
                elapsed_ms,
            )

    # -------------------------------------------------------------------------
    # RESOURCE MANAGEMENT
    # -------------------------------------------------------------------------

    def unload(self) -> None:
        """Release OCR resources.

        Tesseract is invoked as an external process and does not keep an
        in-process model resident, so there is no ML model to unload.
        """
        with self._load_lock:
            self._loaded_device = None
            logger.info("Tesseract OCR resources released")

    def shutdown(self) -> None:
        """Release OCR resources."""
        self.unload()

        logger.info(
            "AdvancedOCR shutdown complete"
        )


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_ocr_engine: Optional[
    AdvancedOCR
] = None

_ocr_lock = threading.Lock()


def get_ocr_engine(
    languages: Optional[List[str]] = None,
    use_gpu: bool = False,
    enable_preprocessing: bool = True,
) -> AdvancedOCR:
    """Return the process-wide OCR engine."""
    global _ocr_engine

    with _ocr_lock:
        if _ocr_engine is None:
            _ocr_engine = AdvancedOCR(
                languages=languages,
                use_gpu=use_gpu,
                enable_preprocessing=enable_preprocessing,
            )

        return _ocr_engine


def reset_ocr_engine() -> None:
    """Reset the global OCR engine."""
    global _ocr_engine

    with _ocr_lock:
        engine = _ocr_engine
        _ocr_engine = None

    if engine is not None:
        engine.shutdown()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def extract_text_from_image(
    image_path: Union[str, Path],
    preprocess: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Extract text from an image."""
    return get_ocr_engine().extract_text_from_image(
        image_path,
        preprocess=preprocess,
    )


def extract_text_from_pdf(
    pdf_path: Union[str, Path],
    pages: Optional[List[int]] = None,
    dpi: int = 300,
) -> Optional[Dict[str, Any]]:
    """Extract text from a PDF."""
    return get_ocr_engine().extract_text_from_pdf(
        pdf_path,
        pages=pages,
        dpi=dpi,
    )


def extract_text_from_bytes(
    image_bytes: bytes,
    filename: str = "image.png",
) -> Optional[Dict[str, Any]]:
    """Extract text from image bytes."""
    return get_ocr_engine().extract_text_from_bytes(
        image_bytes,
        filename=filename,
    )


def is_ocr_available() -> bool:
    """Return whether an OCR backend is available."""
    return get_ocr_engine().is_available


def get_ocr_stats() -> Dict[str, Any]:
    """Return OCR statistics."""
    return get_ocr_engine().get_stats()


# =============================================================================
# TEST / DEMO
# =============================================================================

def test_ocr_engine() -> None:
    """Run a lightweight OCR smoke test."""
    print(
        "\n🔍 Testing Advanced OCR Engine\n"
    )
    print(
        "=" * 70
    )

    ocr = get_ocr_engine()

    print(
        f"Languages: {ocr.languages}"
    )
    print(
        f"GPU requested: {ocr.use_gpu}"
    )
    print(
        f"Preprocessing: {ocr.enable_preprocessing}"
    )

    # Loading is lazy, so explicitly probe availability here.
    available = ocr._ensure_engine()

    print(
        f"Available: {available}"
    )
    print(
        f"Active engine: {ocr.active_engine}"
    )
    print(
        f"Stats: {ocr.get_stats()}"
    )

    image_dir = (
        project_root
        / "data"
        / "images"
    )

    if image_dir.exists():
        image_files = [
            path
            for path in image_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in ocr.SUPPORTED_IMAGE_EXTENSIONS
            )
        ]

        for image_file in image_files[:2]:
            print(
                f"\nProcessing: {image_file.name}"
            )

            result = (
                ocr.extract_text_from_image(
                    image_file
                )
            )

            if result and result.get(
                "text"
            ):
                print(
                    "  Text:",
                    len(
                        result["text"]
                    ),
                    "chars",
                )
                print(
                    "  Confidence:",
                    result.get(
                        "confidence",
                        0.0,
                    ),
                )
                print(
                    "  Engine:",
                    result.get(
                        "engine"
                    ),
                )
            else:
                print(
                    "  No text extracted"
                )

    print(
        "\n📊 Final statistics:"
    )

    for key, value in (
        ocr.get_stats().items()
    ):
        print(
            f"  {key}: {value}"
        )

    print(
        "\n" + "=" * 70
    )


if __name__ == "__main__":
    test_ocr_engine()
