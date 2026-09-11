import os
import cv2
import numpy as np
import easyocr
from PIL import Image
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

class AdvancedOCRProcessor:
    def __init__(self):
        # Initialize EasyOCR (supports 80+ languages)
        # gpu=False ensures it runs on CPU (stable for interviews/demo)
        # If you have NVIDIA GPU, set gpu=True
        try:
            self.reader = easyocr.Reader(['en'], gpu=False) 
            logger.info("✅ Advanced OCR Engine Initialized (EasyOCR)")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Enhance image quality before OCR."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        # Thresholding for better contrast
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text with bounding boxes and confidence scores using EasyOCR."""
        try:
            # EasyOCR handles preprocessing internally, but we can pass our own if needed
            # For best results, let EasyOCR handle it directly or pass the preprocessed image
            result = self.reader.readtext(image_path, detail=1) # detail=1 returns coordinates
            
            if not result:
                return ""
            
            # Format output: Sort by vertical position (reading order)
            lines = []
            for (bbox, text, conf) in result:
                if conf > 0.5:  # Confidence threshold
                    # bbox[0] is top-left y-coordinate
                    lines.append((bbox[0][1], text)) 
            
            # Sort lines top-to-bottom
            lines.sort(key=lambda x: x[0])
            return "\n".join([text for _, text in lines])
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return f"[OCR Error: {str(e)}]"

    def process_pdf_with_images(self, pdf_path: Path) -> list[dict]:
        """Extract text from PDFs containing scanned images/pages."""
        from pdf2image import convert_from_path
        
        chunks = []
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path, dpi=300) # High DPI for accuracy
            
            for i, image in enumerate(images):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    image.save(tmp.name)
                    text = self.extract_text_from_image(tmp.name)
                    os.unlink(tmp.name)
                
                if text.strip():
                    chunks.append({
                        "content": f"[Page {i+1} - Scanned Content]:\n{text}",
                        "metadata": {"source": pdf_path.name, "page": i+1, "type": "scanned_image"}
                    })
        except Exception as e:
            logger.error(f"PDF OCR processing failed: {e}")
            
        return chunks