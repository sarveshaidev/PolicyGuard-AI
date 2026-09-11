import pandas as pd
from pypdf import PdfReader
from docx import Document
import json
import logging
from io import BytesIO

logger = logging.getLogger(__name__)

class UniversalParser:
    """Parses multiple enterprise file formats into clean text."""

    @staticmethod
    def parse_pdf(file_bytes) -> str:
        # Convert bytes to file-like object
        pdf_file = BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    @staticmethod
    def parse_docx(file_bytes) -> str:
        # Convert bytes to file-like object
        docx_file = BytesIO(file_bytes)
        doc = Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs])

    @staticmethod
    def parse_xlsx(file_bytes) -> str:
        """Converts Excel tables into readable text context."""
        # Convert bytes to file-like object
        excel_file = BytesIO(file_bytes)
        df = pd.read_excel(excel_file, engine='openpyxl')
        # Convert dataframe to a clean string representation
        return df.to_string(index=False)

    @staticmethod
    def parse_json(file_bytes) -> str:
        # Convert bytes to file-like object
        json_file = BytesIO(file_bytes)
        data = json.load(json_file)
        # Recursively flatten JSON to text (simplified for demo)
        return json.dumps(data, indent=2)

    @staticmethod
    def parse_txt(file_bytes) -> str:
        return file_bytes.decode('utf-8')

    @classmethod
    def parse_file(cls, uploaded_file) -> str:
        """Router to parse any supported file type."""
        ext = uploaded_file.name.split('.')[-1].lower()
        file_bytes = uploaded_file.getvalue()
        
        parsers = {
            'pdf': cls.parse_pdf,
            'docx': cls.parse_docx,
            'doc': cls.parse_docx,
            'xlsx': cls.parse_xlsx,
            'xls': cls.parse_xlsx,
            'json': cls.parse_json,
            'txt': cls.parse_txt,
        }
        
        if ext in parsers:
            logger.info(f"Parsing {ext} file: {uploaded_file.name}")
            return parsers[ext](file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {ext}")