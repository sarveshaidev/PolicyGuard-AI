import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VISION_MODEL = "llava-hf/llava-1.5-7b-hf" # For image captioning
GENERATION_MODEL = "meta-llama/llama-3.1-8b-instruct"
JUDGE_MODEL = "meta-llama/llama-3.1-8b-instruct"

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "documents"
TRACE_DIR = BASE_DIR / "data" / "traces"
VECTOR_DB_PATH = BASE_DIR / "data" / "vector_db.faiss"

DATA_DIR.mkdir(parents=True, exist_ok=True)
TRACE_DIR.mkdir(parents=True, exist_ok=True)

# Cache Settings
CACHE_SIMILARITY_THRESHOLD = 0.95 # If > 95% similar, return cached answer