from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL
import logging
import os
import torch

# Suppress unnecessary logs from transformers/huggingface
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

class EmbedderSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info(f"🚀 Initializing Embedding Model: {EMBEDDING_MODEL}")
            
            # Determine Device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("✅ GPU detected! Using CUDA for embeddings.")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps" # For Apple Silicon Macs
                logger.info("✅ Apple Silicon detected! Using MPS for embeddings.")
            else:
                logger.warning("⚠️ No GPU found. Using CPU (this will be slower).")

            try:
                # Load the model with the determined device
                cls._instance = SentenceTransformer(EMBEDDING_MODEL, device=device)
                logger.info(f"✅ Embedding model loaded successfully on {device.upper()}.")
                logger.info(f"   Dimension: {cls._instance.get_embedding_dimension()}")
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model: {e}")
                raise e
                
        return cls._instance

    def get_embedding_dimension(self):
        """Returns the dimension of the embedding vector."""
        if self._instance is None:
            raise RuntimeError("Embedder not initialized!")
        return self._instance.get_embedding_dimension()

    def encode(self, sentences, **kwargs):
        """
        Encodes sentences into embeddings.
        Automatically handles device placement and normalization.
        """
        if self._instance is None:
            raise RuntimeError("Embedder not initialized!")
        
        # Ensure default arguments
        if 'device' not in kwargs:
            # Let the model handle device internally based on how it was loaded
            pass 
        
        if 'normalize_embeddings' not in kwargs:
            kwargs['normalize_embeddings'] = True
            
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 32 # Optimize batch size
            
        if 'show_progress_bar' not in kwargs:
            kwargs['show_progress_bar'] = False # Hide progress bar in logs
            
        try:
            return self._instance.encode(sentences, **kwargs)
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise e

# Create the global singleton instance
# This line triggers the __new__ method only once
embedder = EmbedderSingleton()