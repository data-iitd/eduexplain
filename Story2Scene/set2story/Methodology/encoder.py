import threading

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerEncoder:
    """Frozen sentence encoder E_φ (all-roberta-large-v1). Thread-safe."""

    def __init__(self, model_path: str, device: str = None):
        self.model = SentenceTransformer(model_path, local_files_only=True, device=device)
        self.d = self.model.get_sentence_embedding_dimension()
        self._lock = threading.Lock()

    def encode(self, texts):
        single = isinstance(texts, str)
        with self._lock:
            out = self.model.encode([texts] if single else list(texts), normalize_embeddings=True,
                                    batch_size=128, show_progress_bar=False)
        out = np.asarray(out, dtype=np.float32)
        return out[0] if single else out
