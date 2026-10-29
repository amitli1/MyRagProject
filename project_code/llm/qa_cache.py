import json
import time
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class QACache:
    """Semantic question/answer cache backed by FAISS (vector similarity)
    + a JSON metadata file, persisted to disk across runs."""

    def __init__(
        self,
        cache_dir: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.92,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "qa_cache.faiss"
        self.meta_path = self.cache_dir / "qa_cache.json"

        self.embedding_model_name = embedding_model_name
        self._model: Optional[SentenceTransformer] = None
        self.similarity_threshold = similarity_threshold

        self.entries = []   # list of {"question": str, "answer": dict, "ts": float}
        self.index = None   # faiss.IndexFlatIP

        self._load()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model '%s' for QA cache", self.embedding_model_name)
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _embed(self, text: str) -> np.ndarray:
        vec = self._get_model().encode([text], convert_to_numpy=True, show_progress_bar=False)
        vec = vec.astype("float32")
        faiss.normalize_L2(vec)
        return vec

    def _load(self):
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.entries = json.load(f)
        else:
            self.entries = []

        if self.index_path.exists() and self.entries:
            self.index = faiss.read_index(str(self.index_path))
            logger.info("Loaded QA cache: %d entries", len(self.entries))
        else:
            self.index = None
            logger.info("Starting with an empty QA cache")

    def _save(self):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))

    def lookup(self, question: str) -> Optional[dict]:
        """Return the cached answer dict if a similar-enough question exists, else None."""
        if self.index is None or self.index.ntotal == 0:
            return None

        qvec = self._embed(question)
        scores, idxs = self.index.search(qvec, 1)
        best_score = float(scores[0][0])
        best_idx = int(idxs[0][0])

        if best_idx == -1 or best_score < self.similarity_threshold:
            return None

        entry = self.entries[best_idx]
        logger.info(
            "QA cache HIT (score=%.3f >= %.3f): '%s' -> cached '%s'",
            best_score, self.similarity_threshold, question, entry["question"],
        )
        return entry["answer"]

    def add(self, question: str, answer: dict):
        """Add a new question/answer pair to the cache and persist it immediately."""
        qvec = self._embed(question)
        if self.index is None:
            dim = qvec.shape[1]
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(qvec)
        self.entries.append({"question": question, "answer": answer, "ts": time.time()})
        self._save()
        logger.info("QA cache: added new entry (%d total)", len(self.entries))