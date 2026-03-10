"""Epistemic Synthesis — Embedding Client

Thin wrapper around Qwen3-Embedding-8B HTTP API (OpenAI-compatible /v1/embeddings).
Handles single and batch embedding, normalization, and error recovery.
"""

import struct
import time
from typing import Optional

import numpy as np
import requests

from . import config


class EmbedError(Exception):
    """Raised when embedding fails after retries."""
    pass


class EmbedClient:
    """Client for Qwen3-Embedding-8B text embedding service."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        dim: Optional[int] = None,
    ):
        self.base_url = (base_url or config.EMBED_BASE_URL).rstrip("/")
        self.model = model or config.EMBED_MODEL
        self.timeout = timeout or config.EMBED_TIMEOUT
        self.dim = dim or config.EMBED_DIM
        self._url = f"{self.base_url}/v1/embeddings"

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns L2-normalized float32 vector."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        vectors = self._call_api([text])
        return vectors[0]

    def embed_batch(self, texts: list[str], batch_size: int = 8) -> list[np.ndarray]:
        """Embed multiple texts. Returns list of L2-normalized float32 vectors.

        Filters out empty strings (returns None in their positions).
        Batches requests to avoid overloading the server.
        """
        if not texts:
            return []

        results: list[Optional[np.ndarray]] = [None] * len(texts)
        non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]

        for chunk_start in range(0, len(non_empty), batch_size):
            chunk = non_empty[chunk_start : chunk_start + batch_size]
            chunk_texts = [t for _, t in chunk]
            vectors = self._call_api(chunk_texts)
            for (orig_idx, _), vec in zip(chunk, vectors):
                results[orig_idx] = vec

        return results

    def _call_api(self, texts: list[str], retries: int = 2) -> list[np.ndarray]:
        """Call the embedding API with retry logic."""
        payload = {"model": self.model, "input": texts}

        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post(self._url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()

                vectors = []
                for item in sorted(data["data"], key=lambda x: x["index"]):
                    vec = np.array(item["embedding"], dtype=np.float32)
                    # L2 normalize
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    vectors.append(vec)

                return vectors

            except (requests.RequestException, KeyError, ValueError) as e:
                last_err = e
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))

        raise EmbedError(f"Embedding failed after {retries + 1} attempts: {last_err}")

    @staticmethod
    def vec_to_blob(vec: np.ndarray) -> bytes:
        """Convert numpy vector to bytes for SQLite BLOB storage."""
        return vec.astype(np.float32).tobytes()

    @staticmethod
    def blob_to_vec(blob: bytes) -> np.ndarray:
        """Convert SQLite BLOB back to numpy vector."""
        return np.frombuffer(blob, dtype=np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized vectors (= dot product)."""
        return float(np.dot(a, b))
