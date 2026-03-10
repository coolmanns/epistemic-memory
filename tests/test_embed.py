"""Tests for embed.py — T2.* test cases from phase1-plan.md"""

import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embed import EmbedClient, EmbedError
from src import config


@pytest.fixture
def client():
    """EmbedClient pointing at the real nomic-embed service."""
    return EmbedClient()


@pytest.fixture
def mock_client():
    """EmbedClient with a fake base URL for offline tests."""
    return EmbedClient(base_url="http://localhost:99999")


# --- T2.1: Embed a normal summary ---
class TestT2_1_NormalEmbed:
    def test_returns_vector(self, client):
        vec = client.embed_one("This is a test summary about OpenClaw configuration changes.")
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32

    def test_correct_dimensions(self, client):
        vec = client.embed_one("This is a test summary about OpenClaw configuration changes.")
        assert vec.shape == (config.EMBED_DIM,)

    def test_normalized(self, client):
        vec = client.embed_one("This is a test summary about OpenClaw configuration changes.")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"Vector not normalized: norm={norm}"


# --- T2.2: Embed an empty string ---
class TestT2_2_EmptyString:
    def test_empty_string_raises(self, client):
        with pytest.raises(ValueError, match="empty"):
            client.embed_one("")

    def test_whitespace_only_raises(self, client):
        with pytest.raises(ValueError, match="empty"):
            client.embed_one("   \n\t  ")

    def test_batch_skips_empty(self, client):
        results = client.embed_batch(["hello", "", "world"])
        assert results[0] is not None
        assert results[1] is None  # empty string skipped
        assert results[2] is not None


# --- T2.3: Embed when nomic-embed is down ---
class TestT2_3_ServiceDown:
    def test_timeout_raises_embed_error(self, mock_client):
        with pytest.raises(EmbedError):
            mock_client.embed_one("test")

    def test_retries_before_failing(self, mock_client):
        start = time.time()
        with pytest.raises(EmbedError):
            mock_client.embed_one("test")
        elapsed = time.time() - start
        # Should have retried (2 retries * 0.5s+ delay = at least 1s)
        assert elapsed >= 0.5, f"Too fast — didn't retry? elapsed={elapsed:.2f}s"


# --- T2.4: Embed 100 summaries in batch ---
class TestT2_4_BatchPerformance:
    def test_batch_10(self, client):
        texts = [f"Summary number {i} about topic {i % 5}" for i in range(10)]
        start = time.time()
        results = client.embed_batch(texts)
        elapsed = time.time() - start

        assert len(results) == 10
        assert all(r is not None for r in results)
        assert all(r.shape == (config.EMBED_DIM,) for r in results)
        assert elapsed < 120, f"Batch embed too slow: {elapsed:.1f}s"
        print(f"  10 embeddings in {elapsed:.1f}s")


# --- T2.5: Same summary embedded twice (deterministic) ---
class TestT2_5_Deterministic:
    def test_same_input_same_output(self, client):
        text = "The lobster plugin is alive and running in this system."
        vec1 = client.embed_one(text)
        vec2 = client.embed_one(text)
        similarity = EmbedClient.cosine_similarity(vec1, vec2)
        assert similarity > 0.9999, f"Not deterministic: similarity={similarity}"


# --- Blob serialization ---
class TestBlobSerialization:
    def test_roundtrip(self):
        vec = np.random.randn(768).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        blob = EmbedClient.vec_to_blob(vec)
        recovered = EmbedClient.blob_to_vec(blob)
        assert np.allclose(vec, recovered)

    def test_blob_size(self):
        vec = np.zeros(768, dtype=np.float32)
        blob = EmbedClient.vec_to_blob(vec)
        assert len(blob) == 768 * 4  # float32 = 4 bytes

    def test_cosine_similarity_identical(self):
        vec = np.random.randn(768).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        assert abs(EmbedClient.cosine_similarity(vec, vec) - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        a = np.zeros(768, dtype=np.float32)
        b = np.zeros(768, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        assert abs(EmbedClient.cosine_similarity(a, b)) < 1e-5
