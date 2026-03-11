"""Tests for contradictions.py — Contradiction detection pipeline.

Test plan:
  CD-1: Candidate pair found when claim embeddings are similar
  CD-2: No candidates when embeddings are dissimilar
  CD-3: Already-checked pairs are skipped
  CD-4: Direct contradiction classified and stored
  CD-5: Temporal evolution classified with supersedes
  CD-6: Compatible claims stored correctly
  CD-7: LLM failure counted as error, no crash
  CD-8: Claims without embeddings get embedded before comparison
  CD-9: Topic-scoped run only checks relevant claims
  CD-10: format_report produces readable output
  CD-11: Schema extension (new columns) is idempotent
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import init_epistemic_db
from src.embed import EmbedClient
from src.contradictions import ContradictionDetector, format_report


# --- Helpers ---

DIM = 16


def make_vec(seed: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def make_close_vec(base_seed: int, noise: float = 0.02, dim: int = DIM) -> np.ndarray:
    base = make_vec(base_seed, dim)
    rng = np.random.RandomState(base_seed + 10000)
    v = base + rng.randn(dim).astype(np.float32) * noise
    return v / np.linalg.norm(v)


def make_far_vec(seed: int, away_from: int, dim: int = DIM) -> np.ndarray:
    target = make_vec(away_from, dim)
    v = -target + np.random.RandomState(seed + 50000).randn(dim).astype(np.float32) * 0.1
    return v / np.linalg.norm(v)


class FakeEmbedClient(EmbedClient):
    """Returns pre-assigned vectors by text content."""

    def __init__(self, text_to_vec: dict[str, np.ndarray] = None, dim: int = DIM):
        self._map = text_to_vec or {}
        self._default = make_vec(9999, dim)
        self.dim = dim

    def embed_one(self, text: str) -> np.ndarray:
        return self._map.get(text, self._default)

    def embed_batch(self, texts, batch_size=8) -> list[Optional[np.ndarray]]:
        return [self._map.get(t, self._default) for t in texts]


def _insert_topic(edb_path: Path, label: str) -> int:
    conn = sqlite3.connect(str(edb_path))
    cur = conn.execute(
        "INSERT INTO topics (label, centroid) VALUES (?, ?)",
        (label, b"\x00" * (DIM * 4)),
    )
    tid = cur.lastrowid
    conn.commit()
    conn.close()
    return tid


def _insert_claim(
    edb_path: Path,
    topic_id: int,
    text: str,
    embedding: np.ndarray = None,
    first_seen: str = "2026-03-10",
) -> int:
    conn = sqlite3.connect(str(edb_path))
    blob = EmbedClient.vec_to_blob(embedding) if embedding is not None else None
    cur = conn.execute(
        "INSERT INTO claims (topic_id, text, claim_type, confidence, status, embedding, first_seen) "
        "VALUES (?, ?, 'factual', 'MED', 'active', ?, ?)",
        (topic_id, text, blob, first_seen),
    )
    cid = cur.lastrowid
    conn.execute(
        "INSERT INTO claim_sources (claim_id, summary_id, excerpt) VALUES (?, ?, ?)",
        (cid, "sum_test", text[:50]),
    )
    conn.commit()
    conn.close()
    return cid


def _get_contradictions(edb_path: Path) -> list[dict]:
    conn = sqlite3.connect(str(edb_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM claim_contradictions ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _mock_llm_response(result_json: dict):
    """Create a mock requests.post that returns the given JSON classification."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps(result_json)}}]
    }
    return mock_resp


@pytest.fixture
def env(tmp_path):
    edb_path = tmp_path / "epistemic.db"
    init_epistemic_db(edb_path).close()
    return {"edb": edb_path}


# --- Tests ---


class TestCandidateFinding:
    """CD-1 and CD-2: candidate pair identification."""

    def test_similar_claims_become_candidates(self, env):
        """CD-1: Claims with similar embeddings are flagged as candidates."""
        tid = _insert_topic(env["edb"], "Test")
        vec_a = make_vec(1)
        vec_b = make_close_vec(1, noise=0.02)  # Very similar to A

        _insert_claim(env["edb"], tid, "Claim A about podcasts", vec_a)
        _insert_claim(env["edb"], tid, "Claim B about podcasts", vec_b)

        detector = ContradictionDetector(
            epistemic_db=env["edb"],
            embed_client=FakeEmbedClient(),
            candidate_threshold=0.75,
        )

        with patch("requests.post") as mock_post:
            mock_post.return_value = _mock_llm_response({
                "type": "compatible",
                "explanation": "Both about podcasts",
                "supersedes": None,
            })
            stats = detector.run()

        assert stats["candidate_pairs"] == 1

    def test_dissimilar_claims_not_candidates(self, env):
        """CD-2: Claims with dissimilar embeddings are not candidates."""
        tid = _insert_topic(env["edb"], "Test")
        vec_a = make_vec(1)
        vec_b = make_far_vec(2, away_from=1)

        _insert_claim(env["edb"], tid, "Claim about SEO", vec_a)
        _insert_claim(env["edb"], tid, "Claim about cooking", vec_b)

        detector = ContradictionDetector(
            epistemic_db=env["edb"],
            embed_client=FakeEmbedClient(),
            candidate_threshold=0.75,
        )

        stats = detector.run()
        assert stats["candidate_pairs"] == 0


class TestAlreadyChecked:
    """CD-3: pairs already in claim_contradictions are skipped."""

    def test_skip_existing_pairs(self, env):
        """CD-3: Previously checked pairs are not re-evaluated."""
        tid = _insert_topic(env["edb"], "Test")
        vec_a = make_vec(1)
        vec_b = make_close_vec(1, noise=0.02)

        cid_a = _insert_claim(env["edb"], tid, "Claim A", vec_a)
        cid_b = _insert_claim(env["edb"], tid, "Claim B", vec_b)

        # Pre-populate a result
        conn = sqlite3.connect(str(env["edb"]))
        # Ensure extended columns exist
        detector_tmp = ContradictionDetector(epistemic_db=env["edb"], embed_client=FakeEmbedClient())
        tmp_conn = init_epistemic_db(env["edb"])
        detector_tmp._ensure_schema(tmp_conn)
        tmp_conn.close()

        conn.execute(
            "INSERT INTO claim_contradictions (claim_a_id, claim_b_id, contradiction_type) VALUES (?, ?, ?)",
            (min(cid_a, cid_b), max(cid_a, cid_b), "compatible"),
        )
        conn.commit()
        conn.close()

        detector = ContradictionDetector(
            epistemic_db=env["edb"],
            embed_client=FakeEmbedClient(),
            candidate_threshold=0.75,
        )

        stats = detector.run()
        assert stats["candidate_pairs"] == 0  # Skipped because already exists


class TestClassification:
    """CD-4, CD-5, CD-6: LLM classification types."""

    def test_direct_contradiction(self, env):
        """CD-4: Direct contradiction is classified and stored."""
        tid = _insert_topic(env["edb"], "Martin Ball")
        vec_a = make_vec(1)
        vec_b = make_close_vec(1, noise=0.02)

        cid_a = _insert_claim(env["edb"], tid, "Martin has 800 episodes", vec_a, "2026-03-08")
        cid_b = _insert_claim(env["edb"], tid, "Martin has 600 episodes", vec_b, "2026-03-06")

        detector = ContradictionDetector(
            epistemic_db=env["edb"],
            embed_client=FakeEmbedClient(),
            candidate_threshold=0.75,
        )

        with patch("requests.post") as mock_post:
            mock_post.return_value = _mock_llm_response({
                "type": "direct_contradiction",
                "explanation": "Episode counts conflict",
                "supersedes": "A",
            })
            stats = detector.run()

        assert stats["contradictions_found"] == 1
        results = _get_contradictions(env["edb"])
        assert len(results) == 1
        assert results[0]["contradiction_type"] == "direct_contradiction"

    def test_temporal_evolution(self, env):
        """CD-5: Temporal evolution classified with supersedes."""
        tid = _insert_topic(env["edb"], "Martin Ball")
        vec_a = make_vec(1)
        vec_b = make_close_vec(1, noise=0.02)

        cid_a = _insert_claim(env["edb"], tid, "Spring cohort has 5 signups", vec_a, "2026-02-01")
        cid_b = _insert_claim(env["edb"], tid, "Spring cohort didn't fill", vec_b, "2026-03-01")

        detector = ContradictionDetector(
            epistemic_db=env["edb"],
            embed_client=FakeEmbedClient(),
            candidate_threshold=0.75,
        )

        with patch("requests.post") as mock_post:
            mock_post.return_value = _mock_llm_response({
                "type": "temporal_evolution",
                "explanation": "Situation evolved — cohort ultimately didn't fill",
                "supersedes": "B",
            })
            stats = detector.run()

        assert stats["temporal_evolutions"] == 1
        results = _get_contradictions(env["edb"])
        assert results[0]["supersedes"] == str(cid_b)

    def test_compatible_claims(self, env):
        """CD-6: Compatible claims stored correctly."""
        tid = _insert_topic(env["edb"], "Test")
        vec_a = make_vec(1)
        vec_b = make_close_vec(1, noise=0.02)

        _insert_claim(env["edb"], tid, "Martin uses PodOmatic", vec_a)
        _insert_claim(env["edb"], tid, "Podcast distributes to Spotify", vec_b)

        detector = ContradictionDetector(
            epistemic_db=env["edb"],
            embed_client=FakeEmbedClient(),
            candidate_threshold=0.75,
        )

        with patch("requests.post") as mock_post:
            mock_post.return_value = _mock_llm_response({
                "type": "compatible",
                "explanation": "Both true — PodOmatic distributes to Spotify",
                "supersedes": None,
            })
            stats = detector.run()

        assert stats["compatible"] == 1
        results = _get_contradictions(env["edb"])
        assert results[0]["contradiction_type"] == "compatible"
        assert results[0]["supersedes"] is None


class TestLLMFailure:
    """CD-7: LLM errors are handled gracefully."""

    def test_llm_error_counted(self, env):
        """CD-7: Request failure increments error count, doesn't crash."""
        tid = _insert_topic(env["edb"], "Test")
        vec_a = make_vec(1)
        vec_b = make_close_vec(1, noise=0.02)

        _insert_claim(env["edb"], tid, "Claim A", vec_a)
        _insert_claim(env["edb"], tid, "Claim B", vec_b)

        detector = ContradictionDetector(
            epistemic_db=env["edb"],
            embed_client=FakeEmbedClient(),
            candidate_threshold=0.75,
        )

        import requests as req
        with patch("requests.post") as mock_post:
            mock_post.side_effect = req.ConnectionError("Connection refused")
            stats = detector.run()

        assert stats["errors"] == 1
        assert stats["contradictions_found"] == 0
        # Should not crash
        assert _get_contradictions(env["edb"]) == []


class TestEmbeddingGap:
    """CD-8: Claims without embeddings get embedded on the fly."""

    def test_missing_embeddings_filled(self, env):
        """CD-8: Claims inserted without embeddings are embedded before comparison."""
        tid = _insert_topic(env["edb"], "Test")

        # Insert claims WITHOUT embeddings
        cid_a = _insert_claim(env["edb"], tid, "Claim needing embed A", embedding=None)
        cid_b = _insert_claim(env["edb"], tid, "Claim needing embed B", embedding=None)

        # Fake embed client returns similar vectors for both
        vec = make_vec(1)
        embed = FakeEmbedClient({
            "Claim needing embed A": vec,
            "Claim needing embed B": make_close_vec(1, noise=0.02),
        })

        detector = ContradictionDetector(
            epistemic_db=env["edb"],
            embed_client=embed,
            candidate_threshold=0.75,
        )

        with patch("requests.post") as mock_post:
            mock_post.return_value = _mock_llm_response({
                "type": "compatible",
                "explanation": "Both fine",
                "supersedes": None,
            })
            stats = detector.run()

        assert stats["claims_checked"] == 2
        assert stats["candidate_pairs"] >= 1

        # Verify embeddings were written back
        conn = sqlite3.connect(str(env["edb"]))
        blobs = conn.execute("SELECT embedding FROM claims WHERE embedding IS NOT NULL").fetchall()
        conn.close()
        assert len(blobs) == 2


class TestTopicScoped:
    """CD-9: Topic-scoped runs focus on relevant claims."""

    def test_topic_scoped_run(self, env):
        """CD-9: Passing topic_id still loads all claims (for cross-topic checking)."""
        tid_a = _insert_topic(env["edb"], "Topic A")
        tid_b = _insert_topic(env["edb"], "Topic B")

        vec_a = make_vec(1)
        vec_b = make_close_vec(1, noise=0.02)

        _insert_claim(env["edb"], tid_a, "Claim in A", vec_a)
        _insert_claim(env["edb"], tid_b, "Claim in B", vec_b)

        detector = ContradictionDetector(
            epistemic_db=env["edb"],
            embed_client=FakeEmbedClient(),
            candidate_threshold=0.75,
        )

        with patch("requests.post") as mock_post:
            mock_post.return_value = _mock_llm_response({
                "type": "nuance_difference",
                "explanation": "Different contexts",
                "supersedes": None,
            })
            stats = detector.run(topic_id=tid_a)

        # Both claims loaded (cross-topic)
        assert stats["claims_checked"] == 2
        assert stats["nuance_differences"] == 1


class TestFormatReport:
    """CD-10: Report generation."""

    def test_report_with_contradictions(self, env):
        """CD-10: format_report produces readable markdown."""
        tid = _insert_topic(env["edb"], "Test Topic")
        vec_a = make_vec(1)
        vec_b = make_close_vec(1, noise=0.02)

        cid_a = _insert_claim(env["edb"], tid, "800 episodes", vec_a, "2026-03-08")
        cid_b = _insert_claim(env["edb"], tid, "600 episodes", vec_b, "2026-03-06")

        # Manually insert a contradiction result
        conn = sqlite3.connect(str(env["edb"]))
        detector = ContradictionDetector(epistemic_db=env["edb"], embed_client=FakeEmbedClient())
        detector._ensure_schema(conn)
        conn.execute(
            "INSERT INTO claim_contradictions (claim_a_id, claim_b_id, contradiction_type, explanation, embedding_similarity) "
            "VALUES (?, ?, ?, ?, ?)",
            (min(cid_a, cid_b), max(cid_a, cid_b), "direct_contradiction", "Episode counts conflict", 0.95),
        )
        conn.commit()

        report = format_report(conn)
        conn.close()

        assert "Contradiction" in report or "contradiction" in report
        assert "800 episodes" in report
        assert "600 episodes" in report
        assert "Episode counts conflict" in report

    def test_report_empty(self, env):
        """CD-10b: Empty report when no contradictions."""
        conn = sqlite3.connect(str(env["edb"]))
        detector = ContradictionDetector(epistemic_db=env["edb"], embed_client=FakeEmbedClient())
        detector._ensure_schema(conn)
        report = format_report(conn)
        conn.close()
        assert "No contradictions" in report


class TestSchemaIdempotent:
    """CD-11: Schema extension is safe to run multiple times."""

    def test_ensure_schema_twice(self, env):
        """CD-11: Calling _ensure_schema multiple times doesn't error."""
        conn = init_epistemic_db(env["edb"])
        detector = ContradictionDetector(
            epistemic_db=env["edb"],
            embed_client=FakeEmbedClient(),
        )

        # Should not raise
        detector._ensure_schema(conn)
        detector._ensure_schema(conn)
        detector._ensure_schema(conn)

        # Columns should exist
        cols = {r[1] for r in conn.execute("PRAGMA table_info(claim_contradictions)")}
        assert "contradiction_type" in cols
        assert "explanation" in cols
        assert "supersedes" in cols
        assert "embedding_similarity" in cols
        conn.close()
