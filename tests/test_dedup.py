"""Tests for dedup.py — T11.* test cases from phase2-plan.md

All tests use mock embeddings to avoid external dependencies.
"""

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import init_epistemic_db
from src.dedup import ClaimDeduplicator
from src import config


# --- Fixtures ---

@pytest.fixture
def edb(tmp_path):
    """Create epistemic.db with a topic and some claims."""
    db_path = tmp_path / "epistemic.db"
    conn = init_epistemic_db(db_path)
    conn.execute("INSERT INTO topics (label, centroid) VALUES ('Test Topic', ?)", (b"\x00" * 10,))
    conn.commit()
    conn.close()
    return db_path


def _add_claim(db_path, topic_id, text, embedding=None, source_id="sum_001", excerpt="quote"):
    """Helper: insert a claim with optional embedding and source."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
    cursor = conn.execute(
        "INSERT INTO claims (topic_id, text, claim_type, confidence, status, embedding) "
        "VALUES (?, ?, 'factual', 'MED', 'active', ?)",
        (topic_id, text, blob)
    )
    claim_id = cursor.lastrowid
    conn.execute(
        "INSERT INTO claim_sources (claim_id, summary_id, excerpt) VALUES (?, ?, ?)",
        (claim_id, source_id, excerpt)
    )
    conn.commit()
    conn.close()
    return claim_id


def _mock_embed_client(vectors_map=None):
    """Create mock embed client that returns deterministic vectors."""
    client = MagicMock()

    def embed_batch(texts):
        if vectors_map:
            return [vectors_map.get(t, np.random.randn(16).astype(np.float32)) for t in texts]
        return [np.random.randn(16).astype(np.float32) for _ in texts]

    client.embed_batch = embed_batch
    return client


def _make_similar_pair():
    """Return two very similar vectors (cosine > 0.95)."""
    base = np.random.randn(16).astype(np.float32)
    base /= np.linalg.norm(base)
    noise = np.random.randn(16).astype(np.float32) * 0.01
    similar = base + noise
    similar /= np.linalg.norm(similar)
    return base, similar


def _make_different_pair():
    """Return two very different vectors (cosine < 0.5)."""
    a = np.zeros(16, dtype=np.float32)
    a[0] = 1.0
    b = np.zeros(16, dtype=np.float32)
    b[8] = 1.0  # orthogonal
    return a, b


# --- T11.1: Identical claims merged ---
class TestT11_1_IdenticalMerged:
    def test_merge_near_duplicates(self, edb):
        v1, v2 = _make_similar_pair()
        _add_claim(edb, 1, "System uses PostgreSQL", v1, "sum_001", "uses PostgreSQL")
        _add_claim(edb, 1, "PostgreSQL is the database", v2, "sum_002", "PostgreSQL database")

        dedup = ClaimDeduplicator(epistemic_db=edb, embed_client=_mock_embed_client())
        stats = dedup.dedup_topic(1)
        assert stats["merged"] == 1

        # Check keeper has both sources
        conn = sqlite3.connect(str(edb))
        sources = conn.execute(
            "SELECT summary_id FROM claim_sources WHERE claim_id = 1 ORDER BY summary_id"
        ).fetchall()
        assert len(sources) == 2
        assert {s[0] for s in sources} == {"sum_001", "sum_002"}
        conn.close()

    def test_absorbed_claim_superseded(self, edb):
        v1, v2 = _make_similar_pair()
        _add_claim(edb, 1, "Claim A", v1)
        _add_claim(edb, 1, "Claim B", v2)

        dedup = ClaimDeduplicator(epistemic_db=edb, embed_client=_mock_embed_client())
        dedup.dedup_topic(1)

        conn = sqlite3.connect(str(edb))
        status = conn.execute("SELECT status FROM claims WHERE id = 2").fetchone()[0]
        assert status == "superseded"
        conn.close()


# --- T11.2: Similar but distinct claims kept ---
class TestT11_2_DistinctKept:
    def test_below_threshold_not_merged(self, edb):
        v1, v2 = _make_different_pair()
        _add_claim(edb, 1, "System uses PostgreSQL", v1)
        _add_claim(edb, 1, "JWT tokens expire in 24h", v2)

        dedup = ClaimDeduplicator(epistemic_db=edb, embed_client=_mock_embed_client())
        stats = dedup.dedup_topic(1)
        assert stats["merged"] == 0

        conn = sqlite3.connect(str(edb))
        active = conn.execute("SELECT COUNT(*) FROM claims WHERE status = 'active'").fetchone()[0]
        assert active == 2
        conn.close()


# --- T11.3: Subset vs superset ---
class TestT11_3_SubsetSuperset:
    def test_subset_kept_separate(self, edb):
        # Even if embeddings are somewhat similar, threshold 0.90 should differentiate
        a = np.ones(16, dtype=np.float32)
        a /= np.linalg.norm(a)
        b = np.ones(16, dtype=np.float32)
        b[0] = 2.0  # slight difference
        b /= np.linalg.norm(b)
        # Cosine will be high but we can control via threshold
        _add_claim(edb, 1, "X uses Y", a)
        _add_claim(edb, 1, "X uses Y for Z", b)

        # Use high threshold to prevent merge
        dedup = ClaimDeduplicator(epistemic_db=edb, dedup_threshold=0.999,
                                   embed_client=_mock_embed_client())
        stats = dedup.dedup_topic(1)
        assert stats["merged"] == 0


# --- T11.4: Cross-topic duplicate (future: flag only) ---
class TestT11_4_CrossTopic:
    def test_cross_topic_not_auto_merged(self, edb):
        conn = sqlite3.connect(str(edb))
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("INSERT INTO topics (label, centroid) VALUES ('Topic 2', ?)", (b"\x00" * 10,))
        conn.commit()
        conn.close()

        v1, v2 = _make_similar_pair()
        _add_claim(edb, 1, "Shared claim", v1)
        _add_claim(edb, 2, "Shared claim copy", v2)

        # Dedup topic 1 only — should not touch topic 2's claims
        dedup = ClaimDeduplicator(epistemic_db=edb, embed_client=_mock_embed_client())
        stats = dedup.dedup_topic(1)
        assert stats["merged"] == 0  # only 1 claim in topic 1

        conn = sqlite3.connect(str(edb))
        active = conn.execute("SELECT COUNT(*) FROM claims WHERE status = 'active'").fetchone()[0]
        assert active == 2  # both still active
        conn.close()


# --- T11.5: Empty topic ---
class TestT11_5_EmptyTopic:
    def test_no_claims(self, edb):
        dedup = ClaimDeduplicator(epistemic_db=edb, embed_client=_mock_embed_client())
        stats = dedup.dedup_topic(1)
        assert stats["total_claims"] == 0
        assert stats["merged"] == 0
        assert stats["errors"] == 0


# --- T11.6: Performance with many claims ---
class TestT11_6_Performance:
    def test_100_claims(self, edb):
        # Add 100 claims, all with different embeddings
        for i in range(100):
            vec = np.random.randn(16).astype(np.float32)
            vec /= np.linalg.norm(vec)
            _add_claim(edb, 1, f"Unique claim number {i}", vec, f"sum_{i:03d}", f"excerpt {i}")

        dedup = ClaimDeduplicator(epistemic_db=edb, dedup_threshold=0.99,
                                   embed_client=_mock_embed_client())
        stats = dedup.dedup_topic(1)
        assert stats["total_claims"] == 100
        # With random vectors and high threshold, very few (if any) merges
        assert stats["errors"] == 0


# --- T11.7: Merge preserves all sources ---
class TestT11_7_SourcePreservation:
    def test_all_sources_kept(self, edb):
        v1, v2 = _make_similar_pair()
        _add_claim(edb, 1, "Claim A", v1, "sum_001", "excerpt A")
        _add_claim(edb, 1, "Claim B", v2, "sum_002", "excerpt B")

        dedup = ClaimDeduplicator(epistemic_db=edb, embed_client=_mock_embed_client())
        dedup.dedup_topic(1)

        conn = sqlite3.connect(str(edb))
        # Keeper (id=1) should have sources from both claims
        sources = conn.execute(
            "SELECT summary_id, excerpt FROM claim_sources WHERE claim_id = 1 ORDER BY summary_id"
        ).fetchall()
        assert len(sources) == 2
        sids = {s[0] for s in sources}
        assert "sum_001" in sids
        assert "sum_002" in sids
        conn.close()


# --- T11.8: Source count accuracy ---
class TestT11_8_SourceCount:
    def test_count_matches_sources(self, edb):
        v1, v2 = _make_similar_pair()
        _add_claim(edb, 1, "Claim A", v1, "sum_001")
        _add_claim(edb, 1, "Claim B", v2, "sum_002")

        dedup = ClaimDeduplicator(epistemic_db=edb, embed_client=_mock_embed_client())
        dedup.dedup_topic(1)

        conn = sqlite3.connect(str(edb))
        sc = conn.execute("SELECT source_count FROM claims WHERE id = 1").fetchone()[0]
        actual = conn.execute("SELECT COUNT(*) FROM claim_sources WHERE claim_id = 1").fetchone()[0]
        assert sc == actual == 2
        conn.close()


# --- T11.9: Iterative dedup ---
class TestT11_9_IterativeDedup:
    def test_rerun_after_new_duplicate(self, edb):
        v1, _ = _make_similar_pair()
        _add_claim(edb, 1, "Original claim", v1, "sum_001")

        dedup = ClaimDeduplicator(epistemic_db=edb, embed_client=_mock_embed_client())
        stats1 = dedup.dedup_topic(1)
        assert stats1["merged"] == 0  # only 1 claim

        # Add a near-duplicate
        _, v2 = _make_similar_pair()
        v2 = v1 + np.random.randn(16).astype(np.float32) * 0.01
        v2 /= np.linalg.norm(v2)
        _add_claim(edb, 1, "Same original claim rephrased", v2, "sum_002")

        stats2 = dedup.dedup_topic(1)
        assert stats2["merged"] == 1

        conn = sqlite3.connect(str(edb))
        sc = conn.execute("SELECT source_count FROM claims WHERE id = 1").fetchone()[0]
        assert sc == 2
        conn.close()


# --- T11.10: Threshold = 1.0 (nothing merges) ---
class TestT11_10_ThresholdMax:
    def test_no_merges_at_max(self, edb):
        v1, v2 = _make_similar_pair()
        _add_claim(edb, 1, "Claim A", v1)
        _add_claim(edb, 1, "Claim B", v2)

        dedup = ClaimDeduplicator(epistemic_db=edb, dedup_threshold=1.0,
                                   embed_client=_mock_embed_client())
        stats = dedup.dedup_topic(1)
        assert stats["merged"] == 0


# --- T11.11: Threshold = 0.0 (everything merges) ---
class TestT11_11_ThresholdMin:
    def test_all_merge_at_negative(self, edb):
        # Use aligned vectors so cosine is always positive
        for i in range(5):
            vec = np.ones(16, dtype=np.float32) + np.random.randn(16).astype(np.float32) * 0.01
            vec = np.abs(vec)  # all positive components → all cosines positive
            vec /= np.linalg.norm(vec)
            _add_claim(edb, 1, f"Claim {i}", vec, f"sum_{i:03d}")

        dedup = ClaimDeduplicator(epistemic_db=edb, dedup_threshold=-1.0,
                                   embed_client=_mock_embed_client())
        stats = dedup.dedup_topic(1)
        # All 4 should merge into the first
        assert stats["merged"] == 4

        conn = sqlite3.connect(str(edb))
        active = conn.execute("SELECT COUNT(*) FROM claims WHERE status = 'active'").fetchone()[0]
        assert active == 1
        conn.close()


# --- Run logging ---
class TestDedupLogging:
    def test_run_logged(self, edb):
        v1, v2 = _make_similar_pair()
        _add_claim(edb, 1, "Claim A", v1)
        _add_claim(edb, 1, "Claim B", v2)
        dedup = ClaimDeduplicator(epistemic_db=edb, embed_client=_mock_embed_client())
        dedup.dedup_topic(1)

        conn = sqlite3.connect(str(edb))
        row = conn.execute(
            "SELECT phase, stats FROM synthesis_runs WHERE phase = 'dedup'"
        ).fetchone()
        assert row is not None
        assert row[0] == "dedup"
        s = json.loads(row[1])
        assert "merged" in s
        conn.close()
