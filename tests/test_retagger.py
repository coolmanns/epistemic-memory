"""Tests for retagger.py — re-tagging pipeline for seeded and neighbor-aware topics.

Test plan:
  RT-1: Full scan tags summaries above threshold
  RT-2: Full scan skips summaries below threshold
  RT-3: Already-tagged summaries are not double-tagged
  RT-4: Topic with no matching summaries: zero tags, no errors
  RT-5: Summary count updated after retag
  RT-6: Content-derived centroid beats label-only centroid
  RT-7: retag_new_topics only processes topics with 0 summaries
  RT-8: Neighbor scan finds candidates from neighbor topics
  RT-9: Neighbor scan skips non-neighbor topics (low centroid similarity)
  RT-10: Tagging log entry written for each retag run
"""

import sqlite3
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import init_epistemic_db
from src.embed import EmbedClient
from src.retagger import Retagger


# --- Helpers ---

DIM = 16  # Small dimension for fast tests


def make_vec(seed: int, dim: int = DIM) -> np.ndarray:
    """Deterministic normalized vector."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def make_close_vec(base_seed: int, noise: float = 0.05, dim: int = DIM) -> np.ndarray:
    """Vector close to base seed's vector (high cosine similarity)."""
    base = make_vec(base_seed, dim)
    rng = np.random.RandomState(base_seed + 10000)
    perturbation = rng.randn(dim).astype(np.float32) * noise
    v = base + perturbation
    return v / np.linalg.norm(v)


def make_far_vec(seed: int, away_from_seed: int, dim: int = DIM) -> np.ndarray:
    """Vector guaranteed to be far from away_from_seed's vector."""
    target = make_vec(away_from_seed, dim)
    # Flip the target and add noise for a dissimilar vector
    v = -target + np.random.RandomState(seed + 50000).randn(dim).astype(np.float32) * 0.1
    return v / np.linalg.norm(v)


class FakeEmbedClient(EmbedClient):
    """Embed client that returns pre-assigned vectors by summary content."""

    def __init__(self, content_to_vec: dict[str, np.ndarray], dim: int = DIM):
        self._map = content_to_vec
        self._default = make_vec(9999, dim)
        self.dim = dim

    def embed_one(self, text: str) -> np.ndarray:
        return self._map.get(text, self._default)

    def embed_batch(self, texts, batch_size=8) -> list[Optional[np.ndarray]]:
        return [self._map.get(t, self._default) for t in texts]

    @staticmethod
    def vec_to_blob(vec):
        return vec.astype(np.float32).tobytes()

    @staticmethod
    def blob_to_vec(blob):
        return np.frombuffer(blob, dtype=np.float32)


def _create_lcm_db(path: Path) -> sqlite3.Connection:
    """Create a minimal LCM database."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE summaries (summary_id TEXT PRIMARY KEY, content TEXT, depth INTEGER DEFAULT 0)"
    )
    conn.commit()
    return conn


def _insert_summary(lcm_path: Path, sid: str, content: str, depth: int = 0):
    conn = sqlite3.connect(str(lcm_path))
    conn.execute("INSERT INTO summaries VALUES (?, ?, ?)", (sid, content, depth))
    conn.commit()
    conn.close()


def _insert_topic(edb_path: Path, label: str, centroid: np.ndarray) -> int:
    conn = sqlite3.connect(str(edb_path))
    cur = conn.execute(
        "INSERT INTO topics (label, centroid, summary_count) VALUES (?, ?, 0)",
        (label, EmbedClient.vec_to_blob(centroid)),
    )
    tid = cur.lastrowid
    conn.commit()
    conn.close()
    return tid


def _get_tags(edb_path: Path, topic_id: int) -> list[str]:
    conn = sqlite3.connect(str(edb_path))
    rows = conn.execute(
        "SELECT summary_id FROM topic_summaries WHERE topic_id = ?", (topic_id,)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _get_summary_count(edb_path: Path, topic_id: int) -> int:
    conn = sqlite3.connect(str(edb_path))
    row = conn.execute("SELECT summary_count FROM topics WHERE id = ?", (topic_id,)).fetchone()
    conn.close()
    return row[0] if row else 0


def _get_tagging_log(edb_path: Path) -> list[dict]:
    conn = sqlite3.connect(str(edb_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM tagging_log ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@pytest.fixture
def env(tmp_path):
    """Set up epistemic.db + lcm.db with test data."""
    edb_path = tmp_path / "epistemic.db"
    lcm_path = tmp_path / "lcm.db"

    init_epistemic_db(edb_path).close()
    _create_lcm_db(lcm_path).close()

    return {"edb": edb_path, "lcm": lcm_path}


# --- Tests ---


class TestFullScan:
    """RT-1 through RT-6: full scan retag_topic behavior."""

    def test_tags_above_threshold(self, env):
        """RT-1: Summaries with high similarity get tagged."""
        centroid = make_vec(1)
        tid = _insert_topic(env["edb"], "Test Topic", centroid)

        # Insert summaries — 3 close, 2 far
        close_vecs = {}
        for i in range(3):
            sid = f"sum_close_{i}"
            content = f"close content {i}"
            _insert_summary(env["lcm"], sid, content)
            close_vecs[content] = make_close_vec(1, noise=0.02)

        for i in range(2):
            sid = f"sum_far_{i}"
            content = f"far content {i}"
            _insert_summary(env["lcm"], sid, content)
            close_vecs[content] = make_far_vec(i + 100, away_from_seed=1)

        embed = FakeEmbedClient(close_vecs)
        retagger = Retagger(
            epistemic_db=env["edb"],
            lcm_db=env["lcm"],
            embed_client=embed,
            similarity_threshold=0.65,
        )

        stats = retagger.retag_topic(tid)
        tags = _get_tags(env["edb"], tid)

        assert stats["tagged"] == 3
        assert stats["below_threshold"] == 2
        assert len(tags) == 3
        assert all(t.startswith("sum_close_") for t in tags)

    def test_skips_below_threshold(self, env):
        """RT-2: Summaries below similarity threshold are not tagged."""
        centroid = make_vec(1)
        tid = _insert_topic(env["edb"], "Test Topic", centroid)

        # Only far summaries
        vecs = {}
        for i in range(5):
            sid = f"sum_far_{i}"
            content = f"unrelated content {i}"
            _insert_summary(env["lcm"], sid, content)
            vecs[content] = make_far_vec(i + 200, away_from_seed=1)

        embed = FakeEmbedClient(vecs)
        retagger = Retagger(
            epistemic_db=env["edb"],
            lcm_db=env["lcm"],
            embed_client=embed,
            similarity_threshold=0.65,
        )

        stats = retagger.retag_topic(tid)
        assert stats["tagged"] == 0
        assert stats["below_threshold"] == 5
        assert _get_tags(env["edb"], tid) == []

    def test_no_double_tagging(self, env):
        """RT-3: Already-tagged summaries are counted but not re-inserted."""
        centroid = make_vec(1)
        tid = _insert_topic(env["edb"], "Test Topic", centroid)

        content = "already tagged content"
        _insert_summary(env["lcm"], "sum_existing", content)
        vecs = {content: make_close_vec(1, noise=0.02)}

        # Pre-tag it
        conn = sqlite3.connect(str(env["edb"]))
        conn.execute(
            "INSERT INTO topic_summaries (topic_id, summary_id, similarity) VALUES (?, ?, ?)",
            (tid, "sum_existing", 0.9),
        )
        conn.commit()
        conn.close()

        embed = FakeEmbedClient(vecs)
        retagger = Retagger(
            epistemic_db=env["edb"],
            lcm_db=env["lcm"],
            embed_client=embed,
            similarity_threshold=0.65,
        )

        stats = retagger.retag_topic(tid)
        assert stats["already_tagged"] == 1
        assert stats["tagged"] == 0

        # Still only one entry
        tags = _get_tags(env["edb"], tid)
        assert len(tags) == 1

    def test_no_matching_summaries(self, env):
        """RT-4: Topic with empty LCM gets zero tags, no errors."""
        centroid = make_vec(1)
        tid = _insert_topic(env["edb"], "Empty Topic", centroid)
        # No summaries in LCM at all

        embed = FakeEmbedClient({})
        retagger = Retagger(
            epistemic_db=env["edb"],
            lcm_db=env["lcm"],
            embed_client=embed,
            similarity_threshold=0.65,
        )

        stats = retagger.retag_topic(tid)
        assert stats["tagged"] == 0
        assert stats["scanned"] == 0
        assert stats["errors"] == 0

    def test_summary_count_updated(self, env):
        """RT-5: topic.summary_count reflects actual tags after retag."""
        centroid = make_vec(1)
        tid = _insert_topic(env["edb"], "Count Topic", centroid)

        vecs = {}
        for i in range(4):
            content = f"matching {i}"
            _insert_summary(env["lcm"], f"sum_{i}", content)
            vecs[content] = make_close_vec(1, noise=0.02)

        embed = FakeEmbedClient(vecs)
        retagger = Retagger(
            epistemic_db=env["edb"],
            lcm_db=env["lcm"],
            embed_client=embed,
            similarity_threshold=0.65,
        )

        retagger.retag_topic(tid)
        assert _get_summary_count(env["edb"], tid) == 4

    def test_content_centroid_beats_label(self, env):
        """RT-6: A centroid derived from content matches more than a label centroid.

        This is the bug we found: seeding centroids from "Martin Ball" (2 words)
        misses most real Martin Ball content.
        """
        # The "real" content direction
        content_direction = make_vec(42)

        # Label centroid: a different direction (simulates embedding "Martin Ball")
        label_centroid = make_vec(99)

        # Content centroid: derived from actual content (same direction as summaries)
        content_centroid = make_close_vec(42, noise=0.01)

        tid_label = _insert_topic(env["edb"], "Label Topic", label_centroid)
        tid_content = _insert_topic(env["edb"], "Content Topic", content_centroid)

        # Insert 10 summaries that match the content direction
        vecs = {}
        for i in range(10):
            content = f"real content about topic {i}"
            _insert_summary(env["lcm"], f"sum_{i}", content)
            vecs[content] = make_close_vec(42, noise=0.03 + i * 0.005)

        embed = FakeEmbedClient(vecs)

        retagger_label = Retagger(
            epistemic_db=env["edb"],
            lcm_db=env["lcm"],
            embed_client=embed,
            similarity_threshold=0.65,
        )
        retagger_content = Retagger(
            epistemic_db=env["edb"],
            lcm_db=env["lcm"],
            embed_client=embed,
            similarity_threshold=0.65,
        )

        stats_label = retagger_label.retag_topic(tid_label)
        stats_content = retagger_content.retag_topic(tid_content)

        # Content centroid should tag more summaries
        assert stats_content["tagged"] > stats_label["tagged"]


class TestRetagNewTopics:
    """RT-7: retag_new_topics only processes topics with 0 summaries."""

    def test_only_empty_topics(self, env):
        """RT-7: Topics with existing summaries are skipped."""
        centroid_a = make_vec(1)
        centroid_b = make_vec(2)

        tid_a = _insert_topic(env["edb"], "Has Tags", centroid_a)
        tid_b = _insert_topic(env["edb"], "No Tags", centroid_b)

        # Tag one summary to topic A
        _insert_summary(env["lcm"], "sum_a", "content a")
        conn = sqlite3.connect(str(env["edb"]))
        conn.execute(
            "INSERT INTO topic_summaries (topic_id, summary_id, similarity) VALUES (?, ?, ?)",
            (tid_a, "sum_a", 0.9),
        )
        conn.commit()
        conn.close()

        # Add a summary that matches topic B
        content_b = "content for topic b"
        _insert_summary(env["lcm"], "sum_b", content_b)
        vecs = {
            "content a": make_close_vec(1, noise=0.02),
            content_b: make_close_vec(2, noise=0.02),
        }

        embed = FakeEmbedClient(vecs)
        retagger = Retagger(
            epistemic_db=env["edb"],
            lcm_db=env["lcm"],
            embed_client=embed,
            similarity_threshold=0.65,
        )

        results = retagger.retag_new_topics()

        # Only topic B should have been processed
        assert len(results) == 1
        assert results[0]["topic_id"] == tid_b


class TestNeighborScan:
    """RT-8 through RT-9: neighbor-aware re-tagging."""

    def test_finds_neighbor_candidates(self, env):
        """RT-8: Summaries tagged to a neighbor topic are re-evaluated."""
        # Two similar topic centroids (neighbors)
        centroid_a = make_vec(1)
        centroid_b = make_close_vec(1, noise=0.1)  # Close to A

        tid_a = _insert_topic(env["edb"], "Topic A", centroid_a)
        tid_b = _insert_topic(env["edb"], "Topic B", centroid_b)

        # Tag a summary to topic A
        content = "shared domain content"
        _insert_summary(env["lcm"], "sum_shared", content)
        conn = sqlite3.connect(str(env["edb"]))
        conn.execute(
            "INSERT INTO topic_summaries (topic_id, summary_id, similarity) VALUES (?, ?, ?)",
            (tid_a, "sum_shared", 0.8),
        )
        conn.commit()
        conn.close()

        # The summary is also close to topic B
        vecs = {content: make_close_vec(1, noise=0.05)}
        embed = FakeEmbedClient(vecs)

        retagger = Retagger(
            epistemic_db=env["edb"],
            lcm_db=env["lcm"],
            embed_client=embed,
            similarity_threshold=0.65,
            neighbor_sim_threshold=0.4,
        )

        stats = retagger.retag_neighbors(tid_b)
        assert stats["neighbors_checked"] >= 1
        assert stats["candidates"] >= 1
        assert stats["tagged"] >= 1

        # Summary should now be tagged to both topics
        tags_b = _get_tags(env["edb"], tid_b)
        assert "sum_shared" in tags_b

    def test_skips_distant_topics(self, env):
        """RT-9: Topics with dissimilar centroids are not considered neighbors."""
        centroid_a = make_vec(1)
        centroid_far = make_far_vec(500, away_from_seed=1)  # Far from A

        tid_a = _insert_topic(env["edb"], "Topic A", centroid_a)
        tid_far = _insert_topic(env["edb"], "Distant Topic", centroid_far)

        # Tag a summary to distant topic
        content = "distant content"
        _insert_summary(env["lcm"], "sum_distant", content)
        conn = sqlite3.connect(str(env["edb"]))
        conn.execute(
            "INSERT INTO topic_summaries (topic_id, summary_id, similarity) VALUES (?, ?, ?)",
            (tid_far, "sum_distant", 0.8),
        )
        conn.commit()
        conn.close()

        vecs = {content: make_far_vec(501, away_from_seed=1)}
        embed = FakeEmbedClient(vecs)

        retagger = Retagger(
            epistemic_db=env["edb"],
            lcm_db=env["lcm"],
            embed_client=embed,
            similarity_threshold=0.65,
            neighbor_sim_threshold=0.4,
        )

        stats = retagger.retag_neighbors(tid_a)
        assert stats["neighbors_checked"] == 0
        assert stats["candidates"] == 0
        assert stats["tagged"] == 0


class TestLogging:
    """RT-10: Tagging log entries."""

    def test_log_entry_written(self, env):
        """RT-10: Each retag writes a tagging_log entry."""
        centroid = make_vec(1)
        tid = _insert_topic(env["edb"], "Logged Topic", centroid)
        _insert_summary(env["lcm"], "sum_1", "some content")

        embed = FakeEmbedClient({"some content": make_close_vec(1, noise=0.02)})
        retagger = Retagger(
            epistemic_db=env["edb"],
            lcm_db=env["lcm"],
            embed_client=embed,
            similarity_threshold=0.65,
        )

        retagger.retag_topic(tid)
        logs = _get_tagging_log(env["edb"])
        assert len(logs) >= 1
        assert logs[-1]["run_type"] == "retag"
        assert logs[-1]["summaries_tagged"] == 1


class TestNonexistentTopic:
    """Edge case: retag a topic that doesn't exist."""

    def test_missing_topic_returns_error(self, env):
        embed = FakeEmbedClient({})
        retagger = Retagger(
            epistemic_db=env["edb"],
            lcm_db=env["lcm"],
            embed_client=embed,
            similarity_threshold=0.65,
        )

        stats = retagger.retag_topic(9999)
        assert stats["errors"] == 1
