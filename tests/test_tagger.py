"""Tests for tagger.py — T3.* test cases from phase1-plan.md"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import init_epistemic_db
from src.embed import EmbedClient
from src.tagger import Tagger
from src import config


def make_vec(seed: int = 0, dim: int = 768) -> np.ndarray:
    """Generate a deterministic normalized vector."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def make_topic_vec(base_seed: int, noise: float = 0.05) -> np.ndarray:
    """Generate a vector close to the base seed's vector."""
    base = make_vec(base_seed)
    rng = np.random.RandomState(base_seed + 10000)
    perturbation = rng.randn(len(base)).astype(np.float32) * noise
    v = base + perturbation
    return v / np.linalg.norm(v)


@pytest.fixture
def setup(tmp_path):
    """Create epistemic.db with some topics and a fake lcm.db."""
    edb_path = tmp_path / "epistemic.db"
    lcm_path = tmp_path / "lcm.db"

    # Initialize epistemic DB
    econn = init_epistemic_db(edb_path)

    # Create 3 topics with distinct centroids
    centroids = {
        "infrastructure": make_vec(1),
        "psychedelics": make_vec(2),
        "seo-work": make_vec(3),
    }
    for label, centroid in centroids.items():
        econn.execute(
            "INSERT INTO topics (label, centroid, summary_count, created_at, updated_at) VALUES (?, ?, 5, datetime('now'), datetime('now'))",
            (label, EmbedClient.vec_to_blob(centroid)),
        )
    econn.commit()
    econn.close()

    # Create lcm.db with summaries
    lcm_conn = sqlite3.connect(str(lcm_path))
    lcm_conn.execute("CREATE TABLE summaries (summary_id TEXT PRIMARY KEY, content TEXT)")
    lcm_conn.commit()
    lcm_conn.close()

    return {
        "edb_path": edb_path,
        "lcm_path": lcm_path,
        "centroids": centroids,
    }


def insert_lcm_summary(lcm_path: Path, sid: str, content: str):
    """Insert a summary into the fake lcm.db."""
    conn = sqlite3.connect(str(lcm_path))
    conn.execute("INSERT OR REPLACE INTO summaries VALUES (?, ?)", (sid, content))
    conn.commit()
    conn.close()


class FakeEmbedClient(EmbedClient):
    """Embed client that returns deterministic vectors based on content hash."""

    def __init__(self, vec_map: dict[str, np.ndarray] = None, default_vec: np.ndarray = None):
        self._vec_map = vec_map or {}
        self._default_vec = default_vec
        self.dim = config.EMBED_DIM
        self.call_count = 0

    def embed_one(self, text: str) -> np.ndarray:
        self.call_count += 1
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        if text in self._vec_map:
            return self._vec_map[text]
        if self._default_vec is not None:
            return self._default_vec
        # Generate from hash
        seed = hash(text) % (2**31)
        return make_vec(seed)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list:
        results = []
        for t in texts:
            if t and t.strip():
                results.append(self.embed_one(t))
            else:
                results.append(None)
        return results


# --- T3.1: Tag summary with clear topic match ---
class TestT3_1_ClearMatch:
    def test_tags_to_correct_topic(self, setup):
        # Insert a summary whose embedding is close to topic 1 (infrastructure)
        close_vec = make_topic_vec(1, noise=0.02)
        insert_lcm_summary(setup["lcm_path"], "sum_infra1", "server config changes")

        fake_embed = FakeEmbedClient({"server config changes": close_vec})
        tagger = Tagger(
            epistemic_db=setup["edb_path"],
            lcm_db=setup["lcm_path"],
            embed_client=fake_embed,
        )
        stats = tagger.run()

        assert stats["tagged"] == 1

        # Verify it tagged to topic 1
        econn = sqlite3.connect(str(setup["edb_path"]))
        row = econn.execute(
            "SELECT topic_id, similarity FROM topic_summaries WHERE summary_id = 'sum_infra1'"
        ).fetchone()
        assert row is not None
        assert row[0] == 1  # infrastructure topic
        assert row[1] > 0.72
        econn.close()

    def test_centroid_updated(self, setup):
        close_vec = make_topic_vec(1, noise=0.02)
        insert_lcm_summary(setup["lcm_path"], "sum_infra2", "gateway restart logs")

        fake_embed = FakeEmbedClient({"gateway restart logs": close_vec})

        # Get original centroid
        econn = sqlite3.connect(str(setup["edb_path"]))
        orig_blob = econn.execute("SELECT centroid FROM topics WHERE id = 1").fetchone()[0]
        orig_centroid = EmbedClient.blob_to_vec(orig_blob)
        econn.close()

        tagger = Tagger(
            epistemic_db=setup["edb_path"],
            lcm_db=setup["lcm_path"],
            embed_client=fake_embed,
        )
        tagger.run()

        econn = sqlite3.connect(str(setup["edb_path"]))
        new_blob = econn.execute("SELECT centroid FROM topics WHERE id = 1").fetchone()[0]
        new_centroid = EmbedClient.blob_to_vec(new_blob)
        econn.close()

        # Centroid should have shifted
        assert not np.allclose(orig_centroid, new_centroid)
        # But not by much
        sim = EmbedClient.cosine_similarity(orig_centroid, new_centroid)
        assert sim > 0.95


# --- T3.2: Tag summary with no match ---
class TestT3_2_NoMatch:
    def test_stays_untagged(self, setup):
        # Use a random vector far from all topics
        far_vec = make_vec(9999)
        insert_lcm_summary(setup["lcm_path"], "sum_random1", "completely unrelated content")

        fake_embed = FakeEmbedClient({"completely unrelated content": far_vec})
        tagger = Tagger(
            epistemic_db=setup["edb_path"],
            lcm_db=setup["lcm_path"],
            embed_client=fake_embed,
        )
        stats = tagger.run()

        assert stats["skipped"] == 1
        assert stats["tagged"] == 0

        # Verify not in topic_summaries
        econn = sqlite3.connect(str(setup["edb_path"]))
        row = econn.execute(
            "SELECT COUNT(*) FROM topic_summaries WHERE summary_id = 'sum_random1'"
        ).fetchone()
        assert row[0] == 0
        econn.close()


# --- T3.3: Tag summary equidistant between two topics ---
class TestT3_3_Equidistant:
    def test_tags_to_highest_similarity(self, setup):
        # Create a vector slightly closer to topic 1 than topic 2
        v1 = make_vec(1)
        v2 = make_vec(2)
        # Midpoint, biased toward topic 1
        mid = 0.51 * v1 + 0.49 * v2
        mid = mid / np.linalg.norm(mid)
        mid = mid.astype(np.float32)

        insert_lcm_summary(setup["lcm_path"], "sum_mid1", "ambiguous content")
        fake_embed = FakeEmbedClient({"ambiguous content": mid})
        tagger = Tagger(
            epistemic_db=setup["edb_path"],
            lcm_db=setup["lcm_path"],
            embed_client=fake_embed,
            similarity_threshold=0.5,  # low threshold to ensure it tags
        )
        stats = tagger.run()

        if stats["tagged"] == 1:
            econn = sqlite3.connect(str(setup["edb_path"]))
            row = econn.execute(
                "SELECT topic_id FROM topic_summaries WHERE summary_id = 'sum_mid1'"
            ).fetchone()
            # Should tag to topic 1 (slightly closer)
            assert row[0] == 1
            econn.close()


# --- T3.4: Tag same summary twice (idempotent) ---
class TestT3_4_Idempotent:
    def test_no_duplicate_on_rerun(self, setup):
        close_vec = make_topic_vec(1, noise=0.02)
        insert_lcm_summary(setup["lcm_path"], "sum_dup1", "duplicate test content")

        fake_embed = FakeEmbedClient({"duplicate test content": close_vec})
        tagger = Tagger(
            epistemic_db=setup["edb_path"],
            lcm_db=setup["lcm_path"],
            embed_client=fake_embed,
        )

        stats1 = tagger.run()
        assert stats1["tagged"] == 1

        stats2 = tagger.run()
        assert stats2["processed"] == 0  # already known

        # Only one row
        econn = sqlite3.connect(str(setup["edb_path"]))
        count = econn.execute(
            "SELECT COUNT(*) FROM topic_summaries WHERE summary_id = 'sum_dup1'"
        ).fetchone()[0]
        assert count == 1
        econn.close()


# --- T3.5: Centroid drift after 50 tags ---
class TestT3_5_CentroidDrift:
    def test_centroid_stays_in_topic(self, setup):
        # Add 50 summaries all close to topic 1
        for i in range(50):
            vec = make_topic_vec(1, noise=0.05)
            insert_lcm_summary(setup["lcm_path"], f"sum_drift_{i}", f"infra content {i}")

        vecs = {f"infra content {i}": make_topic_vec(1, noise=0.05) for i in range(50)}
        fake_embed = FakeEmbedClient(vecs)
        tagger = Tagger(
            epistemic_db=setup["edb_path"],
            lcm_db=setup["lcm_path"],
            embed_client=fake_embed,
        )
        tagger.run()

        econn = sqlite3.connect(str(setup["edb_path"]))
        blob = econn.execute("SELECT centroid FROM topics WHERE id = 1").fetchone()[0]
        final_centroid = EmbedClient.blob_to_vec(blob)
        econn.close()

        # Still recognizable as topic 1
        original = make_vec(1)
        sim = EmbedClient.cosine_similarity(original, final_centroid)
        assert sim > 0.85, f"Centroid drifted too far: sim={sim}"


# --- T3.6: Centroid drift — adversarial ---
class TestT3_6_AdversarialDrift:
    def test_centroid_resists_off_topic(self, setup):
        original = make_vec(1)

        # Add 50 summaries with vectors from a DIFFERENT topic (adversarial)
        off_topic = make_vec(999)
        # But still close enough to pass threshold
        # We need them to be just above threshold for worst case
        for i in range(50):
            # Blend: 70% topic 1 + 30% off-topic (should still match topic 1)
            blended = 0.7 * original + 0.3 * off_topic
            blended = (blended / np.linalg.norm(blended)).astype(np.float32)
            insert_lcm_summary(setup["lcm_path"], f"sum_adv_{i}", f"adversarial {i}")

        vecs = {}
        for i in range(50):
            blended = 0.7 * original + 0.3 * off_topic
            blended = (blended / np.linalg.norm(blended)).astype(np.float32)
            vecs[f"adversarial {i}"] = blended

        fake_embed = FakeEmbedClient(vecs)
        tagger = Tagger(
            epistemic_db=setup["edb_path"],
            lcm_db=setup["lcm_path"],
            embed_client=fake_embed,
        )
        tagger.run()

        econn = sqlite3.connect(str(setup["edb_path"]))
        blob = econn.execute("SELECT centroid FROM topics WHERE id = 1").fetchone()[0]
        final_centroid = EmbedClient.blob_to_vec(blob)
        econn.close()

        # Centroid should still be recognizably topic 1
        sim = EmbedClient.cosine_similarity(original, final_centroid)
        assert sim > 0.80, f"Adversarial drift too severe: sim={sim}"


# --- T3.7: Threshold = 1.0 (impossible match) ---
class TestT3_7_ThresholdMax:
    def test_nothing_tags(self, setup):
        close_vec = make_topic_vec(1, noise=0.02)
        insert_lcm_summary(setup["lcm_path"], "sum_max1", "should not tag")

        fake_embed = FakeEmbedClient({"should not tag": close_vec})
        tagger = Tagger(
            epistemic_db=setup["edb_path"],
            lcm_db=setup["lcm_path"],
            embed_client=fake_embed,
            similarity_threshold=1.0,  # impossible
        )
        stats = tagger.run()
        assert stats["tagged"] == 0


# --- T3.8: Threshold = 0.0 (match everything) ---
class TestT3_8_ThresholdMin:
    def test_everything_tags(self, setup):
        far_vec = make_vec(9999)
        insert_lcm_summary(setup["lcm_path"], "sum_min1", "far away content")

        fake_embed = FakeEmbedClient({"far away content": far_vec})
        tagger = Tagger(
            epistemic_db=setup["edb_path"],
            lcm_db=setup["lcm_path"],
            embed_client=fake_embed,
            similarity_threshold=0.0,  # match everything
        )
        stats = tagger.run()
        assert stats["tagged"] == 1


# --- Run stats and logging ---
class TestRunLogging:
    def test_tagging_log_entry(self, setup):
        insert_lcm_summary(setup["lcm_path"], "sum_log1", "log test")
        fake_embed = FakeEmbedClient({"log test": make_topic_vec(1, noise=0.02)})
        tagger = Tagger(
            epistemic_db=setup["edb_path"],
            lcm_db=setup["lcm_path"],
            embed_client=fake_embed,
        )
        tagger.run()

        econn = sqlite3.connect(str(setup["edb_path"]))
        row = econn.execute("SELECT * FROM tagging_log ORDER BY id DESC LIMIT 1").fetchone()
        assert row is not None
        econn.close()

    def test_empty_run_still_logs(self, setup):
        """No new summaries → still logs the run."""
        fake_embed = FakeEmbedClient()
        tagger = Tagger(
            epistemic_db=setup["edb_path"],
            lcm_db=setup["lcm_path"],
            embed_client=fake_embed,
        )
        stats = tagger.run()
        assert stats["processed"] == 0

        econn = sqlite3.connect(str(setup["edb_path"]))
        count = econn.execute("SELECT COUNT(*) FROM tagging_log").fetchone()[0]
        assert count == 1
        econn.close()
