"""Tests for orphans.py — T6.* test cases from phase1-plan.md"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import init_epistemic_db
from src.embed import EmbedClient
from src.orphans import OrphanReconciler
from src import config


def make_vec(seed: int = 0, dim: int = 768) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedClient(EmbedClient):
    def __init__(self):
        self.dim = config.EMBED_DIM

    def embed_one(self, text):
        return make_vec(hash(text) % (2**31))

    def embed_batch(self, texts, batch_size=32):
        return [self.embed_one(t) if t and t.strip() else None for t in texts]


@pytest.fixture
def setup(tmp_path):
    """Create epistemic.db with a topic + tagged summaries, and matching lcm.db."""
    edb_path = tmp_path / "epistemic.db"
    lcm_path = tmp_path / "lcm.db"

    # Epistemic DB with 1 topic and 5 tagged summaries
    econn = init_epistemic_db(edb_path)
    econn.execute(
        "INSERT INTO topics (label, centroid, summary_count, created_at, updated_at) VALUES (?, ?, 5, datetime('now'), datetime('now'))",
        ("test-topic", EmbedClient.vec_to_blob(make_vec(1))),
    )
    for i in range(5):
        econn.execute(
            "INSERT INTO topic_summaries (topic_id, summary_id, similarity) VALUES (1, ?, 0.85)",
            (f"sum_{i}",),
        )
    econn.commit()
    econn.close()

    # LCM DB with all 5 summaries present
    lcm_conn = sqlite3.connect(str(lcm_path))
    lcm_conn.execute("CREATE TABLE summaries (summary_id TEXT PRIMARY KEY, content TEXT)")
    for i in range(5):
        lcm_conn.execute(f"INSERT INTO summaries VALUES ('sum_{i}', 'content {i}')")
    lcm_conn.commit()
    lcm_conn.close()

    return {"edb_path": edb_path, "lcm_path": lcm_path}


def remove_from_lcm(lcm_path: Path, ids: list[str]):
    """Simulate LCM re-compaction by deleting summaries."""
    conn = sqlite3.connect(str(lcm_path))
    for sid in ids:
        conn.execute("DELETE FROM summaries WHERE summary_id = ?", (sid,))
    conn.commit()
    conn.close()


# --- T6.1: Summary exists in both DBs ---
class TestT6_1_AllPresent:
    def test_no_orphans(self, setup):
        rec = OrphanReconciler(
            epistemic_db=setup["edb_path"],
            lcm_db=setup["lcm_path"],
            embed_client=FakeEmbedClient(),
        )
        stats = rec.run()

        assert stats["checked"] == 5
        assert stats["orphaned"] == 0

        # Verify no orphan flags set
        econn = sqlite3.connect(str(setup["edb_path"]))
        orphaned = econn.execute(
            "SELECT COUNT(*) FROM topic_summaries WHERE orphaned = 1"
        ).fetchone()[0]
        assert orphaned == 0
        econn.close()


# --- T6.2: Summary deleted from lcm.db ---
class TestT6_2_Deleted:
    def test_marks_orphaned(self, setup):
        remove_from_lcm(setup["lcm_path"], ["sum_0", "sum_1"])

        rec = OrphanReconciler(
            epistemic_db=setup["edb_path"],
            lcm_db=setup["lcm_path"],
            embed_client=FakeEmbedClient(),
        )
        stats = rec.run()

        assert stats["orphaned"] == 2

        econn = sqlite3.connect(str(setup["edb_path"]))
        orphaned_ids = [
            r[0] for r in econn.execute(
                "SELECT summary_id FROM topic_summaries WHERE orphaned = 1"
            ).fetchall()
        ]
        assert set(orphaned_ids) == {"sum_0", "sum_1"}
        econn.close()


# --- T6.3: Summary deleted, no parent ---
class TestT6_3_NoParent:
    def test_flagged_for_review(self, setup):
        # Remove 1 summary — no parent concept in our simple lcm mock
        remove_from_lcm(setup["lcm_path"], ["sum_2"])

        rec = OrphanReconciler(
            epistemic_db=setup["edb_path"],
            lcm_db=setup["lcm_path"],
            embed_client=FakeEmbedClient(),
        )
        stats = rec.run()
        assert stats["orphaned"] == 1


# --- T6.4: >30% orphaned ---
class TestT6_4_HighOrphanRatio:
    def test_topic_flagged(self, setup):
        # Remove 2 of 5 = 40% > 30% threshold
        remove_from_lcm(setup["lcm_path"], ["sum_0", "sum_1"])

        rec = OrphanReconciler(
            epistemic_db=setup["edb_path"],
            lcm_db=setup["lcm_path"],
            embed_client=FakeEmbedClient(),
            max_orphan_ratio=0.3,
        )
        stats = rec.run()

        assert len(stats["topics_flagged"]) == 1
        assert stats["topics_flagged"][0]["id"] == 1
        assert stats["topics_flagged"][0]["ratio"] == 0.4


# --- T6.5: 100% orphaned ---
class TestT6_5_AllOrphaned:
    def test_topic_flagged_dead_not_deleted(self, setup):
        # Remove all 5
        remove_from_lcm(setup["lcm_path"], [f"sum_{i}" for i in range(5)])

        rec = OrphanReconciler(
            epistemic_db=setup["edb_path"],
            lcm_db=setup["lcm_path"],
            embed_client=FakeEmbedClient(),
        )
        stats = rec.run()

        assert len(stats["topics_dead"]) == 1
        assert stats["topics_dead"][0]["id"] == 1

        # Topic still exists — NOT auto-deleted
        econn = sqlite3.connect(str(setup["edb_path"]))
        count = econn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
        assert count == 1
        econn.close()


# --- Logging ---
class TestOrphanLogging:
    def test_logs_run(self, setup):
        rec = OrphanReconciler(
            epistemic_db=setup["edb_path"],
            lcm_db=setup["lcm_path"],
            embed_client=FakeEmbedClient(),
        )
        rec.run()

        econn = sqlite3.connect(str(setup["edb_path"]))
        row = econn.execute(
            "SELECT run_type FROM tagging_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row[0] == "orphan"
        econn.close()
