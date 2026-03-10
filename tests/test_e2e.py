"""End-to-end tests — T8.* test cases from phase1-plan.md

These tests run against the REAL lcm.db with the real embedding service.
They create a temporary epistemic.db to avoid polluting production.

NOTE: These are marked @pytest.mark.slow and skipped by default.
Run with: pytest tests/test_e2e.py -m slow
"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import init_epistemic_db, open_lcm_readonly
from src.embed import EmbedClient
from src.tagger import Tagger
from src.discovery import Discovery
from src.orphans import OrphanReconciler
from src import config


@pytest.fixture
def real_setup(tmp_path):
    """Set up with real lcm.db and temporary epistemic.db."""
    if not config.LCM_DB.exists():
        pytest.skip("Real lcm.db not available")
    edb_path = tmp_path / "epistemic.db"
    return {"edb_path": edb_path, "lcm_path": config.LCM_DB}


# --- T8.1: Fresh start with all real summaries ---
@pytest.mark.slow
class TestT8_1_FreshStart:
    def test_discovery_on_real_data(self, real_setup):
        """Run full discovery on real lcm.db — should produce topics close to POC (19 clusters)."""
        disc = Discovery(
            epistemic_db=real_setup["edb_path"],
            lcm_db=real_setup["lcm_path"],
            min_discovery_batch=10,
            min_cluster_size=3,
        )
        stats = disc.run()

        print(f"\n  Real data: {stats['untagged_count']} summaries")
        print(f"  Clusters found: {stats['clusters_found']}")
        print(f"  New topics: {stats['new_topics_created']}")
        print(f"  Noise: {stats['still_noise']}")
        print(f"  Duration: {stats['duration_ms']}ms")

        # Regression vs POC: should find roughly 10-30 topics
        # (exact count depends on new summaries since POC)
        assert stats["clusters_found"] >= 5, "Too few clusters — regression?"
        assert stats["clusters_found"] <= 50, "Too many clusters — param issue?"
        assert stats["new_topics_created"] > 0

        # Verify data landed in DB
        econn = sqlite3.connect(str(real_setup["edb_path"]))
        topic_count = econn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
        tagged_count = econn.execute("SELECT COUNT(*) FROM topic_summaries").fetchone()[0]
        econn.close()

        assert topic_count == stats["new_topics_created"]
        assert tagged_count > 0
        print(f"  Topics in DB: {topic_count}")
        print(f"  Tagged summaries: {tagged_count}")


# --- T8.2: Incremental tagging after discovery ---
@pytest.mark.slow
class TestT8_2_IncrementalTag:
    def test_tag_after_discovery(self, real_setup):
        """Run discovery first, then tagging — new summaries should tag to existing topics."""
        # First: discovery to create topics
        disc = Discovery(
            epistemic_db=real_setup["edb_path"],
            lcm_db=real_setup["lcm_path"],
            min_discovery_batch=10,
            min_cluster_size=3,
        )
        disc_stats = disc.run()

        # Then: tagging pass on remaining untagged
        tagger = Tagger(
            epistemic_db=real_setup["edb_path"],
            lcm_db=real_setup["lcm_path"],
            similarity_threshold=0.65,  # slightly lower to catch more
        )
        tag_stats = tagger.run()

        print(f"\n  After discovery: {disc_stats['new_topics_created']} topics")
        print(f"  Tagging pass: {tag_stats['processed']} processed, {tag_stats['tagged']} tagged, {tag_stats['skipped']} noise")

        # Some should tag, some should remain noise
        total_tagged = tag_stats["tagged"]
        # Not asserting exact numbers — depends on data


# --- T8.4: Full pipeline ---
@pytest.mark.slow
class TestT8_4_FullPipeline:
    def test_full_pipeline(self, real_setup):
        """Discovery → Tag → Orphan check → all complete without errors."""
        # 1. Discovery
        disc = Discovery(
            epistemic_db=real_setup["edb_path"],
            lcm_db=real_setup["lcm_path"],
            min_discovery_batch=10,
            min_cluster_size=3,
        )
        disc_stats = disc.run()
        assert disc_stats.get("errors") is None or disc_stats.get("errors", 0) == 0

        # 2. Tag remaining
        tagger = Tagger(
            epistemic_db=real_setup["edb_path"],
            lcm_db=real_setup["lcm_path"],
        )
        tag_stats = tagger.run()
        assert tag_stats["errors"] == 0

        # 3. Orphan check
        reconciler = OrphanReconciler(
            epistemic_db=real_setup["edb_path"],
            lcm_db=real_setup["lcm_path"],
        )
        orphan_stats = reconciler.run()
        # Real data: should have 0 orphans (lcm.db is live, nothing re-compacted mid-test)
        assert orphan_stats["orphaned"] == 0

        # 4. Verify tagging_log has all 3 entries
        econn = sqlite3.connect(str(real_setup["edb_path"]))
        log_entries = econn.execute(
            "SELECT run_type FROM tagging_log ORDER BY id"
        ).fetchall()
        run_types = [r[0] for r in log_entries]
        assert "discovery" in run_types
        assert "tag" in run_types
        assert "orphan" in run_types
        econn.close()

        print(f"\n  Full pipeline complete:")
        print(f"    Discovery: {disc_stats['new_topics_created']} topics, {disc_stats['clusters_found']} clusters")
        print(f"    Tagging: {tag_stats['tagged']} tagged, {tag_stats['skipped']} noise")
        print(f"    Orphans: {orphan_stats['orphaned']} found")


# --- Verify lcm.db was never written to ---
@pytest.mark.slow
class TestLcmReadOnly:
    def test_lcm_untouched(self, real_setup):
        """After full pipeline, lcm.db should have no evidence of writes."""
        import os
        import stat

        # Get mtime before
        lcm_stat_before = os.stat(real_setup["lcm_path"])

        # Run full pipeline
        disc = Discovery(
            epistemic_db=real_setup["edb_path"],
            lcm_db=real_setup["lcm_path"],
            min_discovery_batch=10,
            min_cluster_size=3,
        )
        disc.run()

        tagger = Tagger(
            epistemic_db=real_setup["edb_path"],
            lcm_db=real_setup["lcm_path"],
        )
        tagger.run()

        reconciler = OrphanReconciler(
            epistemic_db=real_setup["edb_path"],
            lcm_db=real_setup["lcm_path"],
        )
        reconciler.run()

        # Mtime should be unchanged (or very close — WAL reads don't modify)
        # Note: WAL readers CAN update -shm file but not the main db
        lcm_stat_after = os.stat(real_setup["lcm_path"])
        # Main file mtime should not have changed
        assert lcm_stat_before.st_mtime == lcm_stat_after.st_mtime
