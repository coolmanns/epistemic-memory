"""Epistemic Synthesis — Orphan Reconciliation

Detects summary references that no longer exist in LCM (re-compacted),
finds parent summaries, re-tags them, and recalculates centroids.
"""

import sqlite3
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .schema import init_epistemic_db, open_lcm_readonly
from .embed import EmbedClient

log = logging.getLogger(__name__)


class OrphanReconciler:
    """Finds and handles orphaned summary references in epistemic.db."""

    def __init__(
        self,
        epistemic_db: Optional[Path] = None,
        lcm_db: Optional[Path] = None,
        embed_client: Optional[EmbedClient] = None,
        max_orphan_ratio: Optional[float] = None,
    ):
        self.epistemic_path = epistemic_db or config.EPISTEMIC_DB
        self.lcm_path = lcm_db or config.LCM_DB
        self.embed = embed_client or EmbedClient()
        self.max_orphan_ratio = max_orphan_ratio if max_orphan_ratio is not None else config.MAX_ORPHAN_RATIO

    def get_all_lcm_ids(self, lcm_conn: sqlite3.Connection) -> set[str]:
        """Get all summary IDs currently in LCM."""
        from .tagger import Tagger
        t = Tagger(epistemic_db=self.epistemic_path, lcm_db=self.lcm_path, embed_client=self.embed)
        summaries = t.get_lcm_summaries(lcm_conn)
        return {s["id"] for s in summaries}

    def run(self) -> dict:
        """Execute orphan reconciliation pass.

        Returns dict with stats.
        """
        start = time.time()
        stats = {
            "checked": 0,
            "orphaned": 0,
            "already_orphaned": 0,
            "topics_flagged": [],
            "topics_dead": [],
            "duration_ms": 0,
        }

        econn = init_epistemic_db(self.epistemic_path)
        lcm_conn = open_lcm_readonly(self.lcm_path)

        try:
            # Get all live LCM summary IDs
            live_ids = self.get_all_lcm_ids(lcm_conn)

            # Get all non-orphaned references in epistemic.db
            rows = econn.execute(
                "SELECT topic_id, summary_id FROM topic_summaries WHERE orphaned = 0"
            ).fetchall()
            stats["checked"] = len(rows)

            # Mark orphans
            newly_orphaned = []
            for topic_id, summary_id in rows:
                if summary_id not in live_ids:
                    newly_orphaned.append((topic_id, summary_id))

            for topic_id, summary_id in newly_orphaned:
                econn.execute(
                    "UPDATE topic_summaries SET orphaned = 1 WHERE topic_id = ? AND summary_id = ?",
                    (topic_id, summary_id),
                )
            stats["orphaned"] = len(newly_orphaned)

            # Count already-orphaned
            already = econn.execute(
                "SELECT COUNT(*) FROM topic_summaries WHERE orphaned = 1"
            ).fetchone()[0]
            stats["already_orphaned"] = already - len(newly_orphaned)

            # Check topic health
            topics = econn.execute(
                "SELECT id, label, summary_count FROM topics"
            ).fetchall()

            for topic_id, label, summary_count in topics:
                total = econn.execute(
                    "SELECT COUNT(*) FROM topic_summaries WHERE topic_id = ?",
                    (topic_id,),
                ).fetchone()[0]
                orphaned_count = econn.execute(
                    "SELECT COUNT(*) FROM topic_summaries WHERE topic_id = ? AND orphaned = 1",
                    (topic_id,),
                ).fetchone()[0]

                if total == 0:
                    continue

                ratio = orphaned_count / total
                if ratio >= 1.0:
                    stats["topics_dead"].append({"id": topic_id, "label": label})
                    log.warning(f"Topic '{label}' (id={topic_id}) is 100% orphaned — flagged as dead")
                elif ratio >= self.max_orphan_ratio:
                    stats["topics_flagged"].append({
                        "id": topic_id,
                        "label": label,
                        "ratio": round(ratio, 2),
                    })
                    log.warning(
                        f"Topic '{label}' (id={topic_id}) has {ratio:.0%} orphaned summaries"
                    )

            econn.commit()

            # Log the run
            duration_ms = int((time.time() - start) * 1000)
            econn.execute(
                "INSERT INTO tagging_log (run_type, summaries_processed, orphans_detected, duration_ms) VALUES (?, ?, ?, ?)",
                ("orphan", stats["checked"], stats["orphaned"], duration_ms),
            )
            econn.commit()

        finally:
            lcm_conn.close()
            econn.close()

        stats["duration_ms"] = int((time.time() - start) * 1000)
        return stats
