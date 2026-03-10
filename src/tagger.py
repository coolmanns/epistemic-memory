"""Epistemic Synthesis — Tagging Pipeline

Core loop: fetch untagged summaries from LCM → embed → match against topic centroids → tag.
Reads lcm.db read-only. Writes to epistemic.db only.
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


class Tagger:
    """Tags LCM summaries with topic IDs based on embedding similarity."""

    def __init__(
        self,
        epistemic_db: Optional[Path] = None,
        lcm_db: Optional[Path] = None,
        embed_client: Optional[EmbedClient] = None,
        similarity_threshold: Optional[float] = None,
        centroid_weight: Optional[float] = None,
    ):
        self.epistemic_path = epistemic_db or config.EPISTEMIC_DB
        self.lcm_path = lcm_db or config.LCM_DB
        self.embed = embed_client or EmbedClient()
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else config.SIMILARITY_THRESHOLD
        self.centroid_weight = centroid_weight if centroid_weight is not None else config.CENTROID_UPDATE_WEIGHT

    def get_known_summary_ids(self, econn: sqlite3.Connection) -> set[str]:
        """Get all summary IDs already in epistemic.db (tagged or attempted)."""
        rows = econn.execute("SELECT DISTINCT summary_id FROM topic_summaries").fetchall()
        return {r[0] for r in rows}

    def get_lcm_summaries(self, lcm_conn: sqlite3.Connection) -> list[dict]:
        """Fetch summaries from lcm.db. Returns list of {id, content}."""
        # LCM stores summaries — find the actual table structure
        tables = [
            r["name"]
            for r in lcm_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]

        summaries = []

        # LCM schema: summaries table has summary_id (not id) + content
        if "summaries" in tables:
            rows = lcm_conn.execute("SELECT summary_id, content FROM summaries").fetchall()
            for r in rows:
                summaries.append({"id": r["summary_id"], "content": r["content"]})
        else:
            # Fallback: look for any table with summary_id + content columns
            for table in tables:
                try:
                    cols = {
                        r[1]
                        for r in lcm_conn.execute(f"PRAGMA table_info({table})").fetchall()
                    }
                    if "summary_id" in cols and "content" in cols:
                        rows = lcm_conn.execute(
                            f"SELECT summary_id, content FROM [{table}]"
                        ).fetchall()
                        for r in rows:
                            sid = r["summary_id"]
                            if isinstance(sid, str) and sid.startswith("sum_"):
                                summaries.append({"id": sid, "content": r["content"]})
                        if summaries:
                            break
                except sqlite3.OperationalError:
                    continue

        return summaries

    def get_topics_with_centroids(self, econn: sqlite3.Connection) -> list[dict]:
        """Load all topics with their centroid vectors."""
        rows = econn.execute("SELECT id, label, centroid, summary_count FROM topics").fetchall()
        topics = []
        for r in rows:
            topics.append({
                "id": r[0],
                "label": r[1],
                "centroid": EmbedClient.blob_to_vec(r[2]),
                "summary_count": r[3],
            })
        return topics

    def find_best_topic(
        self, vec: np.ndarray, topics: list[dict]
    ) -> tuple[Optional[dict], float]:
        """Find the most similar topic for a vector. Returns (topic, similarity)."""
        if not topics:
            return None, 0.0

        best_topic = None
        best_sim = -1.0

        for topic in topics:
            sim = EmbedClient.cosine_similarity(vec, topic["centroid"])
            if sim > best_sim:
                best_sim = sim
                best_topic = topic

        return best_topic, best_sim

    def update_centroid(
        self, old_centroid: np.ndarray, new_vec: np.ndarray
    ) -> np.ndarray:
        """EMA update: blend new vector into existing centroid."""
        w = self.centroid_weight
        updated = (1 - w) * old_centroid + w * new_vec
        # Re-normalize
        norm = np.linalg.norm(updated)
        if norm > 0:
            updated = updated / norm
        return updated

    def run(self) -> dict:
        """Execute one tagging pass.

        Returns dict with run stats: {processed, tagged, skipped, duration_ms}
        """
        start = time.time()
        stats = {
            "processed": 0,
            "tagged": 0,
            "skipped": 0,
            "errors": 0,
        }

        # Open connections
        econn = init_epistemic_db(self.epistemic_path)
        lcm_conn = open_lcm_readonly(self.lcm_path)

        try:
            # Get already-known summaries
            known_ids = self.get_known_summary_ids(econn)

            # Get all LCM summaries
            all_summaries = self.get_lcm_summaries(lcm_conn)
            log.info(f"LCM has {len(all_summaries)} summaries, {len(known_ids)} already known")

            # Filter to new ones
            new_summaries = [s for s in all_summaries if s["id"] not in known_ids]
            if not new_summaries:
                log.info("No new summaries to tag")
                self._log_run(econn, "tag", stats, start)
                return stats

            log.info(f"Processing {len(new_summaries)} new summaries")

            # Embed new summaries
            texts = [s["content"] for s in new_summaries]
            vectors = self.embed.embed_batch(texts)

            # Load current topics
            topics = self.get_topics_with_centroids(econn)

            # Tag each summary
            for summary, vec in zip(new_summaries, vectors):
                stats["processed"] += 1

                if vec is None:
                    log.warning(f"Skipping {summary['id']} — empty content")
                    stats["skipped"] += 1
                    continue

                best_topic, best_sim = self.find_best_topic(vec, topics)

                if best_topic is not None and best_sim >= self.similarity_threshold:
                    # Tag it
                    try:
                        econn.execute(
                            "INSERT OR IGNORE INTO topic_summaries (topic_id, summary_id, similarity) VALUES (?, ?, ?)",
                            (best_topic["id"], summary["id"], best_sim),
                        )
                        # Update centroid
                        new_centroid = self.update_centroid(best_topic["centroid"], vec)
                        econn.execute(
                            "UPDATE topics SET centroid = ?, summary_count = summary_count + 1, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                            (EmbedClient.vec_to_blob(new_centroid), best_topic["id"]),
                        )
                        # Update in-memory centroid for subsequent comparisons
                        best_topic["centroid"] = new_centroid
                        best_topic["summary_count"] += 1
                        stats["tagged"] += 1
                    except Exception as e:
                        log.error(f"Error tagging {summary['id']}: {e}")
                        stats["errors"] += 1
                else:
                    # Below threshold — stays as noise for discovery pass
                    stats["skipped"] += 1

            econn.commit()
            self._log_run(econn, "tag", stats, start)

        finally:
            lcm_conn.close()
            econn.close()

        stats["duration_ms"] = int((time.time() - start) * 1000)
        return stats

    def _log_run(self, econn: sqlite3.Connection, run_type: str, stats: dict, start: float):
        """Write a run entry to tagging_log."""
        duration_ms = int((time.time() - start) * 1000)
        econn.execute(
            "INSERT INTO tagging_log (run_type, summaries_processed, summaries_tagged, duration_ms) VALUES (?, ?, ?, ?)",
            (run_type, stats["processed"], stats["tagged"], duration_ms),
        )
        econn.commit()
