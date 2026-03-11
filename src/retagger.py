"""Epistemic Synthesis — Re-tagging Pipeline

Handles two scenarios:
  1. New topic seeded → full scan of ALL LCM summaries against that topic
  2. Neighbor-aware incremental → when new summaries arrive, check neighbor topics

The standard Tagger only processes summaries it hasn't seen before.
Retagger re-evaluates already-processed summaries against specific topics.
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


class Retagger:
    """Re-tags LCM summaries against specific or newly-seeded topics."""

    def __init__(
        self,
        epistemic_db: Optional[Path] = None,
        lcm_db: Optional[Path] = None,
        embed_client: Optional[EmbedClient] = None,
        similarity_threshold: Optional[float] = None,
        neighbor_sim_threshold: float = 0.4,
    ):
        self.epistemic_path = epistemic_db or config.EPISTEMIC_DB
        self.lcm_path = lcm_db or config.LCM_DB
        self.embed = embed_client or EmbedClient()
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else config.SIMILARITY_THRESHOLD
        )
        # How similar two topic centroids must be to count as "neighbors"
        self.neighbor_sim_threshold = neighbor_sim_threshold

    def retag_topic(self, topic_id: int) -> dict:
        """Full scan: evaluate ALL LCM summaries against one topic.

        Used when a new topic is seeded and needs historical coverage.
        Returns stats dict.
        """
        start = time.time()
        stats = {"topic_id": topic_id, "scanned": 0, "tagged": 0, "already_tagged": 0, "below_threshold": 0, "errors": 0}

        econn = init_epistemic_db(self.epistemic_path)
        lcm_conn = open_lcm_readonly(self.lcm_path)

        try:
            # Load the target topic
            row = econn.execute(
                "SELECT id, label, centroid FROM topics WHERE id = ?", (topic_id,)
            ).fetchone()
            if row is None:
                log.error(f"Topic {topic_id} not found")
                stats["errors"] = 1
                return stats

            topic_label = row[1]
            topic_centroid = EmbedClient.blob_to_vec(row[2])
            log.info(f"Re-tagging topic {topic_id} ({topic_label}): full scan")

            # Get already-tagged summary IDs for this topic
            already = set(
                r[0]
                for r in econn.execute(
                    "SELECT summary_id FROM topic_summaries WHERE topic_id = ?",
                    (topic_id,),
                ).fetchall()
            )

            # Get ALL LCM summaries (depth 0 preferred, all depths included)
            summaries = self._get_all_summaries(lcm_conn)
            log.info(f"Scanning {len(summaries)} summaries against '{topic_label}'")

            # Batch embed
            texts = [s["content"] for s in summaries]
            vectors = self.embed.embed_batch(texts, batch_size=8)

            for summary, vec in zip(summaries, vectors):
                stats["scanned"] += 1

                if summary["id"] in already:
                    stats["already_tagged"] += 1
                    continue

                if vec is None:
                    continue

                sim = EmbedClient.cosine_similarity(vec, topic_centroid)

                if sim >= self.similarity_threshold:
                    try:
                        econn.execute(
                            "INSERT OR IGNORE INTO topic_summaries (topic_id, summary_id, similarity) VALUES (?, ?, ?)",
                            (topic_id, summary["id"], sim),
                        )
                        stats["tagged"] += 1
                    except Exception as e:
                        log.error(f"Error tagging {summary['id']} to topic {topic_id}: {e}")
                        stats["errors"] += 1
                else:
                    stats["below_threshold"] += 1

            econn.commit()

            # Update topic summary count
            count = econn.execute(
                "SELECT COUNT(*) FROM topic_summaries WHERE topic_id = ?",
                (topic_id,),
            ).fetchone()[0]
            econn.execute(
                "UPDATE topics SET summary_count = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                (count, topic_id),
            )
            econn.commit()

            self._log_run(econn, "retag", stats, start, topic_id=topic_id)

        finally:
            lcm_conn.close()
            econn.close()

        stats["duration_ms"] = int((time.time() - start) * 1000)
        log.info(
            f"Re-tag topic {topic_id} ({topic_label}): "
            f"{stats['tagged']} new tags, {stats['already_tagged']} already tagged, "
            f"{stats['below_threshold']} below threshold ({stats['duration_ms']}ms)"
        )
        return stats

    def retag_new_topics(self) -> list[dict]:
        """Find all topics with 0 tagged summaries and run full scan on each.

        This is the automatic catch-up: any freshly seeded topic gets historical coverage.
        """
        econn = init_epistemic_db(self.epistemic_path)
        try:
            rows = econn.execute(
                "SELECT t.id, t.label FROM topics t "
                "LEFT JOIN topic_summaries ts ON t.id = ts.topic_id "
                "GROUP BY t.id "
                "HAVING COUNT(ts.summary_id) = 0"
            ).fetchall()
        finally:
            econn.close()

        if not rows:
            log.info("No topics need re-tagging (all have summaries)")
            return []

        log.info(f"Found {len(rows)} topics with 0 summaries — running re-tag")
        results = []
        for row in rows:
            stats = self.retag_topic(row[0])
            results.append(stats)

        return results

    def retag_neighbors(self, topic_id: int) -> dict:
        """Neighbor-aware re-tag: find topics similar to topic_id,
        pull their tagged summaries, evaluate against topic_id.

        Cheaper than full scan — only checks summaries already in the neighborhood.
        """
        start = time.time()
        stats = {"topic_id": topic_id, "neighbors_checked": 0, "candidates": 0, "tagged": 0, "already_tagged": 0, "below_threshold": 0, "errors": 0}

        econn = init_epistemic_db(self.epistemic_path)
        lcm_conn = open_lcm_readonly(self.lcm_path)

        try:
            # Load target topic
            row = econn.execute(
                "SELECT id, label, centroid FROM topics WHERE id = ?", (topic_id,)
            ).fetchone()
            if row is None:
                log.error(f"Topic {topic_id} not found")
                return stats

            topic_label = row[1]
            target_centroid = EmbedClient.blob_to_vec(row[2])

            # Already tagged to this topic
            already = set(
                r[0]
                for r in econn.execute(
                    "SELECT summary_id FROM topic_summaries WHERE topic_id = ?",
                    (topic_id,),
                ).fetchall()
            )

            # Find neighbor topics
            all_topics = econn.execute(
                "SELECT id, label, centroid FROM topics WHERE id != ?", (topic_id,)
            ).fetchall()

            neighbor_sids = set()
            for t in all_topics:
                t_centroid = EmbedClient.blob_to_vec(t[2])
                sim = EmbedClient.cosine_similarity(target_centroid, t_centroid)
                if sim >= self.neighbor_sim_threshold:
                    stats["neighbors_checked"] += 1
                    sids = [
                        r[0]
                        for r in econn.execute(
                            "SELECT summary_id FROM topic_summaries WHERE topic_id = ?",
                            (t[0],),
                        ).fetchall()
                    ]
                    neighbor_sids.update(sids)

            # Remove already-tagged
            candidates = neighbor_sids - already
            stats["candidates"] = len(candidates)

            if not candidates:
                log.info(f"No neighbor candidates for topic {topic_id} ({topic_label})")
                return stats

            log.info(
                f"Re-tag neighbors for topic {topic_id} ({topic_label}): "
                f"{stats['neighbors_checked']} neighbors, {len(candidates)} candidates"
            )

            # Fetch and embed candidates
            for sid in candidates:
                content_row = lcm_conn.execute(
                    "SELECT content FROM summaries WHERE summary_id = ?", (sid,)
                ).fetchone()
                if not content_row:
                    continue

                vec = self.embed.embed_one(content_row["content"])
                sim = EmbedClient.cosine_similarity(vec, target_centroid)

                if sim >= self.similarity_threshold:
                    try:
                        econn.execute(
                            "INSERT OR IGNORE INTO topic_summaries (topic_id, summary_id, similarity) VALUES (?, ?, ?)",
                            (topic_id, sid, sim),
                        )
                        stats["tagged"] += 1
                    except Exception as e:
                        log.error(f"Error tagging {sid}: {e}")
                        stats["errors"] += 1
                else:
                    stats["below_threshold"] += 1

            econn.commit()

            # Update summary count
            count = econn.execute(
                "SELECT COUNT(*) FROM topic_summaries WHERE topic_id = ?",
                (topic_id,),
            ).fetchone()[0]
            econn.execute(
                "UPDATE topics SET summary_count = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                (count, topic_id),
            )
            econn.commit()

            self._log_run(econn, "retag-neighbor", stats, start, topic_id=topic_id)

        finally:
            lcm_conn.close()
            econn.close()

        stats["duration_ms"] = int((time.time() - start) * 1000)
        log.info(
            f"Re-tag neighbors for topic {topic_id} ({topic_label}): "
            f"{stats['tagged']} new tags from {stats['candidates']} candidates ({stats['duration_ms']}ms)"
        )
        return stats

    def _get_all_summaries(self, lcm_conn: sqlite3.Connection) -> list[dict]:
        """Get all summaries from LCM, preferring depth 0 but including all."""
        rows = lcm_conn.execute(
            "SELECT summary_id, content, depth FROM summaries ORDER BY depth ASC"
        ).fetchall()
        return [{"id": r["summary_id"], "content": r["content"], "depth": r["depth"]} for r in rows]

    def _log_run(self, econn: sqlite3.Connection, run_type: str, stats: dict, start: float, topic_id: int = None):
        """Write a run entry to tagging_log."""
        duration_ms = int((time.time() - start) * 1000)
        econn.execute(
            "INSERT INTO tagging_log (run_type, summaries_processed, summaries_tagged, duration_ms) VALUES (?, ?, ?, ?)",
            (run_type, stats.get("scanned", stats.get("candidates", 0)), stats["tagged"], duration_ms),
        )
        econn.commit()
