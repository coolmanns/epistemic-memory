"""Epistemic Synthesis — Claim Deduplicator

Embeds claims and merges near-duplicates within topics using cosine similarity.
Cross-topic duplicates are flagged, not auto-merged.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .embed import EmbedClient
from .schema import init_epistemic_db

logger = logging.getLogger("epistemic")


class ClaimDeduplicator:
    """Deduplicates claims within a topic using embedding similarity."""

    def __init__(
        self,
        epistemic_db: Optional[Path] = None,
        dedup_threshold: Optional[float] = None,
        embed_client: Optional[EmbedClient] = None,
    ):
        self.epistemic_db = epistemic_db or config.EPISTEMIC_DB
        self.threshold = dedup_threshold if dedup_threshold is not None else config.CLAIM_DEDUP_THRESHOLD
        self.embed = embed_client or EmbedClient()

    def _get_claims(self, topic_id: int, conn: sqlite3.Connection) -> list[dict]:
        """Get active claims for a topic."""
        rows = conn.execute(
            """SELECT id, text, source_count, embedding
               FROM claims
               WHERE topic_id = ? AND status IN ('active', 'verified')
               ORDER BY id""",
            (topic_id,)
        ).fetchall()
        return [{"id": r[0], "text": r[1], "source_count": r[2], "embedding": r[3]} for r in rows]

    def _embed_claims(self, claims: list[dict]) -> list[dict]:
        """Embed claims that don't have embeddings yet. Sets _vector on all claims."""
        to_embed = [c for c in claims if c["embedding"] is None]

        vectors = {}
        if to_embed:
            texts = [c["text"] for c in to_embed]
            results = self.embed.embed_batch(texts)
            for c, vec in zip(to_embed, results):
                vectors[c["id"]] = vec

        for c in claims:
            if c["id"] in vectors:
                c["_vector"] = vectors[c["id"]]
                c["_new_blob"] = vectors[c["id"]].astype(np.float32).tobytes()
            else:
                c["_vector"] = np.frombuffer(c["embedding"], dtype=np.float32)
                c["_new_blob"] = None
        return claims

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _merge_claims(self, keeper_id: int, absorbed_id: int, conn: sqlite3.Connection):
        """Merge absorbed claim into keeper. Moves sources, updates counts."""
        # Move claim_sources from absorbed to keeper (skip if duplicate summary_id)
        sources = conn.execute(
            "SELECT summary_id, excerpt FROM claim_sources WHERE claim_id = ?",
            (absorbed_id,)
        ).fetchall()

        for summary_id, excerpt in sources:
            try:
                conn.execute(
                    "INSERT INTO claim_sources (claim_id, summary_id, excerpt) VALUES (?, ?, ?)",
                    (keeper_id, summary_id, excerpt)
                )
            except sqlite3.IntegrityError:
                pass  # already exists

        # Update keeper's source_count
        total = conn.execute(
            "SELECT COUNT(*) FROM claim_sources WHERE claim_id = ?",
            (keeper_id,)
        ).fetchone()[0]
        conn.execute(
            "UPDATE claims SET source_count = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
            (total, keeper_id)
        )

        # Mark absorbed as superseded
        conn.execute(
            "UPDATE claims SET status = 'superseded', updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
            (absorbed_id,)
        )

    def dedup_topic(self, topic_id: int) -> dict:
        """Deduplicate claims within a single topic. Returns stats."""
        start = time.time()
        conn = init_epistemic_db(self.epistemic_db)
        stats = {"topic_id": topic_id, "total_claims": 0, "merged": 0,
                 "cross_topic_flags": 0, "errors": 0}

        claims = self._get_claims(topic_id, conn)
        stats["total_claims"] = len(claims)

        if len(claims) < 2:
            conn.close()
            return stats

        try:
            claims = self._embed_claims(claims)
        except Exception as e:
            logger.error(f"Embedding failed for topic {topic_id}: {e}")
            conn.close()
            stats["errors"] = 1
            return stats

        # Store new embeddings
        for c in claims:
            if c.get("_new_blob"):
                conn.execute(
                    "UPDATE claims SET embedding = ? WHERE id = ?",
                    (c["_new_blob"], c["id"])
                )

        # Build similarity matrix and find merges
        merged_ids = set()
        for i in range(len(claims)):
            if claims[i]["id"] in merged_ids:
                continue
            for j in range(i + 1, len(claims)):
                if claims[j]["id"] in merged_ids:
                    continue
                sim = self._cosine_sim(claims[i]["_vector"], claims[j]["_vector"])
                if sim >= self.threshold:
                    # Keep the older one (lower id), absorb the newer
                    keeper = claims[i]
                    absorbed = claims[j]
                    self._merge_claims(keeper["id"], absorbed["id"], conn)
                    merged_ids.add(absorbed["id"])
                    stats["merged"] += 1
                    logger.debug(
                        f"Merged claim {absorbed['id']} into {keeper['id']} (sim={sim:.3f})"
                    )

        conn.commit()

        # Log run
        duration = int((time.time() - start) * 1000)
        conn.execute(
            """INSERT INTO synthesis_runs (topic_id, phase, stats, finished_at, duration_ms)
               VALUES (?, 'dedup', ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?)""",
            (topic_id, json.dumps(stats), duration),
        )
        conn.commit()
        stats["duration_ms"] = duration

        conn.close()
        logger.info(
            f"Topic {topic_id}: dedup {stats['total_claims']} claims, merged {stats['merged']}"
        )
        return stats

    def run(self, topic_ids: Optional[list[int]] = None) -> list[dict]:
        """Deduplicate all topics (or specified list)."""
        conn = init_epistemic_db(self.epistemic_db)
        if topic_ids is None:
            rows = conn.execute("SELECT id FROM topics ORDER BY id").fetchall()
            topic_ids = [r[0] for r in rows]
        conn.close()

        return [self.dedup_topic(tid) for tid in topic_ids]
