"""Epistemic Synthesis — Topic Discovery

Daily HDBSCAN pass over untagged summaries to discover new topics.
Checks new clusters against existing topics to prevent duplicates.
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

try:
    import hdbscan
except ImportError:
    hdbscan = None


class DiscoveryError(Exception):
    """Raised when discovery fails."""
    pass


class Discovery:
    """Discovers new topics from untagged summaries using HDBSCAN."""

    def __init__(
        self,
        epistemic_db: Optional[Path] = None,
        lcm_db: Optional[Path] = None,
        embed_client: Optional[EmbedClient] = None,
        min_discovery_batch: Optional[int] = None,
        min_cluster_size: Optional[int] = None,
        merge_threshold: Optional[float] = None,
        max_topics_per_run: Optional[int] = None,
    ):
        self.epistemic_path = epistemic_db or config.EPISTEMIC_DB
        self.lcm_path = lcm_db or config.LCM_DB
        self.embed = embed_client or EmbedClient()
        self.min_discovery_batch = min_discovery_batch if min_discovery_batch is not None else config.MIN_DISCOVERY_BATCH
        self.min_cluster_size = min_cluster_size if min_cluster_size is not None else config.HDBSCAN_MIN_CLUSTER_SIZE
        self.merge_threshold = merge_threshold if merge_threshold is not None else config.MERGE_THRESHOLD
        self.max_topics_per_run = max_topics_per_run if max_topics_per_run is not None else config.MAX_TOPICS_PER_RUN

    def get_untagged_summaries(
        self, econn: sqlite3.Connection, lcm_conn: sqlite3.Connection
    ) -> list[dict]:
        """Get LCM summaries that aren't in any topic yet."""
        # Get all tagged summary IDs
        tagged = {
            r[0]
            for r in econn.execute("SELECT DISTINCT summary_id FROM topic_summaries").fetchall()
        }

        # Get all LCM summaries
        from .tagger import Tagger
        t = Tagger(epistemic_db=self.epistemic_path, lcm_db=self.lcm_path, embed_client=self.embed)
        all_summaries = t.get_lcm_summaries(lcm_conn)

        return [s for s in all_summaries if s["id"] not in tagged]

    def get_existing_topics(self, econn: sqlite3.Connection) -> list[dict]:
        """Load existing topics with centroids."""
        rows = econn.execute("SELECT id, label, centroid FROM topics").fetchall()
        return [
            {"id": r[0], "label": r[1], "centroid": EmbedClient.blob_to_vec(r[2])}
            for r in rows
        ]

    def find_matching_topic(
        self, centroid: np.ndarray, existing_topics: list[dict]
    ) -> Optional[dict]:
        """Check if a centroid matches an existing topic above merge threshold."""
        for topic in existing_topics:
            sim = EmbedClient.cosine_similarity(centroid, topic["centroid"])
            if sim >= self.merge_threshold:
                return topic
        return None

    def cluster_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Run HDBSCAN on vectors. Returns cluster labels (-1 = noise)."""
        if hdbscan is None:
            raise DiscoveryError("hdbscan package not installed. Install with: pip install hdbscan")

        # L2 normalize for euclidean distance (equivalent to cosine on normalized vecs)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = vectors / norms

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(normalized)
        return labels

    def compute_centroid(self, vectors: list[np.ndarray]) -> np.ndarray:
        """Compute L2-normalized centroid from a set of vectors."""
        centroid = np.mean(vectors, axis=0).astype(np.float32)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid

    def run(self) -> dict:
        """Execute one discovery pass.

        Returns dict with run stats.
        """
        start = time.time()
        stats = {
            "untagged_count": 0,
            "clusters_found": 0,
            "new_topics_created": 0,
            "merged_into_existing": 0,
            "rejected_max_cap": 0,
            "still_noise": 0,
            "duration_ms": 0,
        }

        econn = init_epistemic_db(self.epistemic_path)
        lcm_conn = open_lcm_readonly(self.lcm_path)

        try:
            # Get untagged summaries
            untagged = self.get_untagged_summaries(econn, lcm_conn)
            stats["untagged_count"] = len(untagged)

            if len(untagged) < self.min_discovery_batch:
                log.info(
                    f"Only {len(untagged)} untagged summaries, need {self.min_discovery_batch}. Skipping."
                )
                self._log_run(econn, stats, start)
                return stats

            log.info(f"Running discovery on {len(untagged)} untagged summaries")

            # Embed all untagged
            texts = [s["content"] for s in untagged]
            vectors = self.embed.embed_batch(texts)

            # Filter out None (empty content)
            valid = [(s, v) for s, v in zip(untagged, vectors) if v is not None]
            if len(valid) < self.min_discovery_batch:
                log.info(f"Only {len(valid)} valid embeddings after filtering. Skipping.")
                self._log_run(econn, stats, start)
                return stats

            valid_summaries, valid_vectors = zip(*valid)
            matrix = np.stack(valid_vectors)

            # Cluster
            labels = self.cluster_vectors(matrix)
            unique_labels = set(labels)
            unique_labels.discard(-1)
            stats["clusters_found"] = len(unique_labels)
            stats["still_noise"] = int(np.sum(labels == -1))

            log.info(f"Found {len(unique_labels)} clusters, {stats['still_noise']} noise points")

            # Sanity check
            if len(unique_labels) > self.max_topics_per_run:
                log.warning(
                    f"Too many clusters ({len(unique_labels)} > {self.max_topics_per_run}). "
                    f"Rejecting all — likely bad HDBSCAN params."
                )
                stats["rejected_max_cap"] = len(unique_labels)
                self._log_run(econn, stats, start)
                return stats

            # Process each cluster
            existing_topics = self.get_existing_topics(econn)
            new_topics_created = 0

            for cluster_id in sorted(unique_labels):
                member_mask = labels == cluster_id
                member_indices = np.where(member_mask)[0]
                member_vectors = [valid_vectors[i] for i in member_indices]
                member_summaries = [valid_summaries[i] for i in member_indices]

                centroid = self.compute_centroid(member_vectors)

                # Check against existing topics
                match = self.find_matching_topic(centroid, existing_topics)

                if match is not None:
                    # Merge into existing topic
                    for summary in member_summaries:
                        sim = EmbedClient.cosine_similarity(
                            self.embed.embed_one(summary["content"])
                            if hasattr(summary["content"], "strip")
                            else member_vectors[member_summaries.index(summary)],
                            match["centroid"],
                        )
                        econn.execute(
                            "INSERT OR IGNORE INTO topic_summaries (topic_id, summary_id, similarity) VALUES (?, ?, ?)",
                            (match["id"], summary["id"], float(sim)),
                        )
                    econn.execute(
                        "UPDATE topics SET summary_count = summary_count + ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                        (len(member_summaries), match["id"]),
                    )
                    stats["merged_into_existing"] += 1
                    log.info(f"Merged {len(member_summaries)} summaries into existing topic '{match['label']}' (id={match['id']})")

                else:
                    # Create new topic
                    cursor = econn.execute(
                        "INSERT INTO topics (centroid, summary_count, created_at, updated_at) VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
                        (EmbedClient.vec_to_blob(centroid), len(member_summaries)),
                    )
                    topic_id = cursor.lastrowid

                    for i, summary in enumerate(member_summaries):
                        sim = EmbedClient.cosine_similarity(member_vectors[i], centroid)
                        econn.execute(
                            "INSERT OR IGNORE INTO topic_summaries (topic_id, summary_id, similarity) VALUES (?, ?, ?)",
                            (topic_id, summary["id"], float(sim)),
                        )

                    new_topics_created += 1
                    # Add to existing_topics so subsequent clusters check against it
                    existing_topics.append({
                        "id": topic_id,
                        "label": None,
                        "centroid": centroid,
                    })
                    log.info(f"Created new topic (id={topic_id}) with {len(member_summaries)} summaries")

            stats["new_topics_created"] = new_topics_created
            econn.commit()
            self._log_run(econn, stats, start)

        finally:
            lcm_conn.close()
            econn.close()

        stats["duration_ms"] = int((time.time() - start) * 1000)
        return stats

    def _log_run(self, econn: sqlite3.Connection, stats: dict, start: float):
        """Write a run entry to tagging_log."""
        duration_ms = int((time.time() - start) * 1000)
        econn.execute(
            "INSERT INTO tagging_log (run_type, summaries_processed, summaries_tagged, new_topics_created, duration_ms) VALUES (?, ?, ?, ?, ?)",
            ("discovery", stats["untagged_count"], stats.get("merged_into_existing", 0), stats["new_topics_created"], duration_ms),
        )
        econn.commit()
