"""Epistemic Synthesis — Output Writer

Writes injection briefs to memory/topics/{label}.md for memorySearch indexing.
"""

import logging
import re
import sqlite3
from pathlib import Path
from typing import Optional

from . import config
from .schema import init_epistemic_db

logger = logging.getLogger("epistemic")


def sanitize_filename(label: str) -> str:
    """Convert topic label to safe filename."""
    # Lowercase, replace non-alphanumeric with hyphens, collapse multiples
    name = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    # Limit length
    if len(name) > 80:
        name = name[:80].rstrip("-")
    return name or "unnamed-topic"


class OutputWriter:
    """Writes synthesis artifacts to memory/topics/ directory."""

    def __init__(
        self,
        epistemic_db: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        self.epistemic_db = epistemic_db or config.EPISTEMIC_DB
        self.output_dir = output_dir or config.SYNTHESIS_OUTPUT_DIR

    def _get_latest_synthesis(self, topic_id: int, conn: sqlite3.Connection) -> Optional[dict]:
        """Get the latest synthesis for a topic."""
        row = conn.execute(
            """SELECT s.id, s.topic_id, s.version, s.injection_brief, s.claim_count, s.created_at,
                      t.label
               FROM syntheses s
               JOIN topics t ON s.topic_id = t.id
               WHERE s.topic_id = ?
               ORDER BY s.version DESC
               LIMIT 1""",
            (topic_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0], "topic_id": row[1], "version": row[2],
            "injection_brief": row[3], "claim_count": row[4],
            "created_at": row[5], "label": row[6],
        }

    def write_topic(self, topic_id: int) -> Optional[Path]:
        """Write injection brief for a topic. Returns file path or None."""
        conn = init_epistemic_db(self.epistemic_db)
        synth = self._get_latest_synthesis(topic_id, conn)
        conn.close()

        if not synth:
            logger.debug(f"Topic {topic_id}: no synthesis yet, skipping write")
            return None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build file content with metadata header
        filename = sanitize_filename(synth["label"] or f"topic-{topic_id}")
        filepath = self.output_dir / f"{filename}.md"

        content = (
            f"---\n"
            f"topic_id: {synth['topic_id']}\n"
            f"version: {synth['version']}\n"
            f"claim_count: {synth['claim_count']}\n"
            f"generated_at: {synth['created_at']}\n"
            f"---\n\n"
            f"# {synth['label'] or f'Topic {topic_id}'}\n\n"
            f"{synth['injection_brief']}\n"
        )

        filepath.write_text(content, encoding="utf-8")
        logger.info(f"Wrote {filepath} (v{synth['version']}, {synth['claim_count']} claims)")
        return filepath

    def write_all(self, topic_ids: Optional[list[int]] = None) -> list[Path]:
        """Write all topics. Returns list of paths written."""
        conn = init_epistemic_db(self.epistemic_db)
        if topic_ids is None:
            rows = conn.execute("SELECT id FROM topics ORDER BY id").fetchall()
            topic_ids = [r[0] for r in rows]
        conn.close()

        paths = []
        for tid in topic_ids:
            p = self.write_topic(tid)
            if p:
                paths.append(p)
        return paths
