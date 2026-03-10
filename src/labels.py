"""Epistemic Synthesis — Topic Labeling

LLM pass to generate human-readable labels for unlabeled topics.
Uses local Qwen3 via OpenAI-compatible chat completions API.
"""

import sqlite3
import time
import logging
from pathlib import Path
from typing import Optional

import requests

from . import config
from .schema import init_epistemic_db, open_lcm_readonly

log = logging.getLogger(__name__)


class LabelError(Exception):
    """Raised when labeling fails."""
    pass


class TopicLabeler:
    """Generates human-readable labels for topics using LLM."""

    def __init__(
        self,
        epistemic_db: Optional[Path] = None,
        lcm_db: Optional[Path] = None,
        llm_base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_timeout: Optional[int] = None,
    ):
        self.epistemic_path = epistemic_db or config.EPISTEMIC_DB
        self.lcm_path = lcm_db or config.LCM_DB
        self.llm_url = (llm_base_url or config.LLM_BASE_URL).rstrip("/") + "/v1/chat/completions"
        self.llm_model = llm_model or config.LLM_MODEL
        self.llm_timeout = llm_timeout or config.LLM_TIMEOUT

    def get_unlabeled_topics(self, econn: sqlite3.Connection) -> list[dict]:
        """Get topics with no label."""
        rows = econn.execute(
            "SELECT id, summary_count FROM topics WHERE label IS NULL"
        ).fetchall()
        return [{"id": r[0], "summary_count": r[1]} for r in rows]

    def get_topic_summary_texts(
        self, econn: sqlite3.Connection, lcm_conn: sqlite3.Connection, topic_id: int, limit: int = 5
    ) -> list[str]:
        """Get summary texts for a topic (for LLM context)."""
        rows = econn.execute(
            "SELECT summary_id FROM topic_summaries WHERE topic_id = ? AND orphaned = 0 ORDER BY similarity DESC LIMIT ?",
            (topic_id, limit),
        ).fetchall()
        summary_ids = [r[0] for r in rows]

        if not summary_ids:
            return []

        # Fetch content from LCM
        from .tagger import Tagger
        t = Tagger(epistemic_db=self.epistemic_path, lcm_db=self.lcm_path)
        all_summaries = t.get_lcm_summaries(lcm_conn)
        id_to_content = {s["id"]: s["content"] for s in all_summaries}

        return [id_to_content[sid] for sid in summary_ids if sid in id_to_content]

    def generate_label(self, summary_texts: list[str]) -> str:
        """Ask LLM to generate a short topic label from sample summaries."""
        # Truncate each summary to ~500 chars to stay within Qwen context window
        truncated = [t[:500] for t in summary_texts[:5]]
        samples = "\n---\n".join(truncated)
        prompt = f"""These are summaries from conversations that cluster together as a single topic.
Generate a short label (2-5 words) that captures what this topic is about.
Return ONLY the label, nothing else.

Summaries:
{samples}"""

        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 20,
        }

        try:
            resp = requests.post(self.llm_url, json=payload, timeout=self.llm_timeout)
            resp.raise_for_status()
            data = resp.json()
            label = data["choices"][0]["message"]["content"].strip().strip('"\'')
            # Truncate if somehow too long
            if len(label) > 100:
                label = label[:100]
            return label
        except Exception as e:
            raise LabelError(f"LLM labeling failed: {e}")

    def run(self) -> dict:
        """Label all unlabeled topics."""
        start = time.time()
        stats = {
            "unlabeled": 0,
            "labeled": 0,
            "errors": 0,
            "duration_ms": 0,
        }

        econn = init_epistemic_db(self.epistemic_path)
        lcm_conn = open_lcm_readonly(self.lcm_path)

        try:
            unlabeled = self.get_unlabeled_topics(econn)
            stats["unlabeled"] = len(unlabeled)

            if not unlabeled:
                log.info("No unlabeled topics")
                return stats

            for topic in unlabeled:
                texts = self.get_topic_summary_texts(econn, lcm_conn, topic["id"])
                if not texts:
                    log.warning(f"Topic {topic['id']} has no readable summaries, skipping")
                    continue

                try:
                    label = self.generate_label(texts)
                    econn.execute(
                        "UPDATE topics SET label = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                        (label, topic["id"]),
                    )
                    econn.commit()
                    stats["labeled"] += 1
                    log.info(f"Topic {topic['id']} labeled: '{label}'")
                except LabelError as e:
                    log.error(f"Failed to label topic {topic['id']}: {e}")
                    stats["errors"] += 1

        finally:
            lcm_conn.close()
            econn.close()

        stats["duration_ms"] = int((time.time() - start) * 1000)
        return stats
