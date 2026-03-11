"""Epistemic Synthesis — Claim Extractor

Extracts atomic claims from topic summaries using Claude Sonnet.
Each claim is a single assertion with type, source excerpt, and optional contradiction flags.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import requests  # kept for fallback / test mocking

from .llm import call_llm_json

from . import config
from .schema import init_epistemic_db, open_lcm_readonly

logger = logging.getLogger("epistemic")

EXTRACTION_PROMPT = """Extract atomic claims from the following summaries about the topic "{topic_label}".

CRITICAL: Only extract claims that are SPECIFICALLY about "{topic_label}". These summaries may contain information about multiple topics discussed in the same conversation. Ignore claims about other entities, projects, or systems — even if they appear prominently in the source material. A claim belongs to this topic ONLY if it would make sense filed under "{topic_label}" and nowhere else.

An atomic claim is a single assertion that represents one piece of learned knowledge.

For each claim:
1. State the claim in ONE sentence (max 500 characters).
2. Classify as: factual (verifiable fact), interpretive (inference or pattern), or analogical (comparison/metaphor).
3. Quote the specific source excerpt that supports this claim (verbatim, 1-2 sentences).
4. Provide the summary_id where you found it.
5. Set initial confidence: HIGH (multiple sources, directly stated), MED (single source, clearly stated), LOW (implied or inferred).
6. If a claim contradicts another claim you've extracted, note both claim numbers.

DO NOT:
- Generate claims beyond what the sources support
- Combine multiple assertions into one claim
- Paraphrase source excerpts (quote verbatim)
- Extract more than {max_claims} claims total
- Extract claims about other topics that happen to appear in the same summaries

Respond with valid JSON:
{{
  "claims": [
    {{
      "text": "claim text here",
      "type": "factual|interpretive|analogical",
      "confidence": "HIGH|MED|LOW",
      "source_excerpt": "verbatim quote from summary",
      "summary_id": "sum_xxx",
      "contradicts": []
    }}
  ]
}}

SUMMARIES:
{summaries}"""


class ClaimExtractor:
    """Extracts atomic claims from LCM summaries per topic."""

    def __init__(
        self,
        epistemic_db: Optional[Path] = None,
        lcm_db: Optional[Path] = None,
        llm_base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_timeout: Optional[int] = None,
        max_claims: Optional[int] = None,
    ):
        self.epistemic_db = epistemic_db or config.EPISTEMIC_DB
        self.lcm_db = lcm_db or config.LCM_DB
        self.llm_base_url = llm_base_url or config.LLM_BASE_URL
        self.llm_model = llm_model or config.LLM_MODEL
        self.llm_timeout = llm_timeout or config.LLM_TIMEOUT
        self.max_claims = max_claims or config.MAX_CLAIMS_PER_EXTRACTION

    def _get_topic_summaries(self, topic_id: int, conn_e: sqlite3.Connection,
                              conn_lcm: sqlite3.Connection) -> list[dict]:
        """Fetch summaries tagged to a topic, with content from lcm.db."""
        tagged = conn_e.execute(
            "SELECT summary_id FROM topic_summaries WHERE topic_id = ? AND orphaned = 0",
            (topic_id,)
        ).fetchall()
        summary_ids = [r[0] for r in tagged]
        if not summary_ids:
            return []

        results = []
        for sid in summary_ids:
            row = conn_lcm.execute(
                "SELECT summary_id, content FROM summaries WHERE summary_id = ?",
                (sid,)
            ).fetchone()
            if row:
                results.append({"summary_id": row["summary_id"], "content": row["content"]})
        return results

    def _get_already_extracted(self, topic_id: int, conn: sqlite3.Connection) -> set[str]:
        """Get summary_ids that already have claims extracted for this topic."""
        rows = conn.execute(
            """SELECT DISTINCT cs.summary_id
               FROM claim_sources cs
               JOIN claims c ON cs.claim_id = c.id
               WHERE c.topic_id = ?""",
            (topic_id,)
        ).fetchall()
        return {r[0] for r in rows}

    def _call_llm(self, prompt: str) -> dict:
        """Call Sonnet for claim extraction. Returns parsed JSON."""
        return call_llm_json(prompt, max_tokens=4096)

    def _validate_claim(self, claim: dict) -> Optional[str]:
        """Validate a single extracted claim. Returns error string or None."""
        if not claim.get("text"):
            return "empty claim text"
        if len(claim["text"]) > config.MAX_CLAIM_LENGTH:
            return f"claim text too long ({len(claim['text'])} > {config.MAX_CLAIM_LENGTH})"
        if claim.get("type") not in ("factual", "interpretive", "analogical"):
            return f"invalid claim type: {claim.get('type')}"
        if claim.get("confidence") not in ("HIGH", "MED", "LOW"):
            return f"invalid confidence: {claim.get('confidence')}"
        if not claim.get("source_excerpt"):
            return "missing source excerpt"
        if not claim.get("summary_id"):
            return "missing summary_id"
        return None

    def extract_topic(self, topic_id: int) -> dict:
        """Extract claims for a single topic. Returns stats dict."""
        start = time.time()
        conn_e = init_epistemic_db(self.epistemic_db)
        stats = {"topic_id": topic_id, "extracted": 0, "skipped": 0,
                 "errors": 0, "already_processed": 0}

        # Get topic label
        row = conn_e.execute("SELECT label FROM topics WHERE id = ?", (topic_id,)).fetchone()
        if not row:
            logger.warning(f"Topic {topic_id} not found")
            conn_e.close()
            stats["errors"] = 1
            return stats
        topic_label = row[0] or f"Topic {topic_id}"

        try:
            conn_lcm = open_lcm_readonly(self.lcm_db)
        except FileNotFoundError:
            logger.error(f"lcm.db not found at {self.lcm_db}")
            conn_e.close()
            stats["errors"] = 1
            return stats

        # Get summaries
        summaries = self._get_topic_summaries(topic_id, conn_e, conn_lcm)
        if not summaries:
            logger.info(f"Topic {topic_id} ({topic_label}): no summaries found")
            conn_lcm.close()
            conn_e.close()
            return stats

        # Filter already-extracted
        already = self._get_already_extracted(topic_id, conn_e)
        new_summaries = [s for s in summaries if s["summary_id"] not in already]
        stats["already_processed"] = len(summaries) - len(new_summaries)

        if not new_summaries:
            logger.info(f"Topic {topic_id} ({topic_label}): all {len(summaries)} summaries already extracted")
            conn_lcm.close()
            conn_e.close()
            return stats

        # Format summaries for prompt — full content, Sonnet handles 200K context
        summary_text = "\n\n".join(
            f"[{s['summary_id']}]\n{s['content']}"
            for s in new_summaries
        )

        prompt = EXTRACTION_PROMPT.format(
            topic_label=topic_label,
            max_claims=self.max_claims,
            summaries=summary_text,
        )

        try:
            result = self._call_llm(prompt)
        except (requests.RequestException, ConnectionError, json.JSONDecodeError, KeyError, Exception) as e:
            logger.error(f"LLM call failed for topic {topic_id}: {e}")
            conn_lcm.close()
            conn_e.close()
            stats["errors"] = 1
            return stats

        claims = result.get("claims", [])
        if len(claims) > self.max_claims:
            logger.warning(
                f"Topic {topic_id}: {len(claims)} claims total, capping at {self.max_claims}"
            )
            claims = claims[:self.max_claims]

        # Insert valid claims
        for claim in claims:
            err = self._validate_claim(claim)
            if err:
                logger.warning(f"Skipping invalid claim: {err}")
                stats["skipped"] += 1
                continue

            try:
                cursor = conn_e.execute(
                    """INSERT INTO claims (topic_id, text, claim_type, confidence, status)
                       VALUES (?, ?, ?, ?, 'active')""",
                    (topic_id, claim["text"], claim["type"], claim["confidence"]),
                )
                claim_id = cursor.lastrowid
                conn_e.execute(
                    """INSERT OR IGNORE INTO claim_sources (claim_id, summary_id, excerpt)
                       VALUES (?, ?, ?)""",
                    (claim_id, claim["summary_id"], claim["source_excerpt"]),
                )
                stats["extracted"] += 1
            except sqlite3.Error as e:
                logger.error(f"DB insert failed for claim: {e}")
                stats["errors"] += 1

        conn_e.commit()

        # Log run
        duration = int((time.time() - start) * 1000)
        conn_e.execute(
            """INSERT INTO synthesis_runs (topic_id, phase, stats, finished_at, duration_ms)
               VALUES (?, 'extract', ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?)""",
            (topic_id, json.dumps(stats), duration),
        )
        conn_e.commit()
        stats["duration_ms"] = duration

        conn_lcm.close()
        conn_e.close()
        logger.info(
            f"Topic {topic_id} ({topic_label}): extracted {stats['extracted']}, "
            f"skipped {stats['skipped']}, errors {stats['errors']}"
        )
        return stats

    def run(self, topic_ids: Optional[list[int]] = None) -> list[dict]:
        """Extract claims for all topics (or specified list). Returns list of stats."""
        conn = init_epistemic_db(self.epistemic_db)
        if topic_ids is None:
            rows = conn.execute("SELECT id FROM topics ORDER BY id").fetchall()
            topic_ids = [r[0] for r in rows]
        conn.close()

        all_stats = []
        for tid in topic_ids:
            stats = self.extract_topic(tid)
            all_stats.append(stats)
        return all_stats
