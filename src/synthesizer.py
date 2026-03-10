"""Epistemic Synthesis — Synthesis Generator

Generates canonical synthesis documents and injection briefs from verified claims.
Uses epistemic register (no first-person). Three confidence tiers + tensions section.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import requests  # kept for test mocking

from .llm import call_llm

from . import config
from .schema import init_epistemic_db

logger = logging.getLogger("epistemic")

SYNTHESIS_PROMPT = """Given the following verified claims about "{topic_label}":

{claims_grouped}

Generate a canonical synthesis document using epistemic register (no first-person "I believe" or "I think").

Structure:
1. **Established Patterns** — Claims with HIGH confidence. Well-supported across multiple sources.
2. **Emerging Understanding** — Claims with MED confidence. Supported but not yet established.
3. **Active Tensions** — Pairs or groups of contradictory claims, presented without resolution.
4. **Provisional Observations** — Claims with LOW confidence or analogical type. Explicitly marked as provisional.
5. **Human-Stated Positions** — Claims attributed to the human's explicitly stated views.
6. **System-Derived Inferences** — Claims the agent has inferred from behavioral patterns.

Include claim IDs inline: [C-xxx]. Do not add claims that aren't in the input set.
Do not resolve tensions — present both sides.
Target length: 800-2000 tokens."""

BRIEF_PROMPT = """Given this canonical synthesis for "{topic_label}":

{canonical}

Generate a compressed injection brief (200-400 tokens) for runtime context.
Include only: HIGH and MED confidence claims, key tensions.
Do not include: LOW confidence claims, full provenance, system-derived inferences.
Write in epistemic register. No first-person voice."""


class Synthesizer:
    """Generates synthesis documents from verified claims."""

    def __init__(
        self,
        epistemic_db: Optional[Path] = None,
        llm_base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        timeout: int = 120,
        min_claims: Optional[int] = None,
    ):
        self.epistemic_db = epistemic_db or config.EPISTEMIC_DB
        self.base_url = llm_base_url or config.LLM_BASE_URL
        self.model = llm_model or "anthropic/claude-sonnet-4-6"
        self.api_key = llm_api_key
        self.timeout = timeout
        self.min_claims = min_claims if min_claims is not None else config.MIN_CLAIMS_FOR_SYNTHESIS

    def _get_verified_claims(self, topic_id: int, conn: sqlite3.Connection) -> list[dict]:
        """Get verified claims grouped by confidence."""
        rows = conn.execute(
            """SELECT id, text, claim_type, confidence
               FROM claims
               WHERE topic_id = ? AND status = 'verified'
               ORDER BY confidence DESC, id""",
            (topic_id,)
        ).fetchall()
        return [{"id": r[0], "text": r[1], "type": r[2], "confidence": r[3]} for r in rows]

    def _group_claims(self, claims: list[dict]) -> str:
        """Group claims by confidence for the synthesis prompt."""
        groups = {"HIGH": [], "MED": [], "LOW": []}
        for c in claims:
            conf = c["confidence"] if c["confidence"] in groups else "MED"
            groups[conf].append(c)

        parts = []
        for level, label in [("HIGH", "HIGH CONFIDENCE"), ("MED", "MEDIUM CONFIDENCE"),
                              ("LOW", "LOW CONFIDENCE")]:
            if groups[level]:
                items = "\n".join(
                    f"  [C-{c['id']}] ({c['type']}): {c['text']}"
                    for c in groups[level]
                )
                parts.append(f"{label}:\n{items}")
        return "\n\n".join(parts)

    def _get_contradictions(self, topic_id: int, conn: sqlite3.Connection) -> list[tuple]:
        """Get contradiction pairs for this topic's claims."""
        return conn.execute(
            """SELECT cc.claim_a_id, cc.claim_b_id, cc.resolution
               FROM claim_contradictions cc
               JOIN claims ca ON cc.claim_a_id = ca.id
               WHERE ca.topic_id = ?""",
            (topic_id,)
        ).fetchall()

    def _call_llm(self, prompt: str) -> str:
        """Call Sonnet and return text response."""
        return call_llm(prompt, max_tokens=4096)

    def _next_version(self, topic_id: int, conn: sqlite3.Connection) -> int:
        """Get next synthesis version number for a topic."""
        row = conn.execute(
            "SELECT MAX(version) FROM syntheses WHERE topic_id = ?",
            (topic_id,)
        ).fetchone()
        return (row[0] or 0) + 1

    def synthesize_topic(self, topic_id: int) -> dict:
        """Generate synthesis for a single topic. Returns stats."""
        start = time.time()
        conn = init_epistemic_db(self.epistemic_db)
        stats = {"topic_id": topic_id, "claim_count": 0, "version": 0,
                 "canonical_tokens": 0, "brief_tokens": 0, "errors": 0}

        # Get topic label
        row = conn.execute("SELECT label FROM topics WHERE id = ?", (topic_id,)).fetchone()
        if not row:
            conn.close()
            stats["errors"] = 1
            return stats
        topic_label = row[0] or f"Topic {topic_id}"

        # Get verified claims
        claims = self._get_verified_claims(topic_id, conn)
        stats["claim_count"] = len(claims)

        if len(claims) < self.min_claims:
            logger.info(
                f"Topic {topic_id} ({topic_label}): {len(claims)} verified claims "
                f"< minimum {self.min_claims}, skipping synthesis"
            )
            conn.close()
            return stats

        # Generate canonical synthesis
        claims_grouped = self._group_claims(claims)
        synthesis_prompt = SYNTHESIS_PROMPT.format(
            topic_label=topic_label,
            claims_grouped=claims_grouped,
        )

        try:
            canonical = self._call_llm(synthesis_prompt)
        except Exception as e:
            logger.error(f"Canonical synthesis failed for topic {topic_id}: {e}")
            conn.close()
            stats["errors"] = 1
            return stats

        # Check for first-person voice violations
        first_person = any(phrase in canonical.lower() for phrase in
                          ["i believe", "i think", "i've learned", "i noticed", "in my experience"])
        if first_person:
            logger.warning(f"Topic {topic_id}: canonical synthesis contains first-person voice")

        stats["canonical_tokens"] = len(canonical.split())

        # Generate injection brief
        brief_prompt = BRIEF_PROMPT.format(
            topic_label=topic_label,
            canonical=canonical,
        )

        try:
            brief = self._call_llm(brief_prompt)
        except Exception as e:
            logger.error(f"Brief generation failed for topic {topic_id}: {e}")
            conn.close()
            stats["errors"] = 1
            return stats

        stats["brief_tokens"] = len(brief.split())

        if stats["brief_tokens"] > 600:
            logger.warning(
                f"Topic {topic_id}: injection brief is {stats['brief_tokens']} tokens (target 200-400)"
            )

        # Store synthesis
        version = self._next_version(topic_id, conn)
        stats["version"] = version

        conn.execute(
            """INSERT INTO syntheses (topic_id, version, canonical_text, injection_brief, claim_count)
               VALUES (?, ?, ?, ?, ?)""",
            (topic_id, version, canonical, brief, len(claims)),
        )

        # Link claims to synthesis
        synth_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        for claim in claims:
            conn.execute(
                "INSERT INTO synthesis_claims (synthesis_id, claim_id, weight) VALUES (?, ?, 1.0)",
                (synth_id, claim["id"]),
            )

        conn.commit()

        # Log run
        duration = int((time.time() - start) * 1000)
        conn.execute(
            """INSERT INTO synthesis_runs (topic_id, phase, stats, finished_at, duration_ms)
               VALUES (?, 'synthesize', ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?)""",
            (topic_id, json.dumps(stats), duration),
        )
        conn.commit()
        stats["duration_ms"] = duration

        conn.close()
        logger.info(
            f"Topic {topic_id} ({topic_label}): synthesis v{version}, "
            f"{len(claims)} claims, {stats['canonical_tokens']} canonical tokens, "
            f"{stats['brief_tokens']} brief tokens"
        )
        return stats

    def run(self, topic_ids: Optional[list[int]] = None) -> list[dict]:
        """Synthesize all topics (or specified list)."""
        conn = init_epistemic_db(self.epistemic_db)
        if topic_ids is None:
            rows = conn.execute("SELECT id FROM topics ORDER BY id").fetchall()
            topic_ids = [r[0] for r in rows]
        conn.close()

        return [self.synthesize_topic(tid) for tid in topic_ids]
