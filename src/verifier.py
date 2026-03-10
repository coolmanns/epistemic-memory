"""Epistemic Synthesis — Claim Verifier

Verifies claims against their source excerpts using Sonnet.
Outputs: VERIFIED | UNSUPPORTED | OVERSTATED | MISATTRIBUTED
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import requests  # kept for test mocking

from .llm import call_llm_json

from . import config
from .schema import init_epistemic_db

logger = logging.getLogger("epistemic")

VERIFICATION_PROMPT = """You are verifying claims extracted from conversation summaries about "{topic_label}".

For each claim below, examine whether the source excerpts actually support it.

Classify each claim as:
- VERIFIED: The source excerpts directly support this claim.
- UNSUPPORTED: The source excerpts do not contain evidence for this claim.
- OVERSTATED: The claim goes beyond what the sources actually say (over-generalization).
- MISATTRIBUTED: The claim attributes something incorrectly (wrong person, wrong system, etc.).

For analogical claims (comparisons/metaphors), apply higher scrutiny — require the analogy to be explicitly present in sources, not inferred.

Respond with valid JSON:
{{
  "results": [
    {{
      "claim_id": <id>,
      "verdict": "VERIFIED|UNSUPPORTED|OVERSTATED|MISATTRIBUTED",
      "reasoning": "brief explanation"
    }}
  ]
}}

CLAIMS TO VERIFY:
{claims_text}"""

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds


class ClaimVerifier:
    """Verifies claims against source material using Sonnet."""

    def __init__(
        self,
        epistemic_db: Optional[Path] = None,
        sonnet_base_url: Optional[str] = None,
        sonnet_model: Optional[str] = None,
        sonnet_api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        self.epistemic_db = epistemic_db or config.EPISTEMIC_DB
        # For testing, allow direct URL override; production uses OpenClaw routing
        self.base_url = sonnet_base_url or config.LLM_BASE_URL
        self.model = sonnet_model or "anthropic/claude-sonnet-4-6"
        self.api_key = sonnet_api_key
        self.timeout = timeout

    def _get_claims_to_verify(self, topic_id: int, conn: sqlite3.Connection) -> list[dict]:
        """Get active (unverified) claims with their source excerpts."""
        rows = conn.execute(
            """SELECT c.id, c.text, c.claim_type, c.confidence
               FROM claims c
               WHERE c.topic_id = ? AND c.status = 'active'
               ORDER BY c.id""",
            (topic_id,)
        ).fetchall()

        claims = []
        for r in rows:
            sources = conn.execute(
                "SELECT summary_id, excerpt FROM claim_sources WHERE claim_id = ?",
                (r[0],)
            ).fetchall()
            claims.append({
                "id": r[0],
                "text": r[1],
                "type": r[2],
                "confidence": r[3],
                "sources": [{"summary_id": s[0], "excerpt": s[1]} for s in sources],
            })
        return claims

    def _format_claims(self, claims: list[dict]) -> str:
        """Format claims for the verification prompt."""
        parts = []
        for c in claims:
            sources_text = "\n".join(
                f"  - [{s['summary_id']}]: \"{s['excerpt']}\""
                for s in c["sources"]
            )
            parts.append(
                f"Claim ID {c['id']} ({c['type']}, confidence {c['confidence']}):\n"
                f"  \"{c['text']}\"\n"
                f"  Sources:\n{sources_text}"
            )
        return "\n\n".join(parts)

    def _call_llm(self, prompt: str) -> dict:
        """Call Sonnet for verification. Returns parsed JSON."""
        return call_llm_json(prompt, max_tokens=8192)

    def _call_llm_legacy(self, prompt: str) -> dict:
        """Legacy OpenAI-compat call. Kept for reference/fallback."""
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }

        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                if resp.status_code == 429:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(f"Rate limited (429), retrying in {delay}s")
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return json.loads(content)
            except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
                last_err = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (2 ** attempt))

        raise last_err or RuntimeError("Verification LLM call failed")

    def _apply_verdict(self, claim_id: int, verdict: str, conn: sqlite3.Connection):
        """Apply verification verdict to a claim."""
        valid_verdicts = {"VERIFIED", "UNSUPPORTED", "OVERSTATED", "MISATTRIBUTED"}
        if verdict not in valid_verdicts:
            logger.warning(f"Invalid verdict '{verdict}' for claim {claim_id}, keeping active")
            return

        status = "verified" if verdict == "VERIFIED" else verdict.lower()
        conn.execute(
            "UPDATE claims SET status = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
            (status, claim_id)
        )

    def verify_topic(self, topic_id: int) -> dict:
        """Verify all active claims for a topic. Returns stats."""
        start = time.time()
        conn = init_epistemic_db(self.epistemic_db)
        stats = {"topic_id": topic_id, "verified": 0, "unsupported": 0,
                 "overstated": 0, "misattributed": 0, "errors": 0, "total": 0}

        # Get topic label
        row = conn.execute("SELECT label FROM topics WHERE id = ?", (topic_id,)).fetchone()
        if not row:
            conn.close()
            stats["errors"] = 1
            return stats
        topic_label = row[0] or f"Topic {topic_id}"

        claims = self._get_claims_to_verify(topic_id, conn)
        stats["total"] = len(claims)

        if not claims:
            logger.info(f"Topic {topic_id}: no claims to verify")
            conn.close()
            return stats

        # Format and call LLM
        claims_text = self._format_claims(claims)
        prompt = VERIFICATION_PROMPT.format(
            topic_label=topic_label,
            claims_text=claims_text,
        )

        try:
            result = self._call_llm(prompt)
        except Exception as e:
            logger.error(f"Verification failed for topic {topic_id}: {e}")
            conn.close()
            stats["errors"] = 1
            return stats

        # Apply verdicts
        results = result.get("results", [])
        result_map = {r["claim_id"]: r["verdict"] for r in results if "claim_id" in r and "verdict" in r}

        for claim in claims:
            verdict = result_map.get(claim["id"])
            if verdict:
                self._apply_verdict(claim["id"], verdict, conn)
                key = verdict.lower()
                if key in stats:
                    stats[key] += 1

        conn.commit()

        # Log run
        duration = int((time.time() - start) * 1000)
        conn.execute(
            """INSERT INTO synthesis_runs (topic_id, phase, stats, finished_at, duration_ms)
               VALUES (?, 'verify', ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?)""",
            (topic_id, json.dumps(stats), duration),
        )
        conn.commit()
        stats["duration_ms"] = duration

        conn.close()
        logger.info(
            f"Topic {topic_id} ({topic_label}): {stats['verified']} verified, "
            f"{stats['unsupported']} unsupported, {stats['overstated']} overstated, "
            f"{stats['misattributed']} misattributed"
        )
        return stats

    def run(self, topic_ids: Optional[list[int]] = None) -> list[dict]:
        """Verify all topics (or specified list)."""
        conn = init_epistemic_db(self.epistemic_db)
        if topic_ids is None:
            rows = conn.execute("SELECT id FROM topics ORDER BY id").fetchall()
            topic_ids = [r[0] for r in rows]
        conn.close()

        return [self.verify_topic(tid) for tid in topic_ids]
