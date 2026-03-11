"""Epistemic Synthesis — Contradiction Detection

Finds claims that conflict within or across topics.
Three-stage pipeline:
  1. Candidate pairs: embedding similarity to find semantically related claims
  2. LLM classification: direct contradiction / temporal evolution / compatible
  3. Store results in claim_contradictions table

Designed to run after extraction — needs claims with embeddings.
"""

import sqlite3
import json
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from . import config
from .schema import init_epistemic_db
from .embed import EmbedClient

log = logging.getLogger(__name__)

# How similar two claim embeddings must be to warrant LLM comparison
CANDIDATE_SIM_THRESHOLD = 0.75

# LLM classification prompt
CLASSIFY_PROMPT = """You are analyzing two knowledge claims for contradictions.

Claim A (from {topic_a}, first seen {date_a}):
"{claim_a}"

Claim B (from {topic_b}, first seen {date_b}):
"{claim_b}"

Classify the relationship between these claims. Respond with ONLY valid JSON:

{{
  "type": "direct_contradiction" | "temporal_evolution" | "nuance_difference" | "compatible",
  "explanation": "One sentence explaining the relationship",
  "supersedes": "A" | "B" | null
}}

Definitions:
- direct_contradiction: Claims cannot both be true at the same time
- temporal_evolution: Both were true at different times — situation changed
- nuance_difference: Claims appear to conflict but describe different aspects/contexts
- compatible: Claims are related but not in conflict

If temporal_evolution, set "supersedes" to whichever claim is newer/more current.
If direct_contradiction, set "supersedes" to whichever has stronger evidence (or null if unclear).
Otherwise set "supersedes" to null."""


class ContradictionDetector:
    """Finds and classifies contradictions between claims."""

    def __init__(
        self,
        epistemic_db: Optional[Path] = None,
        embed_client: Optional[EmbedClient] = None,
        candidate_threshold: float = CANDIDATE_SIM_THRESHOLD,
        llm_base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        self.epistemic_path = epistemic_db or config.EPISTEMIC_DB
        self.embed = embed_client or EmbedClient()
        self.candidate_threshold = candidate_threshold
        self.llm_url = (llm_base_url or config.LLM_BASE_URL).rstrip("/")
        self.llm_model = llm_model or config.LLM_MODEL

    def run(self, topic_id: Optional[int] = None) -> dict:
        """Run contradiction detection.

        If topic_id given, only check claims in that topic + cross-topic.
        Otherwise check all topics.

        Returns stats dict.
        """
        start = time.time()
        stats = {
            "claims_checked": 0,
            "candidate_pairs": 0,
            "contradictions_found": 0,
            "temporal_evolutions": 0,
            "nuance_differences": 0,
            "compatible": 0,
            "already_checked": 0,
            "errors": 0,
        }

        econn = init_epistemic_db(self.epistemic_path)

        try:
            # Ensure extended schema exists
            self._ensure_schema(econn)

            # Load claims with embeddings
            claims = self._load_claims(econn, topic_id)
            stats["claims_checked"] = len(claims)

            if len(claims) < 2:
                log.info("Fewer than 2 claims — nothing to compare")
                stats["duration_ms"] = int((time.time() - start) * 1000)
                return stats

            log.info(f"Loaded {len(claims)} claims for contradiction check")

            # Embed claims that don't have embeddings yet
            claims = self._ensure_embeddings(econn, claims)

            # Get already-checked pairs
            existing_pairs = self._get_existing_pairs(econn)

            # Find candidate pairs by embedding similarity
            candidates = self._find_candidates(claims, existing_pairs)
            stats["candidate_pairs"] = len(candidates)
            log.info(f"Found {len(candidates)} candidate pairs (threshold {self.candidate_threshold})")

            # Classify each candidate via LLM
            for claim_a, claim_b, sim in candidates:
                result = self._classify_pair(econn, claim_a, claim_b)
                if result is None:
                    stats["errors"] += 1
                    continue

                ctype = result.get("type", "compatible")
                if ctype == "direct_contradiction":
                    stats["contradictions_found"] += 1
                elif ctype == "temporal_evolution":
                    stats["temporal_evolutions"] += 1
                elif ctype == "nuance_difference":
                    stats["nuance_differences"] += 1
                else:
                    stats["compatible"] += 1

                # Store result
                self._store_result(econn, claim_a, claim_b, sim, result)

            econn.commit()

        finally:
            econn.close()

        stats["duration_ms"] = int((time.time() - start) * 1000)
        log.info(
            f"Contradiction detection done: {stats['contradictions_found']} contradictions, "
            f"{stats['temporal_evolutions']} temporal, {stats['nuance_differences']} nuance, "
            f"{stats['compatible']} compatible, {stats['errors']} errors ({stats['duration_ms']}ms)"
        )
        return stats

    def _ensure_schema(self, econn: sqlite3.Connection):
        """Add extended columns to claim_contradictions if missing."""
        cols = {r[1] for r in econn.execute("PRAGMA table_info(claim_contradictions)")}
        if "contradiction_type" not in cols:
            econn.execute("ALTER TABLE claim_contradictions ADD COLUMN contradiction_type TEXT DEFAULT 'unknown'")
        if "explanation" not in cols:
            econn.execute("ALTER TABLE claim_contradictions ADD COLUMN explanation TEXT")
        if "supersedes" not in cols:
            econn.execute("ALTER TABLE claim_contradictions ADD COLUMN supersedes TEXT")
        if "embedding_similarity" not in cols:
            econn.execute("ALTER TABLE claim_contradictions ADD COLUMN embedding_similarity REAL")
        econn.commit()

    def _load_claims(self, econn: sqlite3.Connection, topic_id: Optional[int] = None) -> list[dict]:
        """Load active claims with their topic info."""
        if topic_id:
            # Claims in this topic + claims in neighboring topics
            rows = econn.execute(
                "SELECT c.id, c.text, c.topic_id, c.embedding, c.first_seen, c.confidence, t.label "
                "FROM claims c JOIN topics t ON c.topic_id = t.id "
                "WHERE c.status IN ('active', 'verified') AND (c.topic_id = ? OR c.topic_id IN "
                "  (SELECT id FROM topics WHERE id != ?))",
                (topic_id, topic_id),
            ).fetchall()
        else:
            rows = econn.execute(
                "SELECT c.id, c.text, c.topic_id, c.embedding, c.first_seen, c.confidence, t.label "
                "FROM claims c JOIN topics t ON c.topic_id = t.id "
                "WHERE c.status IN ('active', 'verified')"
            ).fetchall()

        claims = []
        for r in rows:
            claim = {
                "id": r[0],
                "text": r[1],
                "topic_id": r[2],
                "embedding": EmbedClient.blob_to_vec(r[3]) if r[3] else None,
                "first_seen": r[4],
                "confidence": r[5],
                "topic_label": r[6],
            }
            claims.append(claim)
        return claims

    def _ensure_embeddings(self, econn: sqlite3.Connection, claims: list[dict]) -> list[dict]:
        """Embed claims that are missing embeddings."""
        missing = [c for c in claims if c["embedding"] is None]
        if not missing:
            return claims

        log.info(f"Embedding {len(missing)} claims without embeddings")
        texts = [c["text"] for c in missing]
        vectors = self.embed.embed_batch(texts)

        for claim, vec in zip(missing, vectors):
            if vec is not None:
                claim["embedding"] = vec
                econn.execute(
                    "UPDATE claims SET embedding = ? WHERE id = ?",
                    (EmbedClient.vec_to_blob(vec), claim["id"]),
                )

        econn.commit()
        return [c for c in claims if c["embedding"] is not None]

    def _get_existing_pairs(self, econn: sqlite3.Connection) -> set[tuple[int, int]]:
        """Get pairs already in claim_contradictions to avoid re-checking."""
        rows = econn.execute("SELECT claim_a_id, claim_b_id FROM claim_contradictions").fetchall()
        return {(r[0], r[1]) for r in rows}

    def _find_candidates(
        self, claims: list[dict], existing_pairs: set[tuple[int, int]]
    ) -> list[tuple[dict, dict, float]]:
        """Find claim pairs with high embedding similarity.

        Only compares claims that could plausibly conflict — same topic
        or cross-topic with high similarity. Skips already-checked pairs.
        """
        candidates = []

        for i, ca in enumerate(claims):
            for j, cb in enumerate(claims):
                if j <= i:
                    continue

                # Ensure ordered IDs for the CHECK constraint
                id_lo = min(ca["id"], cb["id"])
                id_hi = max(ca["id"], cb["id"])

                if (id_lo, id_hi) in existing_pairs:
                    continue

                sim = EmbedClient.cosine_similarity(ca["embedding"], cb["embedding"])

                if sim >= self.candidate_threshold:
                    candidates.append((ca, cb, sim))

        # Sort by similarity descending — check most likely conflicts first
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    def _classify_pair(self, econn: sqlite3.Connection, claim_a: dict, claim_b: dict) -> Optional[dict]:
        """Use LLM to classify the relationship between two claims."""
        prompt = CLASSIFY_PROMPT.format(
            topic_a=claim_a["topic_label"],
            date_a=claim_a["first_seen"][:10] if claim_a["first_seen"] else "unknown",
            claim_a=claim_a["text"],
            topic_b=claim_b["topic_label"],
            date_b=claim_b["first_seen"][:10] if claim_b["first_seen"] else "unknown",
            claim_b=claim_b["text"],
        )

        try:
            resp = requests.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 200,
                },
                timeout=config.LLM_TIMEOUT,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            # Parse JSON from response (handle markdown code blocks)
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            # Also handle /think blocks from Qwen
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content)

            # Validate
            valid_types = {"direct_contradiction", "temporal_evolution", "nuance_difference", "compatible"}
            if result.get("type") not in valid_types:
                log.warning(f"Invalid type '{result.get('type')}' — defaulting to compatible")
                result["type"] = "compatible"

            return result

        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            log.error(f"LLM classification failed for claims {claim_a['id']} vs {claim_b['id']}: {e}")
            return None

    def _store_result(
        self,
        econn: sqlite3.Connection,
        claim_a: dict,
        claim_b: dict,
        similarity: float,
        result: dict,
    ):
        """Store a contradiction check result."""
        # Ensure ordered IDs for CHECK constraint
        id_lo = min(claim_a["id"], claim_b["id"])
        id_hi = max(claim_a["id"], claim_b["id"])

        # Map supersedes to claim ID
        supersedes_id = None
        if result.get("supersedes") == "A":
            supersedes_id = str(claim_a["id"])
        elif result.get("supersedes") == "B":
            supersedes_id = str(claim_b["id"])

        try:
            econn.execute(
                """INSERT OR IGNORE INTO claim_contradictions
                   (claim_a_id, claim_b_id, contradiction_type, explanation, supersedes, embedding_similarity)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    id_lo,
                    id_hi,
                    result.get("type", "unknown"),
                    result.get("explanation", ""),
                    supersedes_id,
                    similarity,
                ),
            )
        except Exception as e:
            log.error(f"Error storing contradiction {id_lo} vs {id_hi}: {e}")


def format_report(econn: sqlite3.Connection) -> str:
    """Generate a human-readable contradiction report."""
    rows = econn.execute(
        """SELECT cc.*, 
           ca.text as text_a, ca.topic_id as tid_a, ca.first_seen as seen_a,
           cb.text as text_b, cb.topic_id as tid_b, cb.first_seen as seen_b,
           ta.label as label_a, tb.label as label_b
        FROM claim_contradictions cc
        JOIN claims ca ON cc.claim_a_id = ca.id
        JOIN claims cb ON cc.claim_b_id = cb.id
        JOIN topics ta ON ca.topic_id = ta.id
        JOIN topics tb ON cb.topic_id = tb.id
        WHERE cc.contradiction_type != 'compatible'
        ORDER BY 
            CASE cc.contradiction_type 
                WHEN 'direct_contradiction' THEN 1
                WHEN 'temporal_evolution' THEN 2
                WHEN 'nuance_difference' THEN 3
            END"""
    ).fetchall()

    if not rows:
        return "No contradictions or temporal evolutions found."

    lines = ["# Epistemic Contradiction Report\n"]

    # Group by type
    by_type = {}
    for r in rows:
        # Column indices: id=0, claim_a_id=1, claim_b_id=2, resolution=3, resolved_at=4,
        # created_at=5, contradiction_type=6, explanation=7, supersedes=8, embedding_similarity=9,
        # text_a=10, tid_a=11, seen_a=12, text_b=13, tid_b=14, seen_b=15, label_a=16, label_b=17
        ctype = r[6] if len(r) > 6 else "unknown"
        by_type.setdefault(ctype, []).append(r)

    type_emoji = {
        "direct_contradiction": "⚡",
        "temporal_evolution": "📅",
        "nuance_difference": "🔍",
    }

    for ctype, entries in by_type.items():
        emoji = type_emoji.get(ctype, "❓")
        lines.append(f"\n## {emoji} {ctype.replace('_', ' ').title()} ({len(entries)})\n")

        for r in entries:
            text_a = r[10] if len(r) > 10 else "?"
            text_b = r[13] if len(r) > 13 else "?"
            label_a = r[16] if len(r) > 16 else "?"
            label_b = r[17] if len(r) > 17 else "?"
            seen_a = (r[12] or "?")[:10] if len(r) > 12 else "?"
            seen_b = (r[15] or "?")[:10] if len(r) > 15 else "?"
            explanation = r[7] if len(r) > 7 else ""
            sim = r[9] if len(r) > 9 else 0

            lines.append(f"**Claim A** [{label_a}, {seen_a}]: \"{text_a}\"")
            lines.append(f"**Claim B** [{label_b}, {seen_b}]: \"{text_b}\"")
            if explanation:
                lines.append(f"→ {explanation}")
            lines.append(f"_(similarity: {sim:.3f})_\n")

    return "\n".join(lines)
