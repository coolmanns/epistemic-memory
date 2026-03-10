"""Epistemic Synthesis — Trace & Scoped Search

Two-tier search within the epistemic graph:
  Tier 1: Search claims and source summaries within a topic (fast, local DB)
  Tier 2: Return LCM summary IDs for deeper expansion when claims aren't enough

Usage:
  trace("SEO Growth", "revenue numbers")
  trace_topic_id(5, "revenue numbers")
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import Optional

from . import config
from .schema import init_epistemic_db, open_lcm_readonly


def _connect_epistemic(db_path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path or config.EPISTEMIC_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _find_topic(conn: sqlite3.Connection, query: str) -> Optional[dict]:
    """Find a topic by label (case-insensitive substring match)."""
    q = f"%{query}%"
    row = conn.execute(
        "SELECT id, label, summary_count FROM topics WHERE label LIKE ? ORDER BY summary_count DESC LIMIT 1",
        (q,),
    ).fetchone()
    if row:
        return dict(row)
    return None


def _find_topic_by_id(conn: sqlite3.Connection, topic_id: int) -> Optional[dict]:
    row = conn.execute(
        "SELECT id, label, summary_count FROM topics WHERE id = ?",
        (topic_id,),
    ).fetchone()
    if row:
        return dict(row)
    return None


def _search_claims(conn: sqlite3.Connection, topic_id: int, query: str) -> list[dict]:
    """Search claims within a topic by text substring match."""
    q = f"%{query}%"
    rows = conn.execute(
        """SELECT c.id, c.text, c.claim_type, c.confidence, c.status,
                  c.source_count, c.first_seen, c.last_reinforced
           FROM claims c
           WHERE c.topic_id = ? AND c.status != 'decayed'
             AND c.text LIKE ?
           ORDER BY
             CASE c.confidence WHEN 'HIGH' THEN 1 WHEN 'MED' THEN 2 WHEN 'LOW' THEN 3 END,
             c.last_reinforced DESC""",
        (topic_id, q),
    ).fetchall()
    return [dict(r) for r in rows]


def _get_all_claims(conn: sqlite3.Connection, topic_id: int, active_only: bool = True) -> list[dict]:
    """Get all claims for a topic, ordered by confidence then recency."""
    status_filter = "AND c.status NOT IN ('decayed', 'merged')" if active_only else ""
    rows = conn.execute(
        f"""SELECT c.id, c.text, c.claim_type, c.confidence, c.status,
                   c.source_count, c.first_seen, c.last_reinforced
            FROM claims c
            WHERE c.topic_id = ? {status_filter}
            ORDER BY
              CASE c.confidence WHEN 'HIGH' THEN 1 WHEN 'MED' THEN 2 WHEN 'LOW' THEN 3 END,
              c.last_reinforced DESC""",
        (topic_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _get_claim_sources(conn: sqlite3.Connection, claim_ids: list[int]) -> dict[int, list[dict]]:
    """Get source summary IDs and excerpts for a list of claim IDs."""
    if not claim_ids:
        return {}

    placeholders = ",".join("?" for _ in claim_ids)
    rows = conn.execute(
        f"""SELECT cs.claim_id, cs.summary_id, cs.excerpt, cs.extracted_at
            FROM claim_sources cs
            WHERE cs.claim_id IN ({placeholders})
            ORDER BY cs.claim_id, cs.extracted_at""",
        claim_ids,
    ).fetchall()

    sources: dict[int, list[dict]] = {}
    for r in rows:
        d = dict(r)
        cid = d.pop("claim_id")
        sources.setdefault(cid, []).append(d)
    return sources


def _get_topic_summary_ids(conn: sqlite3.Connection, topic_id: int) -> list[str]:
    """Get all LCM summary IDs tagged to a topic."""
    rows = conn.execute(
        "SELECT summary_id FROM topic_summaries WHERE topic_id = ? AND orphaned = 0",
        (topic_id,),
    ).fetchall()
    return [r["summary_id"] for r in rows]


def _enrich_with_lcm(summary_ids: list[str], lcm_path: Optional[Path] = None) -> dict[str, str]:
    """Pull summary content snippets from lcm.db for given IDs."""
    if not summary_ids:
        return {}
    try:
        lcm = open_lcm_readonly(lcm_path)
    except FileNotFoundError:
        return {}

    placeholders = ",".join("?" for _ in summary_ids)
    rows = lcm.execute(
        f"SELECT summary_id, content FROM summaries WHERE summary_id IN ({placeholders})",
        summary_ids,
    ).fetchall()
    lcm.close()

    return {r["summary_id"]: r["content"][:500] for r in rows}


def trace(topic_query: str, search_query: Optional[str] = None,
          include_lcm: bool = False, max_claims: int = 20) -> dict:
    """Two-tier scoped search.

    Args:
        topic_query: Topic label to search for (substring match)
        search_query: Optional text to filter claims by. If None, returns all active claims.
        include_lcm: If True, enriches results with LCM summary content snippets
        max_claims: Maximum claims to return

    Returns:
        {
            "topic": {id, label, summary_count},
            "claims": [{id, text, type, confidence, status, sources: [{summary_id, excerpt}]}],
            "summary_ids": [unique LCM summary IDs for tier 2 expansion],
            "lcm_snippets": {summary_id: content_preview}  # only if include_lcm=True
        }
    """
    conn = _connect_epistemic()

    topic = _find_topic(conn, topic_query)
    if not topic:
        conn.close()
        return {"error": f"No topic matching '{topic_query}'", "topics": list_topics()}

    topic_id = topic["id"]

    # Tier 1: search or list claims
    if search_query:
        claims = _search_claims(conn, topic_id, search_query)
    else:
        claims = _get_all_claims(conn, topic_id)

    claims = claims[:max_claims]

    # Get provenance for matched claims
    claim_ids = [c["id"] for c in claims]
    sources = _get_claim_sources(conn, claim_ids)

    # Attach sources to claims
    for c in claims:
        c["sources"] = sources.get(c["id"], [])

    # Collect unique summary IDs (for tier 2 expansion)
    summary_ids = set()
    for c in claims:
        for s in c["sources"]:
            summary_ids.add(s["summary_id"])

    # Also include topic-level summary IDs not yet represented in claims
    topic_summary_ids = _get_topic_summary_ids(conn, topic_id)

    result = {
        "topic": topic,
        "claims": claims,
        "claim_summary_ids": sorted(summary_ids),
        "all_topic_summary_ids": topic_summary_ids,
    }

    # Optional: enrich with LCM content previews
    if include_lcm:
        all_ids = sorted(set(list(summary_ids) + topic_summary_ids))
        result["lcm_snippets"] = _enrich_with_lcm(all_ids)

    conn.close()
    return result


def trace_topic_id(topic_id: int, search_query: Optional[str] = None,
                   include_lcm: bool = False, max_claims: int = 20) -> dict:
    """Same as trace() but by topic ID instead of label search."""
    conn = _connect_epistemic()
    topic = _find_topic_by_id(conn, topic_id)
    conn.close()

    if not topic:
        return {"error": f"No topic with id {topic_id}"}

    return trace(topic["label"], search_query, include_lcm, max_claims)


def list_topics() -> list[dict]:
    """List all topics with basic stats."""
    conn = _connect_epistemic()
    rows = conn.execute(
        """SELECT t.id, t.label, t.summary_count,
                  (SELECT COUNT(*) FROM claims c WHERE c.topic_id = t.id AND c.status NOT IN ('decayed', 'merged')) as claim_count,
                  (SELECT MAX(version) FROM syntheses s WHERE s.topic_id = t.id) as latest_version
           FROM topics t
           ORDER BY t.summary_count DESC"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def provenance_chain(claim_id: int) -> dict:
    """Full provenance chain for a single claim: claim → sources → LCM summaries."""
    conn = _connect_epistemic()

    claim = conn.execute(
        """SELECT c.id, c.text, c.claim_type, c.confidence, c.status, c.topic_id,
                  c.first_seen, c.last_reinforced, t.label as topic_label
           FROM claims c JOIN topics t ON c.topic_id = t.id
           WHERE c.id = ?""",
        (claim_id,),
    ).fetchone()

    if not claim:
        conn.close()
        return {"error": f"No claim with id {claim_id}"}

    claim_dict = dict(claim)

    sources = conn.execute(
        "SELECT summary_id, excerpt, extracted_at FROM claim_sources WHERE claim_id = ?",
        (claim_id,),
    ).fetchall()
    claim_dict["sources"] = [dict(s) for s in sources]

    # Get synthesis usage
    synth_rows = conn.execute(
        """SELECT s.topic_id, s.version, sc.weight
           FROM synthesis_claims sc
           JOIN syntheses s ON sc.synthesis_id = s.id
           WHERE sc.claim_id = ?""",
        (claim_id,),
    ).fetchall()
    claim_dict["used_in_syntheses"] = [dict(r) for r in synth_rows]

    conn.close()

    # Enrich with LCM content
    summary_ids = [s["summary_id"] for s in claim_dict["sources"]]
    claim_dict["lcm_content"] = _enrich_with_lcm(summary_ids)

    return claim_dict


def main():
    """CLI entry point for trace."""
    import argparse

    parser = argparse.ArgumentParser(description="Epistemic trace — scoped search")
    sub = parser.add_subparsers(dest="command")

    # trace search
    search_p = sub.add_parser("search", help="Search within a topic")
    search_p.add_argument("topic", help="Topic label (substring)")
    search_p.add_argument("query", nargs="?", help="Search query within claims")
    search_p.add_argument("--lcm", action="store_true", help="Include LCM summary snippets")
    search_p.add_argument("--max", type=int, default=20, help="Max claims")

    # list topics
    sub.add_parser("topics", help="List all topics")

    # provenance
    prov_p = sub.add_parser("provenance", help="Full provenance for a claim")
    prov_p.add_argument("claim_id", type=int, help="Claim ID")

    args = parser.parse_args()

    if args.command == "search":
        result = trace(args.topic, args.query, include_lcm=args.lcm, max_claims=args.max)
    elif args.command == "topics":
        result = list_topics()
    elif args.command == "provenance":
        result = provenance_chain(args.claim_id)
    else:
        parser.print_help()
        return

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
