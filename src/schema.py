"""Epistemic Synthesis — Database Schema

Creates and manages epistemic.db. Idempotent — safe to call multiple times.
lcm.db is NEVER written to. Read-only access only.
"""

import sqlite3
from pathlib import Path
from typing import Optional

from . import config

SCHEMA_VERSION = 2

_SCHEMA_SQL = """
-- Topics: emergent clusters of related summaries
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY,
    label TEXT,
    centroid BLOB NOT NULL,
    summary_count INTEGER DEFAULT 0,
    depth INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Summary-to-topic mappings (references only — no content copied from LCM)
CREATE TABLE IF NOT EXISTS topic_summaries (
    topic_id INTEGER NOT NULL REFERENCES topics(id),
    summary_id TEXT NOT NULL,
    similarity REAL NOT NULL,
    tagged_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    orphaned INTEGER DEFAULT 0,
    PRIMARY KEY (topic_id, summary_id)
);

-- Topic graph edges (weighted, typed)
CREATE TABLE IF NOT EXISTS topic_edges (
    topic_a INTEGER NOT NULL REFERENCES topics(id),
    topic_b INTEGER NOT NULL REFERENCES topics(id),
    weight REAL NOT NULL,
    edge_type TEXT DEFAULT 'co-occurrence',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (topic_a, topic_b)
);

-- Audit log for every pipeline run
CREATE TABLE IF NOT EXISTS tagging_log (
    id INTEGER PRIMARY KEY,
    run_type TEXT NOT NULL DEFAULT 'tag',
    run_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    summaries_processed INTEGER DEFAULT 0,
    summaries_tagged INTEGER DEFAULT 0,
    new_topics_created INTEGER DEFAULT 0,
    orphans_detected INTEGER DEFAULT 0,
    duration_ms INTEGER DEFAULT 0
);

-- Phase 2: Atomic claims with provenance
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY,
    topic_id INTEGER NOT NULL REFERENCES topics(id),
    text TEXT NOT NULL,
    claim_type TEXT NOT NULL DEFAULT 'factual',
    confidence TEXT NOT NULL DEFAULT 'MED',
    status TEXT NOT NULL DEFAULT 'active',
    source_count INTEGER DEFAULT 1,
    first_seen TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    last_reinforced TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    embedding BLOB,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Provenance: which summaries support each claim
CREATE TABLE IF NOT EXISTS claim_sources (
    id INTEGER PRIMARY KEY,
    claim_id INTEGER NOT NULL REFERENCES claims(id),
    summary_id TEXT NOT NULL,
    excerpt TEXT NOT NULL,
    extracted_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(claim_id, summary_id)
);

-- Explicit contradiction tracking
CREATE TABLE IF NOT EXISTS claim_contradictions (
    id INTEGER PRIMARY KEY,
    claim_a_id INTEGER NOT NULL REFERENCES claims(id),
    claim_b_id INTEGER NOT NULL REFERENCES claims(id),
    resolution TEXT,
    resolved_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK(claim_a_id < claim_b_id)
);

-- Synthesis artifacts (versioned per topic)
CREATE TABLE IF NOT EXISTS syntheses (
    id INTEGER PRIMARY KEY,
    topic_id INTEGER NOT NULL REFERENCES topics(id),
    version INTEGER NOT NULL DEFAULT 1,
    canonical_text TEXT NOT NULL,
    injection_brief TEXT NOT NULL,
    claim_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(topic_id, version)
);

-- Which claims fed into which synthesis
CREATE TABLE IF NOT EXISTS synthesis_claims (
    synthesis_id INTEGER NOT NULL REFERENCES syntheses(id),
    claim_id INTEGER NOT NULL REFERENCES claims(id),
    weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY(synthesis_id, claim_id)
);

-- Track synthesis pipeline runs
CREATE TABLE IF NOT EXISTS synthesis_runs (
    id INTEGER PRIMARY KEY,
    topic_id INTEGER,
    phase TEXT NOT NULL,
    stats TEXT,
    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    finished_at TEXT,
    duration_ms INTEGER
);

-- Phase 2 indexes
CREATE INDEX IF NOT EXISTS idx_claims_topic
    ON claims(topic_id);

CREATE INDEX IF NOT EXISTS idx_claims_status
    ON claims(status);

CREATE INDEX IF NOT EXISTS idx_claim_sources_claim
    ON claim_sources(claim_id);

CREATE INDEX IF NOT EXISTS idx_claim_sources_summary
    ON claim_sources(summary_id);

CREATE INDEX IF NOT EXISTS idx_syntheses_topic
    ON syntheses(topic_id);

CREATE INDEX IF NOT EXISTS idx_synthesis_runs_topic
    ON synthesis_runs(topic_id);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_topic_summaries_summary
    ON topic_summaries(summary_id);

CREATE INDEX IF NOT EXISTS idx_topic_summaries_orphaned
    ON topic_summaries(orphaned) WHERE orphaned = 1;

CREATE INDEX IF NOT EXISTS idx_tagging_log_run_at
    ON tagging_log(run_at);
"""


def init_epistemic_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Initialize epistemic.db with schema. Idempotent.

    Returns an open connection in WAL mode.
    """
    path = db_path or config.EPISTEMIC_DB
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA_SQL)

    # Set schema version if not present
    existing = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()

    return conn


def open_lcm_readonly(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open lcm.db in read-only mode. Raises if file doesn't exist."""
    path = db_path or config.LCM_DB
    if not path.exists():
        raise FileNotFoundError(f"lcm.db not found at {path}")

    # file: URI with mode=ro ensures read-only access
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Get current schema version from epistemic.db."""
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        return 0
    return int(row[0] if isinstance(row, tuple) else row["value"])


def get_all_tables(conn: sqlite3.Connection) -> list[str]:
    """List all tables in the database."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r[0] if isinstance(r, tuple) else r["name"] for r in rows]
