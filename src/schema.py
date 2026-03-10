"""Epistemic Synthesis — Database Schema

Creates and manages epistemic.db. Idempotent — safe to call multiple times.
lcm.db is NEVER written to. Read-only access only.
"""

import sqlite3
from pathlib import Path
from typing import Optional

from . import config

SCHEMA_VERSION = 1

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
