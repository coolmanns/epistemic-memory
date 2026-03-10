"""Tests for schema.py — T1.* test cases from phase1-plan.md"""

import os
import sqlite3
import tempfile
import threading
from pathlib import Path

import pytest

# Allow imports from src/
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import (
    init_epistemic_db,
    open_lcm_readonly,
    get_schema_version,
    get_all_tables,
    SCHEMA_VERSION,
)
from src import config


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary epistemic.db path."""
    return tmp_path / "epistemic.db"


@pytest.fixture
def tmp_lcm(tmp_path):
    """Create a minimal lcm.db for read-only testing."""
    db_path = tmp_path / "lcm.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE summaries (summary_id TEXT PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO summaries VALUES ('sum_test1', 'Test summary content')")
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def real_lcm():
    """Path to the real lcm.db (skip if not available)."""
    path = config.LCM_DB
    if not path.exists():
        pytest.skip("Real lcm.db not available")
    return path


# --- T1.1: Create epistemic.db from scratch ---
class TestT1_1_CreateFromScratch:
    def test_creates_db_file(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        assert tmp_db.exists()
        conn.close()

    def test_creates_all_tables(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        tables = get_all_tables(conn)
        expected = {"topics", "topic_summaries", "topic_edges", "tagging_log", "schema_meta"}
        assert expected.issubset(set(tables))
        conn.close()

    def test_schema_version_set(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        assert get_schema_version(conn) == SCHEMA_VERSION
        conn.close()

    def test_wal_mode_enabled(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()

    def test_foreign_keys_enabled(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
        conn.close()

    def test_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "epistemic.db"
        conn = init_epistemic_db(deep_path)
        assert deep_path.exists()
        conn.close()


# --- T1.2: Run schema creation twice (idempotent) ---
class TestT1_2_Idempotent:
    def test_double_init_no_error(self, tmp_db):
        conn1 = init_epistemic_db(tmp_db)
        conn1.close()
        conn2 = init_epistemic_db(tmp_db)
        conn2.close()

    def test_double_init_preserves_data(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        # Insert a topic
        conn.execute(
            "INSERT INTO topics (label, centroid, created_at, updated_at) VALUES (?, ?, datetime('now'), datetime('now'))",
            ("test-topic", b"\x00" * 3072),
        )
        conn.commit()
        conn.close()

        # Re-init
        conn = init_epistemic_db(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
        assert count == 1
        conn.close()

    def test_double_init_same_schema_version(self, tmp_db):
        conn1 = init_epistemic_db(tmp_db)
        v1 = get_schema_version(conn1)
        conn1.close()

        conn2 = init_epistemic_db(tmp_db)
        v2 = get_schema_version(conn2)
        conn2.close()

        assert v1 == v2 == SCHEMA_VERSION


# --- T1.3: Open lcm.db in read-only mode ---
class TestT1_3_LcmReadOnly:
    def test_opens_readonly(self, tmp_lcm):
        conn = open_lcm_readonly(tmp_lcm)
        row = conn.execute("SELECT content FROM summaries WHERE summary_id = 'sum_test1'").fetchone()
        assert row["content"] == "Test summary content"
        conn.close()

    def test_readonly_blocks_writes(self, tmp_lcm):
        conn = open_lcm_readonly(tmp_lcm)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO summaries VALUES ('sum_bad', 'should fail')")
        conn.close()

    def test_readonly_blocks_deletes(self, tmp_lcm):
        conn = open_lcm_readonly(tmp_lcm)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM summaries WHERE id = 'sum_test1'")
        conn.close()

    def test_readonly_blocks_drops(self, tmp_lcm):
        conn = open_lcm_readonly(tmp_lcm)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DROP TABLE summaries")
        conn.close()

    def test_missing_lcm_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            open_lcm_readonly(tmp_path / "nonexistent.db")


# --- T1.4: Open lcm.db while gateway is running (real DB) ---
class TestT1_4_RealLcm:
    def test_read_real_lcm(self, real_lcm):
        """Read real lcm.db — should work even if gateway has it open."""
        conn = open_lcm_readonly(real_lcm)
        # Just check we can query without SQLITE_BUSY
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert len(tables) > 0
        conn.close()

    def test_real_lcm_no_write(self, real_lcm):
        """Confirm we truly cannot write to the real lcm.db."""
        conn = open_lcm_readonly(real_lcm)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE evil (id INTEGER)")
        conn.close()

    def test_concurrent_reads(self, real_lcm):
        """Multiple readers should not block each other."""
        results = []
        errors = []

        def read_db():
            try:
                conn = open_lcm_readonly(real_lcm)
                count = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                ).fetchone()[0]
                results.append(count)
                conn.close()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_db) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Concurrent read errors: {errors}"
        assert len(results) == 5
        assert all(r > 0 for r in results)


# --- Table structure validation ---
class TestTableStructure:
    def test_topics_columns(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        info = conn.execute("PRAGMA table_info(topics)").fetchall()
        col_names = {row[1] for row in info}
        expected = {"id", "label", "centroid", "summary_count", "depth", "created_at", "updated_at"}
        assert expected == col_names
        conn.close()

    def test_topic_summaries_columns(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        info = conn.execute("PRAGMA table_info(topic_summaries)").fetchall()
        col_names = {row[1] for row in info}
        expected = {"topic_id", "summary_id", "similarity", "tagged_at", "orphaned"}
        assert expected == col_names
        conn.close()

    def test_topic_edges_columns(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        info = conn.execute("PRAGMA table_info(topic_edges)").fetchall()
        col_names = {row[1] for row in info}
        expected = {"topic_a", "topic_b", "weight", "edge_type", "created_at", "updated_at"}
        assert expected == col_names
        conn.close()

    def test_foreign_key_enforcement(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO topic_summaries (topic_id, summary_id, similarity) VALUES (999, 'sum_x', 0.9)"
            )
        conn.close()

    def test_primary_key_uniqueness(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        conn.execute(
            "INSERT INTO topics (label, centroid, created_at, updated_at) VALUES ('t1', ?, datetime('now'), datetime('now'))",
            (b"\x00" * 3072,),
        )
        conn.execute(
            "INSERT INTO topic_summaries (topic_id, summary_id, similarity) VALUES (1, 'sum_a', 0.8)"
        )
        conn.commit()

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO topic_summaries (topic_id, summary_id, similarity) VALUES (1, 'sum_a', 0.9)"
            )
        conn.close()


# --- T9: Phase 2 Schema Migration ---

def _make_phase1_db(tmp_db):
    """Helper: create a Phase 1 db with some data."""
    conn = init_epistemic_db(tmp_db)
    conn.execute(
        "INSERT INTO topics (label, centroid, created_at, updated_at) VALUES ('topic-1', ?, datetime('now'), datetime('now'))",
        (b"\x00" * 3072,),
    )
    conn.execute(
        "INSERT INTO topic_summaries (topic_id, summary_id, similarity) VALUES (1, 'sum_abc', 0.85)"
    )
    conn.commit()
    return conn


class TestT9_1_MigrationPreservesData:
    def test_phase1_data_survives(self, tmp_db):
        conn = _make_phase1_db(tmp_db)
        # Phase 2 tables should exist (created by init)
        tables = get_all_tables(conn)
        assert "claims" in tables
        # Phase 1 data intact
        count = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
        assert count == 1
        ts_count = conn.execute("SELECT COUNT(*) FROM topic_summaries").fetchone()[0]
        assert ts_count == 1
        conn.close()

    def test_new_tables_created(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        tables = set(get_all_tables(conn))
        phase2_tables = {"claims", "claim_sources", "claim_contradictions",
                         "syntheses", "synthesis_claims", "synthesis_runs"}
        assert phase2_tables.issubset(tables)
        conn.close()


class TestT9_2_MigrationIdempotent:
    def test_double_init_with_phase2(self, tmp_db):
        conn1 = init_epistemic_db(tmp_db)
        conn1.execute(
            "INSERT INTO topics (label, centroid) VALUES ('t', ?)", (b"\x00" * 10,)
        )
        conn1.execute(
            "INSERT INTO claims (topic_id, text, claim_type, confidence, status) "
            "VALUES (1, 'test claim', 'factual', 'MED', 'active')"
        )
        conn1.commit()
        conn1.close()

        conn2 = init_epistemic_db(tmp_db)
        assert conn2.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1
        conn2.close()


class TestT9_3_ClaimFK:
    def test_claim_bad_topic_id(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO claims (topic_id, text, claim_type, confidence, status) "
                "VALUES (999, 'orphan claim', 'factual', 'MED', 'active')"
            )
        conn.close()


class TestT9_4_ClaimSourcesUnique:
    def test_duplicate_claim_source_rejected(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        conn.execute("INSERT INTO topics (label, centroid) VALUES ('t', ?)", (b"\x00" * 10,))
        conn.execute(
            "INSERT INTO claims (topic_id, text) VALUES (1, 'claim 1')"
        )
        conn.execute(
            "INSERT INTO claim_sources (claim_id, summary_id, excerpt) VALUES (1, 'sum_x', 'some text')"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO claim_sources (claim_id, summary_id, excerpt) VALUES (1, 'sum_x', 'different text')"
            )
        conn.close()


class TestT9_5_SynthesisVersionUnique:
    def test_duplicate_topic_version_rejected(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        conn.execute("INSERT INTO topics (label, centroid) VALUES ('t', ?)", (b"\x00" * 10,))
        conn.execute(
            "INSERT INTO syntheses (topic_id, version, canonical_text, injection_brief, claim_count) "
            "VALUES (1, 1, 'text', 'brief', 5)"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO syntheses (topic_id, version, canonical_text, injection_brief, claim_count) "
                "VALUES (1, 1, 'other text', 'other brief', 3)"
            )
        conn.close()

    def test_different_versions_ok(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        conn.execute("INSERT INTO topics (label, centroid) VALUES ('t', ?)", (b"\x00" * 10,))
        conn.execute(
            "INSERT INTO syntheses (topic_id, version, canonical_text, injection_brief, claim_count) "
            "VALUES (1, 1, 'v1', 'brief1', 5)"
        )
        conn.execute(
            "INSERT INTO syntheses (topic_id, version, canonical_text, injection_brief, claim_count) "
            "VALUES (1, 2, 'v2', 'brief2', 7)"
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM syntheses WHERE topic_id = 1").fetchone()[0]
        assert count == 2
        conn.close()


class TestT9_6_ContradictionCheck:
    def test_canonical_ordering_enforced(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        conn.execute("INSERT INTO topics (label, centroid) VALUES ('t', ?)", (b"\x00" * 10,))
        conn.execute("INSERT INTO claims (topic_id, text) VALUES (1, 'claim A')")
        conn.execute("INSERT INTO claims (topic_id, text) VALUES (1, 'claim B')")
        conn.commit()
        # claim_a_id < claim_b_id — (1, 2) is valid
        conn.execute(
            "INSERT INTO claim_contradictions (claim_a_id, claim_b_id) VALUES (1, 2)"
        )
        conn.commit()
        # reversed pair (2, 1) should fail CHECK constraint
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO claim_contradictions (claim_a_id, claim_b_id) VALUES (2, 1)"
            )
        conn.close()


class TestPhase2Columns:
    """Verify all Phase 2 table columns exist with correct names."""

    def test_claims_columns(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        info = conn.execute("PRAGMA table_info(claims)").fetchall()
        cols = {row[1] for row in info}
        expected = {"id", "topic_id", "text", "claim_type", "confidence", "status",
                    "source_count", "first_seen", "last_reinforced", "embedding",
                    "created_at", "updated_at"}
        assert expected == cols
        conn.close()

    def test_claim_sources_columns(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        info = conn.execute("PRAGMA table_info(claim_sources)").fetchall()
        cols = {row[1] for row in info}
        expected = {"id", "claim_id", "summary_id", "excerpt", "extracted_at"}
        assert expected == cols
        conn.close()

    def test_syntheses_columns(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        info = conn.execute("PRAGMA table_info(syntheses)").fetchall()
        cols = {row[1] for row in info}
        expected = {"id", "topic_id", "version", "canonical_text", "injection_brief",
                    "claim_count", "created_at"}
        assert expected == cols
        conn.close()

    def test_synthesis_runs_columns(self, tmp_db):
        conn = init_epistemic_db(tmp_db)
        info = conn.execute("PRAGMA table_info(synthesis_runs)").fetchall()
        cols = {row[1] for row in info}
        expected = {"id", "topic_id", "phase", "stats", "started_at", "finished_at", "duration_ms"}
        assert expected == cols
        conn.close()
