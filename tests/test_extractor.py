"""Tests for extractor.py — T10.* test cases from phase2-plan.md

Most tests mock the LLM to avoid external dependencies.
Tests marked @slow hit the real Qwen3-30B endpoint.
"""

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import init_epistemic_db
from src.extractor import ClaimExtractor
from src import config


# --- Fixtures ---

@pytest.fixture
def dbs(tmp_path):
    """Create epistemic.db + mock lcm.db with test data."""
    edb = tmp_path / "epistemic.db"
    ldb = tmp_path / "lcm.db"

    # Init epistemic
    conn_e = init_epistemic_db(edb)
    # Add a topic with centroid
    conn_e.execute(
        "INSERT INTO topics (label, centroid) VALUES ('Test Topic', ?)",
        (b"\x00" * 10,)
    )
    conn_e.execute(
        "INSERT INTO topic_summaries (topic_id, summary_id, similarity) VALUES (1, 'sum_001', 0.85)"
    )
    conn_e.execute(
        "INSERT INTO topic_summaries (topic_id, summary_id, similarity) VALUES (1, 'sum_002', 0.80)"
    )
    conn_e.commit()
    conn_e.close()

    # Create lcm.db with summaries
    conn_l = sqlite3.connect(str(ldb))
    conn_l.execute("CREATE TABLE summaries (summary_id TEXT PRIMARY KEY, content TEXT)")
    conn_l.execute(
        "INSERT INTO summaries VALUES ('sum_001', 'The system uses PostgreSQL for data storage. "
        "Migrations are handled by Drizzle ORM.')"
    )
    conn_l.execute(
        "INSERT INTO summaries VALUES ('sum_002', 'Authentication uses JWT tokens with 24h expiry. "
        "Refresh tokens are stored in httpOnly cookies.')"
    )
    conn_l.commit()
    conn_l.close()

    return edb, ldb


@pytest.fixture
def extractor(dbs):
    edb, ldb = dbs
    return ClaimExtractor(epistemic_db=edb, lcm_db=ldb)


def _mock_llm_response(claims):
    """Helper: create a mock LLM response (call_llm_json returns dict directly)."""
    return {"claims": claims}


def _valid_claim(text="Test claim", summary_id="sum_001", **overrides):
    """Helper: create a valid claim dict."""
    c = {
        "text": text,
        "type": "factual",
        "confidence": "MED",
        "source_excerpt": "some verbatim quote",
        "summary_id": summary_id,
        "contradicts": [],
    }
    c.update(overrides)
    return c


# --- T10.1: Extract claims from a single summary ---
class TestT10_1_HappyPath:
    @patch("src.extractor.call_llm_json")
    def test_extracts_claims(self, mock_post, extractor, dbs):
        claims = [
            _valid_claim("PostgreSQL is used for storage", "sum_001"),
            _valid_claim("Drizzle ORM handles migrations", "sum_001"),
            _valid_claim("JWT tokens have 24h expiry", "sum_002"),
        ]
        mock_post.return_value = _mock_llm_response(claims)

        stats = extractor.extract_topic(1)
        assert stats["extracted"] == 3
        assert stats["errors"] == 0

    @patch("src.extractor.call_llm_json")
    def test_claims_in_db(self, mock_post, extractor, dbs):
        edb, _ = dbs
        claims = [_valid_claim("Test claim one", "sum_001")]
        mock_post.return_value = _mock_llm_response(claims)

        extractor.extract_topic(1)

        conn = sqlite3.connect(str(edb))
        rows = conn.execute("SELECT text, claim_type, confidence, status FROM claims").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "Test claim one"
        assert rows[0][1] == "factual"
        assert rows[0][2] == "MED"
        assert rows[0][3] == "active"
        conn.close()

    @patch("src.extractor.call_llm_json")
    def test_sources_in_db(self, mock_post, extractor, dbs):
        edb, _ = dbs
        claims = [_valid_claim("Claim", "sum_001", source_excerpt="exact quote")]
        mock_post.return_value = _mock_llm_response(claims)

        extractor.extract_topic(1)

        conn = sqlite3.connect(str(edb))
        rows = conn.execute("SELECT claim_id, summary_id, excerpt FROM claim_sources").fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "sum_001"
        assert rows[0][2] == "exact quote"
        conn.close()


# --- T10.2: Empty/trivial summary ---
class TestT10_2_EmptyInput:
    @patch("src.extractor.call_llm_json")
    def test_no_claims_returned(self, mock_post, extractor):
        mock_post.return_value = _mock_llm_response([])
        stats = extractor.extract_topic(1)
        assert stats["extracted"] == 0
        assert stats["errors"] == 0


# --- T10.3: Contradictory information ---
class TestT10_3_Contradictions:
    @patch("src.extractor.call_llm_json")
    def test_contradictions_extracted(self, mock_post, extractor, dbs):
        edb, _ = dbs
        claims = [
            _valid_claim("System uses PostgreSQL", "sum_001", contradicts=[1]),
            _valid_claim("System uses MySQL", "sum_002", contradicts=[0]),
        ]
        mock_post.return_value = _mock_llm_response(claims)

        stats = extractor.extract_topic(1)
        assert stats["extracted"] == 2

        # Both claims stored (contradictions tracked separately in later phase)
        conn = sqlite3.connect(str(edb))
        count = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        assert count == 2
        conn.close()


# --- T10.4: Idempotent extraction ---
class TestT10_4_Idempotent:
    @patch("src.extractor.call_llm_json")
    def test_second_run_skips(self, mock_post, extractor, dbs):
        edb, _ = dbs
        claims = [_valid_claim("Claim one", "sum_001")]
        mock_post.return_value = _mock_llm_response(claims)

        # First run
        stats1 = extractor.extract_topic(1)
        assert stats1["extracted"] == 1

        # Second run — sum_001 already has claim_source, but sum_002 doesn't
        # The extractor tracks by which summaries have claims, not which were in a batch
        # Only sum_001 was in claim_sources, so already_processed = 1
        # sum_002 will be sent again but LLM mock returns same claim for sum_001
        claims2 = [_valid_claim("Claim from sum_002", "sum_002")]
        mock_post.return_value = _mock_llm_response(claims2)
        stats2 = extractor.extract_topic(1)
        assert stats2["already_processed"] == 1  # sum_001 already had claims
        assert stats2["extracted"] == 1  # sum_002 now extracted

        conn = sqlite3.connect(str(edb))
        count = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        assert count == 2  # one from each run, no duplicates
        conn.close()


# --- T10.5: LLM down ---
class TestT10_5_LlmDown:
    @patch("src.extractor.call_llm_json")
    def test_graceful_error(self, mock_post, extractor):
        mock_post.side_effect = ConnectionError("Connection refused")
        stats = extractor.extract_topic(1)
        assert stats["errors"] == 1
        assert stats["extracted"] == 0


# --- T10.6: Too many claims ---
class TestT10_6_ClaimCap:
    @patch("src.extractor.call_llm_json")
    def test_capped_at_max(self, mock_post, extractor, dbs):
        edb, _ = dbs
        # Return 60 claims (over the 50 cap)
        claims = [_valid_claim(f"Claim {i}", "sum_001") for i in range(60)]
        mock_post.return_value = _mock_llm_response(claims)

        stats = extractor.extract_topic(1)
        assert stats["extracted"] == 50  # capped

        conn = sqlite3.connect(str(edb))
        count = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        assert count == 50
        conn.close()


# --- T10.7: Malformed JSON ---
class TestT10_7_MalformedJson:
    @patch("src.extractor.call_llm_json")
    def test_bad_json_handled(self, mock_post, extractor):
        mock_post.side_effect = json.JSONDecodeError("bad json", "", 0)

        stats = extractor.extract_topic(1)
        assert stats["errors"] == 1
        assert stats["extracted"] == 0


# --- T10.8: Topic with no summaries ---
class TestT10_8_NoSummaries:
    @patch("src.extractor.call_llm_json")
    def test_empty_topic(self, mock_post, dbs):
        edb, ldb = dbs
        # Add a topic with no tagged summaries
        conn = sqlite3.connect(str(edb))
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("INSERT INTO topics (label, centroid) VALUES ('Empty', ?)", (b"\x00" * 10,))
        conn.commit()
        conn.close()

        ext = ClaimExtractor(epistemic_db=edb, lcm_db=ldb)
        stats = ext.extract_topic(2)  # topic 2 = Empty
        assert stats["extracted"] == 0
        assert stats["errors"] == 0
        mock_post.assert_not_called()


# --- T10.9: Incremental extraction ---
class TestT10_9_Incremental:
    @patch("src.extractor.call_llm_json")
    def test_only_new_summaries(self, mock_post, dbs):
        edb, ldb = dbs
        claims_round1 = [_valid_claim("First claim", "sum_001")]
        mock_post.return_value = _mock_llm_response(claims_round1)

        ext = ClaimExtractor(epistemic_db=edb, lcm_db=ldb)
        stats1 = ext.extract_topic(1)
        assert stats1["extracted"] == 1

        # Add a new summary
        conn_e = sqlite3.connect(str(edb))
        conn_e.execute("PRAGMA foreign_keys=ON")
        conn_e.execute(
            "INSERT INTO topic_summaries (topic_id, summary_id, similarity) VALUES (1, 'sum_003', 0.78)"
        )
        conn_e.commit()
        conn_e.close()

        conn_l = sqlite3.connect(str(ldb))
        conn_l.execute("INSERT INTO summaries VALUES ('sum_003', 'New info about Redis caching.')")
        conn_l.commit()
        conn_l.close()

        # Second run should only see sum_002 (not yet extracted) and sum_003 (new)
        claims_round2 = [_valid_claim("Redis used for caching", "sum_003")]
        mock_post.return_value = _mock_llm_response(claims_round2)

        stats2 = ext.extract_topic(1)
        # sum_001 already extracted, so already_processed >= 1
        assert stats2["already_processed"] >= 1
        assert stats2["extracted"] == 1


# --- T10.10: Empty claim text ---
class TestT10_10_EmptyClaimText:
    @patch("src.extractor.call_llm_json")
    def test_empty_text_rejected(self, mock_post, extractor, dbs):
        edb, _ = dbs
        claims = [
            _valid_claim("", "sum_001"),  # empty text
            _valid_claim("Valid claim", "sum_001"),
        ]
        mock_post.return_value = _mock_llm_response(claims)

        stats = extractor.extract_topic(1)
        assert stats["extracted"] == 1  # only the valid one
        assert stats["skipped"] == 1

        conn = sqlite3.connect(str(edb))
        text = conn.execute("SELECT text FROM claims").fetchone()[0]
        assert text == "Valid claim"
        conn.close()


# --- T10 bonus: Claim validation ---
class TestClaimValidation:
    def test_too_long_text(self, extractor):
        claim = _valid_claim("x" * 501)
        err = extractor._validate_claim(claim)
        assert err is not None
        assert "too long" in err

    def test_invalid_type(self, extractor):
        claim = _valid_claim(type="opinion")
        err = extractor._validate_claim(claim)
        assert "invalid claim type" in err

    def test_invalid_confidence(self, extractor):
        claim = _valid_claim(confidence="VERY_HIGH")
        err = extractor._validate_claim(claim)
        assert "invalid confidence" in err

    def test_missing_excerpt(self, extractor):
        claim = _valid_claim(source_excerpt="")
        err = extractor._validate_claim(claim)
        assert "missing source excerpt" in err

    def test_missing_summary_id(self, extractor):
        claim = _valid_claim(summary_id="")
        err = extractor._validate_claim(claim)
        assert "missing summary_id" in err

    def test_valid_claim_passes(self, extractor):
        claim = _valid_claim()
        err = extractor._validate_claim(claim)
        assert err is None


# --- Synthesis run logging ---
class TestRunLogging:
    @patch("src.extractor.call_llm_json")
    def test_run_logged(self, mock_post, extractor, dbs):
        edb, _ = dbs
        mock_post.return_value = _mock_llm_response([_valid_claim()])
        extractor.extract_topic(1)

        conn = sqlite3.connect(str(edb))
        row = conn.execute(
            "SELECT topic_id, phase, stats FROM synthesis_runs WHERE phase = 'extract'"
        ).fetchone()
        assert row is not None
        assert row[0] == 1
        assert row[1] == "extract"
        stats = json.loads(row[2])
        assert stats["extracted"] == 1
        conn.close()
