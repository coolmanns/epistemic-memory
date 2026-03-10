"""Tests for verifier.py — T12.* test cases from phase2-plan.md"""

import json
import sqlite3
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import init_epistemic_db
from src.verifier import ClaimVerifier


# --- Fixtures ---

@pytest.fixture
def edb(tmp_path):
    """Create epistemic.db with a topic and claims."""
    db_path = tmp_path / "epistemic.db"
    conn = init_epistemic_db(db_path)
    conn.execute("INSERT INTO topics (label, centroid) VALUES ('Test Topic', ?)", (b"\x00" * 10,))
    conn.commit()
    conn.close()
    return db_path


def _add_claim(db_path, topic_id, text, claim_type="factual", confidence="MED",
               status="active", source_id="sum_001", excerpt="source quote"):
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    cursor = conn.execute(
        "INSERT INTO claims (topic_id, text, claim_type, confidence, status) VALUES (?, ?, ?, ?, ?)",
        (topic_id, text, claim_type, confidence, status)
    )
    claim_id = cursor.lastrowid
    conn.execute(
        "INSERT INTO claim_sources (claim_id, summary_id, excerpt) VALUES (?, ?, ?)",
        (claim_id, source_id, excerpt)
    )
    conn.commit()
    conn.close()
    return claim_id


def _mock_verification_response(results):
    """call_llm_json returns dict directly."""
    return {"results": results}


@pytest.fixture
def verifier(edb):
    return ClaimVerifier(epistemic_db=edb)


# --- T12.1: Clear source support → VERIFIED ---
class TestT12_1_Verified:
    @patch("src.verifier.call_llm_json")
    def test_verified_status(self, mock_post, verifier, edb):
        cid = _add_claim(edb, 1, "System uses PostgreSQL", excerpt="The system uses PostgreSQL")
        mock_post.return_value = _mock_verification_response([
            {"claim_id": cid, "verdict": "VERIFIED", "reasoning": "directly stated"}
        ])

        stats = verifier.verify_topic(1)
        assert stats["verified"] == 1

        conn = sqlite3.connect(str(edb))
        status = conn.execute("SELECT status FROM claims WHERE id = ?", (cid,)).fetchone()[0]
        assert status == "verified"
        conn.close()


# --- T12.2: Overgeneralization → OVERSTATED ---
class TestT12_2_Overstated:
    @patch("src.verifier.call_llm_json")
    def test_overstated_status(self, mock_post, verifier, edb):
        cid = _add_claim(edb, 1, "All databases use PostgreSQL",
                        excerpt="Our system uses PostgreSQL")
        mock_post.return_value = _mock_verification_response([
            {"claim_id": cid, "verdict": "OVERSTATED", "reasoning": "generalizes beyond source"}
        ])

        stats = verifier.verify_topic(1)
        assert stats["overstated"] == 1

        conn = sqlite3.connect(str(edb))
        status = conn.execute("SELECT status FROM claims WHERE id = ?", (cid,)).fetchone()[0]
        assert status == "overstated"
        conn.close()


# --- T12.3: No supporting evidence → UNSUPPORTED ---
class TestT12_3_Unsupported:
    @patch("src.verifier.call_llm_json")
    def test_unsupported_status(self, mock_post, verifier, edb):
        cid = _add_claim(edb, 1, "System uses Redis", excerpt="PostgreSQL is the database")
        mock_post.return_value = _mock_verification_response([
            {"claim_id": cid, "verdict": "UNSUPPORTED", "reasoning": "no mention of Redis"}
        ])

        stats = verifier.verify_topic(1)
        assert stats["unsupported"] == 1

        conn = sqlite3.connect(str(edb))
        status = conn.execute("SELECT status FROM claims WHERE id = ?", (cid,)).fetchone()[0]
        assert status == "unsupported"
        conn.close()


# --- T12.4: Misattribution → MISATTRIBUTED ---
class TestT12_4_Misattributed:
    @patch("src.verifier.call_llm_json")
    def test_misattributed_status(self, mock_post, verifier, edb):
        cid = _add_claim(edb, 1, "John built the auth system",
                        excerpt="Sarah built the auth system")
        mock_post.return_value = _mock_verification_response([
            {"claim_id": cid, "verdict": "MISATTRIBUTED", "reasoning": "wrong person"}
        ])

        stats = verifier.verify_topic(1)
        assert stats["misattributed"] == 1


# --- T12.5: Idempotent verification ---
class TestT12_5_Idempotent:
    @patch("src.verifier.call_llm_json")
    def test_verified_not_reverified(self, mock_post, verifier, edb):
        _add_claim(edb, 1, "Claim A", status="verified")  # already verified
        _add_claim(edb, 1, "Claim B", status="active")     # needs verification

        mock_post.return_value = _mock_verification_response([
            {"claim_id": 2, "verdict": "VERIFIED", "reasoning": "ok"}
        ])

        stats = verifier.verify_topic(1)
        assert stats["total"] == 1  # only active claims, not already-verified
        assert stats["verified"] == 1


# --- T12.6: API down ---
class TestT12_6_ApiDown:
    @patch("src.verifier.call_llm_json")
    @patch("src.verifier.time.sleep")  # don't actually sleep in tests
    def test_graceful_error(self, mock_sleep, mock_post, verifier, edb):
        _add_claim(edb, 1, "Some claim")
        mock_post.side_effect = ConnectionError("refused")

        stats = verifier.verify_topic(1)
        assert stats["errors"] == 1
        assert stats["verified"] == 0


# --- T12.7: Batch of 20 claims ---
class TestT12_7_BatchProcessing:
    @patch("src.verifier.call_llm_json")
    def test_batch_20(self, mock_post, verifier, edb):
        for i in range(20):
            _add_claim(edb, 1, f"Claim {i}", source_id=f"sum_{i:03d}")

        results = [
            {"claim_id": i + 1, "verdict": "VERIFIED", "reasoning": "ok"}
            for i in range(20)
        ]
        mock_post.return_value = _mock_verification_response(results)

        stats = verifier.verify_topic(1)
        assert stats["total"] == 20
        assert stats["verified"] == 20


# --- T12.8: Analogical claim higher scrutiny ---
class TestT12_8_AnalogicalScrutiny:
    @patch("src.verifier.call_llm_json")
    def test_analogical_sent_with_type(self, mock_post, verifier, edb):
        cid = _add_claim(edb, 1, "Memory is like a filing cabinet",
                        claim_type="analogical", excerpt="we organize memories")
        mock_post.return_value = _mock_verification_response([
            {"claim_id": cid, "verdict": "OVERSTATED", "reasoning": "analogy not in source"}
        ])

        stats = verifier.verify_topic(1)
        assert stats["overstated"] == 1

        # Check that the prompt included the claim type
        call_args = mock_post.call_args
        prompt_text = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        assert "analogical" in prompt_text


# --- T12.9: Unsupported excluded from synthesis input ---
class TestT12_9_UnsupportedExcluded:
    @patch("src.verifier.call_llm_json")
    def test_unsupported_stays_in_db(self, mock_post, verifier, edb):
        cid = _add_claim(edb, 1, "Unsupported claim")
        mock_post.return_value = _mock_verification_response([
            {"claim_id": cid, "verdict": "UNSUPPORTED", "reasoning": "no evidence"}
        ])

        verifier.verify_topic(1)

        conn = sqlite3.connect(str(edb))
        # Claim exists but is unsupported
        row = conn.execute("SELECT status FROM claims WHERE id = ?", (cid,)).fetchone()
        assert row[0] == "unsupported"
        # A synthesis query for verified claims would exclude it
        verified = conn.execute(
            "SELECT COUNT(*) FROM claims WHERE topic_id = 1 AND status = 'verified'"
        ).fetchone()[0]
        assert verified == 0
        conn.close()


# --- T12.10: Genuine contradiction preserved ---
class TestT12_10_ContradictionPreserved:
    @patch("src.verifier.call_llm_json")
    def test_both_sides_verified(self, mock_post, verifier, edb):
        c1 = _add_claim(edb, 1, "System uses PostgreSQL", source_id="sum_001",
                        excerpt="We use PostgreSQL")
        c2 = _add_claim(edb, 1, "System uses MySQL", source_id="sum_002",
                        excerpt="We migrated to MySQL")

        mock_post.return_value = _mock_verification_response([
            {"claim_id": c1, "verdict": "VERIFIED", "reasoning": "stated in source"},
            {"claim_id": c2, "verdict": "VERIFIED", "reasoning": "stated in source"},
        ])

        stats = verifier.verify_topic(1)
        assert stats["verified"] == 2

        # Both remain, contradiction tracking is separate
        conn = sqlite3.connect(str(edb))
        statuses = conn.execute(
            "SELECT status FROM claims WHERE topic_id = 1"
        ).fetchall()
        assert all(s[0] == "verified" for s in statuses)
        conn.close()


# --- Run logging ---
class TestVerifyLogging:
    @patch("src.verifier.call_llm_json")
    def test_run_logged(self, mock_post, verifier, edb):
        _add_claim(edb, 1, "Test claim")
        mock_post.return_value = _mock_verification_response([
            {"claim_id": 1, "verdict": "VERIFIED", "reasoning": "ok"}
        ])

        verifier.verify_topic(1)

        conn = sqlite3.connect(str(edb))
        row = conn.execute(
            "SELECT phase, stats FROM synthesis_runs WHERE phase = 'verify'"
        ).fetchone()
        assert row is not None
        s = json.loads(row[1])
        assert s["verified"] == 1
        conn.close()
