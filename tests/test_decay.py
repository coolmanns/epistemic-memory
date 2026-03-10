"""Tests for decay.py — T16.* test cases from phase2-plan.md"""

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import init_epistemic_db
from src.decay import ClaimDecay


# --- Fixtures ---

@pytest.fixture
def edb(tmp_path):
    db_path = tmp_path / "epistemic.db"
    conn = init_epistemic_db(db_path)
    conn.execute("INSERT INTO topics (label, centroid) VALUES ('Test', ?)", (b"\x00" * 10,))
    conn.commit()
    conn.close()
    return db_path


def _add_claim(db_path, text, confidence="HIGH", status="verified", last_reinforced=None):
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    lr = last_reinforced or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    conn.execute(
        "INSERT INTO claims (topic_id, text, claim_type, confidence, status, last_reinforced) "
        "VALUES (1, ?, 'factual', ?, ?, ?)",
        (text, confidence, status, lr)
    )
    conn.commit()
    conn.close()


def _get_claim(db_path, claim_id):
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT confidence, status FROM claims WHERE id = ?", (claim_id,)
    ).fetchone()
    conn.close()
    return {"confidence": row[0], "status": row[1]}


# --- T16.1: HIGH → MED after 90 days ---
class TestT16_1_HighToMed:
    def test_decay(self, edb):
        old = (datetime.utcnow() - timedelta(days=100)).strftime("%Y-%m-%dT%H:%M:%S")
        _add_claim(edb, "Old HIGH claim", confidence="HIGH", last_reinforced=old)

        now = datetime.utcnow()
        decay = ClaimDecay(epistemic_db=edb, now=now)
        stats = decay.run()
        assert stats["high_to_med"] == 1

        c = _get_claim(edb, 1)
        assert c["confidence"] == "MED"


# --- T16.2: MED → LOW after 60 days ---
class TestT16_2_MedToLow:
    def test_decay(self, edb):
        old = (datetime.utcnow() - timedelta(days=70)).strftime("%Y-%m-%dT%H:%M:%S")
        _add_claim(edb, "Old MED claim", confidence="MED", last_reinforced=old)

        decay = ClaimDecay(epistemic_db=edb, now=datetime.utcnow())
        stats = decay.run()
        assert stats["med_to_low"] == 1

        c = _get_claim(edb, 1)
        assert c["confidence"] == "LOW"


# --- T16.3: LOW → decayed after 60 days ---
class TestT16_3_LowToDecayed:
    def test_decay(self, edb):
        old = (datetime.utcnow() - timedelta(days=70)).strftime("%Y-%m-%dT%H:%M:%S")
        _add_claim(edb, "Old LOW claim", confidence="LOW", last_reinforced=old)

        decay = ClaimDecay(epistemic_db=edb, now=datetime.utcnow())
        stats = decay.run()
        assert stats["low_to_decayed"] == 1

        c = _get_claim(edb, 1)
        assert c["status"] == "decayed"


# --- T16.4: Decayed excluded from injection brief queries ---
class TestT16_4_DecayedExcluded:
    def test_not_in_verified(self, edb):
        old = (datetime.utcnow() - timedelta(days=70)).strftime("%Y-%m-%dT%H:%M:%S")
        _add_claim(edb, "Decayed claim", confidence="LOW", last_reinforced=old)
        _add_claim(edb, "Active claim", confidence="HIGH",
                  last_reinforced=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"))

        decay = ClaimDecay(epistemic_db=edb, now=datetime.utcnow())
        decay.run()

        conn = sqlite3.connect(str(edb))
        active = conn.execute(
            "SELECT COUNT(*) FROM claims WHERE status = 'verified'"
        ).fetchone()[0]
        assert active == 1  # only the active HIGH claim
        conn.close()


# --- T16.5: Decayed stays in evidence ledger ---
class TestT16_5_StillInDB:
    def test_not_deleted(self, edb):
        old = (datetime.utcnow() - timedelta(days=70)).strftime("%Y-%m-%dT%H:%M:%S")
        _add_claim(edb, "Decayed claim", confidence="LOW", last_reinforced=old)

        decay = ClaimDecay(epistemic_db=edb, now=datetime.utcnow())
        decay.run()

        conn = sqlite3.connect(str(edb))
        total = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        assert total == 1  # still exists
        status = conn.execute("SELECT status FROM claims WHERE id = 1").fetchone()[0]
        assert status == "decayed"
        conn.close()


# --- T16.6: Reinforcement resets timer ---
class TestT16_6_Reinforcement:
    def test_reinforce_prevents_decay(self, edb):
        old = (datetime.utcnow() - timedelta(days=100)).strftime("%Y-%m-%dT%H:%M:%S")
        _add_claim(edb, "Reinforced claim", confidence="HIGH", last_reinforced=old)

        # Reinforce it
        ClaimDecay.reinforce(1, edb)

        # Now run decay — should not decay because reinforcement updated timestamp
        decay = ClaimDecay(epistemic_db=edb, now=datetime.utcnow())
        stats = decay.run()
        assert stats["high_to_med"] == 0

        c = _get_claim(edb, 1)
        assert c["confidence"] == "HIGH"


# --- T16.7: Empty claims table ---
class TestT16_7_EmptyTable:
    def test_no_op(self, edb):
        decay = ClaimDecay(epistemic_db=edb, now=datetime.utcnow())
        stats = decay.run()
        assert stats["total_processed"] == 0
        assert stats["high_to_med"] == 0


# --- Full cascade test ---
class TestFullCascade:
    def test_high_to_decayed(self, edb):
        """Simulate full decay lifecycle."""
        day0 = datetime(2026, 1, 1)
        _add_claim(edb, "Aging claim", confidence="HIGH",
                  last_reinforced=day0.strftime("%Y-%m-%dT%H:%M:%S"))

        # After 91 days: HIGH → MED
        day91 = day0 + timedelta(days=91)
        decay = ClaimDecay(epistemic_db=edb, now=day91)
        stats = decay.run()
        assert stats["high_to_med"] == 1
        assert _get_claim(edb, 1)["confidence"] == "MED"

        # After 91+61=152 days: MED → LOW
        day152 = day0 + timedelta(days=152)
        decay = ClaimDecay(epistemic_db=edb, now=day152)
        stats = decay.run()
        assert stats["med_to_low"] == 1
        assert _get_claim(edb, 1)["confidence"] == "LOW"

        # After 152+61=213 days: LOW → decayed
        day213 = day0 + timedelta(days=213)
        decay = ClaimDecay(epistemic_db=edb, now=day213)
        stats = decay.run()
        assert stats["low_to_decayed"] == 1
        assert _get_claim(edb, 1)["status"] == "decayed"
