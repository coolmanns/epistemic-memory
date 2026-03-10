"""Tests for synthesizer.py — T13.* test cases from phase2-plan.md"""

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import init_epistemic_db
from src.synthesizer import Synthesizer


# --- Fixtures ---

@pytest.fixture
def edb(tmp_path):
    db_path = tmp_path / "epistemic.db"
    conn = init_epistemic_db(db_path)
    conn.execute("INSERT INTO topics (label, centroid) VALUES ('Test Topic', ?)", (b"\x00" * 10,))
    conn.commit()
    conn.close()
    return db_path


def _add_verified_claim(db_path, topic_id, text, confidence="MED", claim_type="factual"):
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    cursor = conn.execute(
        "INSERT INTO claims (topic_id, text, claim_type, confidence, status) VALUES (?, ?, ?, ?, 'verified')",
        (topic_id, text, claim_type, confidence)
    )
    conn.commit()
    cid = cursor.lastrowid
    conn.close()
    return cid


def _mock_llm_text(text):
    """call_llm returns text directly."""
    return text


@pytest.fixture
def synth(edb):
    return Synthesizer(epistemic_db=edb, min_claims=3)


# --- T13.1: Generate canonical from 10 claims ---
class TestT13_1_HappyPath:
    @patch("src.synthesizer.call_llm")
    def test_synthesis_generated(self, mock_post, synth, edb):
        for i in range(10):
            _add_verified_claim(edb, 1, f"Claim {i} about the system [C-{i+1}]",
                               confidence=["HIGH", "MED", "LOW"][i % 3])

        canonical = "## Established Patterns\n[C-1] The system...\n## Emerging Understanding\n[C-2]..."
        brief = "Key findings: system uses PostgreSQL, JWT auth with 24h expiry."
        mock_post.side_effect = [_mock_llm_text(canonical), _mock_llm_text(brief)]

        stats = synth.synthesize_topic(1)
        assert stats["claim_count"] == 10
        assert stats["version"] == 1
        assert stats["errors"] == 0

    @patch("src.synthesizer.call_llm")
    def test_stored_in_db(self, mock_post, synth, edb):
        for i in range(5):
            _add_verified_claim(edb, 1, f"Claim {i}")

        mock_post.side_effect = [_mock_llm_text("Canonical text"), _mock_llm_text("Brief text")]

        synth.synthesize_topic(1)

        conn = sqlite3.connect(str(edb))
        row = conn.execute(
            "SELECT topic_id, version, canonical_text, injection_brief, claim_count FROM syntheses"
        ).fetchone()
        assert row[0] == 1
        assert row[1] == 1
        assert row[2] == "Canonical text"
        assert row[3] == "Brief text"
        assert row[4] == 5
        conn.close()


# --- T13.2: Mixed confidence levels ---
class TestT13_2_MixedConfidence:
    @patch("src.synthesizer.call_llm")
    def test_all_levels_in_prompt(self, mock_post, synth, edb):
        _add_verified_claim(edb, 1, "High confidence claim", confidence="HIGH")
        _add_verified_claim(edb, 1, "Medium confidence claim", confidence="MED")
        _add_verified_claim(edb, 1, "Low confidence claim", confidence="LOW")

        mock_post.side_effect = [_mock_llm_text("Canonical"), _mock_llm_text("Brief")]
        synth.synthesize_topic(1)

        # Check that the prompt includes all confidence levels
        first_call = mock_post.call_args_list[0]
        prompt = first_call[0][0] if first_call[0] else first_call[1].get("prompt", "")
        assert "HIGH CONFIDENCE" in prompt
        assert "MEDIUM CONFIDENCE" in prompt
        assert "LOW CONFIDENCE" in prompt


# --- T13.3: Below minimum claims ---
class TestT13_3_BelowMinimum:
    @patch("src.synthesizer.call_llm")
    def test_skipped(self, mock_post, synth, edb):
        _add_verified_claim(edb, 1, "Only one claim")
        _add_verified_claim(edb, 1, "Two claims total")

        stats = synth.synthesize_topic(1)
        assert stats["claim_count"] == 2
        assert stats["version"] == 0  # no synthesis created
        mock_post.assert_not_called()


# --- T13.4: Injection brief from canonical ---
class TestT13_4_BriefGeneration:
    @patch("src.synthesizer.call_llm")
    def test_brief_stored(self, mock_post, synth, edb):
        for i in range(3):
            _add_verified_claim(edb, 1, f"Claim {i}")

        mock_post.side_effect = [_mock_llm_text("Full canonical"), _mock_llm_text("Short brief")]
        synth.synthesize_topic(1)

        conn = sqlite3.connect(str(edb))
        brief = conn.execute("SELECT injection_brief FROM syntheses WHERE topic_id = 1").fetchone()[0]
        assert brief == "Short brief"
        conn.close()


# --- T13.5: Version increments ---
class TestT13_5_Versioning:
    @patch("src.synthesizer.call_llm")
    def test_version_increments(self, mock_post, synth, edb):
        for i in range(5):
            _add_verified_claim(edb, 1, f"Claim {i}")

        mock_post.side_effect = [
            _mock_llm_text("v1 canonical"), _mock_llm_text("v1 brief"),
            _mock_llm_text("v2 canonical"), _mock_llm_text("v2 brief"),
        ]

        stats1 = synth.synthesize_topic(1)
        assert stats1["version"] == 1

        # Add more claims and re-synthesize
        _add_verified_claim(edb, 1, "New claim")
        stats2 = synth.synthesize_topic(1)
        assert stats2["version"] == 2

        conn = sqlite3.connect(str(edb))
        versions = conn.execute(
            "SELECT version FROM syntheses WHERE topic_id = 1 ORDER BY version"
        ).fetchall()
        assert [v[0] for v in versions] == [1, 2]
        conn.close()


# --- T13.6: Only verified claims in synthesis ---
class TestT13_6_OnlyVerified:
    @patch("src.synthesizer.call_llm")
    def test_excludes_non_verified(self, mock_post, synth, edb):
        _add_verified_claim(edb, 1, "Verified one")
        _add_verified_claim(edb, 1, "Verified two")
        _add_verified_claim(edb, 1, "Verified three")

        # Add non-verified claims directly
        conn = sqlite3.connect(str(edb))
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "INSERT INTO claims (topic_id, text, claim_type, confidence, status) "
            "VALUES (1, 'Unsupported claim', 'factual', 'MED', 'unsupported')"
        )
        conn.execute(
            "INSERT INTO claims (topic_id, text, claim_type, confidence, status) "
            "VALUES (1, 'Overstated claim', 'factual', 'MED', 'overstated')"
        )
        conn.commit()
        conn.close()

        mock_post.side_effect = [_mock_llm_text("Canonical"), _mock_llm_text("Brief")]
        stats = synth.synthesize_topic(1)
        assert stats["claim_count"] == 3  # only verified


# --- T13.7: API down ---
class TestT13_7_ApiDown:
    @patch("src.synthesizer.call_llm")
    def test_graceful_error(self, mock_post, synth, edb):
        for i in range(3):
            _add_verified_claim(edb, 1, f"Claim {i}")

        mock_post.side_effect = ConnectionError("refused")
        stats = synth.synthesize_topic(1)
        assert stats["errors"] == 1


# --- T13.8: Re-synthesis after new claims ---
class TestT13_8_Incremental:
    @patch("src.synthesizer.call_llm")
    def test_new_claims_included(self, mock_post, synth, edb):
        for i in range(3):
            _add_verified_claim(edb, 1, f"Original claim {i}")

        mock_post.side_effect = [_mock_llm_text("v1"), _mock_llm_text("b1")]
        synth.synthesize_topic(1)

        # Add 5 more
        for i in range(5):
            _add_verified_claim(edb, 1, f"New claim {i}")

        mock_post.side_effect = [_mock_llm_text("v2"), _mock_llm_text("b2")]
        stats = synth.synthesize_topic(1)
        assert stats["claim_count"] == 8  # all 8 verified claims


# --- T13.9: synthesis_claims populated ---
class TestT13_9_SynthesisClaims:
    @patch("src.synthesizer.call_llm")
    def test_claim_links(self, mock_post, synth, edb):
        ids = []
        for i in range(4):
            ids.append(_add_verified_claim(edb, 1, f"Claim {i}"))

        mock_post.side_effect = [_mock_llm_text("Canonical"), _mock_llm_text("Brief")]
        synth.synthesize_topic(1)

        conn = sqlite3.connect(str(edb))
        links = conn.execute(
            "SELECT claim_id FROM synthesis_claims ORDER BY claim_id"
        ).fetchall()
        linked_ids = [l[0] for l in links]
        assert linked_ids == ids
        conn.close()


# --- T13.10: Epistemic register (no first-person) ---
class TestT13_10_EpistemicRegister:
    @patch("src.synthesizer.call_llm")
    def test_first_person_warning(self, mock_post, synth, edb):
        for i in range(3):
            _add_verified_claim(edb, 1, f"Claim {i}")

        # Simulate LLM violating epistemic register
        bad_canonical = "I believe the system uses PostgreSQL. I think it's fast."
        mock_post.side_effect = [_mock_llm_text(bad_canonical), _mock_llm_text("Brief")]

        # Synthesis still completes (warning logged, not an error)
        stats = synth.synthesize_topic(1)
        assert stats["errors"] == 0


# --- Run logging ---
class TestSynthesisLogging:
    @patch("src.synthesizer.call_llm")
    def test_run_logged(self, mock_post, synth, edb):
        for i in range(3):
            _add_verified_claim(edb, 1, f"Claim {i}")

        mock_post.side_effect = [_mock_llm_text("Canonical"), _mock_llm_text("Brief")]
        synth.synthesize_topic(1)

        conn = sqlite3.connect(str(edb))
        row = conn.execute(
            "SELECT phase, stats FROM synthesis_runs WHERE phase = 'synthesize'"
        ).fetchone()
        assert row is not None
        s = json.loads(row[1])
        assert s["version"] == 1
        conn.close()
