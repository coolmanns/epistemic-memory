"""Tests for full pipeline integration — T15.* test cases from phase2-plan.md

Tests the extract → dedup → verify → synthesize → write chain.
All LLM calls mocked.
"""

import json
import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import init_epistemic_db
from src.extractor import ClaimExtractor
from src.dedup import ClaimDeduplicator
from src.verifier import ClaimVerifier
from src.synthesizer import Synthesizer
from src.writer import OutputWriter


# --- Fixtures ---

@pytest.fixture
def pipeline_env(tmp_path):
    """Full pipeline environment with epistemic.db, lcm.db, and output dir."""
    edb = tmp_path / "epistemic.db"
    ldb = tmp_path / "lcm.db"
    out = tmp_path / "topics"

    # Init epistemic with topic
    conn_e = init_epistemic_db(edb)
    conn_e.execute("INSERT INTO topics (label, centroid) VALUES ('Test Pipeline Topic', ?)", (b"\x00" * 10,))
    for i in range(10):
        conn_e.execute(
            "INSERT INTO topic_summaries (topic_id, summary_id, similarity) VALUES (1, ?, 0.85)",
            (f"sum_{i:03d}",)
        )
    conn_e.commit()
    conn_e.close()

    # Create lcm.db with summaries
    conn_l = sqlite3.connect(str(ldb))
    conn_l.execute("CREATE TABLE summaries (summary_id TEXT PRIMARY KEY, content TEXT)")
    for i in range(10):
        conn_l.execute(
            "INSERT INTO summaries VALUES (?, ?)",
            (f"sum_{i:03d}", f"Summary {i}: The system component {i} handles task {i}. "
             f"It was configured on day {i} and processes {i*10} requests per hour.")
        )
    conn_l.commit()
    conn_l.close()

    return {"edb": edb, "ldb": ldb, "out": out}


def _mock_extraction_response(claims):
    """call_llm_json returns dict directly."""
    return {"claims": claims}


def _mock_verification_response(results):
    """call_llm_json returns dict directly."""
    return {"results": results}


def _mock_text_response(text):
    """call_llm returns text directly."""
    return text


def _make_mock_embed_client():
    client = MagicMock()
    dim = 16
    def embed_batch(texts):
        return [np.random.randn(dim).astype(np.float32) for _ in texts]
    client.embed_batch = embed_batch
    return client


# --- T15.1: Full pipeline end-to-end ---
class TestT15_1_EndToEnd:
    def test_full_pipeline(self, pipeline_env):
        env = pipeline_env

        # Step 1: Extract
        claims = [
            {"text": f"Component {i} handles task {i}", "type": "factual",
             "confidence": "MED", "source_excerpt": f"handles task {i}",
             "summary_id": f"sum_{i:03d}", "contradicts": []}
            for i in range(8)
        ]

        with patch("src.extractor.call_llm_json") as mock_ext:
            mock_ext.return_value = _mock_extraction_response(claims)
            ext = ClaimExtractor(epistemic_db=env["edb"], lcm_db=env["ldb"])
            ext_stats = ext.extract_topic(1)
            assert ext_stats["extracted"] == 8

        # Step 2: Dedup
        dd = ClaimDeduplicator(epistemic_db=env["edb"], embed_client=_make_mock_embed_client())
        dd_stats = dd.dedup_topic(1)
        assert dd_stats["errors"] == 0

        # Step 3: Verify — query active claims to get real IDs
        conn = sqlite3.connect(str(env["edb"]))
        active_ids = [r[0] for r in conn.execute(
            "SELECT id FROM claims WHERE topic_id = 1 AND status = 'active' ORDER BY id"
        ).fetchall()]
        conn.close()

        verify_results = [
            {"claim_id": cid, "verdict": "VERIFIED" if i < len(active_ids) - 2 else "UNSUPPORTED",
             "reasoning": "ok"}
            for i, cid in enumerate(active_ids)
        ]

        with patch("src.verifier.call_llm_json") as mock_ver:
            mock_ver.return_value = _mock_verification_response(verify_results)
            ver = ClaimVerifier(epistemic_db=env["edb"])
            ver_stats = ver.verify_topic(1)

        assert ver_stats["total"] == len(active_ids)
        verified_count = ver_stats["verified"]
        assert verified_count >= 3

        # Step 4: Synthesize
        with patch("src.synthesizer.call_llm") as mock_syn:
            mock_syn.side_effect = [
                _mock_text_response("## Established Patterns\n[C-1] Component 0..."),
                _mock_text_response("Key findings: components verified."),
            ]
            syn = Synthesizer(epistemic_db=env["edb"], min_claims=3)
            syn_stats = syn.synthesize_topic(1)

        assert syn_stats["claim_count"] >= 3
        assert syn_stats["version"] == 1

        # Step 5: Write
        w = OutputWriter(epistemic_db=env["edb"], output_dir=env["out"])
        path = w.write_topic(1)
        assert path is not None
        assert path.exists()
        content = path.read_text()
        assert "Key findings" in content
        assert "version: 1" in content


# --- T15.2: No new summaries since last run ---
class TestT15_2_NoNewSummaries:
    def test_extract_skips(self, pipeline_env):
        env = pipeline_env

        # First run
        claims = [{"text": f"Claim {i}", "type": "factual", "confidence": "MED",
                   "source_excerpt": "quote", "summary_id": f"sum_{i:03d}", "contradicts": []}
                  for i in range(10)]
        with patch("src.extractor.call_llm_json") as mock_post:
            mock_post.return_value = _mock_extraction_response(claims)
            ext = ClaimExtractor(epistemic_db=env["edb"], lcm_db=env["ldb"])
            ext.extract_topic(1)

            # Second run — all summaries already processed
            stats = ext.extract_topic(1)
        assert stats["already_processed"] == 10


# --- T15.5: Run logs all phases ---
class TestT15_5_RunLogging:
    def test_all_phases_logged(self, pipeline_env):
        env = pipeline_env

        # Extract
        claims = [{"text": f"Claim {i}", "type": "factual", "confidence": "MED",
                   "source_excerpt": "q", "summary_id": f"sum_{i:03d}", "contradicts": []}
                  for i in range(5)]
        with patch("src.extractor.call_llm_json") as mock_ext:
            mock_ext.return_value = _mock_extraction_response(claims)
            ClaimExtractor(epistemic_db=env["edb"], lcm_db=env["ldb"]).extract_topic(1)

        # Dedup
        ClaimDeduplicator(epistemic_db=env["edb"],
                         embed_client=_make_mock_embed_client()).dedup_topic(1)

        # Verify
        conn_q = sqlite3.connect(str(env["edb"]))
        active_ids = [r[0] for r in conn_q.execute(
            "SELECT id FROM claims WHERE topic_id=1 AND status='active'"
        ).fetchall()]
        conn_q.close()

        results = [{"claim_id": cid, "verdict": "VERIFIED", "reasoning": "ok"}
                   for cid in active_ids]
        with patch("src.verifier.call_llm_json") as mock_ver:
            mock_ver.return_value = _mock_verification_response(results)
            ClaimVerifier(epistemic_db=env["edb"]).verify_topic(1)

        # Synthesize
        with patch("src.synthesizer.call_llm") as mock_syn:
            mock_syn.side_effect = [_mock_text_response("Canonical"), _mock_text_response("Brief")]
            Synthesizer(epistemic_db=env["edb"], min_claims=3).synthesize_topic(1)

        # Check all phases logged
        conn = sqlite3.connect(str(env["edb"]))
        phases = conn.execute(
            "SELECT DISTINCT phase FROM synthesis_runs ORDER BY phase"
        ).fetchall()
        phase_names = {p[0] for p in phases}
        assert {"extract", "dedup", "verify", "synthesize"}.issubset(phase_names)
        conn.close()


# --- T15.7: lcm.db mtime unchanged ---
class TestT15_7_LcmReadOnly:
    def test_lcm_unchanged(self, pipeline_env):
        env = pipeline_env
        mtime_before = os.path.getmtime(env["ldb"])

        claims = [{"text": "Test", "type": "factual", "confidence": "MED",
                   "source_excerpt": "q", "summary_id": "sum_000", "contradicts": []}]
        with patch("src.extractor.call_llm_json") as mock_post:
            mock_post.return_value = _mock_extraction_response(claims)
            ClaimExtractor(epistemic_db=env["edb"], lcm_db=env["ldb"]).extract_topic(1)

        mtime_after = os.path.getmtime(env["ldb"])
        assert mtime_before == mtime_after
