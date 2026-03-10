"""Tests for trace.py — Two-tier scoped search."""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import init_epistemic_db
from src import config
from src.trace import trace, trace_topic_id, list_topics, provenance_chain


@pytest.fixture
def db_path(tmp_path):
    """Create a test epistemic.db with sample data."""
    db = tmp_path / "epistemic.db"
    conn = init_epistemic_db(db)

    # Create topics
    import struct
    centroid = struct.pack(f"{config.EMBED_DIM}f", *([0.1] * config.EMBED_DIM))

    conn.execute("INSERT INTO topics (id, label, centroid, summary_count) VALUES (1, 'SEO Growth Strategy', ?, 9)", (centroid,))
    conn.execute("INSERT INTO topics (id, label, centroid, summary_count) VALUES (2, 'Debug Log Flood', ?, 6)", (centroid,))
    conn.execute("INSERT INTO topics (id, label, centroid, summary_count) VALUES (3, 'Empty Topic', ?, 0)", (centroid,))

    # Topic summaries
    conn.execute("INSERT INTO topic_summaries (topic_id, summary_id, similarity) VALUES (1, 'sum_aaa', 0.85)")
    conn.execute("INSERT INTO topic_summaries (topic_id, summary_id, similarity) VALUES (1, 'sum_bbb', 0.80)")
    conn.execute("INSERT INTO topic_summaries (topic_id, summary_id, similarity) VALUES (1, 'sum_ccc', 0.78)")
    conn.execute("INSERT INTO topic_summaries (topic_id, summary_id, similarity) VALUES (2, 'sum_ddd', 0.90)")

    # Claims for topic 1
    conn.execute("""INSERT INTO claims (id, topic_id, text, claim_type, confidence, status, source_count)
                    VALUES (1, 1, 'Martin Ball SEO generated approximately $89K revenue in 2025', 'factual', 'HIGH', 'verified', 2)""")
    conn.execute("""INSERT INTO claims (id, topic_id, text, claim_type, confidence, status, source_count)
                    VALUES (2, 1, 'Organic traffic grew 340% over 6 months', 'factual', 'HIGH', 'verified', 1)""")
    conn.execute("""INSERT INTO claims (id, topic_id, text, claim_type, confidence, status, source_count)
                    VALUES (3, 1, 'Content strategy focuses on long-tail keywords', 'evaluative', 'MED', 'verified', 1)""")
    conn.execute("""INSERT INTO claims (id, topic_id, text, claim_type, confidence, status, source_count)
                    VALUES (4, 1, 'Old decayed claim', 'factual', 'LOW', 'decayed', 1)""")

    # Claims for topic 2
    conn.execute("""INSERT INTO claims (id, topic_id, text, claim_type, confidence, status, source_count)
                    VALUES (5, 2, 'Cron debug logs flooding Telegram due to missing logging.level config', 'causal', 'HIGH', 'verified', 1)""")

    # Claim sources
    conn.execute("INSERT INTO claim_sources (claim_id, summary_id, excerpt) VALUES (1, 'sum_aaa', 'Revenue was approximately 89K from organic search')")
    conn.execute("INSERT INTO claim_sources (claim_id, summary_id, excerpt) VALUES (1, 'sum_bbb', 'SEO project brought in roughly $89K')")
    conn.execute("INSERT INTO claim_sources (claim_id, summary_id, excerpt) VALUES (2, 'sum_aaa', 'Traffic increased 340%')")
    conn.execute("INSERT INTO claim_sources (claim_id, summary_id, excerpt) VALUES (3, 'sum_ccc', 'Strategy targets long-tail keywords')")
    conn.execute("INSERT INTO claim_sources (claim_id, summary_id, excerpt) VALUES (5, 'sum_ddd', 'Debug log appeared in Telegram')")

    # Synthesis
    conn.execute("INSERT INTO syntheses (id, topic_id, version, canonical_text, injection_brief, claim_count) VALUES (1, 1, 1, 'Full synthesis...', 'Brief...', 3)")
    conn.execute("INSERT INTO synthesis_claims (synthesis_id, claim_id, weight) VALUES (1, 1, 1.0)")
    conn.execute("INSERT INTO synthesis_claims (synthesis_id, claim_id, weight) VALUES (1, 2, 1.0)")
    conn.execute("INSERT INTO synthesis_claims (synthesis_id, claim_id, weight) VALUES (1, 3, 0.8)")

    conn.commit()
    conn.close()
    return db


@pytest.fixture(autouse=True)
def mock_config(db_path, tmp_path):
    """Point config to test DB."""
    with patch.object(config, "EPISTEMIC_DB", db_path), \
         patch.object(config, "LCM_DB", tmp_path / "lcm.db"):
        yield


class TestTraceSearch:
    def test_find_topic_by_substring(self):
        result = trace("SEO")
        assert result["topic"]["label"] == "SEO Growth Strategy"

    def test_find_topic_case_insensitive(self):
        result = trace("seo growth")
        assert result["topic"]["label"] == "SEO Growth Strategy"

    def test_no_topic_found(self):
        result = trace("nonexistent")
        assert "error" in result
        assert "topics" in result  # includes available topics

    def test_all_claims_returned_without_query(self):
        result = trace("SEO")
        # 3 active claims (decayed excluded)
        assert len(result["claims"]) == 3

    def test_decayed_claims_excluded(self):
        result = trace("SEO")
        statuses = [c["status"] for c in result["claims"]]
        assert "decayed" not in statuses

    def test_claims_ordered_by_confidence(self):
        result = trace("SEO")
        confidences = [c["confidence"] for c in result["claims"]]
        assert confidences == ["HIGH", "HIGH", "MED"]

    def test_search_filters_claims(self):
        result = trace("SEO", "revenue")
        assert len(result["claims"]) == 1
        assert "89K" in result["claims"][0]["text"]

    def test_search_no_match(self):
        result = trace("SEO", "blockchain")
        assert len(result["claims"]) == 0

    def test_sources_attached_to_claims(self):
        result = trace("SEO", "revenue")
        claim = result["claims"][0]
        assert len(claim["sources"]) == 2
        sids = {s["summary_id"] for s in claim["sources"]}
        assert sids == {"sum_aaa", "sum_bbb"}

    def test_claim_summary_ids_collected(self):
        result = trace("SEO", "revenue")
        assert "sum_aaa" in result["claim_summary_ids"]
        assert "sum_bbb" in result["claim_summary_ids"]

    def test_all_topic_summary_ids_included(self):
        result = trace("SEO")
        assert set(result["all_topic_summary_ids"]) == {"sum_aaa", "sum_bbb", "sum_ccc"}

    def test_max_claims_respected(self):
        result = trace("SEO", max_claims=1)
        assert len(result["claims"]) == 1


class TestTraceTopicId:
    def test_by_id(self):
        result = trace_topic_id(1)
        assert result["topic"]["label"] == "SEO Growth Strategy"

    def test_invalid_id(self):
        result = trace_topic_id(999)
        assert "error" in result


class TestListTopics:
    def test_returns_all_topics(self):
        topics = list_topics()
        labels = [t["label"] for t in topics]
        assert "SEO Growth Strategy" in labels
        assert "Debug Log Flood" in labels
        assert "Empty Topic" in labels

    def test_includes_claim_count(self):
        topics = list_topics()
        seo = next(t for t in topics if t["label"] == "SEO Growth Strategy")
        assert seo["claim_count"] == 3  # excludes decayed

    def test_includes_synthesis_version(self):
        topics = list_topics()
        seo = next(t for t in topics if t["label"] == "SEO Growth Strategy")
        assert seo["latest_version"] == 1

    def test_ordered_by_summary_count(self):
        topics = list_topics()
        counts = [t["summary_count"] for t in topics]
        assert counts == sorted(counts, reverse=True)


class TestProvenance:
    def test_full_chain(self):
        result = provenance_chain(1)
        assert result["text"] == "Martin Ball SEO generated approximately $89K revenue in 2025"
        assert result["topic_label"] == "SEO Growth Strategy"
        assert len(result["sources"]) == 2
        assert len(result["used_in_syntheses"]) == 1

    def test_lcm_content_attempted(self):
        # LCM DB doesn't exist in test, should return empty
        result = provenance_chain(1)
        assert result["lcm_content"] == {}

    def test_invalid_claim(self):
        result = provenance_chain(999)
        assert "error" in result


class TestLcmEnrichment:
    def test_with_lcm_db(self, tmp_path):
        """Create a fake lcm.db and verify enrichment."""
        lcm_path = tmp_path / "lcm.db"
        lcm = sqlite3.connect(str(lcm_path))
        lcm.execute("""CREATE TABLE summaries (
            summary_id TEXT PRIMARY KEY, conversation_id INTEGER,
            kind TEXT, depth INTEGER, content TEXT, token_count INTEGER,
            earliest_at TEXT, latest_at TEXT, descendant_count INTEGER,
            descendant_token_count INTEGER, source_message_token_count INTEGER,
            created_at TEXT, file_ids TEXT)""")
        lcm.execute("INSERT INTO summaries (summary_id, content) VALUES ('sum_aaa', 'Full summary content about SEO revenue growth and Martin Ball project...')")
        lcm.commit()
        lcm.close()

        with patch.object(config, "LCM_DB", lcm_path):
            result = trace("SEO", "revenue", include_lcm=True)
            assert "sum_aaa" in result["lcm_snippets"]
            assert "SEO revenue" in result["lcm_snippets"]["sum_aaa"]
