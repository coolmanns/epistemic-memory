"""Tests for writer.py — T14.* test cases from phase2-plan.md"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import init_epistemic_db
from src.writer import OutputWriter, sanitize_filename


# --- Fixtures ---

@pytest.fixture
def setup(tmp_path):
    """Create epistemic.db with a topic and synthesis."""
    edb = tmp_path / "epistemic.db"
    out = tmp_path / "topics"
    conn = init_epistemic_db(edb)
    conn.execute("INSERT INTO topics (label, centroid) VALUES ('Test Topic', ?)", (b"\x00" * 10,))
    conn.execute(
        "INSERT INTO syntheses (topic_id, version, canonical_text, injection_brief, claim_count) "
        "VALUES (1, 1, 'Full canonical text here', 'Compressed brief for injection', 8)"
    )
    conn.commit()
    conn.close()
    return edb, out


@pytest.fixture
def writer(setup):
    edb, out = setup
    return OutputWriter(epistemic_db=edb, output_dir=out)


# --- T14.1: Write injection brief ---
class TestT14_1_HappyPath:
    def test_file_created(self, writer, setup):
        _, out = setup
        path = writer.write_topic(1)
        assert path is not None
        assert path.exists()
        assert path.name == "test-topic.md"

    def test_file_content(self, writer, setup):
        path = writer.write_topic(1)
        content = path.read_text()
        assert "Compressed brief for injection" in content
        assert "# Test Topic" in content


# --- T14.2: Special characters in label ---
class TestT14_2_SpecialChars:
    def test_sanitize_ampersand(self):
        assert sanitize_filename("OAuth & GCP") == "oauth-gcp"

    def test_sanitize_slashes(self):
        assert sanitize_filename("path/to/thing") == "path-to-thing"

    def test_sanitize_unicode(self):
        assert sanitize_filename("café résumé") == "caf-r-sum"

    def test_sanitize_long_name(self):
        name = sanitize_filename("a" * 100)
        assert len(name) <= 80

    def test_sanitize_empty(self):
        assert sanitize_filename("") == "unnamed-topic"

    def test_sanitize_all_special(self):
        assert sanitize_filename("@#$%^&*") == "unnamed-topic"

    def test_no_path_traversal(self):
        name = sanitize_filename("../../../etc/passwd")
        assert "/" not in name
        assert ".." not in name


# --- T14.3: Overwrite existing file ---
class TestT14_3_Overwrite:
    def test_update_on_rewrite(self, setup):
        edb, out = setup
        w = OutputWriter(epistemic_db=edb, output_dir=out)
        path1 = w.write_topic(1)
        content1 = path1.read_text()

        # Add new version
        conn = sqlite3.connect(str(edb))
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "INSERT INTO syntheses (topic_id, version, canonical_text, injection_brief, claim_count) "
            "VALUES (1, 2, 'Updated canonical', 'Updated brief v2', 12)"
        )
        conn.commit()
        conn.close()

        path2 = w.write_topic(1)
        assert path1 == path2  # same file
        content2 = path2.read_text()
        assert "Updated brief v2" in content2
        assert "version: 2" in content2


# --- T14.4: Topic with no synthesis ---
class TestT14_4_NoSynthesis:
    def test_no_file_written(self, tmp_path):
        edb = tmp_path / "epistemic.db"
        out = tmp_path / "topics"
        conn = init_epistemic_db(edb)
        conn.execute("INSERT INTO topics (label, centroid) VALUES ('Empty', ?)", (b"\x00" * 10,))
        conn.commit()
        conn.close()

        w = OutputWriter(epistemic_db=edb, output_dir=out)
        path = w.write_topic(1)
        assert path is None


# --- T14.5: Batch write ---
class TestT14_5_BatchWrite:
    def test_17_topics(self, tmp_path):
        edb = tmp_path / "epistemic.db"
        out = tmp_path / "topics"
        conn = init_epistemic_db(edb)
        for i in range(17):
            conn.execute(
                "INSERT INTO topics (label, centroid) VALUES (?, ?)",
                (f"Topic {i}", b"\x00" * 10)
            )
            conn.execute(
                "INSERT INTO syntheses (topic_id, version, canonical_text, injection_brief, claim_count) "
                "VALUES (?, 1, 'canonical', 'brief', 5)",
                (i + 1,)
            )
        conn.commit()
        conn.close()

        w = OutputWriter(epistemic_db=edb, output_dir=out)
        paths = w.write_all()
        assert len(paths) == 17
        assert all(p.exists() for p in paths)
        # No filename collisions
        names = [p.name for p in paths]
        assert len(names) == len(set(names))


# --- T14.6: Output directory doesn't exist ---
class TestT14_6_MissingDir:
    def test_auto_created(self, setup):
        edb, _ = setup
        deep = Path(setup[1]) / "a" / "b" / "c"
        w = OutputWriter(epistemic_db=edb, output_dir=deep)
        path = w.write_topic(1)
        assert path.exists()
        assert deep.exists()


# --- T14.7: Metadata header ---
class TestT14_7_Metadata:
    def test_header_fields(self, writer, setup):
        path = writer.write_topic(1)
        content = path.read_text()
        assert "topic_id: 1" in content
        assert "version: 1" in content
        assert "claim_count: 8" in content
        assert "generated_at:" in content
