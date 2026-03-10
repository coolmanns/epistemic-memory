#!/usr/bin/env python3
"""Epistemic Synthesis — Cron Runner

Entry point for scheduled execution. Modes:
  tag       — Tag new summaries against existing topics (every 30 min)
  discover  — HDBSCAN discovery + tag + label (daily)
  orphan    — Orphan reconciliation (weekly)
  full      — All of the above in sequence
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import init_epistemic_db
from src.tagger import Tagger
from src.discovery import Discovery
from src.orphans import OrphanReconciler
from src.labels import TopicLabeler
from src.extractor import ClaimExtractor
from src.dedup import ClaimDeduplicator
from src.verifier import ClaimVerifier
from src.synthesizer import Synthesizer
from src.writer import OutputWriter
from src.decay import ClaimDecay
from src import config

LOG_FILE = Path.home() / "clawd/logs/epistemic-synthesis.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [epistemic] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def run_tag() -> dict:
    """Incremental tagging pass."""
    log.info("Starting tagging pass")
    t = Tagger(similarity_threshold=config.SIMILARITY_THRESHOLD)
    stats = t.run()
    log.info(f"Tagging done: {stats['tagged']} tagged, {stats['skipped']} noise, {stats['errors']} errors ({stats['duration_ms']}ms)")
    return stats


def run_discover() -> dict:
    """Discovery + tag + label."""
    log.info("Starting discovery pass")

    # 1. Discovery
    d = Discovery(min_discovery_batch=config.MIN_DISCOVERY_BATCH, min_cluster_size=3)
    disc_stats = d.run()
    log.info(f"Discovery: {disc_stats['clusters_found']} clusters, {disc_stats['new_topics_created']} new topics, {disc_stats['still_noise']} noise ({disc_stats['duration_ms']}ms)")

    # 2. Tag remaining
    tag_stats = run_tag()

    # 3. Label unlabeled topics
    log.info("Starting label pass")
    try:
        labeler = TopicLabeler()
        label_stats = labeler.run()
        log.info(f"Labeling: {label_stats['labeled']} labeled, {label_stats['errors']} errors ({label_stats['duration_ms']}ms)")
    except Exception as e:
        log.error(f"Labeling failed: {e}")
        label_stats = {"labeled": 0, "errors": 1}

    return {
        "discovery": disc_stats,
        "tagging": tag_stats,
        "labeling": label_stats,
    }


def run_orphan() -> dict:
    """Orphan reconciliation."""
    log.info("Starting orphan reconciliation")
    r = OrphanReconciler()
    stats = r.run()
    log.info(f"Orphans: {stats['checked']} checked, {stats['orphaned']} orphaned, {len(stats.get('topics_flagged', []))} flagged, {len(stats.get('topics_dead', []))} dead ({stats['duration_ms']}ms)")
    return stats


def run_extract() -> list[dict]:
    """Extract claims from all topics with new summaries."""
    log.info("Starting claim extraction")
    ex = ClaimExtractor()
    stats = ex.run()
    total = sum(s["extracted"] for s in stats)
    errors = sum(s["errors"] for s in stats)
    log.info(f"Extraction: {total} claims from {len(stats)} topics, {errors} errors")
    return stats


def run_dedup() -> list[dict]:
    """Deduplicate claims within all topics."""
    log.info("Starting claim deduplication")
    dd = ClaimDeduplicator()
    stats = dd.run()
    total_merged = sum(s["merged"] for s in stats)
    log.info(f"Dedup: {total_merged} merged across {len(stats)} topics")
    return stats


def run_verify() -> list[dict]:
    """Verify all active claims."""
    log.info("Starting claim verification")
    v = ClaimVerifier()
    stats = v.run()
    total_v = sum(s["verified"] for s in stats)
    total_u = sum(s["unsupported"] for s in stats)
    log.info(f"Verify: {total_v} verified, {total_u} unsupported across {len(stats)} topics")
    return stats


def run_synthesize() -> list[dict]:
    """Generate synthesis documents for all topics."""
    log.info("Starting synthesis generation")
    s = Synthesizer()
    stats = s.run()
    generated = sum(1 for st in stats if st["version"] > 0)
    log.info(f"Synthesis: {generated}/{len(stats)} topics synthesized")
    return stats


def run_write() -> list:
    """Write injection briefs to memory/topics/."""
    log.info("Starting output write")
    w = OutputWriter()
    paths = w.write_all()
    log.info(f"Write: {len(paths)} topic files written to {config.SYNTHESIS_OUTPUT_DIR}")
    return [str(p) for p in paths]


def run_decay() -> dict:
    """Process claim decay."""
    log.info("Starting claim decay")
    d = ClaimDecay()
    stats = d.run()
    log.info(f"Decay: {stats['total_processed']} claims processed")
    return stats


def run_full_synthesis() -> dict:
    """Full synthesis pipeline: extract → dedup → verify → synthesize → write."""
    log.info("Starting full synthesis pipeline")
    results = {}
    results["extract"] = run_extract()
    results["dedup"] = run_dedup()
    results["verify"] = run_verify()
    results["synthesize"] = run_synthesize()
    results["write"] = run_write()
    return results


def run_full() -> dict:
    """Full pipeline: discover + tag + label + orphan + synthesis."""
    results = {}
    results["discovery"] = run_discover()
    results["orphans"] = run_orphan()
    results["synthesis"] = run_full_synthesis()
    return results


def main():
    parser = argparse.ArgumentParser(description="Epistemic Synthesis runner")
    parser.add_argument("mode", choices=["tag", "discover", "orphan", "extract", "dedup",
                                         "verify", "synthesize", "write", "decay",
                                         "full-synthesis", "full"], help="Run mode")
    parser.add_argument("--json", action="store_true", help="Output stats as JSON")
    args = parser.parse_args()

    # Ensure epistemic.db exists
    init_epistemic_db(config.EPISTEMIC_DB)

    start = time.time()
    try:
        if args.mode == "tag":
            stats = run_tag()
        elif args.mode == "discover":
            stats = run_discover()
        elif args.mode == "orphan":
            stats = run_orphan()
        elif args.mode == "extract":
            stats = run_extract()
        elif args.mode == "dedup":
            stats = run_dedup()
        elif args.mode == "verify":
            stats = run_verify()
        elif args.mode == "synthesize":
            stats = run_synthesize()
        elif args.mode == "write":
            stats = run_write()
        elif args.mode == "decay":
            stats = run_decay()
        elif args.mode == "full-synthesis":
            stats = run_full_synthesis()
        elif args.mode == "full":
            stats = run_full()

        elapsed = int((time.time() - start) * 1000)
        log.info(f"Run complete ({args.mode}) in {elapsed}ms")

        if args.json:
            print(json.dumps(stats, indent=2))

    except Exception as e:
        log.error(f"Run failed ({args.mode}): {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
