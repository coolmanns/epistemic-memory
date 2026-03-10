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


def run_full() -> dict:
    """Full pipeline: discover + tag + label + orphan."""
    results = {}
    results["discovery"] = run_discover()
    results["orphans"] = run_orphan()
    return results


def main():
    parser = argparse.ArgumentParser(description="Epistemic Synthesis runner")
    parser.add_argument("mode", choices=["tag", "discover", "orphan", "full"], help="Run mode")
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
