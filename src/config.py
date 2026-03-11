"""Epistemic Synthesis — Configuration

All thresholds, paths, and connection details in one place.
Override via environment variables prefixed with ES_ (e.g., ES_SIMILARITY_THRESHOLD=0.75).
"""

import os
from pathlib import Path


def _env(key: str, default, cast=str):
    """Read from environment with ES_ prefix, cast to type."""
    val = os.environ.get(f"ES_{key}", None)
    if val is None:
        return default
    return cast(val)


# --- Paths ---
LCM_DB = Path(_env("LCM_DB", os.path.expanduser("~/.openclaw/lcm.db")))
EPISTEMIC_DB = Path(_env("EPISTEMIC_DB", os.path.expanduser("~/.openclaw/data/epistemic.db")))

# --- Embedding ---
EMBED_BASE_URL = _env("EMBED_BASE_URL", "http://localhost:8086")
EMBED_MODEL = _env("EMBED_MODEL", "Qwen3-Embedding-4B-Q6_K.gguf")
EMBED_TIMEOUT = _env("EMBED_TIMEOUT", 60, cast=int)

# --- LLM (topic labeling) ---
LLM_BASE_URL = _env("LLM_BASE_URL", "http://localhost:8084")
LLM_MODEL = _env("LLM_MODEL", "qwen3-30b")
LLM_TIMEOUT = _env("LLM_TIMEOUT", 60, cast=int)

# --- Tagging ---
SIMILARITY_THRESHOLD = _env("SIMILARITY_THRESHOLD", 0.65, cast=float)
CENTROID_UPDATE_WEIGHT = _env("CENTROID_UPDATE_WEIGHT", 0.15, cast=float)

# --- Discovery (HDBSCAN) ---
MERGE_THRESHOLD = _env("MERGE_THRESHOLD", 0.82, cast=float)
MIN_DISCOVERY_BATCH = _env("MIN_DISCOVERY_BATCH", 15, cast=int)
HDBSCAN_MIN_CLUSTER_SIZE = _env("HDBSCAN_MIN_CLUSTER_SIZE", 2, cast=int)
MAX_TOPICS_PER_RUN = _env("MAX_TOPICS_PER_RUN", 20, cast=int)

# --- Orphan Reconciliation ---
MAX_ORPHAN_RATIO = _env("MAX_ORPHAN_RATIO", 0.3, cast=float)

# --- Embedding Dimensions ---
EMBED_DIM = _env("EMBED_DIM", 2560, cast=int)

# --- Phase 2: Synthesis ---
CLAIM_DEDUP_THRESHOLD = _env("CLAIM_DEDUP_THRESHOLD", 0.90, cast=float)
MIN_CLAIMS_FOR_SYNTHESIS = _env("MIN_CLAIMS_FOR_SYNTHESIS", 3, cast=int)
MAX_CLAIMS_PER_EXTRACTION = _env("MAX_CLAIMS_PER_EXTRACTION", 15, cast=int)
MAX_CLAIM_LENGTH = _env("MAX_CLAIM_LENGTH", 500, cast=int)
SYNTHESIS_OUTPUT_DIR = Path(_env("SYNTHESIS_OUTPUT_DIR", os.path.expanduser("~/clawd/memory/topics")))

# --- Claim Decay ---
CLAIM_DECAY_HIGH_DAYS = _env("CLAIM_DECAY_HIGH_DAYS", 90, cast=int)
CLAIM_DECAY_MED_DAYS = _env("CLAIM_DECAY_MED_DAYS", 60, cast=int)
CLAIM_DECAY_LOW_DAYS = _env("CLAIM_DECAY_LOW_DAYS", 60, cast=int)
