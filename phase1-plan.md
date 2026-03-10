# Epistemic Synthesis — Phase 1: Topic Tagging

**Goal:** Every LCM summary gets tagged with a topic. New topics emerge organically from data.
**Constraint:** Zero impact on LCM's compaction performance. Read-only access to lcm.db.
**Output:** `epistemic.db` with topics, centroids, and summary→topic mappings.

---

## Architecture Decision: Python Service (not JS plugin)

- We have working clustering code (`topic-discovery-test.py`)
- Embedding infra already proven (nomic-embed on port 8082)
- Runs on schedule via cron, not as a gateway plugin — decoupled from gateway lifecycle
- Reads `lcm.db` in read-only mode (WAL, no lock contention)
- If this works, can port to JS plugin later for real-time tagging

---

## Components

### 1. Schema (`epistemic.db`)

```sql
CREATE TABLE topics (
    id INTEGER PRIMARY KEY,
    label TEXT,                    -- LLM-generated label (nullable until labeling pass)
    centroid BLOB NOT NULL,        -- float32 embedding vector
    summary_count INTEGER DEFAULT 0,
    depth INTEGER DEFAULT 0,       -- future: hierarchy
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE topic_summaries (
    topic_id INTEGER NOT NULL REFERENCES topics(id),
    summary_id TEXT NOT NULL,      -- LCM summary ID (e.g., sum_abc123)
    similarity REAL NOT NULL,      -- cosine similarity at tag time
    tagged_at TEXT NOT NULL,
    orphaned INTEGER DEFAULT 0,    -- 1 if LCM re-compacted this summary away
    PRIMARY KEY (topic_id, summary_id)
);

CREATE TABLE topic_edges (
    topic_a INTEGER NOT NULL REFERENCES topics(id),
    topic_b INTEGER NOT NULL REFERENCES topics(id),
    weight REAL NOT NULL,
    edge_type TEXT DEFAULT 'co-occurrence',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (topic_a, topic_b)
);

CREATE TABLE tagging_log (
    id INTEGER PRIMARY KEY,
    run_at TEXT NOT NULL,
    summaries_processed INTEGER,
    summaries_tagged INTEGER,
    new_topics_created INTEGER,
    orphans_detected INTEGER,
    duration_ms INTEGER
);
```

### 2. Tagging Pipeline (runs every compaction or on schedule)

```
1. Query lcm.db for summaries not yet in topic_summaries
2. Embed each new summary via nomic-embed (port 8082)
3. For each embedding:
   a. Compute cosine similarity against all topic centroids
   b. If best_match >= SIMILARITY_THRESHOLD (0.72): tag it
   c. If best_match < SIMILARITY_THRESHOLD: mark as untagged (noise)
4. Update centroid: weighted average of old centroid + new embedding
5. Update summary_count on topic
6. Log the run in tagging_log
```

### 3. Topic Discovery (daily HDBSCAN pass)

```
1. Collect all untagged summaries (noise from tagging pipeline)
2. If count < MIN_DISCOVERY_BATCH (15): skip — not enough data
3. Embed all untagged (or pull cached embeddings)
4. Run HDBSCAN (min_cluster_size=3, metric=euclidean on L2-normalized)
5. For each new cluster:
   a. Compute centroid from member embeddings
   b. Check centroid against existing topics (prevent duplicates)
   c. If no match >= MERGE_THRESHOLD (0.82): create new topic
   d. If match >= MERGE_THRESHOLD: tag members into existing topic
6. Label new topics via LLM (Qwen3, port 8084) — async, non-blocking
7. Log results
```

### 4. Orphan Reconciliation (weekly or on-demand)

```
1. For each summary_id in topic_summaries:
   a. Check if it still exists in lcm.db
   b. If not: mark orphaned=1
   c. Find parent summary (if LCM merged it)
   d. If parent exists and isn't tagged: tag parent to same topic
2. If >30% of a topic's summaries are orphaned: flag for review
3. Recalculate centroid excluding orphaned summaries
```

---

## Configuration

```python
CONFIG = {
    "SIMILARITY_THRESHOLD": 0.72,      # min cosine to tag to existing topic
    "MERGE_THRESHOLD": 0.82,           # min cosine to merge new cluster into existing topic
    "MIN_DISCOVERY_BATCH": 15,         # min untagged summaries before running HDBSCAN
    "HDBSCAN_MIN_CLUSTER_SIZE": 3,     # minimum cluster members
    "CENTROID_UPDATE_WEIGHT": 0.15,    # how much new summary shifts centroid (EMA)
    "EMBED_MODEL": "nomic-embed",
    "EMBED_PORT": 8082,
    "LLM_PORT": 8084,                  # Qwen3 for topic labeling
    "LCM_DB": "~/.openclaw/lcm.db",
    "EPISTEMIC_DB": "~/.openclaw/data/epistemic.db",
    "MAX_ORPHAN_RATIO": 0.3,           # alert if >30% orphaned
}
```

---

## Test Cases

### T1: Schema & Initialization
| ID | Test | Expected | Risk Mitigated |
|----|------|----------|----------------|
| T1.1 | Create epistemic.db from scratch | All tables created, no errors | First-run failure |
| T1.2 | Run schema creation twice | Idempotent — no errors, no data loss | Accidental re-init |
| T1.3 | Open lcm.db in read-only mode | No WAL locks, no writes | Gateway contention |
| T1.4 | Open lcm.db while gateway is running | Read succeeds, no SQLITE_BUSY | Production safety |

### T2: Embedding
| ID | Test | Expected | Risk Mitigated |
|----|------|----------|----------------|
| T2.1 | Embed a normal summary (~200 tokens) | 768-dim float32 vector | Basic functionality |
| T2.2 | Embed an empty string | Graceful skip or error, not crash | Edge case |
| T2.3 | Embed when nomic-embed is down | Timeout, log error, skip run | Infra failure |
| T2.4 | Embed 100 summaries in batch | All succeed, <10s total | Performance ceiling |
| T2.5 | Same summary embedded twice | Identical vectors (deterministic) | Reproducibility |

### T3: Tagging
| ID | Test | Expected | Risk Mitigated |
|----|------|----------|----------------|
| T3.1 | Tag summary with clear topic match (sim > 0.72) | Tagged to correct topic, centroid updated | Happy path |
| T3.2 | Tag summary with no match (sim < 0.72 for all) | Remains untagged (noise pool) | False positives |
| T3.3 | Tag summary equidistant between two topics | Tagged to highest similarity only | Ambiguity handling |
| T3.4 | Tag same summary twice | Idempotent — no duplicate rows | Re-run safety |
| T3.5 | Centroid drift after 50 tags | Centroid moves but stays within topic | Gradual topic shift |
| T3.6 | Centroid drift — adversarial (50 off-topic summaries) | Centroid doesn't drift past recognition | Topic hijacking |
| T3.7 | Tag with threshold = 1.0 (impossible match) | Everything goes to noise | Threshold sanity |
| T3.8 | Tag with threshold = 0.0 (match everything) | Everything tags to closest topic | Threshold sanity |

### T4: Topic Discovery (HDBSCAN)
| ID | Test | Expected | Risk Mitigated |
|----|------|----------|----------------|
| T4.1 | 20 untagged summaries, 2 clear clusters | 2 new topics created, members tagged | Happy path |
| T4.2 | 14 untagged summaries (below MIN_DISCOVERY_BATCH) | Skip — no discovery run | Premature clustering |
| T4.3 | New cluster centroid matches existing topic (>0.82) | Members tagged to existing topic, no duplicate | Topic duplication |
| T4.4 | All summaries are noise (no clusters found) | No new topics, summaries stay untagged | HDBSCAN returns -1 |
| T4.5 | Run discovery twice on same data | Idempotent — same topics, no duplicates | Re-run safety |
| T4.6 | HDBSCAN parameters produce 100+ micro-clusters | Sanity check: reject if clusters > MAX_TOPICS_PER_RUN (20) | Hyperparameter failure |
| T4.7 | Discovery with mixed tagged + untagged | Only processes untagged, ignores tagged | Data isolation |

### T5: Centroid Management
| ID | Test | Expected | Risk Mitigated |
|----|------|----------|----------------|
| T5.1 | EMA update with weight 0.15 | Centroid shifts 15% toward new vector | Math correctness |
| T5.2 | Centroid after 1 summary vs after 50 | Early centroids more volatile, later stable | Convergence |
| T5.3 | Recalculate centroid from scratch vs EMA | Results are similar (within 0.05 cosine) | Drift accumulation |
| T5.4 | Centroid recalc after orphan removal | Centroid reflects only live summaries | Orphan contamination |

### T6: Orphan Reconciliation
| ID | Test | Expected | Risk Mitigated |
|----|------|----------|----------------|
| T6.1 | Summary exists in both DBs | orphaned=0, no changes | Happy path |
| T6.2 | Summary deleted from lcm.db (re-compacted) | orphaned=1, parent found and tagged | Normal LCM lifecycle |
| T6.3 | Summary deleted, no parent found | orphaned=1, flagged for review | Broken reference |
| T6.4 | >30% of topic's summaries orphaned | Alert raised, centroid recalculated | Topic decay |
| T6.5 | 100% of topic's summaries orphaned | Topic flagged as dead, not auto-deleted | Accidental topic deletion |

### T7: Concurrency & Safety
| ID | Test | Expected | Risk Mitigated |
|----|------|----------|----------------|
| T7.1 | Tagging runs while gateway is compacting LCM | No SQLITE_BUSY on lcm.db (read-only) | Lock contention |
| T7.2 | Two tagging runs execute simultaneously | Second run waits or skips (file lock) | Race condition |
| T7.3 | Tagging runs during HDBSCAN discovery | No conflicts (operate on different sets) | Pipeline collision |
| T7.4 | Power failure mid-tagging | WAL recovery, no corrupt state | Crash safety |
| T7.5 | epistemic.db grows to 100MB | Performance still acceptable (<5s per run) | Scale ceiling |

### T8: End-to-End
| ID | Test | Expected | Risk Mitigated |
|----|------|----------|----------------|
| T8.1 | Fresh start: 168 summaries from current lcm.db | ~19 topics (matching POC), all summaries tagged or noise | Regression vs POC |
| T8.2 | Add 10 new summaries about a known topic | All 10 tagged correctly, centroid stable | Incremental tagging |
| T8.3 | Add 10 summaries about a brand new topic | Stay as noise → next discovery creates topic | New topic emergence |
| T8.4 | Full pipeline: tag → discover → orphan check → log | All stages complete, log entry written | Integration |
| T8.5 | Run pipeline 7 days simulated (daily batches) | Topic count stabilizes, no runaway growth | Stability over time |

---

## Failure Modes We're Explicitly Guarding Against

| Failure | How It Happens | Guard |
|---------|---------------|-------|
| **Topic duplication** | HDBSCAN creates cluster that overlaps existing topic | T4.3: merge threshold check against existing centroids |
| **Centroid hijacking** | Many off-topic summaries slowly drift a centroid | T3.6: EMA weight limits drift rate; T5.3: periodic full recalc |
| **Orphan cascade** | LCM mass-recompacts, 80% of references break | T6.4/T6.5: alert threshold + centroid recalc, never auto-delete topics |
| **Lock contention** | Writing epistemic.db while reading lcm.db blocks gateway | T7.1: lcm.db always opened read-only; separate DB files |
| **False positive tags** | Low threshold tags everything, noise becomes signal | T3.7/T3.8: threshold boundary tests; T8.1: regression vs POC |
| **Runaway topics** | Bad HDBSCAN params create 200 micro-clusters | T4.6: max topics per run cap |
| **Stale centroids** | Topics that haven't received new summaries in months | Future: topic decay (Phase 3), not Phase 1 concern |
| **Self-contamination** | Our own pipeline logs/queries get tagged as topics | Read-only from lcm.db + pipeline runs in cron, not in conversation |

---

## File Structure

```
projects/epistemic-synthesis/
├── whitepaper-v0.3.md          # architecture doc
├── phase1-plan.md              # this file
├── src/
│   ├── schema.py               # DB initialization
│   ├── embed.py                # nomic-embed client
│   ├── tagger.py               # real-time tagging pipeline
│   ├── discovery.py            # HDBSCAN topic discovery
│   ├── orphans.py              # reconciliation
│   ├── labels.py               # LLM topic labeling
│   └── config.py               # all thresholds and paths
├── tests/
│   ├── test_schema.py          # T1.*
│   ├── test_embed.py           # T2.*
│   ├── test_tagger.py          # T3.*
│   ├── test_discovery.py       # T4.*
│   ├── test_centroids.py       # T5.*
│   ├── test_orphans.py         # T6.*
│   ├── test_concurrency.py     # T7.*
│   └── test_e2e.py             # T8.*
└── scripts/
    ├── topic-discovery-test.py  # existing POC (reference)
    └── topic-label-pass.py      # existing POC (reference)
```

---

## Implementation Order

1. `config.py` + `schema.py` — get the DB right first
2. `embed.py` — thin wrapper around nomic-embed HTTP API
3. `tests/test_schema.py` + `tests/test_embed.py` — run green before proceeding
4. `tagger.py` — the core loop: fetch untagged → embed → match → tag
5. `tests/test_tagger.py` — run against snapshot of real lcm.db
6. `discovery.py` — port HDBSCAN from POC, add merge-with-existing check
7. `tests/test_discovery.py` — regression against POC results (19 clusters)
8. `orphans.py` — reconciliation pass
9. `tests/test_orphans.py` — simulate LCM re-compaction
10. `labels.py` — LLM labeling (port from POC)
11. `tests/test_e2e.py` — full pipeline on real data
12. Cron entry: tagging every 30 min, discovery daily, orphans weekly

---

## Definition of Done (Phase 1)

- [ ] epistemic.db created and populated from current lcm.db snapshot
- [ ] All 168 existing summaries tagged or classified as noise
- [ ] Topic count within ±3 of POC result (19 clusters)
- [ ] All test suites passing (T1–T8)
- [ ] Cron entries installed and running without errors for 48h
- [ ] tagging_log shows consistent runs with zero crashes
- [ ] lcm.db never written to (verified by `lsof` during runs)
- [ ] No gateway performance impact (verified by response time comparison)
