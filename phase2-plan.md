# Epistemic Synthesis — Phase 2: Claim-First Synthesis

**Goal:** Extract atomic claims from tagged topic summaries, verify them, generate three artifacts per topic (Evidence Ledger, Canonical Synthesis, Injection Brief).
**Constraint:** Read-only lcm.db. All writes to epistemic.db. LLM calls must be idempotent and retryable.
**Output:** `memory/topics/{topic-label}.md` files for memorySearch injection.

---

## Architecture

### Claim-First Pipeline (per topic)

```
Summaries (tagged in Phase 1)
  → Claim Extraction (Qwen3-30B local, port 8084)
    → Claim Deduplication (embed + cosine, Qwen3-Embedding-4B port 8086)
      → Claim Verification (Sonnet via API)
        → Canonical Synthesis Generation (Sonnet via API)
          → Injection Brief Generation (Sonnet via API)
            → Write to memory/topics/{label}.md
```

### LLM Routing

| Task | Model | Location | Why |
|------|-------|----------|-----|
| Claim extraction | Qwen3-30B | localhost:8084 | Structured, mechanical — extract don't reason |
| Claim dedup labels | Qwen3-30B | localhost:8084 | Simple similarity judgment |
| Claim embedding | Qwen3-Embedding-4B | localhost:8086 | Cosine similarity for dedup |
| Claim verification | Sonnet | OpenClaw API | Needs reasoning about grounding |
| Canonical synthesis | Sonnet | OpenClaw API | Needs epistemic register, nuance |
| Injection brief | Sonnet | OpenClaw API | Compression with quality |

### Three Artifacts

| Artifact | Purpose | Size Target | Updated When |
|----------|---------|-------------|--------------|
| **Evidence Ledger** | Source of truth — atomic claims with provenance | Grows over time | Every claim extraction pass |
| **Canonical Synthesis** | Structured understanding in epistemic register | 800-2000 tokens | When claim set changes materially |
| **Injection Brief** | Runtime context — compressed, query-optimized | 200-400 tokens | When canonical synthesis updates |

---

## New Schema (extend epistemic.db)

```sql
-- Atomic claims extracted from summaries
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY,
    topic_id INTEGER NOT NULL REFERENCES topics(id),
    text TEXT NOT NULL,                    -- single-sentence assertion
    claim_type TEXT NOT NULL DEFAULT 'factual',  -- factual|interpretive|analogical
    confidence TEXT NOT NULL DEFAULT 'MED', -- HIGH|MED|LOW
    status TEXT NOT NULL DEFAULT 'active',  -- active|verified|unsupported|overstated|contradicted|decayed|superseded
    source_count INTEGER DEFAULT 1,
    first_seen TEXT NOT NULL,
    last_reinforced TEXT NOT NULL,
    embedding BLOB,                         -- float32 vector for dedup
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Provenance: which summaries support each claim
CREATE TABLE IF NOT EXISTS claim_sources (
    id INTEGER PRIMARY KEY,
    claim_id INTEGER NOT NULL REFERENCES claims(id),
    summary_id TEXT NOT NULL,              -- references lcm.db summary_id
    excerpt TEXT NOT NULL,                 -- verbatim source passage
    extracted_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(claim_id, summary_id)
);

-- Explicit contradiction tracking
CREATE TABLE IF NOT EXISTS claim_contradictions (
    id INTEGER PRIMARY KEY,
    claim_a_id INTEGER NOT NULL REFERENCES claims(id),
    claim_b_id INTEGER NOT NULL REFERENCES claims(id),
    resolution TEXT,                       -- null = unresolved, text = how resolved
    resolved_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK(claim_a_id < claim_b_id)        -- canonical ordering, no duplicates
);

-- Synthesis artifacts (versioned per topic)
CREATE TABLE IF NOT EXISTS syntheses (
    id INTEGER PRIMARY KEY,
    topic_id INTEGER NOT NULL REFERENCES topics(id),
    version INTEGER NOT NULL DEFAULT 1,
    canonical_text TEXT NOT NULL,          -- full epistemic register document
    injection_brief TEXT NOT NULL,         -- compressed runtime version
    claim_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(topic_id, version)
);

-- Which claims fed into which synthesis
CREATE TABLE IF NOT EXISTS synthesis_claims (
    synthesis_id INTEGER NOT NULL REFERENCES syntheses(id),
    claim_id INTEGER NOT NULL REFERENCES claims(id),
    weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY(synthesis_id, claim_id)
);

-- Track synthesis runs
CREATE TABLE IF NOT EXISTS synthesis_runs (
    id INTEGER PRIMARY KEY,
    topic_id INTEGER,                     -- null = all topics
    phase TEXT NOT NULL,                  -- extract|dedup|verify|synthesize|brief
    stats TEXT,                           -- JSON blob
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    duration_ms INTEGER
);
```

---

## Components

### S1: Schema Migration (`schema.py` extension)
- Add new tables to existing `init_epistemic_db()`
- Must be additive — don't touch Phase 1 tables
- Idempotent (IF NOT EXISTS everywhere)

### S2: Claim Extractor (`extractor.py`)
- Input: topic_id → fetch all tagged summaries from lcm.db
- For each summary: send to Qwen3-30B with extraction prompt
- Parse structured output: claim text, type, source excerpt, contradictions
- Insert into `claims` + `claim_sources`
- Skip summaries already processed (track in synthesis_runs or claim_sources)
- **Incremental:** only process summaries tagged since last extraction run

### S3: Claim Deduplicator (`dedup.py`)
- Embed all claims for a topic via Qwen3-Embedding-4B
- Cosine similarity matrix within topic
- Merge claims above threshold (0.90): keep older, increment source_count, merge sources
- Cross-topic dedup: flag (don't auto-merge) claims that appear in multiple topics
- Output: deduplicated claim set with merged provenance

### S4: Claim Verifier (`verifier.py`)
- Input: claims with status='active' for a topic
- For each claim: send to Sonnet with source excerpts
- Sonnet outputs: VERIFIED | UNSUPPORTED | OVERSTATED | MISATTRIBUTED
- Update claim status accordingly
- UNSUPPORTED claims stay in evidence ledger, excluded from synthesis
- Only VERIFIED claims feed into synthesis generation
- **Rate limiting:** batch claims per topic, serialize across topics

### S5: Synthesizer (`synthesizer.py`)
- Input: verified claims for a topic, grouped by confidence
- Generate canonical synthesis in epistemic register (Sonnet)
- Generate injection brief from canonical (Sonnet, separate call)
- Insert into `syntheses` + `synthesis_claims`
- Version numbering: auto-increment per topic

### S6: Output Writer (`writer.py`)
- Read latest injection brief per topic from `syntheses`
- Write to `memory/topics/{sanitized-label}.md`
- Include metadata header: topic_id, version, claim_count, generated_at
- These files are what memorySearch indexes for runtime injection

### S7: Runner Integration (`runner.py` extension)
- New commands: `extract`, `dedup`, `verify`, `synthesize`, `full-synthesis`
- `full-synthesis` = extract → dedup → verify → synthesize → write (per topic)
- Cron: nightly `full-synthesis` for topics with new tagged summaries since last run

---

## Configuration Additions

```python
# Phase 2: Synthesis
CLAIM_DEDUP_THRESHOLD = 0.90        # cosine similarity to merge claims
MIN_CLAIMS_FOR_SYNTHESIS = 3        # don't synthesize topics with <3 verified claims
MAX_CLAIMS_PER_EXTRACTION = 50      # cap claims per summary batch (safety)
SONNET_MODEL = "anthropic/claude-sonnet-4-6"
QWEN_CHAT_URL = "http://localhost:8084"
QWEN_CHAT_MODEL = "qwen3-30b"
SYNTHESIS_OUTPUT_DIR = "~/clawd/memory/topics"
CLAIM_DECAY_HIGH_DAYS = 90          # HIGH → MED after 90 days without reinforcement
CLAIM_DECAY_MED_DAYS = 60           # MED → LOW after 60 more days
CLAIM_DECAY_LOW_DAYS = 60           # LOW → decayed after 60 more days
```

---

## Test Cases

### T9: Schema Migration
| ID | Test | Expected | Risk Mitigated |
|----|------|----------|----------------|
| T9.1 | Add Phase 2 tables to existing epistemic.db with Phase 1 data | New tables created, Phase 1 data intact | Migration safety |
| T9.2 | Run migration twice | Idempotent — no errors, no data loss | Re-run safety |
| T9.3 | Foreign key: claim references nonexistent topic_id | INSERT rejected (FK constraint) | Referential integrity |
| T9.4 | Unique constraint: same claim_id + summary_id in claim_sources | Second INSERT rejected | Duplicate provenance |
| T9.5 | Unique constraint: same topic_id + version in syntheses | Second INSERT rejected | Duplicate synthesis versions |
| T9.6 | claim_contradictions CHECK constraint: claim_a_id < claim_b_id | Reversed pair rejected | Canonical ordering |

### T10: Claim Extraction
| ID | Test | Expected | Risk Mitigated |
|----|------|----------|----------------|
| T10.1 | Extract claims from a single summary with clear content | 3-8 atomic claims with type + excerpt | Happy path |
| T10.2 | Extract from summary with no substantive content (e.g. "heartbeat ok") | 0 claims, no error | Empty input |
| T10.3 | Extract from summary with contradictory information | Claims extracted with contradiction flags | Contradiction detection |
| T10.4 | Same summary extracted twice | Second run skips (already processed) | Idempotent extraction |
| T10.5 | Extraction with Qwen3 down (port 8084 unreachable) | Graceful error, logged, no partial writes | Infra failure |
| T10.6 | Extraction produces >50 claims from one batch | Capped at MAX_CLAIMS_PER_EXTRACTION, warning logged | Runaway extraction |
| T10.7 | Extraction prompt returns malformed JSON | Parse error caught, summary skipped, logged | LLM output variance |
| T10.8 | Extract from topic with 0 tagged summaries | No-op, logged | Edge case |
| T10.9 | Incremental: only new summaries since last run processed | Old summaries skipped, new ones extracted | Efficiency |
| T10.10 | Claim text is empty string | Rejected, not inserted | Data quality |

### T11: Claim Deduplication
| ID | Test | Expected | Risk Mitigated |
|----|------|----------|----------------|
| T11.1 | Two identical claims (cosine > 0.95) | Merged: older kept, source_count incremented, sources merged | Happy path dedup |
| T11.2 | Two similar but distinct claims (cosine 0.85) | Both kept (below 0.90 threshold) | False merge prevention |
| T11.3 | Claim A is subset of Claim B (A: "X uses Y", B: "X uses Y for Z") | Both kept — semantic similarity may be high but they're distinct assertions | Over-merging |
| T11.4 | Same claim appears in two different topics | Flagged as cross-topic duplicate, not auto-merged | Cross-topic awareness |
| T11.5 | Dedup with no claims in topic | No-op, logged | Empty topic |
| T11.6 | Dedup 100+ claims in one topic | Completes within 60s, correct merge count | Performance |
| T11.7 | Merge preserves all source excerpts from both claims | Merged claim has union of claim_sources | Provenance preservation |
| T11.8 | Merge updates source_count correctly | source_count = sum of merged claims' source_counts | Counter accuracy |
| T11.9 | Already-merged claim gets new duplicate | Re-merges cleanly, source_count increments again | Iterative dedup |
| T11.10 | Dedup threshold = 1.0 (nothing merges) | All claims kept | Threshold boundary |
| T11.11 | Dedup threshold = 0.0 (everything merges) | One claim per topic | Threshold boundary |

### T12: Claim Verification
| ID | Test | Expected | Risk Mitigated |
|----|------|----------|----------------|
| T12.1 | Claim with clear source support | Status → VERIFIED, confidence maintained | Happy path |
| T12.2 | Claim that overgeneralizes from source | Status → OVERSTATED | False abstraction catch |
| T12.3 | Claim with no supporting excerpt match | Status → UNSUPPORTED | Hallucinated claim catch |
| T12.4 | Claim that misattributes source | Status → MISATTRIBUTED | Attribution error catch |
| T12.5 | Verify same claim twice | Idempotent — status doesn't flip without new evidence | Re-run stability |
| T12.6 | Verification with Sonnet API down | Graceful error, claims stay at current status, logged | API failure |
| T12.7 | Batch of 20 claims verified in one run | All 20 get status updates, rate limits respected | Batch processing |
| T12.8 | Verification of analogical claim (highest scrutiny) | Requires stronger evidence than factual claim | Type-aware verification |
| T12.9 | UNSUPPORTED claim excluded from synthesis input | Claim stays in evidence ledger, not in synthesis | Pipeline integration |
| T12.10 | Contradiction pair: both claims have evidence | Both flagged in claim_contradictions, neither auto-resolved | Genuine tension preservation |

### T13: Synthesis Generation
| ID | Test | Expected | Risk Mitigated |
|----|------|----------|----------------|
| T13.1 | Generate canonical synthesis from 10 verified claims | Epistemic register doc, 800-2000 tokens, all claim IDs referenced | Happy path |
| T13.2 | Generate with mixed confidence levels | HIGH/MED/LOW sections present, tensions section if contradictions exist | Structure compliance |
| T13.3 | Generate with <3 verified claims | Skipped (below MIN_CLAIMS_FOR_SYNTHESIS), logged | Premature synthesis |
| T13.4 | Generate injection brief from canonical | 200-400 tokens, no low-confidence claims, no provenance | Compression quality |
| T13.5 | Version increments correctly | v1 → v2 on re-synthesis, both stored | Version tracking |
| T13.6 | Synthesis references only VERIFIED claims | No UNSUPPORTED/OVERSTATED claims appear in text | Grounding enforcement |
| T13.7 | Synthesis with Sonnet API down | Graceful error, no partial synthesis written | API failure |
| T13.8 | Re-synthesis after 5 new claims added | New version generated, includes new claims | Incremental update |
| T13.9 | synthesis_claims table populated correctly | Every claim ID in canonical text has a row in synthesis_claims | Provenance tracking |
| T13.10 | Canonical text uses epistemic register (no first-person) | No "I believe", "I think" — uses "established patterns", "recurring themes" | Voice compliance |

### T14: Output Writer
| ID | Test | Expected | Risk Mitigated |
|----|------|----------|----------------|
| T14.1 | Write injection brief to memory/topics/{label}.md | File created with metadata header + brief content | Happy path |
| T14.2 | Topic label with special characters ("OAuth & GCP") | Sanitized filename (oauth-gcp.md), no path traversal | Filename safety |
| T14.3 | Overwrite existing file (re-synthesis) | File updated, old content replaced | Update path |
| T14.4 | Topic with no synthesis yet | No file written, no error | Pre-synthesis state |
| T14.5 | Write 17 topic files in one run | All 17 created, no collisions | Batch write |
| T14.6 | Output directory doesn't exist | Created automatically | First-run setup |
| T14.7 | Metadata header includes topic_id, version, claim_count, generated_at | All fields present and correct | Traceability |

### T15: Pipeline Integration
| ID | Test | Expected | Risk Mitigated |
|----|------|----------|----------------|
| T15.1 | full-synthesis on topic with 10+ summaries | extract → dedup → verify → synthesize → write completes | End-to-end |
| T15.2 | full-synthesis on topic with 0 new summaries since last run | Skips extraction, re-runs verify + synthesize if claims exist | Efficiency |
| T15.3 | full-synthesis on all 17 topics sequentially | All complete, rate limits respected, <30 min total | Batch runtime |
| T15.4 | Pipeline interrupted mid-verification | Next run resumes cleanly (no duplicate claims, no corrupt state) | Crash recovery |
| T15.5 | Pipeline produces consistent results on same input | Same claims, same synthesis structure (content may vary slightly due to LLM) | Reproducibility |
| T15.6 | Runner logs synthesis_runs for each phase | All phases logged with duration_ms and stats JSON | Observability |
| T15.7 | lcm.db mtime unchanged after full pipeline | Read-only access verified | Safety invariant |

### T16: Claim Decay
| ID | Test | Expected | Risk Mitigated |
|----|------|----------|----------------|
| T16.1 | Claim at HIGH, not reinforced for 90 days | Downgraded to MED | Stale confidence prevention |
| T16.2 | Claim at MED, not reinforced for 60 days | Downgraded to LOW | Decay cascade |
| T16.3 | Claim at LOW, not reinforced for 60 days | Status → decayed | Full decay |
| T16.4 | Decayed claim excluded from injection brief | Not present in latest synthesis output | Stale carryover prevention |
| T16.5 | Decayed claim still in evidence ledger | Row exists, queryable, status = decayed | Audit trail preservation |
| T16.6 | Claim reinforced by new evidence resets decay timer | last_reinforced updated, confidence maintained | Reinforcement works |
| T16.7 | Decay run on empty claims table | No-op, no errors | Edge case |

### T17: Error Handling & Safety
| ID | Test | Expected | Risk Mitigated |
|----|------|----------|----------------|
| T17.1 | Qwen3 returns partial JSON | Claim batch skipped, error logged, other batches continue | Partial LLM failure |
| T17.2 | Sonnet 429 rate limit | Exponential backoff, retry up to 3 times | Rate limiting |
| T17.3 | Embedding service returns wrong dimension | Detected, batch rejected | Model mismatch |
| T17.4 | Claim text > 500 characters | Truncated or rejected with warning | Oversized claims |
| T17.5 | Topic has 500+ claims after extraction | System handles gracefully, dedup reduces to manageable set | Scale ceiling |
| T17.6 | Database WAL checkpoint during write | No corruption, transaction integrity | Concurrent access |
| T17.7 | Synthesis output > 3000 tokens | Warning logged, truncated before injection brief generation | Oversized output |

---

## Failure Modes We're Explicitly Guarding Against

| Failure | How It Happens | Guard |
|---------|---------------|-------|
| **False abstraction** | LLM generates claim that goes beyond source material | T12.2: verification catches OVERSTATED claims |
| **Premature convergence** | Synthesis sounds certain when evidence is thin | T13.3: min claims threshold; confidence levels per claim |
| **Contradiction laundering** | LLM picks one side of a genuine tension silently | T12.10: contradictions explicit in claim graph |
| **Hallucinated claims** | Extraction produces claims not in source | T12.3: verification catches UNSUPPORTED |
| **Stale high-confidence carryover** | Old well-supported claim persists past evidence shift | T16.1-T16.3: claim-level decay |
| **Provenance loss** | Can't trace synthesis back to source conversation | T11.7: merge preserves all sources; T13.9: synthesis_claims populated |
| **Self-reinforcing drift** | Synthesis becomes input to future synthesis | Claims always extracted from LCM summaries (raw), never from own synthesis docs |
| **Over-merging** | Dedup merges distinct but similar claims | T11.3: threshold at 0.90 prevents most; T11.4: cross-topic flagging |
| **Rate limit cascade** | Sonnet calls for 17 topics × 50 claims overwhelm API | T17.2: exponential backoff; serialize across topics |

---

## File Structure (additions to Phase 1)

```
projects/epistemic-synthesis/
├── src/
│   ├── schema.py               # EXTEND: add Phase 2 tables
│   ├── config.py               # EXTEND: add Phase 2 config
│   ├── extractor.py            # NEW: claim extraction (Qwen3-30B)
│   ├── dedup.py                # NEW: claim deduplication (embeddings)
│   ├── verifier.py             # NEW: claim verification (Sonnet)
│   ├── synthesizer.py          # NEW: canonical + brief generation (Sonnet)
│   ├── writer.py               # NEW: memory/topics/*.md output
│   ├── decay.py                # NEW: claim decay processor
│   └── runner.py               # EXTEND: add synthesis commands
├── tests/
│   ├── test_schema.py          # EXTEND: T9.*
│   ├── test_extractor.py       # NEW: T10.*
│   ├── test_dedup.py           # NEW: T11.*
│   ├── test_verifier.py        # NEW: T12.*
│   ├── test_synthesizer.py     # NEW: T13.*
│   ├── test_writer.py          # NEW: T14.*
│   ├── test_pipeline.py        # NEW: T15.*
│   ├── test_decay.py           # NEW: T16.*
│   └── test_errors.py          # NEW: T17.*
└── memory/
    └── topics/                 # OUTPUT: injection briefs per topic
```

---

## Implementation Order

1. `schema.py` extension + `test_schema.py` T9.* → green
2. `config.py` additions (new constants)
3. `extractor.py` + `test_extractor.py` T10.* → green
4. `dedup.py` + `test_dedup.py` T11.* → green
5. `verifier.py` + `test_verifier.py` T12.* → green
6. `synthesizer.py` + `test_synthesizer.py` T13.* → green
7. `writer.py` + `test_writer.py` T14.* → green
8. `decay.py` + `test_decay.py` T16.* → green
9. `runner.py` extension + `test_pipeline.py` T15.* → green
10. Error handling sweep + `test_errors.py` T17.* → green
11. Single topic proof-of-concept: run full pipeline on "Epistemic Synthesis" topic
12. All 17 topics: batch run, review output quality
13. Cron entry: nightly full-synthesis for topics with new data

---

## Definition of Done (Phase 2)

- [ ] All Phase 2 tables created, Phase 1 data intact
- [ ] Claims extracted from all 17 topics
- [ ] Deduplication reduces claim count by >10% (validates it's working)
- [ ] Verification produces non-trivial distribution (not 100% VERIFIED — that means verification is rubber-stamping)
- [ ] Canonical synthesis for each topic reads naturally in epistemic register
- [ ] Injection briefs are 200-400 tokens, no low-confidence claims
- [ ] memory/topics/*.md files written and indexable by memorySearch
- [ ] All test suites passing (T9–T17: ~75 tests)
- [ ] Nightly cron running for 48h without errors
- [ ] Claim decay logic tested with time-simulated data
- [ ] No self-referencing: synthesis docs never appear as input to claim extraction
- [ ] lcm.db never written to (invariant from Phase 1, still holds)
