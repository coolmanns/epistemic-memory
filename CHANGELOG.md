# Changelog

## [0.2.0] — 2026-03-10

### Phase 2: Claim-First Synthesis Pipeline

**Added:**
- `src/extractor.py` — Atomic claim extraction from tagged summaries (Sonnet via Anthropic SDK)
- `src/dedup.py` — Claim deduplication via embedding cosine similarity
- `src/verifier.py` — Two-pass claim verification (verdicts: verified, overstated, unsupported, misattributed)
- `src/synthesizer.py` — Canonical synthesis + injection brief generation from verified claims
- `src/writer.py` — Markdown output writer with filename sanitization
- `src/decay.py` — Claim-level confidence decay processor (HIGH → MED → LOW → decayed)
- `src/llm.py` — Shared Anthropic SDK wrapper (replaces per-module requests.post calls)
- `tests/test_extractor.py` — 19 tests
- `tests/test_dedup.py` — 13 tests
- `tests/test_verifier.py` — 11 tests
- `tests/test_synthesizer.py` — 12 tests
- `tests/test_writer.py` — 14 tests
- `tests/test_decay.py` — 8 tests
- `tests/test_pipeline.py` — 4 integration tests
- `phase2-plan.md` — Full build plan with 75 test cases across 9 categories
- Phase 2 schema: `claims`, `claim_sources`, `syntheses`, `synthesis_runs` tables

**Changed:**
- `src/config.py` — Added Phase 2 settings (LLM model, synthesis thresholds, output dir, decay half-life)
- `src/schema.py` — Phase 2 table migrations
- `src/runner.py` — Added extract/dedup/verify/synthesize/write/decay commands
- `src/embed.py` — Batch size reduced 32→8 for timeout safety with larger models
- Embedding engine: Qwen3-Embedding-4B (2560-dim) replaces nomic-embed-text (768-dim)
- LLM routing: Sonnet (Anthropic API) for extraction/verification/synthesis; Qwen3-30B (local) for labeling only
- `max_tokens` default bumped 4096→8192 to prevent JSON truncation on large claim sets
- `MAX_CLAIMS_PER_EXTRACTION` reduced 50→15 (output fits within token limits)

**First real run:**
- 188 summaries → 17 topics → 139 claims → 41 deduplicated → 78 verified → 16 synthesis docs

## [0.1.0] — 2026-03-10

### Phase 1: Topic Discovery Pipeline

**Added:**
- `src/schema.py` — SQLite schema with WAL mode (topics, topic_summaries, topic_edges, tagging_log)
- `src/config.py` — Environment-based configuration
- `src/embed.py` — Embedding client with batch support and blob serialization
- `src/tagger.py` — Cosine similarity tagging against topic centroids with EMA updates
- `src/discovery.py` — HDBSCAN clustering for bottom-up topic discovery
- `src/orphans.py` — Orphan detection and reconciliation for re-compacted summaries
- `src/labels.py` — LLM-powered topic labeling (Qwen3-30B local)
- `src/runner.py` — CLI runner (tag/discover/orphan/full modes)
- Full test suite: 65 tests across schema, embed, tagger, discovery, orphan, and E2E
- `phase1-plan.md` — Build plan with 42 test cases

**Infrastructure:**
- Komodo stack `llama-epistemic` (Qwen3-Embedding-4B-Q6_K on port 8086)
- Cron schedule: tag every 30min, discover daily 3AM, orphans Sunday 4AM
- Persistent DB at `~/.openclaw/data/epistemic.db`

**First real run:**
- 188 summaries → 17 topics discovered, 153 tagged (81%), 35 noise (19%)
