# Changelog

## [0.3.0] — 2026-03-10

### Topic Seeding, Re-tagging, Contradiction Detection, Contamination Fix

**Added:**
- `src/retagger.py` — Two-tier re-tagging: full-scan for newly seeded topics, neighbor-scan for ongoing maintenance
- `src/contradictions.py` — Pairwise contradiction detection with LLM classification (direct conflict / temporal evolution / compatible / nuance difference)
- `tests/test_retagger.py` — 11 tests
- `tests/test_contradictions.py` — 12 tests
- Runner modes: `retag` (with optional `--topic-id`), `contradictions`
- `PROJECT.md` — Cold-start project documentation

**Fixed:**
- **Topic contamination in extraction** — Multi-topic summaries caused cross-topic claim bleed. Fox Valley Plumbing synthesis was 10/13 Martin Ball claims. Added topic-awareness constraint to extraction prompt: `"Only extract claims SPECIFICALLY about {topic_label}"`. Re-extracted all seeded topics clean.
- **Centroid seeding from labels** — Seeded topic centroids built from label text only ("Martin Ball" = 2 words) produced 0.2–0.4 cosine similarity against real content. Fix: rebuild centroids from actual matching summaries via keyword search. Result: 441 tags (was 3).
- `contradictions.py` early return missing `duration_ms` → KeyError in runner
- `contradictions.py` `_load_claims` filtered `status='active'` only, excluding all 523 verified claims. Fixed to `IN ('active', 'verified')`.

**Changed:**
- `src/extractor.py` — Extraction prompt now scopes claims to topic label, with explicit instruction to ignore other entities in multi-topic summaries
- `src/runner.py` — Added `retag` and `contradictions` modes

**Seeded topics (7):**
Martin Ball, Fox Valley Plumbing, Ripples of Impact, Adult in Training, AI Memory Architecture, Madospeakers, Xandeum

**Production results after fix:**
- 24 topics, ~735 claims (active/verified), 61 syntheses, 233 contradictions
- All 7 seeded topics synthesized clean — zero cross-topic contamination

**Design rules established:**
1. Always seed centroids from content, never from label text alone
2. Extraction prompts must include topic-scoping constraint for multi-topic summary sources

## [0.2.1] — 2026-03-10

### Tuning: Tagger Sensitivity + Pipeline Automation

**Problem observed:**
- Tagger running every 30 min but classifying 97% of summaries as noise
- Only 5 claims tagged out of 171 summaries processed across 4 tagging passes
- Discovery (3 AM) found 0 clusters from 36 noise points — too few similar summaries to cluster
- Synthesis pipeline (extract → dedup → verify → synthesize → write) had NO cron — only ran manually
- Net effect: epistemic DB frozen at 486 claims, 17 topics since initial build

**Root cause analysis:**
1. Tagger uses pure embedding cosine similarity (no LLM) — threshold of 0.72 too strict for diverse daily work
2. 17 existing topics heavily biased toward initial build context (image handling, OAuth, Ghost mockups)
3. New work domains (client research, business strategy, plumbing SEO) have no matching topic centroids
4. HDBSCAN min_cluster_size=3 too high — daily work produces 1-2 summaries per niche topic
5. Discovery runs once daily — not enough accumulation time for diverse topics to cluster

**Changed:**
- `SIMILARITY_THRESHOLD`: 0.72 → 0.65 (let near-misses tag against adjacent topics; tagged range was 0.59–0.99, avg 0.86)
- `HDBSCAN_MIN_CLUSTER_SIZE`: 3 → 2 (catch topic pairs from diverse daily work)
- Discovery cron: daily 3 AM → twice daily 3 AM + 3 PM (catch daytime work before it fragments)
- Added `full-synthesis` cron: daily 4 AM (extract → dedup → verify → synthesize → write)
- Orphan cron: Sun 4 AM → Sun 5 AM (runs after synthesis completes)

**New cron schedule:**
```
*/30 * * * *  tag              (unchanged)
0 3,15 * * *  discover         (was: 0 3 * * *)
0 4 * * *     full-synthesis   (NEW)
0 5 * * 0     orphan           (was: 0 4 * * 0)
```

**Expected impact:**
- Lower similarity threshold → more summaries tag against existing topics
- Smaller cluster size → discovery creates topics from 2+ related summaries
- Twice-daily discovery → new topics emerge same-day instead of next-morning
- Automated synthesis → claims extracted, verified, and synthesized overnight without manual runs

**Monitoring plan:**
- Check noise rate in tagging_log after 24h — target <80% (was 97%)
- Check topic count growth after 3 discovery passes
- Check synthesis_runs for new entries after first 4 AM full-synthesis
- If threshold 0.65 floods low-quality tags: raise to 0.68

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
