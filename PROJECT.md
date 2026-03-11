# Epistemic Memory — PROJECT.md

## What It Is

Claim-first epistemic synthesis for personal AI agents. Discovers topics from conversation history (LCM summaries), extracts and verifies atomic claims, detects contradictions, and synthesizes evolving understanding documents per topic. Runs as automated cron pipeline on OpenClaw.

## Current State

**Phase 1 — Topic Discovery**: ✅ Complete, running in production
**Phase 2 — Claim Extraction & Synthesis**: ✅ Complete, running in production
**Phase 3 — Context Integration**: 🔲 Not started (synthesis docs feed back into conversation context)
**Phase 7 — Associative Network**: 🔲 Not started (topic_edges table exists, 0 rows)

### Production Numbers (2026-03-10)
- 24 topics (17 discovered, 7 manually seeded)
- ~735 claims (active/verified after contamination purge + re-extract)
- 61 syntheses across 22 topics
- 233 contradictions logged (206 direct, 12 temporal evolution, 15 nuance)
- LCM coverage: March 8–10 only (230 summaries). Pre-LCM: 456 JSONL session files (Jan 30–now, 370MB)

### Pipeline Modules
| Module | Purpose | LLM |
|--------|---------|-----|
| tagger.py | Cosine similarity tagging against topic centroids | Qwen3-Embedding-4B (local) |
| discovery.py | HDBSCAN clustering for new topic detection | Qwen3-Embedding-4B + Qwen3-30B (local) |
| extractor.py | Atomic claim extraction from tagged summaries | Claude Sonnet (API) |
| dedup.py | Claim deduplication via embedding similarity | Qwen3-Embedding-4B (local) |
| verifier.py | Two-pass claim verification | Claude Sonnet (API) |
| synthesizer.py | Canonical synthesis + injection brief | Claude Sonnet (API) |
| writer.py | Markdown output per topic | None |
| decay.py | Confidence decay (HIGH→MED→LOW→decayed) | None |
| retagger.py | Full-scan re-tagging for newly seeded topics | Qwen3-Embedding-4B (local) |
| contradictions.py | Pairwise contradiction detection + classification | Claude Sonnet (API) |
| orphans.py | Broken reference detection from LCM re-compaction | None |
| trace.py | Two-tier scoped search with provenance | Qwen3-Embedding-4B (local) |
| labels.py | LLM-generated topic labels | Qwen3-30B (local) |

### Cron Schedule
```
*/30 * * * *  tag              (every 30 min)
0 3,15 * * *  discover         (twice daily)
0 4 * * *     full-synthesis   (daily — extract → dedup → verify → synthesize → write)
0 5 * * 0     orphan           (weekly Sunday)
```

## Configuration

- **DB**: `~/.openclaw/data/epistemic.db` (SQLite, WAL mode)
- **LCM DB**: `~/.openclaw/lcm.db` (read-only, summaries source)
- **Embedding server**: localhost:8086 (Qwen3-Embedding-4B-Q6_K via llama.cpp, Komodo stack `llama-epistemic`)
- **Local LLM**: localhost:8080 (Qwen3-30B for labeling only)
- **Anthropic API**: Claude Sonnet for extraction, verification, synthesis, contradictions
- **Output**: `~/.openclaw/memory/topics/` (per-topic markdown files)
- **Config**: `src/config.py` — all thresholds, model names, paths

### Key Thresholds
- `SIMILARITY_THRESHOLD`: 0.65 (tagging — was 0.72, lowered 2026-03-10)
- `HDBSCAN_MIN_CLUSTER_SIZE`: 2 (was 3)
- `MAX_CLAIMS_PER_EXTRACTION`: 15 per LLM call
- Dedup cosine threshold: 0.85
- Decay half-life: configurable per claim type

## Architecture

```
LCM Summaries (lcm.db)
    ↓ tagger (cosine similarity to topic centroids)
topic_summaries (epistemic.db)
    ↓ extractor (Claude Sonnet, topic-aware prompt)
claims + claim_sources
    ↓ dedup (embedding similarity merge)
    ↓ verifier (Claude Sonnet two-pass)
    ↓ synthesizer (Claude Sonnet)
syntheses (canonical_text + injection_brief)
    ↓ writer
~/.openclaw/memory/topics/*.md
```

**Critical dependency**: topic_summaries stores only summary_id as FK to lcm.db. Extractor fetches content from lcm.db at extraction time. If summary_id missing from lcm.db, extractor silently skips.

## Key Decisions

1. **Topic-aware extraction prompt** (2026-03-10): Extraction prompt now includes `"Only extract claims SPECIFICALLY about {topic_label}"`. Without this, multi-topic summaries cause cross-topic contamination — e.g., Fox Valley Plumbing synthesis was 10/13 Martin Ball claims.

2. **Seed centroids from content, not labels** (2026-03-10): 2-word label embeddings produce 0.2–0.4 similarity against real content. Must build centroids from actual matching summaries via keyword search.

3. **Checkpoint + delta model** (2026-03-10): Files are checkpoints (crystallize knowledge to last-modified date). Conversations after that date are the delta. Accepted for multi-source ingestion design. Source weighting model rejected.

4. **Claims stay multi-tagged, specificity migrates** (2026-03-10): A Martin Ball + SEO summary is legitimately both topics. But topic-specific claims should gravitate to the more specific topic as it grows. Not yet implemented.

5. **Wait for upstream LCM backfill** (2026-03-10): Don't hack direct SQLite insertion into lcm.db — risks seq numbering, FTS index, DAG integrity. lossless-claw issues #5 and #18 track this.

6. **15-claim cap is known limitation** (2026-03-10): Extractor sends all topic summaries in one LLM call with MAX_CLAIMS=15. Content-rich topics lose granularity. Task #119 tracks batched extraction fix.

## Open Tasks

- **Task #117**: Evaluate organic topic discovery ~March 17. If psychedelic/AIT topics haven't emerged organically, consider manual seeding or full pipeline rerun.
- **Task #118**: Design file-system/session JSONL ingestion. Blocked pending lossless-claw backfill feature (issues #5, #18).
- **Task #119**: Batched extraction — send summaries in groups of 10-15 to fix 15-claim cap for content-rich topics.
- **Contradiction detection triage**: 206 "direct" contradictions likely includes cross-topic bleed and extractor drift. Needs review.
- **Phase 7 — Associative Network**: topic_edges table exists with 0 rows. Needs topic diversity first (now seeded).
- **AI Memory Architecture topic**: Extraction failed (LLM JSON parse error). Needs retry.

## Next Steps (Cold Start)

1. Read this file and `CHANGELOG.md`
2. Check production state: `sqlite3 ~/.openclaw/data/epistemic.db "SELECT id, label, (SELECT COUNT(*) FROM claims WHERE topic_id=t.id) as claims FROM topics t"`
3. Check cron health: `grep -i "error\|fail" ~/clawd/logs/epistemic*.log | tail -10`
4. Review whitepaper.md for architectural context (especially §10 on multi-source ingestion)
5. Check open tasks in tasks.db: `task.sh list | grep -i epistemic`
