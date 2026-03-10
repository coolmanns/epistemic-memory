# Epistemic Memory

**Claim-first epistemic synthesis for personal AI agents.**

Most AI memory systems store facts or compress conversations. None build evolving *understanding* — the kind of conceptual knowledge that develops across hundreds of conversations over months.

Epistemic Memory bridges that gap. It discovers topics from conversation history, extracts and verifies atomic claims, and synthesizes them into evolving understanding documents that improve over time.

## The Problem

Personal AI agents accumulate thousands of conversation summaries. Current memory architectures offer:

- **RAG**: retrieves relevant chunks, no synthesis
- **Knowledge graphs**: store entity relationships, not conceptual understanding  
- **Temporal compression** (LCM): reduces token cost, loses cross-conversation patterns
- **Fact extraction**: captures `entity.key = value` triples, not comprehension

If you discuss psychedelic integration across 15 sessions, you get 15 temporal summaries — not a synthesized understanding of what integration means to this person, how their thinking evolved, or what tensions remain unresolved.

## Architecture

Epistemic Memory uses a **claim-first** architecture:

```
Conversations → LCM Summaries → Topic Discovery → Claim Extraction
    → Deduplication → Claim Verification → Canonical Synthesis → Markdown Output
```

### Phase 1: Topic Discovery & Tagging ✅

Bottom-up topic discovery using vector embeddings and HDBSCAN clustering. No predefined categories — topics emerge from the data.

- **Embedding**: Qwen3-Embedding-4B (2560-dim, local inference via llama.cpp)
- **Clustering**: HDBSCAN with automatic merge detection
- **Tagging**: Cosine similarity against topic centroids with EMA updates
- **Labeling**: Local LLM (Qwen3-30B) generates 2-5 word topic labels
- **Orphan reconciliation**: Detects broken references when upstream summaries re-compact

### Phase 2: Claim Extraction & Synthesis ✅

The core innovation — extracting atomic, verifiable claims from tagged summaries, then synthesizing them into evolving understanding documents.

Pipeline: **extract → deduplicate → verify → synthesize → write → decay**

- **Extractor**: Pulls atomic claims from summary text with type classification (factual, causal, evaluative, analogical, procedural)
- **Deduplicator**: Embeds claims, cosine similarity merge above threshold
- **Verifier**: Two-pass verification — verdicts: verified, overstated, unsupported, misattributed
- **Synthesizer**: Generates canonical synthesis + injection brief from verified claims only
- **Writer**: Outputs per-topic markdown files with frontmatter metadata
- **Decay**: Claim-level confidence decay (HIGH → MED → LOW → decayed) prevents stale carryover

LLM routing: Sonnet (Anthropic API) for extraction, verification, and synthesis. Qwen3-Embedding-4B (local) for embeddings. Qwen3-30B (local) for topic labeling.

### Phase 3: Context Integration (Planned)

Synthesis documents feed back into conversation context, replacing N temporal summaries with one evolving understanding document per topic.

## First Real Run

```
188 summaries → 17 topics discovered → 139 claims extracted
    → 41 deduplicated → 78 verified (1 overstated, 0 unsupported)
    → 16 synthesis docs written to memory/topics/
```

Topics discovered (bottom-up, no predefined categories):
Cover Image Sync, Frosted Glass Backgrounds, Epistemic Synthesis, LCM Migration,
Emotion Scoring, Image Gallery, OAuth Fix, SEO Growth Strategy, LCM Summary Model,
Plugin Scoping, Mission Control, Anthropic Rate Limiting, Compliance Dashboard,
Debug Log Flood, Image Handling, Audience & Platform Analysis, SEO Site Audit

## Quick Start

```bash
# Requirements: Python 3.10+, embedding model, Anthropic API key
pip install -r requirements.txt

# Configure (see src/config.py for all options)
export ES_LCM_DB=~/.openclaw/lcm.db
export ES_EPISTEMIC_DB=~/.openclaw/data/epistemic.db
export ES_EMBED_BASE_URL=http://localhost:8086

# Run full pipeline
python -m src.runner full --json

# Or individual passes
python -m src.runner tag        # Tag new summaries (every 30 min)
python -m src.runner discover   # Discovery + tag + label (daily)
python -m src.runner orphan     # Orphan reconciliation (weekly)

# Phase 2 pipeline
python -m src.runner extract    # Extract claims from tagged summaries
python -m src.runner dedup      # Deduplicate similar claims
python -m src.runner verify     # Verify claims against source material
python -m src.runner synthesize # Generate synthesis docs from verified claims
python -m src.runner write      # Write markdown files to output directory
python -m src.runner decay      # Apply confidence decay to aging claims
```

## Data Model

```sql
-- Phase 1: Topic Discovery
topics          -- Discovered topic clusters with centroids
topic_edges     -- Graph relationships between topics (weighted, not tree)
topic_summaries -- Links topics to LCM summary IDs

-- Phase 2: Claim Synthesis
claims          -- Atomic claims with type, confidence, verification status
claim_sources   -- Provenance: which summaries sourced each claim
syntheses       -- Versioned synthesis docs per topic (canonical + brief)
synthesis_runs  -- Audit log of synthesis pipeline executions

-- Shared
tagging_log     -- Audit log of all tagging/discovery runs
schema_meta     -- Migration version tracking
```

Separate `epistemic.db` (SQLite, WAL mode) — references LCM summary IDs but never writes to `lcm.db`.

## Design Principles

- **Claim-first, not document-first**: Extract atomic claims → verify → synthesize. Documents are views of the claim graph, not sources of truth.
- **Epistemic register**: No first-person assertions. Categories: established patterns, recurring hypotheses, active tensions, unverified inferences.
- **Named failure modes**: False abstraction, premature convergence, topic aliasing, contradiction laundering, persona leakage, stale high-confidence carryover.
- **Read-only upstream**: Never writes to LCM. Epistemic.db is the only mutable store.
- **Bottom-up topics**: Data drives what topics emerge. No predefined categories.
- **Claim-level decay**: Independent of edge decay — prevents stale high-confidence carryover across sessions.

## Testing

```bash
# Run all unit tests (no live API needed)
pytest tests/ -v -k "not Deterministic"

# Run with slow/live tests
pytest tests/ -v -m slow

# 153 tests total: 60 Phase 1 + 93 Phase 2
```

## White Paper

See [`whitepaper-v0.3.md`](whitepaper-v0.3.md) for the full architectural specification, literature review (8 papers), failure mode analysis, and evaluation framework.

### Prior Work Comparison

| System | Unit of Knowledge | Synthesis | Verification | Temporal |
|--------|------------------|-----------|-------------|----------|
| Generative Agents | Natural language memory | Reflection (importance threshold) | None | Recency scoring |
| Zep/Graphiti | Temporal KG triples | Edge invalidation | Bitemporal modeling | Full |
| Mem0/Mem0g | Facts + graph edges | Deduplication | None | Session-level |
| RMM | Topic-based segments | Prospective reflection | RL-based retrieval | Turn/session |
| HippoRAG 2 | KG triples + PPR | Recognition memory | Triple filtering | Partial |
| A-MEM | Zettelkasten notes | Cross-note linking | None | Implicit |
| **Epistemic Memory** | **Atomic claims** | **Claim-first synthesis** | **Two-pass verify** | **Claim-level decay** |

## License

MIT — see [LICENSE](LICENSE).

## Author

Sascha Kuhlmann ([@coolmanns](https://github.com/coolmanns))

Built with [OpenClaw](https://openclaw.ai) + GandalfOfElgin 🧙‍♂️
