# Epistemic Memory

**Claim-first epistemic synthesis for personal AI agents.**

Most AI memory systems store facts or compress conversations. None build evolving *understanding* — the kind of conceptual knowledge that develops across hundreds of conversations over months.

Epistemic Memory bridges that gap. It discovers topics from conversation history, extracts and verifies atomic claims, and synthesizes them into evolving understanding documents that improve over time.

## The Problem

Personal AI agents accumulate thousands of conversation summaries. Current memory architectures offer:

- **RAG**: retrieves relevant chunks, no synthesis
- **Knowledge graphs**: store entity relationships, not conceptual understanding  
- **Temporal compression** (LCM): reduces token cost, loses cross-conversation patterns
- **Fact extraction** (Metabolism): captures `entity.key = value` triples, not comprehension

If you discuss psychedelic integration across 15 sessions, you get 15 temporal summaries — not a synthesized understanding of what integration means to this person, how their thinking evolved, or what tensions remain unresolved.

## Architecture

Epistemic Memory uses a **claim-first** architecture:

```
Conversations → LCM Summaries → Topic Discovery → Claim Extraction
    → Claim Verification → Canonical Synthesis → Context Injection
```

### Phase 1: Topic Discovery & Tagging (✅ Complete)

Bottom-up topic discovery from conversation summaries using vector embeddings and HDBSCAN clustering. No predefined categories — topics emerge from the data.

- **Embedding**: nomic-embed-text (768-dim, local inference)
- **Clustering**: HDBSCAN with automatic merge detection
- **Tagging**: Cosine similarity against topic centroids with EMA updates
- **Labeling**: Local LLM generates 2-5 word topic labels
- **Orphan reconciliation**: Detects broken references when upstream summaries re-compact

### Phase 2: Claim Extraction & Synthesis (In Progress)

The core innovation — extracting atomic, verifiable claims from tagged summaries and synthesizing them into evolving understanding documents.

Three artifacts per topic:
1. **Evidence Ledger** — atomic claims with provenance, confidence, and decay
2. **Canonical Synthesis** — full understanding document generated from verified claims
3. **Injection Brief** — token-efficient summary for context injection

### Phase 3: Context Integration

Synthesis documents feed back into conversation context, replacing N temporal summaries with one evolving understanding document per topic.

## Quick Start

```bash
# Requirements: Python 3.10+, local embedding model (nomic-embed), local LLM (for labeling)
pip install -r requirements.txt

# Configure (see src/config.py for all options)
export ES_LCM_DB=~/.openclaw/lcm.db
export ES_EPISTEMIC_DB=~/.openclaw/data/epistemic.db
export ES_EMBED_BASE_URL=http://localhost:8082

# Run full pipeline
python -m src.runner full --json

# Or individual passes
python -m src.runner tag        # Tag new summaries (every 30 min)
python -m src.runner discover   # Discovery + tag + label (daily)
python -m src.runner orphan     # Orphan reconciliation (weekly)
```

## Data Model

```sql
topics          -- Discovered topic clusters with centroids
topic_edges     -- Graph relationships between topics (not tree — spreading activation)
topic_summaries -- Links topics to LCM summary IDs
synthesis_versions -- Versioned understanding documents per topic
synthesis_sources  -- Provenance: which summaries contributed to each synthesis
runs            -- Audit log of all pipeline executions
```

Separate `epistemic.db` (SQLite, WAL mode) — references LCM summary IDs but never writes to `lcm.db`.

## Design Principles

- **Claim-first, not document-first**: Extract atomic claims → verify → synthesize. Documents are views of the claim graph, not sources of truth.
- **Epistemic register**: No first-person assertions. Categories: established patterns, recurring hypotheses, active tensions, unverified inferences.
- **Named failure modes**: False abstraction, premature convergence, topic aliasing, contradiction laundering, persona leakage, stale high-confidence carryover.
- **Read-only upstream**: Never writes to LCM. Epistemic.db is the only mutable store.
- **Bottom-up topics**: Data drives what topics emerge. No predefined categories.

## Testing

```bash
pytest tests/ -v
# 65 tests covering schema, embedding, tagging, discovery, orphans, and E2E
```

## White Paper

See [`whitepaper.md`](whitepaper.md) for the full architectural specification, literature review (8 papers), failure mode analysis, and evaluation framework.

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
