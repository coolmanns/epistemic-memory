# Epistemic Synthesis: Claim-Grounded Topic Understanding for Persistent AI Agents

**Authors:** Sascha Kuhlmann & Gandalf (GandalfOfElgin)
**Date:** March 9, 2026
**Status:** Draft v0.4.1 — Design Phase (multi-source architecture + provenance model)
**Changelog:** v0.1 initial design → v0.2 grounding/drift/privacy (peer review) → v0.3 claim-first architecture, typed topics, three-artifact model (peer review) → v0.3.1 deeper literature engagement, A-MEM comparison, benchmark-grounded evaluation → v0.4 multi-source knowledge ingestion, provenance chains, checkpoint+delta retrieval model → v0.4.1 verification failure modes, cost projections, evaluation benchmark sketch, topic stability protocol (peer review)

---

## Abstract

Current AI memory systems store and retrieve information but do not consolidate it into reusable understanding. Systems like LCM (Lossless Context Management) compress conversation history into temporal summaries; vector databases enable semantic search; fact extraction produces entity-key-value triples. None synthesize evolving conceptual understanding across conversations over time.

This paper proposes **Epistemic Synthesis** — a learning architecture for persistent AI agents that discovers topics from conversation data and workspace files, extracts and verifies atomic claims, builds associative networks between topics, and produces auditable, evolving topic models that represent what the agent has learned about a subject.

The core architectural decision is **claim-first synthesis**: the unit of epistemic update is the atomic claim with provenance, not the prose document. Understanding documents are generated *from* verified claim sets, not verified *after* generation. This inverts the synthesis pipeline and makes drift detection, contradiction handling, and evaluation tractable.

The v0.4 design extends the architecture to multi-source knowledge ingestion, recognizing that understanding does not originate in files or conversations alone — it emerges from the cyclical interaction between both. A provenance chain model tracks how knowledge flows from generative conversations through crystallized files and back, while a checkpoint + delta retrieval model eliminates redundant loading by treating files as knowledge checkpoints and post-file conversations as the evolutionary delta.

---

## 1. The Problem

### 1.1 The State of AI Memory

Modern AI agent memory systems have converged on a layered architecture:

| Layer | Function | Example Systems |
|-------|----------|----------------|
| Context window | Active working memory | All LLMs |
| Conversation history | Episodic recall | LCM, MemGPT/Letta |
| Fact extraction | Structured knowledge | Metabolism, LangMem, Mem0 |
| Behavioral tracking | Growth patterns | Stability, Contemplation, RMM |
| Vector search | Semantic retrieval | memorySearch, RAG systems |
| Temporal knowledge graphs | Evolving relationships | Zep/Graphiti |
| Associative retrieval | Graph-structured recall | HippoRAG 2 |
| Linked knowledge networks | Dynamic note interconnection | A-MEM (Zettelkasten) |
| Reflection | Higher-level inference | Generative Agents, RMM |

Each layer handles a specific memory operation — storage, retrieval, consolidation, or behavioral regulation. **None performs epistemic synthesis**: the transformation of accumulated experiences into auditable, evolving topic understanding.

### 1.2 The 15-Conversation Problem

Consider a persistent agent that discusses "psychedelic integration" across 15 separate conversations over two months. Under current architectures, the agent possesses:

- **15 temporal summaries** (LCM) — what happened in each conversation
- **~30 extracted facts** (Metabolism/Mem0) — entity-key-value triples like `integration.common_protocols = ["journaling", "breathwork", "talk therapy"]`
- **Behavioral vectors** (Contemplation/RMM) — "agent has become more nuanced when discussing integration timelines"

What the agent does **not** possess:

- An auditable set of claims about what psychedelic integration *is*, grounded in specific conversations, with tracked confidence, known contradictions, and explicit uncertainty — the kind of accumulated professional knowledge a human therapist would build naturally over months of practice.

The 15 summaries are filed chronologically. The facts are flat labels. The behavioral vectors describe *how* the agent acts, not *what* it has learned. There is no structured representation of the agent's evolving knowledge of integration.

### 1.3 Why This Gap Exists

Three factors explain the absence of epistemic synthesis in production systems:

1. **Market focus on enterprise**: Enterprise AI needs accurate retrieval of facts and procedures, not evolving conceptual understanding.

2. **Evaluation difficulty**: Measuring whether a system "understands" a topic is qualitatively different from measuring retrieval accuracy. Recent benchmarks (LoCoMo, LongMemEval, REALTALK) increasingly separate memory into competencies — extraction, multi-session reasoning, temporal reasoning, knowledge updates, conflict handling — but no benchmark targets synthesis quality directly. (See Section 9.)

3. **Statelessness assumption**: Most AI architectures assume each conversation starts fresh. Personal persistent agents — where understanding accumulates — are a small but growing segment.

---

## 2. Architecture

### 2.1 Overview

Epistemic Synthesis operates as a post-hoc learning layer that consumes conversation summaries and produces three artifacts per topic: an **evidence ledger** of atomic claims, a **canonical synthesis** document, and an **injection brief** for runtime context.

The unit of epistemic update is the **claim**, not the document.

The system has three operational phases:

```
Phase 1: Topic Tagging       (real-time, on LCM compaction events)
Phase 2: Topic Discovery      (scheduled, daily)
Phase 3: Claim Extraction     (overnight, per topic with new summaries)
         & Synthesis
```

### 2.2 Phase 1 — Topic Tagging (Real-Time)

**Trigger:** LCM compaction event (new summary created)
**Operation:** Embed the summary → cosine similarity against existing topic centroids → tag if above threshold
**Cost:** One embedding call + N similarity comparisons (where N = number of topics)
**Latency:** <100ms, non-blocking to conversation

When LCM creates a new summary during or after a conversation, the tagging phase checks whether it belongs to an existing topic. This is a lightweight vector similarity operation — no LLM call required. If the summary's embedding is within threshold of a topic centroid, it gets tagged. If not, it enters the untagged pool for Phase 2 discovery.

### 2.3 Phase 2 — Topic Discovery (Daily)

**Trigger:** Scheduled during low-activity hours
**Operation:** Cluster untagged summaries using HDBSCAN → identify new topic clusters → classify type → generate labels via LLM → compute centroids
**Cost:** One clustering pass + K LLM calls for labeling (where K = new clusters found)
**Output:** New typed topic entries, newly tagged summaries

Topic discovery is bottom-up: the data determines what topics exist, not top-down declaration.

HDBSCAN was chosen because it discovers the number of clusters automatically, handles noise gracefully, and identifies clusters of varying density.

**Proof of concept results:** 168 LCM summaries from ~1 month of conversation → 19 distinct topic clusters with 53 noise points. Topics included client projects (3 sub-clusters for one client), infrastructure work, memory architecture discussions, and psychedelic research — all emerging without pre-defined categories.

#### 2.3.1 Typed Topics

Not all topics should be discovered or synthesized the same way. Embedding similarity clusters by conversational recurrence, which often reflects project context or entity mentions rather than conceptual substance. The system classifies discovered topics into types:

| Type | Discovery Signal | Synthesis Strategy | Example |
|------|-----------------|-------------------|---------|
| **Entity-centric** | Named entity recurrence | Factual profile + relationship map | "Martin Ball" |
| **Project/process** | Shared operational context | Status, decisions, lessons learned | "Ghost CMS migration" |
| **Conceptual** | Abstract theme across varied contexts | Evolving understanding, tensions, frameworks | "Psychedelic integration" |
| **Autobiographical** | Self-referential patterns | Behavioral observations, growth patterns | "My approach to client work" |

Classification happens during discovery via a lightweight LLM call that examines the cluster's representative summaries and assigns a type. Different types use different synthesis prompts (Section 2.4) and different disclosure policies (Section 7).

**Why this matters:** HDBSCAN will reliably discover "Martin Ball SEO work" (entity-centric, high surface similarity) long before it discovers "the evolving concept of integration" (conceptual, distributed across varied conversations). Typed topics acknowledge this asymmetry and handle each type appropriately rather than forcing a single synthesis strategy.

#### 2.3.2 Topic Stability

HDBSCAN cluster boundaries shift as new summaries arrive. Mitigation:

- **Incremental clustering:** New summaries first attempt tagging against existing centroids (Phase 1). Full re-clustering runs only when the untagged pool exceeds a threshold.
- **Centroid drift tolerance:** Each topic tracks drift between current centroid and centroid-at-last-synthesis. Resynthesis triggers only when drift exceeds a configurable threshold.
- **Split detection:** When a topic's internal variance exceeds threshold, the system flags it for potential split and proposes sub-topics.
- **Merge detection:** When two topics converge in centroid space and share significant summary overlap, they're flagged for merge.
- **Versioned snapshots:** Each synthesis version records the topic centroid and member summaries at time of creation.

#### 2.3.2.1 Claim Reassignment Protocol

Topic splits and merges are where the messiest data integrity issues live. Claims, edges, and synthesis versions all need consistent handling. The following protocol specifies the operations:

**On topic split** (Topic A → Topic A' + Topic B'):

```
1. Recluster: Re-embed all claims in Topic A. Assign each claim to A' or B'
   based on embedding proximity to the new centroids.
2. Provenance follows: claim_sources entries move with their claims unchanged.
   The source summaries don't change — only which topic the claim belongs to.
3. Contradictions re-evaluate: If claim_a and claim_b (previously contradicting
   within Topic A) land in different sub-topics, the contradiction is
   demoted to a cross-topic tension — still tracked, but no longer an
   intra-topic Active Tension.
4. Edges redistribute: Topic A's edges to other topics are copied to both A'
   and B', weighted by the proportion of claims each inherited. An edge
   with weight 0.7 where A' got 60% of claims → A' inherits 0.42, B' gets 0.28.
5. Synthesis invalidated: Both A' and B' get status 'needs_synthesis'.
   Previous synthesis versions for Topic A get status 'superseded_by_split'.
6. Human notification: Splits are surfaced for review. Automated splits
   above a configurable claim count threshold require approval.
```

**On topic merge** (Topic A + Topic B → Topic C):

```
1. Union: All claims from A and B move to C.
2. Deduplication: Cross-topic claims with embedding similarity > 0.92 are
   merged. The version with higher support_count survives; the other's
   unique sources are added to claim_sources.
3. Contradiction discovery: Claims that were independent in separate topics
   may contradict each other. Run pairwise contradiction detection on
   the merged claim set.
4. Edges merge: C inherits the union of A and B's edges. Where both had
   edges to the same topic, take max(weight).
5. Centroid recompute: C's centroid is computed from the merged claim set,
   not averaged from A and B's centroids.
6. Synthesis required: C gets status 'needs_synthesis'. Previous versions
   for A and B get status 'superseded_by_merge'.
```

**Invariants that must hold after any split/merge:**
- Every claim belongs to exactly one topic
- Every claim_source entry points to a valid claim
- No orphaned contradiction entries
- Edge weights sum to ≤ 1.0 per pair
- At least one synthesis version exists with status 'current' or 'needs_synthesis' per topic

### 2.4 Phase 3 — Claim Extraction & Synthesis (Overnight)

**Trigger:** Overnight batch, for topics that received new tagged summaries since last synthesis
**Pipeline:**

```
Step 1: Retrieve tagged summaries via LCM expand
Step 2: Extract candidate claims from summaries (LLM)
Step 3: Deduplicate and cluster claims
Step 4: Attach provenance (source summaries) and compute support counts
Step 5: Detect tensions and contradictions between claims
Step 6: Verify claims against source material
Step 7: Generate canonical synthesis from verified claim set
Step 8: Generate injection brief from canonical synthesis
```

**Cost:** One LCM expand + 3-4 LLM calls per updated topic (extraction, verification, synthesis, brief)
**Output:** Updated evidence ledger, new synthesis version, updated injection brief

#### 2.4.1 Claim Extraction

The fundamental unit of epistemic update is the **atomic claim** — a single assertion with provenance, confidence, and type classification.

```
Claim structure:
  id:           unique identifier
  topic_id:     parent topic
  text:         the assertion (one sentence)
  type:         extractive | abductive | analogical
  confidence:   HIGH | MED | LOW
  support:      [sum_xxx, sum_yyy]  -- source summaries
  support_count: number of independent sources
  contradicts:  [claim_id, ...]     -- opposing claims
  first_seen:   timestamp
  last_reinforced: timestamp
  status:       active | decayed | disputed | retracted
```

**Claim types determine verification stringency:**
- **Extractive:** Directly stated in source material. Verification: does the source actually say this? Lowest risk.
- **Abductive:** Inferred from patterns across sources. Verification: is the inference reasonable given the evidence? Medium risk.
- **Analogical:** Drawn from similarities to other topics. Verification: is the analogy valid? Highest risk — these claims require the most scrutiny and should rarely reach `[HIGH]` confidence.

**Extraction prompt (per topic type):**

```
Given these {n} conversation summaries about "{topic_label}" (type: {topic_type}):

Extract atomic claims — single assertions that represent what has been learned
about this topic. For each claim:

1. State the claim in one sentence.
2. Classify: extractive (directly stated), abductive (inferred from patterns),
   or analogical (drawn from similarity to other domains).
3. Cite source summaries: [src:sum_xxx].
4. Rate confidence:
   - HIGH: supported by 3+ independent conversations
   - MED: supported by 1-2 conversations or strong indirect evidence
   - LOW: tentative, partial, or single-source inference
5. Flag any claim that contradicts another claim you've extracted.

DO NOT:
- Generate claims that go beyond what the sources support
- Treat exploratory questions as settled positions
- Merge distinct claims into compound statements
- Assert confidence where the sources are tentative
```

**Known limitation (v0.4.1):** The current extractor sends *all* tagged summaries for a topic in a single LLM call with a hard cap (`MAX_CLAIMS_PER_EXTRACTION`, default 15). For content-rich topics (e.g., 74+ summaries), this forces the LLM to aggressively compress — losing granularity. The planned fix is **batched extraction**: send summaries in groups of 10–15, extract claims per batch, then deduplicate across batches. This preserves fine-grained facts while keeping per-call cost bounded. Note: claims are an intermediate representation feeding synthesis; raw conversation content remains fully accessible via LCM drill-down regardless of claim count.

#### 2.4.2 Claim Verification

After extraction, a separate verification pass checks grounding at the individual claim level:

```
For each claim:
1. Does the cited source actually support this assertion?
   - If extractive: is it stated or clearly implied?
   - If abductive: is the inference pattern reasonable?
   - If analogical: is the analogy justified by the sources?
2. Is the confidence level appropriate given support count and type?
3. Are claimed contradictions genuine, or merely different framings?

Output per claim: VERIFIED | UNSUPPORTED | OVERSTATED | MISATTRIBUTED
```

Claims that fail verification are flagged but not deleted — they enter the evidence ledger with status `unsupported` for human review. Only `VERIFIED` claims feed into synthesis document generation.

**Why claim-level verification beats document-level:** A document-level verifier can pass a synthesis that is "mostly right" while containing subtle overgeneralizations. Claim-level verification catches: incorrect abstraction from true details, style-induced confidence inflation, and confirmation of vaguely-supported assertions. Each failure is localized and traceable.

#### 2.4.2.1 Verification Failure Modes

The verification pass is load-bearing in the anti-drift architecture — and it is itself an LLM call. This creates a fundamental limitation that must be stated explicitly: **an LLM verifying another LLM's extraction is a probabilistic filter, not a guarantee.**

Known failure modes of the verification layer:

| Failure Mode | Description | Risk Level |
|-------------|-------------|------------|
| **Same-model confirmation bias** | If the same model extracts and verifies a claim, it may reproduce its own reasoning patterns rather than independently checking | HIGH if same model; MEDIUM with model diversity |
| **Plausible-sounding confabulation** | A well-formed abductive claim with plausible reasoning may pass verification even when the source doesn't actually support it | MEDIUM — abductive and analogical claims most vulnerable |
| **Source summary compression** | Verification checks claims against LCM summaries, not raw conversation text. If the summary already lost relevant nuance, verification can't recover it | LOW-MEDIUM — depends on LCM compaction quality |
| **Confidence inflation** | The verifier may accept an overstated confidence level because the claim is directionally correct | MEDIUM — mitigated by explicit confidence criteria |

**Mitigations (layered, not reliant on any single check):**

1. **Model diversity:** Use a different model for verification than extraction where feasible. At minimum, use different prompts and temperature settings.
2. **Monthly re-verification:** The primary-source audit (Section 4.3) re-checks HIGH claims on a different temporal context, which changes what the model attends to. This is not independent verification, but it is decorrelated verification.
3. **Human audit surface:** The evidence ledger is designed for human readability. When verification fails silently, the human reviewing the ledger is the actual backstop. This is why auditability matters more than verification accuracy — the system is designed to make failures *visible*, not to prevent them entirely.
4. **Statistical monitoring:** Track verification pass rates per claim type. If abductive claims pass at the same rate as extractive claims, the verifier is likely not discriminating effectively.

**The honest framing:** Verification *reduces* drift probability per claim. It does not *prevent* drift. The actual safety net is the combination of verification + decay + re-verification + human auditability. No single layer is sufficient. The architecture accepts this and stacks probabilistic defenses rather than claiming any one is reliable.

#### 2.4.3 Canonical Synthesis Generation

The canonical synthesis document is generated from the verified claim set, not directly from summaries. This is the core architectural inversion.

```
Given the following verified claims about "{topic_label}" ({topic_type}):

{claims, grouped by confidence level}

Generate a synthesis document using the following structure:

ESTABLISHED PATTERNS
  Claims with HIGH confidence. These are well-supported across multiple
  conversations.

RECURRING HYPOTHESES
  Claims with MED confidence. Supported but not yet established.

ACTIVE TENSIONS
  Pairs or groups of contradictory claims, presented without resolution.
  Both sides cited.

UNVERIFIED INFERENCES
  Claims with LOW confidence or analogical type. Explicitly marked as
  tentative.

USER-STATED POSITIONS
  Claims attributed to the human's explicitly stated views (vs. agent
  inference). Critical distinction for autobiographical topics.

SYSTEM-DERIVED OBSERVATIONS
  Claims the agent has inferred from behavioral patterns, not from explicit
  human statements. These may be accurate but were never validated by the
  human.

Include claim IDs inline: [claim:C-xxx]. Do not add claims that aren't in the
verified set. Do not resolve Active Tensions by choosing sides.
```

This epistemic register replaces first-person narrative voice. It is safer to audit, less prone to epistemic confusion, and explicitly separates what the human said from what the agent inferred.

#### 2.4.4 Injection Brief Generation

The injection brief is a compressed, query-optimized version of the canonical synthesis, designed for runtime context injection:

```
Given this canonical synthesis for "{topic_label}":

Generate a brief (max 400 tokens) optimized for context injection. Include:
- Top 3-5 established patterns
- Any active tensions relevant to typical queries about this topic
- Key connections to other topics

Do not include: low-confidence claims, full provenance, system-derived
observations, or verification metadata.
```

The injection brief is what enters the conversation context. The canonical synthesis is the reference document. The evidence ledger is the source of truth. Three artifacts, three purposes, cleanly separated.

### 2.5 The Three Artifacts

| Artifact | Purpose | Updated | Size | Audience |
|----------|---------|---------|------|----------|
| **Evidence Ledger** | Source of truth — atomic claims with provenance | On every claim extraction pass | Grows over time | Verification, audit, debugging |
| **Canonical Synthesis** | Structured understanding — epistemic register | When claim set changes materially | 800-2000 tokens | Human review, detailed reference |
| **Injection Brief** | Runtime context — compressed, query-optimized | When canonical synthesis updates | 200-400 tokens | LLM context window |

### 2.6 Cost Model

The overnight batch pipeline's cost scales with the number of topics receiving new material, not the total topic count. This distinction is critical for feasibility assessment.

**Per-topic synthesis cost (Phase 3):**

| Operation | LLM Calls | Est. Input Tokens | Est. Output Tokens |
|-----------|-----------|-------------------|-------------------|
| LCM expand (retrieve summaries) | 0 (API call) | — | — |
| Claim extraction | 1 | 2,000–8,000 | 500–2,000 |
| Claim verification | 1 | 1,500–5,000 | 200–800 |
| Synthesis generation | 1 | 1,000–4,000 | 800–2,000 |
| Brief generation | 1 | 800–2,000 | 200–400 |
| **Total per topic** | **4** | **5,300–19,000** | **1,700–5,200** |

**Projected batch costs at scale (using approximate API pricing):**

| Timeframe | Topics | Active/week (~20%) | Batch cost/night | Monthly |
|-----------|--------|-------------------|-----------------|---------|
| 1 month | 20 | 4 | ~$0.03–0.10 | ~$1–3 |
| 6 months | 300 | 60 | ~$0.50–1.50 | ~$15–45 |
| 1 year | 750 | 150 | ~$1.20–3.80 | ~$36–114 |
| 3 years | 3,000 | 600 | ~$5–15 | ~$150–450 |

*Estimates assume Sonnet-class pricing (~$3/M input, ~$15/M output). Local model execution (e.g., llama.cpp for extraction) would reduce costs significantly for the extraction and brief generation steps, reserving frontier model budget for verification and synthesis where quality matters most.*

**Cost mitigation strategies:**
1. **Tiered model assignment:** Use local models for extraction and brief generation (lower-stakes operations). Reserve frontier models for verification and synthesis.
2. **Materiality threshold:** Only trigger resynthesis when new claims materially change the evidence ledger (e.g., >10% claim change or new contradiction detected), not on every new tagged summary.
3. **Batch consolidation:** If a topic receives new material on 5 consecutive days, synthesize once at week's end rather than nightly.
4. **Claim extraction caching:** If a summary has already been processed for claim extraction and hasn't changed, skip it on subsequent runs.

At the 1-year mark with 750 topics, the monthly cost ($36–114) is comparable to a single premium API subscription — manageable for a personal agent where the value proposition is accumulated knowledge. At the 3-year mark, cost optimization becomes mandatory, but by then the tiered model and materiality threshold strategies should be well-tuned from operational data.

---

## 3. Topic Topology — Graph, Not Tree

### 3.1 Why Hierarchy Is Wrong

An early design used hierarchical topic relationships (parent_id). This was rejected because knowledge doesn't organize hierarchically. A topic like "Martin Ball" connects to "SEO client work," "5-MeO-DMT research," "Ghost CMS," "Content strategy," and "Psychedelic community" simultaneously. Forcing this into a tree requires choosing one parent, severing all other connections.

### 3.2 Associative Topic Network

Topics connect through weighted edges:

```
topic_edges:
  topic_a_id          INTEGER
  topic_b_id          INTEGER
  weight              REAL      -- 0.0 to 1.0
  last_reinforced_at  TIMESTAMP
  created_at          TIMESTAMP
  updated_at          TIMESTAMP
```

Edge weights derive from:
- **Co-occurrence:** Topics sharing tagged summaries
- **Embedding proximity:** Centroid closeness in embedding space
- **Claim cross-reference:** When claims in Topic A cite summaries also tagged to Topic B

### 3.3 Spreading Activation for Context Retrieval

When a conversation touches a topic, the system activates its neighbors (Collins & Loftus, 1975):

1. Identify primary topic(s) from current conversation (embedding similarity)
2. Activate direct neighbors, weighted by edge strength
3. Optionally activate second-hop neighbors with decayed weight
4. Inject **injection briefs** for activated topics above threshold

**Starting heuristics:**
- Primary topic activation: cosine similarity > 0.75
- First-hop activation: edge weight > 0.6
- Second-hop activation: edge weight > 0.8
- Maximum activated topics: 5
- Total injection token budget: 2,000 tokens

**Tuning:** Log every activation decision. After 30 days, measure: activation relevance (was the topic referenced in conversation?), miss rate (did the agent retrieve raw summaries for a topic that had an injection brief?), and context efficiency. Adjust thresholds from data.

Future refinement: query-conditioned activation — where activation weights are modulated by the current conversation's semantic content, not just topic-level similarity. For the initial implementation, static thresholds with empirical tuning are sufficient.

### 3.4 Edge Weight Decay

Associations that aren't reinforced weaken over time, preventing the network from becoming undifferentiated.

```
weight_effective = weight_base * 0.995^(days_since_last_reinforcement)
Half-life ≈ 139 days. Prune below 0.1.
```

### 3.5 Claim-Level Decay

Independent of edge decay, individual claims decay:

- Claims not reinforced by new evidence for 90 days: downgrade from `HIGH` → `MED`
- Claims at `MED` without reinforcement for 60 more days: downgrade to `LOW`
- Claims at `LOW` without reinforcement for 60 more days: status → `decayed`
- Decayed claims are excluded from injection briefs but remain in the evidence ledger

This prevents stale high-confidence carryover — one of the most insidious failure modes, where an early well-supported claim persists at `HIGH` confidence long after the evidence landscape has shifted.

---

## 4. Grounding and Drift Prevention

### 4.1 The Feedback Loop Problem

The most serious structural risk is self-reinforcing drift: synthesis outputs become inputs to future syntheses, and any confabulation in an early version gets baked in and amplified.

This is not theoretical. During development, the authors experienced an identical failure mode: the agent hallucinated a claim, a fact-extraction system ingested it as authoritative, a context-injection system served it back in future sessions, and the agent repeated the false claim with increasing certainty across multiple sessions. The loop was only broken by manual human intervention. (See Appendix A.)

### 4.2 How Claim-First Architecture Mitigates Drift

The document-first pipeline (v0.1-v0.2) was vulnerable because the synthesis document was both the unit of creation and the unit of truth. Errors in the document propagated through all downstream operations.

The claim-first pipeline separates concerns:

| Failure Mode | Document-First (v0.2) | Claim-First (v0.3) |
|-------------|----------------------|-------------------|
| False abstraction | Hidden in prose, hard to isolate | Individual claim flagged by verification |
| Premature convergence | Document sounds certain, uncertainty smoothed away | Confidence levels per claim, active tensions preserved |
| Contradiction laundering | Contradictions resolved implicitly by LLM choosing one side | Contradictions explicit in claim graph, both sides maintained |
| Stale confidence | Entire document ages together | Per-claim decay, independent timelines |
| Telephone game drift | New synthesis builds on old synthesis | New synthesis builds on **claims verified against primary sources** |

The critical protection: synthesis generation (Step 7) takes verified claims as input, not previous synthesis documents. Each synthesis version is a fresh rendering of the current claim graph. There is no document-to-document inheritance.

### 4.3 Additional Safeguards

**Monthly primary-source audit:** An automated pass re-verifies `HIGH` confidence claims against their original source summaries (via LCM expand), bypassing any intermediate artifacts.

**Entropy monitoring:** If a topic's claim count or average confidence increases without new source material, flag for review.

**Human-in-the-loop escalation:** When verification fails, when cross-version contradictions appear, or when entropy monitoring flags drift, the system surfaces the issue to the human.

---

## 5. Failure Modes

Naming failure modes explicitly strengthens the architecture's defenses:

| Failure Mode | Description | Detection | Mitigation |
|-------------|-------------|-----------|------------|
| **False abstraction** | Claim generalizes beyond what sources support | Claim-level verification; type classification (abductive/analogical flagged) | Reject or downgrade confidence |
| **Premature convergence** | Topic understanding settles too early on one interpretation | Low claim diversity relative to summary count | Flag for human review; inject "what are we missing?" in next synthesis |
| **Topic aliasing** | Two distinct concepts merged into one topic due to surface similarity | Intra-topic variance monitoring; split detection | Propose topic split |
| **Contradiction laundering** | Genuine tensions smoothed away by synthesis | Active Tensions section in epistemic register; contradiction claim pairs | Contradictions are structurally preserved, not resolved |
| **Persona leakage** | Synthesis captures agent's inference style rather than actual knowledge | Separation of User-Stated Positions vs. System-Derived Observations | Different disclosure policies per section |
| **User-agent belief conflation** | Agent's inferred claims presented as user's stated views | Typed claim sources; autobiographical topic type has stricter attribution | Explicit tagging in evidence ledger |
| **Stale high-confidence carryover** | Old well-supported claim persists at HIGH after evidence shifts | Claim-level decay; monthly re-verification | Time-based confidence downgrade |
| **Feedback loop drift** | Hallucination ingested and amplified across versions | Claims verified against primary sources, not previous syntheses | Claim-first pipeline; no document-to-document inheritance |

---

## 6. Data Architecture

### 6.1 Separation from LCM

The epistemic store is a separate database (`epistemic.db`). Rationale: different write patterns (LCM append-heavy, topics update-heavy), upstream compatibility with lossless-claw, and OpenClaw plugin architecture conventions.

Cross-references use LCM summary IDs (`sum_xxx` strings) as loose foreign keys.

### 6.2 Schema

```sql
-- Core topic store
CREATE TABLE topics (
    id              INTEGER PRIMARY KEY,
    label           TEXT NOT NULL,
    type            TEXT NOT NULL,    -- entity | project | conceptual | autobiographical
    centroid        BLOB,
    summary_count   INTEGER DEFAULT 0,
    centroid_at_last_synthesis BLOB,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

-- Topic-to-summary mapping
CREATE TABLE topic_summaries (
    topic_id    INTEGER REFERENCES topics(id),
    summary_id  TEXT NOT NULL,
    similarity  REAL,
    tagged_at   TEXT NOT NULL,
    PRIMARY KEY (topic_id, summary_id)
);

-- Associative network
CREATE TABLE topic_edges (
    topic_a_id          INTEGER REFERENCES topics(id),
    topic_b_id          INTEGER REFERENCES topics(id),
    weight              REAL DEFAULT 0.5,
    last_reinforced_at  TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    PRIMARY KEY (topic_a_id, topic_b_id)
);

-- Evidence ledger: atomic claims (source of truth)
CREATE TABLE claims (
    id              INTEGER PRIMARY KEY,
    topic_id        INTEGER REFERENCES topics(id),
    text            TEXT NOT NULL,
    type            TEXT NOT NULL,    -- extractive | abductive | analogical
    confidence      TEXT NOT NULL,    -- HIGH | MED | LOW
    support_count   INTEGER DEFAULT 1,
    status          TEXT DEFAULT 'active',  -- active | decayed | disputed | retracted
    first_seen      TEXT NOT NULL,
    last_reinforced TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

-- Claim provenance: which summaries support each claim
CREATE TABLE claim_sources (
    claim_id    INTEGER REFERENCES claims(id),
    summary_id  TEXT NOT NULL,
    PRIMARY KEY (claim_id, summary_id)
);

-- Claim contradictions
CREATE TABLE claim_contradictions (
    claim_a_id  INTEGER REFERENCES claims(id),
    claim_b_id  INTEGER REFERENCES claims(id),
    detected_at TEXT NOT NULL,
    PRIMARY KEY (claim_a_id, claim_b_id)
);

-- Versioned synthesis documents (generated FROM claim set)
CREATE TABLE synthesis_versions (
    id          INTEGER PRIMARY KEY,
    topic_id    INTEGER REFERENCES topics(id),
    version     INTEGER NOT NULL,
    content     TEXT NOT NULL,
    brief       TEXT NOT NULL,        -- injection brief
    token_count INTEGER,
    brief_token_count INTEGER,
    claim_snapshot TEXT,              -- JSON: claim IDs included in this version
    status      TEXT DEFAULT 'current',  -- current | superseded | failed
    created_at  TEXT NOT NULL
);

-- Verification audit log
CREATE TABLE verification_log (
    id          INTEGER PRIMARY KEY,
    claim_id    INTEGER REFERENCES claims(id),
    result      TEXT NOT NULL,    -- verified | unsupported | overstated | misattributed
    detail      TEXT,
    checked_at  TEXT NOT NULL
);

-- Untagged summary pool
CREATE TABLE untagged_summaries (
    summary_id  TEXT PRIMARY KEY,
    embedding   BLOB,
    created_at  TEXT NOT NULL
);

-- Activation decision log (for threshold tuning)
CREATE TABLE activation_log (
    id          INTEGER PRIMARY KEY,
    session_id  TEXT,
    topic_id    INTEGER REFERENCES topics(id),
    similarity  REAL,
    activated   INTEGER,
    referenced  INTEGER,
    logged_at   TEXT NOT NULL
);
```

### 6.3 Storage Interface Abstraction

```
TopicStore (interface)
  ├── tag_summary(summary_id, embedding) → topic_id | null
  ├── discover_topics(untagged_embeddings) → new_topics[]
  ├── get_neighbors(topic_id, depth, min_weight) → topic[]
  ├── activate(query_embedding, top_k) → activated_topics[]
  ├── extract_claims(topic_id, summaries) → claims[]
  ├── verify_claims(claims[]) → verification_results[]
  ├── get_claims(topic_id, min_confidence, status) → claims[]
  ├── store_synthesis(topic_id, content, brief, claim_ids) → version_id
  ├── get_brief(topic_id) → brief_text
  ├── decay_claims(rules) → decayed_count
  ├── update_edges() → void
  └── decay_edges(decay_factor, prune_threshold) → pruned_count
```

Initial: `SQLiteTopicStore`. The abstraction supports future migration without touching plugin logic.

### 6.4 Scale Considerations

| Timeframe | Est. Topics | Est. Claims | Est. Edges | SQLite Viable? |
|-----------|-------------|-------------|------------|----------------|
| 1 month   | 20-50       | 200-500     | 100-300    | Yes            |
| 6 months  | 200-500     | 2,000-5,000 | 2,000-5,000 | Yes           |
| 1 year    | 500-2,000   | 5,000-20,000 | 10,000-50,000 | Yes         |
| 3 years   | 2,000-10,000 | 20,000-100,000 | 50,000-200,000 | Likely   |
| 5+ years  | 10,000+     | 100,000+    | 500,000+   | Monitor        |

The human mind holds thousands of interconnected concepts. A persistent agent running for years should aspire to comparable richness. The storage abstraction exists for when SQLite no longer suffices.

---

## 7. Privacy and Security

### 7.1 The Sensitivity Problem

Synthesis documents distill the evolving essence of topics — potentially spanning months of personal experiences, mental health patterns, relationship dynamics, or professional knowledge. A topic's evidence ledger may contain more actionable personal information than any 50 individual conversation summaries.

### 7.2 Mitigations

- **Encryption at rest:** `epistemic.db` should be encrypted. The information density justifies stronger-than-default protection.
- **Access control:** Evidence ledgers and synthesis documents accessible only to the agent serving the human who generated them.
- **Right to delete:** The human can delete a topic and all associated claims, synthesis versions, and edge weights. Understanding that was synthesized is still the human's data.
- **Typed disclosure policies:**

| Topic Type | System-Derived Observations | Proactive surfacing? |
|-----------|---------------------------|---------------------|
| Entity-centric | Full disclosure | Yes |
| Project/process | Full disclosure | Yes |
| Conceptual | Disclosure with hedging | When HIGH confidence |
| Autobiographical | Flagged, not volunteered | Only on human request |

Autobiographical topics require the most care. "Your integration practice has become more avoidant" may be accurately grounded in evidence — but proactively surfacing it is a different ethical decision than storing it.

### 7.3 The Disagreement Dynamic

When a synthesis contains claims that contradict the human's self-perception:

1. **Surface, don't assert.** Present as observation with cited evidence, not as authoritative truth.
2. **Confidence-gated disclosure.** Only `[HIGH]` contradictions surfaced proactively. `[MED]` and `[LOW]` remain internal.
3. **Human override.** If the human rejects a claim, it gets status `disputed` — excluded from injection briefs but retained in the evidence ledger.

---

## 8. Relationship to Prior Work

The field of agent memory is evolving rapidly. We situate Epistemic Synthesis within the current landscape by engaging deeply with the most relevant systems and benchmarks, organized by the type of memory operation they perform.

### 8.1 Reflection-Based Systems

#### 8.1.1 Generative Agents — Park et al., UIST 2023

The clearest prior art on reflection as a distinct memory operation. Generative Agents (arXiv:2304.03442) maintain a memory stream of natural language observations and periodically synthesize higher-level reflections. The mechanism is specific: when the cumulative importance scores of recent events exceed a threshold (~2-3 times per simulated day), the system retrieves the 100 most recent memories, prompts the LLM for "the 3 most salient high-level questions," then generates "5 high-level insights" with cited source numbers (e.g., "Klaus Mueller is dedicated to his research on gentrification (because of 1, 2, 8, 15)"). Reflections are stored in the memory stream alongside observations and can recursively generate further reflections.

**What they got right:** Reflection as a first-class operation, importance-triggered rather than time-triggered, and source citation in reflections. These are design choices we adopt.

**Where Epistemic Synthesis diverges:** Generative Agents' reflections are behavioral self-assessments about the agent's world ("I am increasingly interested in urban planning"), not structured claims about external topics. They have no verification pass — the LLM's reflection is accepted as-is. They have no contradiction tracking — if Reflection A says X and Reflection B says not-X, both coexist without tension. And critically, reflections are free-text stored in the same stream as observations, with no structural distinction between claim types, confidence levels, or provenance granularity. Epistemic Synthesis produces typed, verified, atomic claims organized into topic networks — a fundamentally different unit of epistemic update.

#### 8.1.2 RMM / Reflective Memory Management — Tan et al., ACL 2025

RMM (arXiv:2503.08026) integrates forward-looking (prospective) and backward-looking (retrospective) reflection for long-term personalized dialogue. Prospective Reflection dynamically summarizes interactions across multiple granularities — utterances, turns, and sessions — into a topic-segmented memory bank. This is closer to our Phase 1 (topic tagging) than a surface reading suggests: RMM decomposes conversations into topic-based segments using an LLM, associating each segment with raw dialogue snippets, rather than using fixed conversational boundaries. Retrospective Reflection then refines retrieval using reinforcement learning, with citation signals from LLM responses serving as reward signals for a reranker.

**What they got right:** Topic-based memory segmentation outperforms fixed granularity (their Figure 1 demonstrates this on LongMemEval). RL-based retrieval refinement using citation signals is elegant — it learns what to retrieve from what the model actually uses.

**Where Epistemic Synthesis diverges:** RMM's topic segments are retrieval-optimized summaries, not understanding artifacts. They improve *what gets retrieved*; they don't consolidate *what has been learned*. RMM has no claim extraction, no contradiction tracking, no cross-topic associative network, and no versioned synthesis. The prospective reflection produces better retrieval targets; Epistemic Synthesis produces evolving knowledge structures. RMM achieves 10%+ accuracy improvement on LongMemEval — but LongMemEval measures retrieval and reasoning, not synthesis quality. Our systems target different evaluation surfaces.

#### 8.1.3 A-MEM — Xu et al., NeurIPS 2025

A-MEM (arXiv:2502.12110) proposes agentic memory inspired by the Zettelkasten method — the personal knowledge management system where notes actively link to and contextualize each other. When a new memory is added, A-MEM generates a structured note with contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Critically, new memories can trigger updates to the contextual representations and attributes of existing memories — the knowledge network continuously refines itself.

**What they got right:** Memory evolution — new information updates existing knowledge, not just appends to it. The Zettelkasten principle that knowledge is a network of interconnected notes, not a filing cabinet, directly parallels our topic graph. Dynamic linking based on semantic similarity is the same intuition as our edge weight computation.

**Where Epistemic Synthesis diverges:** A-MEM operates at the individual memory level — each note is a self-contained unit with links. It does not aggregate memories into topic-level understanding. There is no synthesis step that transforms N related memories into a consolidated claim set. The linking is bidirectional and flat; our topic network has weighted, decaying edges with spreading activation. A-MEM's memory evolution (updating old notes when new ones arrive) is powerful but unverified — there is no grounding check on whether the update is faithful to the original memory's content. Our claim verification pass addresses this gap.

### 8.2 Knowledge Graph / Structured Memory Systems

#### 8.2.1 Zep/Graphiti — Rasmussen et al., 2025

Zep (arXiv:2501.13956) introduces Graphiti, a temporally-aware knowledge graph engine that dynamically synthesizes unstructured conversational data and structured business data while maintaining historical relationships. It uses bitemporal modeling (valid time vs. transaction time) and contradiction resolution via edge invalidation. On the DMR benchmark, Zep achieves 94.8% (vs. MemGPT's 93.4%) and on LongMemEval shows up to 18.5% accuracy improvement with 90% latency reduction over baselines. Its focus is enterprise: cross-session information synthesis and long-term context maintenance.

**Key contrast:** Graphiti operates on entity-predicate-entity triples. When facts change, the old edge gets invalidated (tinvalid set) and a new edge is created. This is temporal fact management, not epistemic synthesis. Graphiti tells you "Martin Ball's website platform changed from WordPress to Ghost on February 15" — it tracks factual transitions. Epistemic Synthesis maintains competing claims with evidence: "Our approach to Martin Ball's web presence has evolved [claim:C-042, HIGH, src:sum_a1f,sum_b2c,sum_d4e]." The unit of knowledge differs: triple vs. grounded claim.

#### 8.2.2 Mem0 — Chhikara et al., 2025

Mem0 (arXiv:2504.19413) is a production-oriented memory architecture that dynamically extracts, consolidates, and retrieves salient information. Its enhanced variant Mem0^g layers a graph-based store to capture relational structures across sessions. On the LOCOMO benchmark, Mem0 achieves 26% higher accuracy than OpenAI's memory, 91% lower p95 latency, and 90% token savings vs. full-context approaches. These are significant production metrics.

**What they got right:** Practical memory at scale — the 91% latency and 90% token cost reductions demonstrate that structured memory is not just more accurate but more deployable. The graph variant (Mem0^g) shows ~2% improvement over base, confirming that relational structure adds value even at production scale.

**Where Epistemic Synthesis diverges:** Mem0's consolidation operates on individual memory entries — extracting, deduplicating, and updating discrete facts. It is sophisticated fact management with excellent engineering. Epistemic Synthesis aims at a different abstraction level: topic-scoped understanding built from verified claim sets. Mem0 would tell you the facts about psychedelic integration; Epistemic Synthesis would tell you what the agent has learned about integration as a concept, with tracked confidence, contradictions, and evolving perspectives. The systems could be complementary: Mem0 for fact-level memory, Epistemic Synthesis for concept-level knowledge.

### 8.3 Retrieval-First Systems

#### 8.3.1 HippoRAG 2 — Gutiérrez et al., ICML 2025

HippoRAG 2 (arXiv:2502.14802) is a neurobiologically inspired framework comprising an artificial neocortex (LLM), parahippocampal region encoder, and open knowledge graph. It builds on Personalized PageRank (PPR) from HippoRAG and enhances it with deeper passage integration and recognition memory for filtering triples. The query-to-triple approach outperforms NER-to-node by 12.5% on Recall@5. It achieves 7% improvement on associative memory tasks while maintaining factual recall — solving the accuracy degradation that plagued earlier graph-augmented RAG systems.

**Relevance to our design:** HippoRAG 2's Personalized PageRank is directly analogous to our spreading activation. Both traverse a graph from seed nodes with decaying influence. The key difference: HippoRAG 2's graph is a knowledge graph of extracted triples, optimized for retrieval. Our graph is a topic network of weighted associations, optimized for context injection of synthesis documents. HippoRAG 2's recognition memory (filtering irrelevant triples before PPR) parallels our claim verification (filtering unsupported claims before synthesis).

**Critical insight from EcphoryRAG (arXiv:2510.08958):** A subsequent paper critiques HippoRAG 2 for not fully replicating cue-based memory retrieval — the principle that partial cues trigger reconstruction of full memories (ecphory). This is relevant to our activation model: our current centroid-based topic matching is a coarse cue. Future refinement should support partial-cue activation where a fragment of a topic triggers retrieval of the full synthesis.

### 8.4 Benchmarks

Understanding the evaluation landscape is critical because Epistemic Synthesis targets a capability that no existing benchmark directly measures.

#### 8.4.1 LoCoMo — Maharana et al., ACL 2024

LoCoMo (arXiv:2402.17753) evaluates very long-term conversational memory with conversations 16x longer than MSC, distributed over 10x more turns and 5x more sessions. Tasks include single-hop QA, multi-hop QA, temporal reasoning, and event summarization. Key finding: even state-of-the-art systems show significant degradation on multi-session reasoning.

**Relevance:** LoCoMo tests retrieval and reasoning over long conversation histories — the *input* to our system. Strong LoCoMo performance means the memory layer successfully retrieves relevant information. It does not test whether that information has been consolidated into reusable understanding. A system could score perfectly on LoCoMo while having zero synthesis capability.

#### 8.4.2 LongMemEval — Wu et al., ICLR 2025

LongMemEval (arXiv:2410.10813) benchmarks five core abilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention. With 500 questions across scalable chat histories (115K to 1.5M tokens), it reveals a 30% accuracy drop for commercial systems on sustained interactions. Its three-stage framework (indexing → retrieval → reading) identifies that session decomposition, fact-augmented key expansion, and time-aware query expansion each independently improve performance.

**Relevance:** LongMemEval's "knowledge updates" task is the closest existing benchmark to our domain — it tests whether the system correctly tracks information that changes over time. But it tests this at the fact level (did the system update a specific data point?), not at the concept level (did the system revise its understanding?). Epistemic Synthesis needs a benchmark that tests conceptual revision — "Given that three recent conversations about integration contradicted earlier assumptions, does the system's current understanding reflect the revised position?"

#### 8.4.3 REALTALK — Lee et al., 2025

REALTALK (arXiv:2502.13270) provides 21 days of authentic messaging app dialogues — real conversations, not synthetic. Its two benchmarks are persona simulation (continue a conversation as a specific user) and memory probing (answer questions requiring long-term memory). Key finding: models struggle to simulate users from dialogue history alone, and face significant challenges leveraging long-term context in real-world (vs. synthetic) conversations.

**Relevance:** REALTALK's use of real conversations exposes brittleness that synthetic benchmarks miss — diverse emotional expressions, persona instability, and natural topic drift. This is the environment Epistemic Synthesis must work in. Our topic discovery must handle messy, real-world conversation patterns, not clean synthetic dialogues. REALTALK's memory probing is recall-oriented; we need a synthesis-probing analog.

### 8.5 Positioning Within the Landscape

| System | Unit of Knowledge | Memory Operation | Verification | Temporal Awareness |
|--------|------------------|-----------------|-------------|-------------------|
| Generative Agents | Free-text reflection | Importance-triggered synthesis | None | Recency scoring |
| RMM | Topic-segmented summary | Multi-granularity prospective + RL retrospective | Citation-based reward | Session boundaries |
| A-MEM | Zettelkasten note | Dynamic linking + mutual update | None | Recency in retrieval |
| Zep/Graphiti | Entity-predicate triple | Bitemporal graph maintenance | Edge invalidation | Full bitemporal model |
| Mem0/Mem0^g | Memory entry + graph edges | Extract, consolidate, retrieve | Deduplication | Session-level |
| HippoRAG 2 | KG triple + passage | PPR graph traversal | Recognition memory filtering | None explicit |
| LCM | Temporal summary | DAG compaction | None | Positional/temporal |
| **Epistemic Synthesis** | **Typed atomic claim** | **Topic-scoped claim extraction + verified synthesis** | **Claim-level verification + primary-source re-audit** | **Claim decay + edge decay** |

The table reveals our niche: Epistemic Synthesis is the only system that combines typed knowledge units, explicit verification, claim-level temporal decay, and topic-scoped synthesis. The tradeoff is complexity and cost — more LLM calls per synthesis cycle than any system listed.

### 8.6 Human Sleep Consolidation — Rasch & Born, 2013

The brain's memory consolidation during slow-wave sleep — hippocampal replay into neocortical schemas — remains the biological motivation for overnight batch synthesis.

We use this analogy to frame the problem, not to claim biological fidelity. The mechanism is engineering; the inspiration is neuroscience. Recent surveys on LLM memory systems (including graph-memory taxonomies from 2025-2026) show a wide range of memory abstractions, and the better papers are careful not to treat a brain analogy as evidence that the architecture is correct. We follow that discipline: the sleep consolidation framing motivates the *why*; the claim-first pipeline specifies the *how*.

---

## 9. Evaluation

### 9.1 The Measurement Problem

Measuring synthesis quality is harder than measuring retrieval accuracy. Existing benchmarks evaluate competencies adjacent to — but distinct from — what Epistemic Synthesis produces:

- **LoCoMo** (Maharana et al., ACL 2024) tests single-hop, multi-hop, temporal, and open-domain QA over long conversations. It measures whether the system can *find* relevant information across sessions.
- **LongMemEval** (Wu et al., ICLR 2025) tests information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention. Its "knowledge updates" task comes closest — testing whether the system tracks changed facts — but operates at the fact level, not the concept level.
- **REALTALK** (Lee et al., 2025) tests persona simulation and memory probing against real (not synthetic) conversations. It reveals that models struggle with real-world conversational patterns, but tests recall, not synthesis.

All three benchmarks ask: "Can the system retrieve and reason over stored information?" None asks: "Has the system consolidated information into reusable understanding that improves future performance?" This is the evaluation gap. No standard benchmark exists for epistemic synthesis quality, and building one requires defining what "good synthesis" means operationally. This remains the single biggest constraint on empirical progress.

### 9.2 Proposed Metrics

**Quantitative (automated):**
- **Token efficiency:** Context tokens with injection briefs vs. equivalent raw summaries. Target: >60% reduction.
- **Grounding rate:** Percentage of claims passing verification. Target: >95%.
- **Claim diversity:** Ratio of unique claims to source summaries per topic. Too low = under-extraction. Too high = over-generation.
- **Contradiction coverage:** Percentage of detected contradictions preserved in Active Tensions vs. laundered into resolution.
- **Retrieval displacement:** When briefs are available, how often does the agent still need raw summaries? Lower is better.

**Qualitative (human-evaluated, periodic):**
- **Blind comparison:** Present the human with two agent responses — one using injection briefs, one using raw summaries. Which demonstrates better topic knowledge?
- **Novel connections:** Does the agent surface cross-topic relationships the human hadn't explicitly made?
- **Productive contradiction:** Does the agent identify tensions in the human's evolving position?

**Structural (automated, periodic):**
- **Drift rate:** Synthesis document changes between versions without new source material.
- **Coverage:** Percentage of summaries tagged to at least one topic.
- **Network health:** Average edges per topic (too low = isolation, too high = undifferentiated). Clustering coefficient. Connected components.

### 9.3 What Would Demonstrate Success

The honest framing: this system produces **auditable, evolving topic models** — not "understanding" in the philosophical sense. Whether this constitutes genuine understanding is academically interesting but practically irrelevant if the outputs improve agent performance.

The test cases that would demonstrate synthesis beyond retrieval:
1. **Cross-topic insight:** Connecting patterns across independently discussed topics
2. **Temporal pattern recognition:** Identifying behavioral shifts over time
3. **Productive contradiction:** Noticing gaps between stated positions and observed patterns

If the system produces these outputs with proper grounding and verifiable provenance, it is doing something that pure retrieval cannot match — regardless of what we call it.

### 9.4 Toward a Synthetic Benchmark for Synthesis Quality

No existing benchmark tests whether a system has consolidated information into reusable understanding. Building one requires controllable ground truth. The following sketch outlines what such a benchmark could look like:

**SynthEval: A Synthetic Benchmark for Epistemic Synthesis**

**Construction:**

```
1. Design 10-20 topics, each with a known "ground truth" understanding
   that includes:
   - 5-10 established claims
   - 2-3 active contradictions (claims that genuinely conflict)
   - 1-2 temporal evolutions (a position that changed over time)
   - 1-2 cross-topic connections

2. Generate synthetic conversation histories (15-30 conversations per
   topic) that embed the ground-truth claims across sessions:
   - Some claims stated explicitly (extractive ground truth)
   - Some inferable from patterns across conversations (abductive)
   - Contradictions distributed across different time periods
   - Position changes occurring at known conversation boundaries

3. Feed the conversation histories through the epistemic pipeline.

4. Evaluate the output against ground truth.
```

**Proposed metrics:**

| Metric | Measures | Ground Truth Available? |
|--------|----------|----------------------|
| **Claim recall** | Did the system extract the known claims? | Yes — compare against planted claims |
| **Claim precision** | Did the system avoid hallucinated claims? | Yes — anything not in the planted set is false |
| **Contradiction preservation** | Are planted contradictions in Active Tensions, not resolved? | Yes — known contradiction pairs |
| **Temporal fidelity** | Does the system reflect the final position after an evolution? | Yes — known evolution endpoints |
| **Confidence calibration** | Do multi-source claims get higher confidence than single-source? | Yes — support count is controlled |
| **Cross-topic linkage** | Did the system discover planted cross-topic connections? | Yes — known connection points |

**Limitations of this approach:**

- Synthetic conversations lack the messiness of real dialogue (REALTALK demonstrates this gap)
- Ground truth is planted by the benchmark designer, introducing designer bias
- Tests the pipeline's ability to recover *known* knowledge, not its ability to discover *novel* understanding — which is arguably the system's primary value proposition

This benchmark is constructable with current tools and would provide the first quantitative evaluation of synthesis quality distinct from retrieval accuracy. It would not validate the system's performance on real conversations — but it would establish a baseline and catch regressions.

---

## 10. Multi-Source Knowledge Ingestion

### 10.1 The Single-Source Limitation

Sections 1–9 assume a single knowledge source: LCM conversation summaries. In practice, a persistent agent's knowledge is distributed across two fundamentally different substrates:

1. **Conversations** — the generative medium where ideas originate, get debated, refined, and stress-tested
2. **Files** — the artifacts where knowledge crystallizes: project documentation, call transcripts, strategy documents, research notes, code

The current architecture ingests only conversation summaries. When a human hands the agent a 45-minute client call transcript, the agent reads it, summarizes it in conversation, and that summary enters LCM. The epistemic system then extracts claims from a summary of a summary of the original material. This is a lossy pipeline — the richness of the source is filtered through two compression layers before reaching the claim extractor.

### 10.2 The Provenance Problem

The naive assumption — "files are primary sources, conversations are secondary" — is wrong. Knowledge does not originate in files. It originates in conversations.

Consider the development of this very system: Epistemic Synthesis emerged from dozens of conversations — architectural debates, whitepaper drafts, peer reviews, pushback sessions where ideas were killed or refined. The whitepaper is not the source of the knowledge; it is a *crystallization* of knowledge that was generated through dialogue. The GitHub README crystallizes the crystallization further.

Conversely, when the human asks the agent to read a client transcript and discuss it, the conversation *about* the file generates new understanding that goes beyond what the file contains. The file is the starting material; the conversation is the refinement.

Neither conversations nor files are "the source." **Understanding emerges from the interaction between both.** The relationship is not hierarchical but cyclical:

```
Conversations (generative)
  → insights emerge
    → files written (crystallized)
      → files discussed further (refined)
        → files updated (evolved)
          → new conversations reference updated files
```

This means the epistemic system cannot simply ingest files alongside conversations and weight them independently. It must model the **provenance chain** — the lineage through which a piece of knowledge traveled from origin to current form.

### 10.3 The Checkpoint + Delta Model

A practical retrieval model emerges from acknowledging the cyclical relationship:

**A file is a checkpoint.** It represents the crystallized state of knowledge about a topic as of its last modification date. Everything the human and agent knew, discussed, and decided about that topic — up to that point — is captured in the file.

**Conversations after the checkpoint are the delta.** Any discussions that occurred after the file was last modified represent evolution, revision, or expansion of the knowledge the file contains.

```
Current understanding of Topic X =
  File(s) for Topic X (checkpoint, last modified T₁)
  + Conversations about Topic X since T₁ (delta)
```

This model has several advantages:

1. **Eliminates redundant retrieval.** Conversations that occurred before T₁ are already absorbed into the file. Retrieving them alongside the file double-counts.
2. **Reduces token cost.** The file is typically more compressed than the sum of conversations it synthesizes.
3. **Naturally handles staleness.** A file modified 6 months ago with 50 conversations since has a large delta — signaling the file needs updating. A file modified yesterday with zero conversations since has no delta — the checkpoint is current.
4. **Matches human intuition.** "Let me check my notes, and what have we discussed since then?" is the natural information-seeking pattern.

### 10.4 Unsolved: Multi-Source Topic Mapping

The checkpoint + delta model works cleanly for the single-file case. Real topics have multiple files and multiple conversation threads. This creates hard problems:

#### 10.4.1 Multiple Files Per Topic

A client topic like "Martin Ball" might have:
- `projects/martin-ball/PROJECT.md` (project status, last modified Mar 8)
- `projects/martin-ball/transcripts/call-2026-03-10.md` (call transcript, Mar 10)
- `projects/martin-ball/strategy.md` (business strategy, Mar 5)
- `projects/martin-ball/website-audit.md` (SEO audit, Feb 28)

Each file has a different last-modified date. Each represents a different facet of the topic. Which is "the checkpoint"? All of them? The newest? Do we merge them into a single baseline?

The answer likely depends on the query. If the question is about SEO performance, the website audit is the relevant checkpoint. If it's about overall client status, the PROJECT.md is primary. **The checkpoint is query-conditioned, not fixed per topic.**

#### 10.4.2 Overlapping and Duplicate Files

The agent writes a strategy document. A week later, the human asks for a "Martin Ball overview" and the agent covers half the same ground. Now two files claim to be checkpoints for overlapping knowledge. Using both as baselines double-counts. Picking one loses what's unique to the other.

This requires **file-level deduplication** — semantic comparison between files within a topic to identify overlap, unique content, and supersession relationships. A file that is a strict subset of a later file can be treated as superseded. Files with partial overlap need their unique contributions extracted.

#### 10.4.3 Multiple Conversation Threads

Conversations about Martin Ball span SEO discussions, business strategy sessions, website redesign threads, and psychedelic space positioning. Some are 90% about Martin Ball; others mention him in passing. Relevance is not binary — it's a spectrum, and a conversation might be relevant to the SEO facet but not the business strategy facet of the same topic.

#### 10.4.4 The Mapping Problem

Before solving retrieval, the system must solve **topic-to-source mapping**: which files and which conversations authoritatively belong to a topic? The current system maps LCM summaries to topics via embedding similarity. Adding files requires a parallel mapping — potentially using semantic search (e.g., GrepAI, which already indexes the workspace) to discover relevant files per topic.

But embedding similarity alone produces fuzzy matches. A file about "SEO best practices" might match both the "Martin Ball" topic (he's an SEO client) and the "SEO Growth Strategy" topic (general methodology). The mapping needs to be **aware of partial relevance** — a file can belong to multiple topics with different weights.

### 10.5 Retrieval Architecture

Assuming the mapping problem is solved, the retrieval stack for topic activation becomes three layers:

```
Topic activated →
  Layer 1: Synthesis brief (epistemic DB — compressed understanding)
  Layer 2: LCM expand (conversation depth — delta since checkpoint)
  Layer 3: File retrieval (file depth — checkpoint content)
```

Each layer goes deeper only when needed. The synthesis brief (Layer 1) handles most queries. When the agent needs more detail, it drills into conversations (Layer 2) or source files (Layer 3).

**Layer 3 retrieval** leverages existing infrastructure: GrepAI provides semantic search over the workspace filesystem with pre-computed embeddings and file-change watching. The epistemic system does not need its own file crawler — it queries GrepAI for files relevant to the activated topic, filtered by the topic-to-file mapping.

**Time-filtered retrieval** applies the checkpoint + delta model: for each relevant file, retrieve only LCM summaries with timestamps *after* that file's last modification. This eliminates the redundancy of loading conversations that the file already represents.

### 10.6 Impact on the Claim Pipeline

File-sourced claims differ from conversation-sourced claims in important ways:

1. **Provenance tracking** expands. Each claim needs a source type (`lcm_summary` | `file`) and, for file sources, a file path and modification timestamp.
2. **Verification changes.** A claim extracted from a file the agent itself wrote has a different verification profile than a claim from a client transcript the human provided. The agent's own files may contain its own prior confabulations — they should not be treated as independent corroboration.
3. **Deduplication across sources.** The same knowledge will appear in both files and conversations. The dedup layer (currently embedding-based cosine similarity) must handle cross-source matches — keeping the most refined version regardless of source type.
4. **The origination chain matters for supersession.** If a conversation claim was later crystallized into a file, and the file was later discussed and revised in a subsequent conversation, the supersession chain is: conversation₁ → file → conversation₂. The latest point in the chain is the current truth, regardless of source type.

### 10.7 Design Principles

From this analysis, several design principles emerge:

1. **Neither source type is inherently primary.** Weight by recency and refinement stage, not by whether it came from a file or a conversation.
2. **The checkpoint is query-conditioned.** Different files serve as checkpoints for different facets of the same topic.
3. **Provenance is a chain, not a label.** The system must track how knowledge flows between conversations and files, not just which source it was extracted from.
4. **Get the storage model right first.** Every downstream operation — sourcing, synthesis, retrieval, injection — depends on the data model. An incorrect storage model creates compounding errors across all layers.
5. **Existing infrastructure handles file retrieval.** GrepAI for file discovery, LCM for conversation history. The epistemic system orchestrates; it doesn't duplicate.

### 10.8 Status

Multi-source ingestion is at the design stage. The current implementation processes only LCM summaries. The problems identified in Sections 10.4–10.6 — multi-file mapping, overlap detection, query-conditioned checkpoints, cross-source deduplication — require formal specification before implementation. The risk of building on an incorrect storage model is that every downstream layer inherits the error, creating multiple adjustment points instead of one.

---

## 11. Open Questions

1. **Topic split/merge mechanics:** How are claims reassigned during splits? How are contradictory synthesis documents merged? This needs formal specification.

2. **Cross-agent synthesis:** Can agents share evidence ledgers? What are the privacy implications?

3. **Cold start threshold:** ~150 summaries (one month) appears sufficient for clustering. Below that, the system should gracefully degrade to raw summaries.

4. **Synthesis prompt evolution:** Should the extraction and synthesis prompts evolve based on evaluation metrics? This creates a meta-optimization problem.

5. **Multi-human agents:** Per-human vs. shared topics in team contexts. Non-trivial privacy implications.

6. **Claim-level query conditioning:** Future activation should weight claims by relevance to the current query, not just topic similarity. This is the refinement from static spreading activation to dynamic, context-aware retrieval.

7. **Topic-to-file mapping:** How does the system discover and maintain authoritative mappings between topics and workspace files? Embedding similarity alone produces fuzzy, potentially incorrect mappings. Manual declaration doesn't scale. The solution likely combines semantic search with co-reference signals (files mentioned in tagged conversations).

8. **Query-conditioned checkpoints:** When multiple files exist for a topic, which serves as the retrieval checkpoint? Static assignment is insufficient — the relevant checkpoint depends on the current query's facet of the topic. This requires either faceted topic modeling or runtime checkpoint selection.

9. **Cross-source claim deduplication:** How to handle identical knowledge appearing in both files and conversations without double-counting, while preserving the most refined version? The origination chain (conversation → file → conversation) complicates simple recency-based supersession.

10. **Agent-authored file verification:** Files written by the agent may contain prior confabulations. Treating agent-authored files as independent corroboration creates a feedback loop analogous to the Lobster Confabulation (Appendix A). The verification pass must distinguish between human-provided files and agent-generated files.

---

## 12. Implementation Roadmap

| Phase | Scope | Dependencies |
|-------|-------|-------------|
| **POC** (completed) | HDBSCAN clustering of LCM summaries, LLM labeling | nomic-embed, Python, lcm.db |
| **Phase 1** | Plugin scaffold, SQLite store, claim schema, Phase 2 (discovery) as cron | OpenClaw plugin API |
| **Phase 2** | Phase 1 (real-time tagging) on LCM compaction hook | lossless-claw event API |
| **Phase 3** | Claim extraction pipeline, verification pass, evidence ledger | LCM expand API |
| **Phase 4** | Canonical synthesis generation, injection brief, passive injection via memory files | Phase 3 stable |
| **Phase 5** | Evaluation framework, activation logging, threshold tuning | 30+ days of operational data |
| **Phase 6** | Active context injection via LCM assembler | lossless-claw PR or hook |
| **Phase 7** | Associative network, spreading activation, edge/claim decay | Phase 1-6 stable |
| **Phase 8** | Multi-source data model: provenance chains, source types, file-to-topic mapping | Phase 7 stable, GrepAI |
| **Phase 9** | File ingestion pipeline: claim extraction from workspace files, cross-source dedup | Phase 8 schema |
| **Phase 10** | Checkpoint + delta retrieval: query-conditioned file checkpoints, time-filtered LCM | Phase 9, GrepAI integration |

---

## 13. Conclusion

Epistemic Synthesis addresses a genuine gap in AI memory architecture: the absence of structured, auditable topic understanding that evolves over time. By treating conversation summaries — and, in future phases, workspace files — as raw material for bottom-up topic discovery, extracting and verifying atomic claims, building associative networks between topics, and generating synthesis documents from verified claim sets, the system bridges the gap between episodic memory (what happened) and semantic knowledge (what has been learned).

The core architectural decision — claim-first synthesis — makes the system auditable, drift-resistant, and evaluable in ways that document-first approaches cannot achieve. The unit of epistemic update is the claim, not the document. Documents are views; claims are truth.

The v0.4 design recognizes that knowledge does not live in one place. Conversations are the generative medium where understanding originates. Files are the crystallized artifacts where it lands. Neither is "the source" — understanding emerges from the interaction between both, flowing cyclically from conversations to files and back. The checkpoint + delta retrieval model reflects this reality: files serve as checkpoints of crystallized knowledge, conversations since the checkpoint serve as the evolutionary delta. Getting this multi-source storage model right is prerequisite to everything downstream — sourcing, synthesis, retrieval, and injection all inherit correctness or error from the foundation.

The architecture is designed for personal persistent agents — systems that serve one human over months or years, where accumulated knowledge is the primary value proposition. It is complementary to existing memory systems rather than a replacement.

The primary risks are synthesis quality (addressed through claim-level extraction and verification), feedback loop drift (addressed through primary-source grounding and no document-to-document inheritance), multi-source complexity (addressed through provenance chains and query-conditioned checkpoints), and evaluation difficulty (addressed through a multi-dimensional metric framework). The honest acknowledgment: measuring whether an AI system "understands" a topic remains an open problem. What we can measure is whether it performs better with structured topic models than without — and that is sufficient to validate the architecture empirically.

---

## Appendix A: Case Study — The Lobster Confabulation Loop

During development, the authors' existing memory architecture experienced a self-reinforcing misinformation loop that directly motivated the claim-first grounding architecture.

**Timeline:**
1. The agent hallucinated: "Lobster CLI was removed in version 2026.3.2."
2. A fact-extraction system (Metabolism) stored this as a structured fact.
3. A context-injection system (Continuity) served it back as authoritative context.
4. The agent repeated the claim with increasing certainty across sessions.
5. The claim was false. The tool existed and was functional.

**Why claim-first prevents this:**
- The hallucination would need to survive extraction as an atomic claim with cited sources.
- The verification pass would check: does the cited source actually state the tool was removed?
- The source summary would contain no such statement. The claim would be flagged `unsupported`.
- Even if the claim survived verification once (verifier error), monthly re-verification against primary sources would catch it.
- Claim-level confidence decay would downgrade it over time without reinforcing evidence.

No architecture eliminates all confabulation. But claim-first synthesis makes each confabulation individually traceable, verifiable, and time-bounded — rather than buried in prose where it compounds silently.

---

*This paper describes work in progress. Phase 1 (topic discovery/tagging) and Phase 2 (claim extraction/synthesis) are implemented and running in production. v0.4.1 incorporates multi-source knowledge ingestion design, provenance chain modeling, checkpoint + delta retrieval architecture, explicit verification failure mode analysis, cost projections, a synthetic benchmark sketch, and topic split/merge protocols — motivated by operational experience and peer review.*

---

## References

1. Park, J.S., O'Brien, J.C., Cai, C.J., Morris, M.R., Liang, P., & Bernstein, M.S. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST 2023*. arXiv:2304.03442.

2. Rasch, B. & Born, J. (2013). About Sleep's Role in Memory. *Physiological Reviews*, 93(2), 681-766.

3. Collins, A.M. & Loftus, E.F. (1975). A Spreading-Activation Theory of Semantic Processing. *Psychological Review*, 82(6), 407-428.

4. Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., & Chalef, D. (2025). Zep: A Temporal Knowledge Graph Architecture for Agent Memory. arXiv:2501.13956.

5. Ehrlich, M. & Blackman, J. (2026). Lossless Context Management for Large Language Models. Voltropy PBC.

6. Maharana, A., Lee, D.H., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y. (2024). Evaluating Very Long-Term Conversational Memory of LLM Agents. *ACL 2024*. arXiv:2402.17753.

7. Wu, D., Wang, H., Yu, W., Zhang, Y., & Deng, Z. (2025). LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. *ICLR 2025*. arXiv:2410.10813.

8. Lee, D.H. et al. (2025). REALTALK: A 21-Day Real-World Dataset for Long-Term Conversation. arXiv:2502.13270.

9. Chhikara, P. et al. (2025). Building Production-Ready AI Agents with Scalable Long-Term Memory [Mem0]. arXiv:2504.19413.

10. Tan, J. et al. (2025). In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents. *ACL 2025*. arXiv:2503.08026.

11. Gutiérrez, B.J. et al. (2025). HippoRAG 2: Towards Versatile Long-Term Memory for LLMs. arXiv:2502.14802.

12. Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., & Zhang, Y. (2025). A-MEM: Agentic Memory for LLM Agents. *NeurIPS 2025*. arXiv:2502.12110.

13. LangChain (2025). LangMem: Long-Term Memory for AI Agents. https://langchain-ai.github.io/long-term-memory/.
