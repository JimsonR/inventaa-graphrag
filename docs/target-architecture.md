# Target Architecture — Existing-KG Reasoning Layer + Graph Memory

> Status: design only (no code changes). Companion to the candidate-selection plan.
> Scope: multi-tenant, multi-domain WhatsApp commerce concierge. Retrieval layer is
> the MCP server (`inventaa-graphrag`); the LLM agent lives in the consuming
> WhatsApp bot.

## Context

The WhatsApp agent already extracts structured fields from each turn
(`{intent, product_name, category, feature, filters, preferences}`) and **persists
them to a DB as per-session state**. On the next request it calls our MCP tool with
that persisted state. So follow-ups are not starting from bare text — they carry a
structured `focused_entity` / `candidate_skus` payload.

The current flow relies on a `get_taxonomy_context(query)` call **on the raw current
utterance** to produce candidates. This is brittle when:
- the user sends a follow-up with no category/product token ("the cheaper one?"), or
- only `category` was persisted (not an exact `product_name`), so a comparative
  follow-up is ambiguous among the products shown.

The fix is not to abandon taxonomy, but to (a) ground queries with persisted state and
(b) add an **existing-KG reasoning layer** that reasons over the already-shown
candidate set, with taxonomy demoted to a normalizer.

This design follows the *"GraphRAG with Existing KGs" + Hybrid Retrieval* branch of the
Awesome-GraphRAG survey (KAG, Think-on-Graph, HybGRAG), not the corpus-graph
extraction branch (Microsoft GraphRAG / RAPTOR / LightRAG text-mode), because we
already have a curated knowledge graph (Neo4j catalog) + SQLite.

## Sample follow-up walkthrough

Setup (real catalog):
- **T1:** "outdoor gate lights under 1500" → agent persists
  `state = {category:"Gate Light Collections", max_price:1500,
  candidate_skus:["18C-2042"(Athena 18W ₹1525/2yr), "Viva.."(₹872/1yr), ...]}`
- **T2 (follow-up):** "which is cheaper and what's the warranty?"

### Current hybrid (taxonomy → LLM → product_name → Lucene)
- `get_taxonomy_context("which is cheaper and what's the warranty")` → no
  category/product token → keyword match returns almost nothing.
- If `product_name` wasn't locked in T1, falls back to
  `search_catalog(category=..., max_price=1500)` → returns **all** gate lights under
  1500, not the 3 shown. LLM guesses "cheaper" from a wider set → can cite a product
  the user never saw. **Inaccurate on comparative follow-ups.**
- ✅ Fast/cheap; ✅ precise only when exact `product_name` was persisted.
- ❌ Brittle for comparisons/filters when state is thin.

### KAG (existing-KG symbolic reasoning)
- Consumes the persisted `candidate_skus` as the scoped subgraph in Neo4j. Solver runs
  `min(price) over candidate set` → Viva (₹872); retrieves `warranty` via the
  `:HAS_SPEC` edge → "1-Year".
- ✅ Exact on comparative/filter follow-ups (symbolic, not guesswork).
- ✅ Multi-tenant (per-tenant KG + schema). ❌ Heavier setup (KG + solver).

### Think-on-Graph (LLM-guided traversal over existing KG)
- Seeds from `candidate_skus`; LLM decides relations to traverse (price → warranty),
  multi-hop, prunes, returns subgraph. Concludes "Viva cheapest, 1-Year warranty."
- ✅ Robust to paraphrase/partial text (reasons on graph structure, not keywords).
- ✅ Uses Neo4j directly. ❌ More LLM calls/latency; needs traversal guardrails.

**Takeaway:** current hybrid is the right *base* and wins when `product_name` is
persisted; KAG/ToG win exactly on the follow-up cases taxonomy struggles with, and the
persisted `candidate_skus` is the perfect scope for them.

## Target architecture

```
                 WhatsApp agent
        ┌─────────────────────────────────┐
        │ LLM extracts → persists STATE→DB │   ← already exists
        │ {intent, product_name, category, │
        │  feature, filters, candidate_skus}│
        └───────────────┬─────────────────┘
                        │ passes persisted state to MCP
                        ▼
        ┌───── Reasoning Layer (NEW, per tenant) ─────┐
        │  given (grounded query + persisted state):  │
        │                                             │
        │  IF product_name / focused entity known     │
        │     → search_catalog(product_name=...)      │  (current hybrid — fast)
        │                                             │
        │  ELSE IF comparative/filter follow-up       │
        │     → KAG-style symbolic reasoning over     │
        │       state.candidate_skus subgraph in Neo4j│  (Cypher: min price,
        │       → narrowed SKUs → hydrate             │   filter by feature)
        │                                             │
        │  ELSE (new discovery / browse)              │
        │     → ToG-style exploration from seed nodes │  (multi-hop traversal)
        │                                             │
        │  get_taxonomy_context → DEMOTED to          │
        │     normalizer/guardrail (label check)      │
        └───────────────┬─────────────────────────────┘
                        ▼
        ┌───── MCP retrieval service (unchanged core) ─────┐
        │  search_catalog (Lucene + SQLite + vector, RRF)  │
        │  get_product_details, get_taxonomy_context       │
        │  + optional scope_skus=[...] from state          │
        └──────────────────────────────────────────────────┘
```

### Layer 1 — Graph memory = persisted state
Per-session state in DB is already a lightweight entity/intent graph (the "graph
memory" HippoRAG2 / Graphiti argue for). No second graph store needed.
**Action:** ensure the agent persists `candidate_skus` (the shown set), not just
`category`, so comparative follow-ups have a scope.

### Layer 2 — Reasoning layer (new, thin, multi-tenant)
Routes by state richness:
- `product_name` / focused entity known → direct `search_catalog(product_name=...)`.
- comparative/filter follow-up → **KAG-style symbolic reasoning** over
  `state.candidate_skus` in Neo4j (Cypher: `min(price)`, filter by feature) → narrowed
  SKUs → hydrate.
- new discovery / browse → **ToG-style exploration** from category/feature seed nodes
  (multi-hop traversal).
- `get_taxonomy_context` → normalizer/guardrail (confirm labels exist in tenant KG),
  not the primary retriever.

### Layer 3 — MCP retrieval service (unchanged core)
Keep the existing 3-channel hybrid (Lucene + SQLite token + vector, RRF) as the base
retriever — the survey explicitly says *"Don't Forget the Base Retriever."* Add one
optional parameter:

#### `search_catalog` interface addition (optional, tenant-safe)
```
search_catalog(
    query: str,
    limit: int,
    tenant_id: str,
    session_id: str,
    intent_data: {...},
    scope_skus: Optional[List[str]] = None   # NEW: restrict results to this
                                             # persisted candidate set (state.candidate_skus)
)
```
When `scope_skus` is provided, fusion/hydration is constrained to that set — this is
what lets the KAG/ToG reasoning layer answer "the cheaper one" precisely over the
products already shown, instead of re-querying the whole catalog.

## Why this fits the constraints
- **Multi-tenant / multi-domain:** every resource (state, Neo4j graph, indexes,
  taxonomy) is per-`tenant_id`. No hardcoded inventaa keywords.
- **Follows the right survey branch:** existing-KG + hybrid retrieval, not corpus
  graph extraction.
- **Evolves, doesn't rewrite:** current hybrid stays as the fast base path; the
  reasoning layer only engages when state is thin or the query is comparative.
- **Graph memory is free:** it's the state the agent already persists.

## Out of scope / deferred
- KAG solver implementation details (Cypher templates per domain).
- ToG traversal guardrails (max hops, prune strategy).
- Evaluation harness on GraphRAG-Bench / PolyG before committing to KAG vs ToG.
