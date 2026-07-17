# Inventaa GraphRAG MCP & REST Test Payloads

This document contains ready-to-use payloads for testing the tools and resources exposed
by the **Inventaa Tri-Store GraphRAG MCP Server** (`mcp_server.py`) and API endpoints.

> **Updated to match the live code.** Earlier versions of this doc drifted from the
> implementation. The corrections below are verified against `mcp_server.py` and
> `src/query/graphrag_engine.py`.

---

## 0. Server surface (what actually exists)

The MCP server (`python mcp_server.py --transport streamable-http --port 8008`) exposes:

**Tools** (call with `tools/call` / `client.call_tool`):
- `search_catalog(query, limit=6, session_id=None, tenant_id=None, intent_data=None, dialogue_state=None)`
- `get_taxonomy_context(query, threshold=0.80)`
- `get_product_details(sku, tenant_id=None)`

**Resources** (read with `resources/read` / `client.read_resource` — NOT tools):
- `data://catalog/status` — system health & DB stats
- `data://catalog/sku/{sku}` — authoritative product data by SKU

> ⚠️ `get_catalog_status` is a **resource**, not a tool. There is **no**
> `verify_whatsapp_status` tool — that entry was removed from this doc because it does
> not exist in `mcp_server.py`.

### Tenant id

The canonical tenant id is **`inventaa`** (`src/config/base_config.yaml`). The value
`tenant_inventaa_led_001` is an *alias* that is only normalised in `router_api.py`. The
MCP engine path does **not** normalise it, and `src/query/retrieval/graph_search.py`
filters `WHERE p.tenant = $tenant_id`, so passing the alias makes the graph channel
return 0 hits. **Always use `tenant_id: "inventaa"` in MCP payloads.**

### Intents

`GraphRAGEngine` recognises these intents (`src/query/models.py`): `find_product`,
`browse_category`, `faq_knowledge`, `get_product_info`, `get_advice`,
`unknown`. (`check_policy` was removed - the per-product card already carries
warranty/policy detail.)

> ⚠️ `get_product_info`, `get_advice`, and `check_policy` are **intentionally
> excluded** from the test set. `get_product_info` and `get_advice` have no dedicated
> branch and silently fall back to `FIND_PRODUCT` (`graphrag_engine.py:97-105`).
> `check_policy` is also dropped because the per-product **card already carries warranty /
> policy detail** (via `Product.description` / `features` → "1-year replacement warranty"),
> so a separate policy-vector lookup is redundant for the product-list flow. `faq_knowledge`
> maps to `FAQ_KNOWLEDGE` and is the only non-product intent we test (blog / FAQ chunks).

---

## 1. Tool: `search_catalog` (All Intent Variations)

The `search_catalog` tool is the primary entry point for AI agents. It accepts a natural
language `query` plus a pre-classified `intent_data` dict. The engine does **no LLM
classification** — it trusts the `intent` you supply.

### A. Intent: `faq_knowledge` (Blog / FAQ Vector Search)

> Vectors are **Neo4j native** (no Pinecone): for `faq_knowledge` the
> engine runs **only** `vector_search`, which queries the Neo4j vector index
> `inventaa_faq_vector` (config `neo4j.faq_index`, `config.py:61`) over
> `(:Chunk)` blog/knowledge nodes (643 embedded articles). The `:FAQ`
> product-FAQ index (`product_faq_vector`) is **skiped** for this intent, so
> no category keyword is needed and no products leak in. `check_policy` is
> intentionally **omitted** from the test set - the per-product card already
> carries warranty/policy text (via `Product.description` / `features`).
> NOTE: the `(:Chunk)` vector index was previously mis-referenced as
> `faq_vector` (which does not exist) - it is `inventaa_faq_vector`.

* **Tool Arguments Dictionary:**
```json
{
  "query": "What are the top wall lighting ideas for a small outdoor garden space?",
  "limit": 5,
  "tenant_id": "inventaa",
  "session_id": "test_session_knowledge_01",
  "intent_data": {
    "intent": "faq_knowledge",
    "category_keywords": ["Outdoor Wall Lamps"],
    "feature_keywords": [],
    "product_name": null,
    "filters": {},
    "preferences": {}
  }
}
```

---

### B. Intent: `find_product` (Product Discovery with Price/Feature Filters)

```json
{
  "query": "Show me waterproof LED gate pillar lights under 1500 rupees",
  "limit": 6,
  "tenant_id": "inventaa",
  "session_id": "test_session_find_01",
  "intent_data": {
    "intent": "find_product",
    "category_keywords": ["Gate Light Collections"],
    "feature_keywords": ["waterproof", "IP65-rated"],
    "product_name": null,
    "filters": { "category": "Gate Light Collections" },
    "preferences": { "max_price": 1500 }
  }
}
```

---

### C. Intent: `browse_category` (Catalog & Collection Navigation)

```json
{
  "query": "Show me what outdoor wall lights collections you have",
  "limit": 10,
  "tenant_id": "inventaa",
  "session_id": "test_session_browse_01",
  "intent_data": {
    "intent": "browse_category",
    "category_keywords": ["Outdoor Wall Lamps"],
    "feature_keywords": [],
    "product_name": null,
    "filters": { "category": "Outdoor Wall Lamps" },
    "preferences": {}
  }
}
```

---

### D. Intent: `get_product_info` (Specific Product Details)

> Falls back to `FIND_PRODUCT` in the engine (no dedicated branch). Best done via the
> `get_product_details` tool with an exact SKU instead.

```json
{
  "query": "What is the wattage of the Tacita Modern Gate Pillar Light?",
  "limit": 3,
  "tenant_id": "inventaa",
  "session_id": "test_session_info_01",
  "intent_data": {
    "intent": "get_product_info",
    "category_keywords": ["Gate Light Collections"],
    "feature_keywords": ["cool-white", "dimmable"],
    "product_name": "Tacita Modern 12W Outdoor Gate Pillar Light",
    "filters": {},
    "preferences": {}
  }
}
```

---

## 2. Tool: `get_product_details` (Authoritative SKU Lookup)

Retrieves full SQLite specs, variants, stock, pricing, and FAQs for an exact SKU. The
lookup is a case-insensitive `ILIKE` substring match, so pass a SKU that actually exists
in `inventaa_catalog.db` (e.g. `18C-2042`, `MAR03C`, `GAT05`). The previously documented
SKU `12M-2026B` is not guaranteed to exist.

* **Tool Arguments Dictionary:**
```json
{
  "sku": "18C-2042",
  "tenant_id": "inventaa"
}
```

---

## 3. Tool: `get_taxonomy_context` (Taxonomy Tag Resolution)

> Taxonomy vectors now live in **Neo4j** (native `taxonomy_vector_index` on
> `(:TaxonomyTag)` nodes, synced in `src/services/agent/taxonomy.py`) - the
> old Pinecone-backed path is gone. `fetch_taxonomy_candidates_fast` does an
> instant string match first, then falls back to the Neo4j vector query.

Resolves natural language into exact DB category/feature/use-case names.

```json
{
  "query": "waterproof solar gate lamps",
  "threshold": 0.80
}
```

---

## 4. Resource: `data://catalog/status` (System Health & DB Stats)

> **This is a RESOURCE, not a tool.** Read it with `resources/read` / `client.read_resource`,
> not `tools/call`. It returns tenant DB connectivity status and product counts.

```
data://catalog/status
```

(Equivalent `data://catalog/sku/{sku}` returns the same shape as `get_product_details`
for a given SKU.)

---

## 5. Removed: `verify_whatsapp_status`

This tool was previously documented but **does not exist** in `mcp_server.py`. It has been
removed. If WhatsApp status verification is needed, it must be implemented as a new tool
in `mcp_server.py` (the legacy agent's WhatsApp webhook lives at `POST /route` via
`src/endpoints/router_api.py`).

---

## Appendix: Testing

- A runnable test script lives at `scripts/test_mcp_payloads.py` (prints the discovered
  surface, runs the aligned payloads, and emits an alignment report).
- An interactive notebook lives at `notebooks/retrieval_tests.ipynb` with one cell
  per payload and inline observations.
- Both assume the server is running:
  ```powershell
  python mcp_server.py --transport streamable-http --port 8008
  ```
