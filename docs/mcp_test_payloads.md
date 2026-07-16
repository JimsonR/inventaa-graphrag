# Inventaa GraphRAG MCP & REST Test Payloads

This document contains ready-to-use JSON payloads for testing all tools and query intents exposed by the **Inventaa GraphRAG MCP Server** (`mcp_server.py`) and API endpoints. 

You can copy and paste these exact payloads into **Postman**, **cURL**, **MCP Inspector**, or **client agents** (`whatsapp_agent`).

---

## 1. Tool: `search_catalog` (All Intent Variations)

The `search_catalog` tool is the primary entry point for AI agents. It accepts a natural language `query` along with structured `intent_data` (`intent`, `category_keywords`, `feature_keywords`, `filters`, `preferences`).

### A. Intent: `faq_knowledge` (New Blog / Knowledge Vector Search)
Used when the customer asks general lighting ideas, blog questions, or how products work.
* **JSON-RPC 2.0 Payload (POST `/mcp` or FastMCP Client):**
```json
{
  "jsonrpc": "2.0",
  "id": 101,
  "method": "tools/call",
  "params": {
    "name": "search_catalog",
    "arguments": {
      "query": "What are the top wall lighting ideas for a small outdoor garden space?",
      "limit": 5,
      "tenant_id": "tenant_inventaa_led_001",
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
  }
}
```

* **Tool Arguments Dictionary (for Python / LangChain Clients):**
```json
{
  "query": "Do solar LED wall lights need wiring to work at night?",
  "limit": 4,
  "tenant_id": "tenant_inventaa_led_001",
  "session_id": "test_session_knowledge_02",
  "intent_data": {
    "intent": "faq_knowledge",
    "category_keywords": [],
    "feature_keywords": ["solar-powered"],
    "product_name": null,
    "filters": {},
    "preferences": {}
  }
}
```

---

### B. Intent: `find_product` (Product Discovery with Price/Feature Filters)
Used when searching for specific product types, budgets, or capabilities.
```json
{
  "jsonrpc": "2.0",
  "id": 102,
  "method": "tools/call",
  "params": {
    "name": "search_catalog",
    "arguments": {
      "query": "Show me waterproof LED gate pillar lights under 1500 rupees",
      "limit": 6,
      "tenant_id": "tenant_inventaa_led_001",
      "session_id": "test_session_find_01",
      "intent_data": {
        "intent": "find_product",
        "category_keywords": ["Gate Light Collections"],
        "feature_keywords": ["waterproof", "IP65-rated"],
        "product_name": null,
        "filters": {
          "category": "Gate Light Collections"
        },
        "preferences": {
          "max_price": 1500
        }
      }
    }
  }
}
```

---

### C. Intent: `browse_category` (Catalog & Collection Navigation)
Used when exploring top-level collections or categories without a specific product in mind.
```json
{
  "jsonrpc": "2.0",
  "id": 103,
  "method": "tools/call",
  "params": {
    "name": "search_catalog",
    "arguments": {
      "query": "Show me what outdoor wall lights collections you have",
      "limit": 10,
      "tenant_id": "tenant_inventaa_led_001",
      "session_id": "test_session_browse_01",
      "intent_data": {
        "intent": "browse_category",
        "category_keywords": ["Outdoor Wall Lamps"],
        "feature_keywords": [],
        "product_name": null,
        "filters": {
          "category": "Outdoor Wall Lamps"
        },
        "preferences": {}
      }
    }
  }
}
```

---

### D. Intent: `check_policy` (Store Policy, Warranty & Shipping Search)
Used when asking about replacement, warranty terms, return rules, or delivery timelines.
```json
{
  "jsonrpc": "2.0",
  "id": 104,
  "method": "tools/call",
  "params": {
    "name": "search_catalog",
    "arguments": {
      "query": "What is your replacement and exchange policy if a light arrives broken?",
      "limit": 4,
      "tenant_id": "tenant_inventaa_led_001",
      "session_id": "test_session_policy_01",
      "intent_data": {
        "intent": "check_policy",
        "category_keywords": [],
        "feature_keywords": ["replacement", "warranty", "exchange"],
        "product_name": null,
        "filters": {},
        "preferences": {}
      }
    }
  }
}
```

---

### E. Intent: `get_product_info` (Specific Product Details via Search)
Used when asking specific questions about an identified product name or model.
```json
{
  "jsonrpc": "2.0",
  "id": 105,
  "method": "tools/call",
  "params": {
    "name": "search_catalog",
    "arguments": {
      "query": "What is the wattage and color temperature of the Tacita Modern Gate Pillar Light?",
      "limit": 3,
      "tenant_id": "tenant_inventaa_led_001",
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
  }
}
```

---

### F. Intent: `get_advice` (Consultative Recommendations)
Used when asking what fixture to choose for architectural or environmental scenarios.
```json
{
  "jsonrpc": "2.0",
  "id": 106,
  "method": "tools/call",
  "params": {
    "name": "search_catalog",
    "arguments": {
      "query": "Which lights would you recommend for illuminating a long dark garden pathway near coastal salt air?",
      "limit": 5,
      "tenant_id": "tenant_inventaa_led_001",
      "session_id": "test_session_advice_01",
      "intent_data": {
        "intent": "get_advice",
        "category_keywords": ["Garden Lights"],
        "feature_keywords": ["IP65-rated", "waterproof"],
        "product_name": null,
        "filters": {
          "application": "garden-pathway"
        },
        "preferences": {}
      }
    }
  }
}
```

---

## 2. Tool: `get_product_details` (Authoritative SKU Lookup)

Directly retrieves full SQLite specifications, available variants, stock status, pricing, and product FAQs for an exact SKU.

```json
{
  "jsonrpc": "2.0",
  "id": 201,
  "method": "tools/call",
  "params": {
    "name": "get_product_details",
    "arguments": {
      "sku": "12M-2026B"
    }
  }
}
```

---

## 3. Tool: `get_taxonomy_context` (Taxonomy Tag Resolution)

Resolves natural language queries into exact database category names, feature tags, and use-case tags (using instant string matching and fallback Neo4j vector search).

```json
{
  "jsonrpc": "2.0",
  "id": 301,
  "method": "tools/call",
  "params": {
    "name": "get_taxonomy_context",
    "arguments": {
      "query": "waterproof solar gate lamps",
      "threshold": 0.80
    }
  }
}
```

---

## 4. Tool: `get_catalog_status` (System Health & DB Stats)

Checks tenant database connections and returns counts of products, categories, features, and vector chunks.

```json
{
  "jsonrpc": "2.0",
  "id": 401,
  "method": "tools/call",
  "params": {
    "name": "get_catalog_status",
    "arguments": {}
  }
}
```

---

## 5. Tool: `verify_whatsapp_status` (WhatsApp API Verification)

Verifies WhatsApp Business Cloud API token, phone number ID, and webhook health for a tenant.

```json
{
  "jsonrpc": "2.0",
  "id": 501,
  "method": "tools/call",
  "params": {
    "name": "verify_whatsapp_status",
    "arguments": {
      "tenant_id": "tenant_inventaa_led_001"
    }
  }
}
```
