# Inventaa GraphRAG — Neo4j Knowledge Graph Documentation

> **Database**: Neo4j AuraDB  
> **URI**: `neo4j+ssc://7feba1e0.databases.neo4j.io`  
> **Total Products**: 125 | **Total Categories**: 12 | **Last Updated**: 2026-06-12

---

## 1. Node Labels

| Label | Count | Description |
|---|---|---|
| `Product` | 125 | Core product nodes. Each has `HAS_PRODUCT` incoming from a `Category`. |
| `Category` | 12 | Product categories. Connected to products via `(Category)-[:HAS_PRODUCT]->(Product)`. |
| `Feature` | 20 | Normalized feature tags (e.g. `waterproof`, `solar-powered`). |
| `UseCase` | 9 | Normalized use-case tags (e.g. `indoor-ceiling`, `gate-pillar`). |
| `Spec` | ~389 | Key-value specification entries (e.g. `Wattage: 12W`). |
| `Warranty` | 9 (unique) | Warranty description nodes, shared across products. |
| `ColorOption` | 5 | Color variants (e.g. `Cool White`, `3-in-1`). |
| `WattageOption` | 24 | Wattage variants (e.g. `12W`, `36W`). |
| `Brand` | 1 | Always `Inventaa`. |
| `FAQ` | ~389 edges | Product-specific FAQ nodes. |
| `Policy` | ~240 edges | Return/shipping policy nodes shared across products. |
| `Chunk` | — | RAG text chunks for vector search (not graph-traversed). |

---

## 2. Relationship Types

```
(Category)   -[:HAS_PRODUCT]-------> (Product)       [125 edges]
(Product)    -[:HAS_FEATURE]-------> (Feature)        [784 edges]
(Product)    -[:HAS_SPEC]----------> (Spec)           [561 edges]
(Product)    -[:HAS_FAQ]-----------> (FAQ)            [389 edges]
(Product)    -[:HAS_POLICY]--------> (Policy)         [240 edges]
(Product)    -[:SUITABLE_FOR]------> (UseCase)        [259 edges]
(Product)    -[:AVAILABLE_IN_COLOR]> (ColorOption)    [187 edges]
(Product)    -[:AVAILABLE_IN_WATTAGE](WattageOption)  [154 edges]
(Product)    -[:MADE_BY]-----------> (Brand)          [125 edges]
(Product)    -[:HAS_WARRANTY]------> (Warranty)       [94 edges]
(Brand)      -[:MAKES]-------------> (Category)       [12 edges]
```

> [!IMPORTANT]
> The relationship between Category and Product is **`(Category)-[:HAS_PRODUCT]->(Product)`** — NOT the reverse. Always start from Category when filtering by product type.

---

## 3. Product Node Properties

| Property | Type | Example | Notes |
|---|---|---|---|
| `id` | string | `inventaa_product_abc123` | Internal unique ID |
| `name` | string | `Tacita Modern 12W Outdoor Gate Pillar Light` | Full display name |
| `sku` | string | `12M-2026B` | Product SKU |
| `price_num` | int | `1805` | Current sale price in INR (paise-less) |
| `regular_price` | string | `"2406.02"` | MRP before discount |
| `discount_percentage` | int | `25` | % discount off MRP |
| `rating_score` | float | `4.93` | Average star rating (0–5) |
| `review_count` | int | `28` | Number of customer reviews |
| `image_url` | string | `http://inventaa.in/cdn/...` | Product image URL |
| `url` | string | `https://inventaa.in/products/tacita` | Product page URL |
| `description` | string | Long text | Full scraped product description |
| `feature_descriptions` | string | Short summary | 1–3 sentence feature summary |
| `has_variants` | bool | `true` | Whether product has wattage/colour variants |
| `tenant` | string | `"inventaa"` | Tenant identifier for multi-tenancy |
| `wattage` | int | `12` | Nominal wattage (may be missing for variant products) |

---

## 4. Categories (Exhaustive List)

| Category Name | Product Count | Keywords that map to it |
|---|---|---|
| `Gate & Pillar Lights` | **43** | gate, pillar, entrance, post, compound |
| `Indoor & Ceiling Lights` | **14** | indoor, ceiling, down light, surface mount, panel |
| `Solar Lights` | **12** | solar |
| `Outdoor Wall Lights` | **12** | wall, elevation, sconce |
| `Bollard & Garden Lights` | **10** | bollard, garden, lawn, yard |
| `Divine & Temple Lights` | **9** | divine, temple, religious, god, murugan, perumal |
| `Street Lights` | **8** | street, road, pathway |
| `Pathway & Step Lights` | **7** | pathway, step, stairway, walkway |
| `Flood Lights` | **3** | flood, stadium |
| `General Purpose Lights` | **3** | general, compound, welcome |
| `Panel Lights` | **2** | panel |
| `Bulkhead Lights` | **2** | bulkhead |

> [!NOTE]
> Always use exact category name strings in Cypher — they are case-sensitive.

---

## 5. Use Cases (Exhaustive List)

| UseCase Node Name | Meaning | Common Product Types |
|---|---|---|
| `gate-pillar` | Installed on gate or compound pillar | Gate lights, pillar lights |
| `indoor-ceiling` | Indoor ceiling / downlight / surface mount | COB lights, panel lights, tube lights |
| `outdoor-wall` | Mounted on exterior walls | Wall lights, sconces, bulkhead |
| `garden-pathway` | Garden / pathway / landscape | Bollards, garden lights |
| `pathway-step` | Stairway / step / walkway | Step lights, pathway lights |
| `street-road` | Roads, streets, public lighting | Street lights |
| `flood-area` | Wide-area floodlighting | Flood lights, stadium lights |
| `solar-outdoor` | Solar-powered outdoor use | Solar gate lights, solar bollards |
| `religious-decorative` | Temple / shrine / decorative religious | Divine lights |

---

## 6. Features (Exhaustive List)

| Feature Node Name | Meaning |
|---|---|
| `outdoor` | Suitable for outdoor use |
| `indoor` | Suitable for indoor use |
| `solar-powered` | Uses solar energy |
| `waterproof` | Weatherproof/waterproof construction |
| `IP65-rated` | IP65 ingress protection rating |
| `IP66-rated` | IP66 ingress protection rating |
| `UV-protected` | UV-resistant coating |
| `motion-sensor` | Has built-in motion/PIR sensor |
| `dimmable` | Supports dimming |
| `energy-efficient` | Energy-efficient LED |
| `warm-white` | Warm white (2700–3000K) colour option |
| `cool-white` | Cool white (5000–6500K) colour option |
| `neutral-white` | Neutral white (4000K) colour option |
| `3-in-1-colour` | 3-in-1 switchable colour mode |
| `aluminium-body` | Aluminium housing/body |
| `polycarbonate-body` | Polycarbonate housing/body |
| `surface-mount` | Surface-mounted installation |
| `wall-mount` | Wall-mounted installation |
| `post-top-mount` | Post-top / bollard-top installation |
| `rustproof` | Rust-resistant material |

---

## 7. Color & Wattage Options

**ColorOption values**: `Cool White` | `Warm White` | `Natural White` | `3-in-1` | `RGB`

**WattageOption values**: `2W` `3W` `4W` `5W` `6W` `7W` `8W` `10W` `12W` `15W` `18W` `18 W` `20W` `22W` `24W` `25W` `30W` `36W` `40W` `50W` `60W` `100W` `150W` `200W`

---

## 8. Spec Node Properties & Common Keys

Each `Spec` node has two properties: `key` (string) and `value` (string).

Common spec keys (not exhaustive):

| Spec Key | Example Value |
|---|---|
| `Wattage` | `12W`, `18W` |
| `Ip Rating` / `Waterproof Rating` | `IP65`, `IP66` |
| `Color Temperature` | `Cool White (6500K)` |
| `Body Material` | `Polycarbonate`, `Aluminium` |
| `Mounting Type` | `Surface`, `Wall`, `Post Top` |
| `Input Voltage` | `220–240V AC` |
| `Beam Angle` | `120°` |
| `Lifespan` | `25,000 hours` |
| `Application` | `Gate pillars, compound walls` |
| `Colour Mode Switching` | `Toggle via wall switch` |
| `1-Year Warranty` | `Included` |

> [!NOTE]
> Spec keys have been largely normalized (e.g. `Wattage`, `Color`, `Dimension`), but always use `CONTAINS` or `toLower()` for spec key matching to be safe against slight variations.

---

## 9. Cypher Query Patterns

### List products by category
```cypher
MATCH (c:Category {name: "Indoor & Ceiling Lights"})-[:HAS_PRODUCT]->(p:Product)
RETURN p.name, p.sku, p.price_num, p.rating_score
ORDER BY p.rating_score DESC
LIMIT 10
```

### Filter products by use case
```cypher
MATCH (p:Product)-[:SUITABLE_FOR]->(uc:UseCase {name: "indoor-ceiling"})
RETURN p.name, p.price_num
ORDER BY p.rating_score DESC
LIMIT 10
```

### Filter products by feature
```cypher
MATCH (p:Product)-[:HAS_FEATURE]->(f:Feature {name: "solar-powered"})
RETURN p.name, p.price_num
```

### Filter products by category + feature (combined)
```cypher
MATCH (c:Category {name: "Gate & Pillar Lights"})-[:HAS_PRODUCT]->(p:Product)
MATCH (p)-[:HAS_FEATURE]->(f:Feature {name: "solar-powered"})
RETURN p.name, p.price_num
```

### Filter products by spec value (fuzzy)
```cypher
MATCH (p:Product)-[:HAS_SPEC]->(s:Spec)
WHERE toLower(s.key) CONTAINS 'wattage' AND s.value CONTAINS '12W'
RETURN DISTINCT p.name, p.price_num
```

### Full-text search on product name (fuzzy)
```cypher
CALL db.index.fulltext.queryNodes("product_name_ft", "athena~") YIELD node AS p, score
RETURN p.name, score
ORDER BY score DESC LIMIT 5
```

### Get full product details
```cypher
MATCH (p:Product) WHERE p.sku = "12M-2026B"
OPTIONAL MATCH (p)-[:HAS_SPEC]->(s:Spec)
OPTIONAL MATCH (p)-[:HAS_WARRANTY]->(w:Warranty)
OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f:Feature)
OPTIONAL MATCH (p)-[:SUITABLE_FOR]->(uc:UseCase)
OPTIONAL MATCH (p)-[:AVAILABLE_IN_COLOR]->(co:ColorOption)
OPTIONAL MATCH (p)-[:AVAILABLE_IN_WATTAGE]->(wo:WattageOption)
OPTIONAL MATCH (cat:Category)-[:HAS_PRODUCT]->(p)
RETURN p, collect(DISTINCT s.key + ": " + s.value) as specs,
       collect(DISTINCT f.name) as features,
       collect(DISTINCT uc.name) as usecases,
       w.description as warranty,
       collect(DISTINCT cat.name) as categories
```

---

## 10. Full-Text Indexes

| Index Name | On Property | Node Label | Use |
|---|---|---|---|
| `product_name_ft` | `name` | `Product` | Fuzzy product name search |

> [!TIP]
> Use `~` suffix for fuzzy Lucene matching: `"athena~"` matches "Athena", "Athena 3 in 1", etc.
> Use `AND` to combine tokens: `"athena~ gate~"`.

---

## 11. Tool → Query Strategy Mapping

> This table documents how `src/services/agent/tools.py` maps user queries to Cypher patterns.

| User Query Pattern | Strategy | Cypher Entry Point |
|---|---|---|
| "indoor lights", "ceiling lights" | Category match | `(Category {name: "Indoor & Ceiling Lights"})-[:HAS_PRODUCT]->(p)` |
| "gate lights", "pillar lights" | Category match | `(Category {name: "Gate & Pillar Lights"})-[:HAS_PRODUCT]->(p)` |
| "solar lights" | Category match | `(Category {name: "Solar Lights"})-[:HAS_PRODUCT]->(p)` |
| "bollard", "garden lights" | Category match | `(Category {name: "Bollard & Garden Lights"})-[:HAS_PRODUCT]->(p)` |
| "street lights", "road lights" | Category match | `(Category {name: "Street Lights"})-[:HAS_PRODUCT]->(p)` |
| "temple lights", "divine lights" | Category match | `(Category {name: "Divine & Temple Lights"})-[:HAS_PRODUCT]->(p)` |
| "waterproof lights" | Feature match | `(p)-[:HAS_FEATURE]->(Feature {name: "waterproof"})` |
| "solar gate light" | Category + Feature | Category `Gate & Pillar Lights` + Feature `solar-powered` |
| "IP65 lights" | Spec match | `(p)-[:HAS_SPEC]->(s)` WHERE `s.value CONTAINS 'IP65'` |
| "12W lights" | Spec match | `(p)-[:AVAILABLE_IN_WATTAGE]->(WattageOption {name: "12W"})` |
| Specific product name | Full-text index | `CALL db.index.fulltext.queryNodes("product_name_ft", ...)` |

---

## 12. Known Issues & Gotchas

> [!CAUTION]
> **Fuzzy Name Matching Bug** — Searching for "indoor" via the full-text index (`indoor~`) incorrectly matches products with "door" in their name (e.g., "Modern Exterior **Door** Light"). Levenshtein distance between "indoor" and "door" = 2 (within fuzzy match threshold). **Fix**: Always prefer Category/UseCase traversal over fuzzy name matching when the user's intent is a product category.

> [!WARNING]
> **Category → Product direction** — The edge goes `(Category)-[:HAS_PRODUCT]->(Product)`, NOT `(Product)-[:IN_CATEGORY]->(Category)`. Using the wrong direction returns 0 results.

> [!NOTE]
> **Spec keys are mostly normalized** — However, variations might still occur in new data. Use `toLower(s.key) CONTAINS 'watt'` pattern to be robust.

> [!NOTE]
> **Duplicate product nodes** — Some products appear twice with slightly different names (e.g., `Tacita Modern  12W...` with double space vs. `Tacita Modern 12W...`). Use `RETURN DISTINCT` in queries.

> [!NOTE]
> **`wattage` property on Product** — The `wattage` int property exists on some products but not all (variant products may only have `WattageOption` nodes). Do not rely on it for filtering; use `AVAILABLE_IN_WATTAGE` relationship instead.

---

## 13. Adding a New Product Category or Use Case

If a new product type is added to Neo4j, update these locations:

1. **`src/services/agent/tools.py`** — `CATEGORY_KEYWORDS`, `USECASE_KEYWORDS`, `FEATURE_KEYWORDS` dicts
2. **`src/services/agent/graph.py`** — `_system_prompt` section `PRODUCT CATEGORIES IN THE DATABASE`
3. This document

---

## 14. Graph Statistics (as of 2026-06-12)

```
Nodes:
  Product:      125
  Feature:       20 distinct values
  UseCase:        9 distinct values
  Category:      12
  Spec:         ~389 (key-value pairs, normalized)
  Warranty:       9 distinct descriptions
  ColorOption:    5 distinct values
  WattageOption: 24 distinct values
  Brand:          1 (Inventaa)

Relationships:
  HAS_FEATURE:          784
  HAS_SPEC:             561
  HAS_FAQ:              389
  SUITABLE_FOR:         259
  HAS_POLICY:           240
  AVAILABLE_IN_COLOR:   187
  AVAILABLE_IN_WATTAGE: 154
  MADE_BY:              125
  HAS_PRODUCT:          125
  HAS_WARRANTY:          94
  MAKES:                 12
```
