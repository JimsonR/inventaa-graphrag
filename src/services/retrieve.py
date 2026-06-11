"""
RAG Retrieval Service — Neo4j Backend
Callable service module adapted from hybrid_rag_agent.py for use in FastAPI routes.
Supports LangChain Hybrid Agent.
"""

import os
import json
import logging
from typing import Optional, List
# from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

# load_dotenv()
logger = logging.getLogger(__name__)

# Singletons
_agent_executor = None

def initialize_agent() -> None:
    """
    Initializes the LangChain Azure OpenAI Agent and tools.
    """
    global _agent_executor
    if _agent_executor is not None:
        return

    logger.info("Initializing Hybrid RAG Agent...")

    NEO4J_URI = os.getenv("NEO4J_URI", "").replace("neo4j+s://", "neo4j+ssc://")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
    from langchain_neo4j import Neo4jGraph, GraphCypherQAChain, Neo4jVector
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import Tool
    from langchain.agents import create_openai_tools_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    # 1. Initialize Azure OpenAI LLM
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_AI_ENDPOINT"),
        api_key=os.getenv("AZURE_AI_API_KEY"),
        api_version="2024-02-15-preview",
        azure_deployment=os.getenv("GPT_4_1_MINI_DEPLOYMENT", "gpt-4.1-mini"),
        temperature=0
    )

    # 2. Initialize Embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_AI_ENDPOINT"),
        api_key=os.getenv("AZURE_AI_API_KEY"),
        api_version="2024-02-15-preview",
        azure_deployment=os.getenv("TEXT_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    )

    # 3. Connect to Graph
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    graph.refresh_schema()

    # 4. Connect to Vector Stores
    general_vector_store = Neo4jVector.from_existing_index(
        embedding=embeddings, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
        index_name="inventaa_faq_vector", text_node_property="text"
    )

    policy_vector_store = Neo4jVector.from_existing_index(
        embedding=embeddings, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
        index_name="policy_vector", text_node_property="text"
    )

    product_faq_vector_store = Neo4jVector.from_existing_index(
        embedding=embeddings, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
        index_name="product_faq_vector", text_node_property="question",
        retrieval_query="""
        MATCH (node)<-[:HAS_FAQ]-(p:Product)
        RETURN "FAQ Match: " + node.question + "\\nAnswer: " + node.answer + 
               "\\n--> This belongs to Product: " + p.name + " (Price: ₹" + toString(p.price_num) + ")" AS text,
               score, {product_url: p.url} AS metadata
        """
    )

    # 5. Build Graph Cypher Chain
    CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Schema:
{schema}

IMPORTANT RULES:
1. NEVER use exact matches (`=`) for strings. ALWAYS use fuzzy matching with `toLower(toString(property)) CONTAINS toLower('value')` or case-insensitive regular expressions. When searching for a product by name, check both `p.name` and `c.name` with `OR` conditions for a SINGLE keyword (e.g., `toLower(p.name) CONTAINS 'athena' OR toLower(c.name) CONTAINS 'athena'`). If the user provides multiple keywords (like 'athena' and 'gate light'), you MUST combine them using `AND` (e.g., `(name/category CONTAINS 'athena') AND (name/category CONTAINS 'gate light')`) so you don't return irrelevant products.
2. Categories are connected via `(c:Category)-[:HAS_PRODUCT]->(p:Product)`.
3. Specs (like IP65) are connected via `(p:Product)-[:HAS_SPEC]->(s:Spec)`.
4. ALWAYS return ALL of the following product fields when your query returns products:
   p.sku, p.name, p.price_num, p.regular_price, p.discount_percentage,
   p.image_url, p.url, p.rating, p.review_count, p.tenant, p.feature_descriptions
   Do NOT return only p.url or a subset — always return all fields listed above.

The question is:
{question}"""
    from typing import TypedDict, Any
    from langgraph.graph import StateGraph, END

    class GraphQAState(TypedDict):
        question: str
        cypher_query: str
        db_results: Any
        validation_feedback: str
        attempts: int
        final_output: Any

    def generate_cypher_node(state: GraphQAState):
        schema = graph.schema
        prompt = CYPHER_GENERATION_TEMPLATE.format(schema=schema, question=state["question"])
        if state.get("validation_feedback"):
            prompt += f"\n\nPrevious attempt failed. Feedback: {state['validation_feedback']}\nFix the query and return ONLY the raw Cypher query string without markdown block formatting."
        else:
            prompt += "\n\nReturn ONLY the raw Cypher query string without markdown block formatting."
            
        res = llm.invoke(prompt)
        cypher = res.content.strip().replace("```cypher", "").replace("```", "").strip()
        return {"cypher_query": cypher, "attempts": state.get("attempts", 0) + 1}
        
    def execute_cypher_node(state: GraphQAState):
        cypher = state["cypher_query"]
        try:
            res = graph.query(cypher)
            return {"db_results": res}
        except Exception as e:
            return {"db_results": str(e)}

    def validate_cypher_node(state: GraphQAState):
        results = state["db_results"]
        # Ask LLM if the result makes sense for the question and isn't an error string
        prompt = f"""Question: {state['question']}
Cypher Query Executed: {state['cypher_query']}
Results from DB: {results}

If the results are a database syntax error string, or if they are completely empty and you think the query was too restrictive, provide feedback on how to fix the cypher query.
If the query returned too many irrelevant products (e.g. returning all gate lights when the user asked for a specific 'Athena' light), provide feedback to make the Cypher query stricter (e.g., use AND instead of OR for multiple keywords).
If the results successfully answer the question, do not contain irrelevant items, or if you believe empty results are genuinely correct because the product doesn't exist, just output 'VALID'.
Otherwise, output the feedback to correct the Cypher query. Return ONLY your feedback or the word VALID."""
        
        res = llm.invoke(prompt).content.strip()
        if res == "VALID":
            if not results or isinstance(results, str):
                 return {"final_output": "No matching products found in the database for your query.", "validation_feedback": ""}
            
            cleaned = [
                {k.split(".", 1)[-1]: v for k, v in row.items()}
                for row in results
            ]
            return {"final_output": json.dumps(cleaned, indent=2, ensure_ascii=False), "validation_feedback": ""}
        else:
            return {"validation_feedback": res}

    def should_continue(state: GraphQAState):
        if state.get("final_output"):
            return END
        if state.get("attempts", 0) >= 3:
            return END
        return "generate_cypher"

    workflow = StateGraph(GraphQAState)
    workflow.add_node("generate_cypher", generate_cypher_node)
    workflow.add_node("execute_cypher", execute_cypher_node)
    workflow.add_node("validate_cypher", validate_cypher_node)
    
    workflow.set_entry_point("generate_cypher")
    workflow.add_edge("generate_cypher", "execute_cypher")
    workflow.add_edge("execute_cypher", "validate_cypher")
    workflow.add_conditional_edges("validate_cypher", should_continue)
    
    cypher_graph = workflow.compile()

    def query_graph(query: str):
        try:
            res = cypher_graph.invoke({"question": query, "attempts": 0})
            if res.get("final_output"):
                return res["final_output"]
            elif res.get("db_results") and not isinstance(res["db_results"], str):
                # Fallback if validation failed but we hit max attempts and we have actual results
                cleaned = [
                    {k.split(".", 1)[-1]: v for k, v in row.items()}
                    for row in res["db_results"]
                ]
                return json.dumps(cleaned, indent=2, ensure_ascii=False)
            else:
                return "No matching products found in the database for your query."
        except Exception as e:
            return f"Error querying graph: {e}"

    def query_policies(query: str):
        # We search the dedicated policy index first
        results = policy_vector_store.similarity_search_with_score(query, k=2)
        if not results:
            # Fallback to the general chunk vector store if no policy matches
            results = general_vector_store.similarity_search_with_score(query, k=2)
            
        if not results:
            return "No relevant policy found."
        text = "\n\n".join([doc.page_content for doc, _ in results])
        # Encode to ASCII-safe to prevent UnicodeEncodeError on Windows console
        return text.encode("ascii", errors="ignore").decode("ascii")

    def query_product_faqs(query: str):
        results = product_faq_vector_store.similarity_search_with_score(query, k=2)
        if not results:
            return "No relevant product FAQ found."
        text = "\n\n".join([doc.page_content for doc, _ in results])
        return text.encode("ascii", errors="ignore").decode("ascii")

    tools = [
        Tool(
            name="GraphProductDatabase",
            func=query_graph,
            description="ALWAYS use this to find a list of products, filter by specifications (like waterproof, IP65, wattage), prices, or categories. It queries the structural database.",
            return_direct=True
        ),
        Tool(
            name="PolicyVectorDatabase",
            func=query_policies,
            description="Use this to answer questions about shipping policies, return policies, warranty rules, and general company guidelines."
        ),
        Tool(
            name="ProductAdviceDatabase",
            func=query_product_faqs,
            description="Use this to answer conversational advice, installation instructions, or troubleshooting for a specific product."
        )
    ]

    system_prompt = """You are an e-commerce assistant for Inventaa. You have access to three tools and you MUST use them — you have NO general knowledge to offer.

ABSOLUTE RULES — NEVER BREAK THESE:
1. You MUST call a tool before giving ANY answer. Never answer directly from your own knowledge.
2. If you do not find relevant information in the first tool, try a SECOND tool before giving up.
3. If no tool returns relevant information, respond ONLY with: "I'm sorry, I don't have that information in our database."
4. NEVER suggest, guess, or recommend anything that was not returned by a tool.
5. NEVER say things like "I recommend contacting support" or "you should..." from your own reasoning.
6. If a tool returns content, use it to form your answer even if it is partial.

TOOL SELECTION RULES:
- Product search/listing/filtering (specs, category, price, wattage, IP rating) → GraphProductDatabase
- General policies: returns, refunds, shipping timelines, cancellations, warranty claims → PolicyVectorDatabase
- Product-specific questions: delivery time for a specific product, installation, troubleshooting, usage tips → ProductAdviceDatabase FIRST, then PolicyVectorDatabase if needed
- Missing parts, damaged packages, order issues → PolicyVectorDatabase FIRST, then ProductAdviceDatabase if needed"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    _agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=False)
    logger.info("Hybrid RAG Agent Initialized!")

def _resolve_collections(tenant_id: Optional[str], collection_name: str) -> List[str]:
    """
    Returns the list of collections to search based on tenant_id and collection_name.

    - If collection_name is a specific name (not 'all'), use it directly.
    - If tenant_id is given, filter COLLECTIONS to those prefixed with '<tenant_id>_'.
    - tenant_id is normalized to lowercase so 'Inventaa', 'INVENTAA', 'inventaa' all work.
    - If tenant_id is None, search across all collections (no tenant scoping).
    """
    if collection_name != "all":
        return [collection_name]

    if tenant_id:
        prefix = f"{tenant_id.lower().strip()}_"   # normalize: "Inventaa" → "inventaa_"
        scoped = [c for c in COLLECTIONS if c.startswith(prefix)]
        return scoped if scoped else COLLECTIONS  # graceful fallback

    return COLLECTIONS

def ask_agent(query_text: str, tenant_id: Optional[str] = None):
    """
    Invokes the agent with the user's query.
    Returns a parsed JSON object (list/dict) for graph queries,
    or a plain string for conversational/policy answers.
    tenant_id is accepted for future tenant-scoped filtering.
    """
    if _agent_executor is None:
        initialize_agent()
    
    try:
        response = _agent_executor.invoke({"input": query_text})
        output = response.get("output", str(response))
        # If the agent returned a JSON string, parse it into a real object
        try:
            return json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return output
    except Exception as e:
        logger.error(f"Error invoking agent: {e}")
        return "I'm sorry, I encountered an error while processing your request. Please try again later."
