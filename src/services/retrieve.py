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
    from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
    from typing import Annotated, Sequence, TypedDict
    import operator
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode

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

    def search_products_db(query: Optional[str] = None, spec: Optional[str] = None, min_price: Optional[int] = None, max_price: Optional[int] = None, sort_by: Optional[str] = None, limit: int = 5):
        try:
            cypher_query = ""
            params = {"limit": limit}
            
            tokens = []
            if query:
                tokens = [t.strip() + "~" for t in query.split() if t.strip() and t.lower() not in ["light", "lights", "lamp", "lamps", "product"]]
                if tokens:
                    lucene_query = " AND ".join(tokens)
                    cypher_query += 'CALL db.index.fulltext.queryNodes("product_name_ft", $lucene_query) YIELD node AS p, score\n'
                    params["lucene_query"] = lucene_query
                else:
                    cypher_query += "MATCH (p:Product)\n"
            else:
                cypher_query += "MATCH (p:Product)\n"
                
            where_clauses = []
            if spec:
                cypher_query += "MATCH (p)-[:HAS_SPEC]->(s:Spec)\n"
                where_clauses.append("(toLower(s.key) CONTAINS toLower($spec) OR toLower(s.value) CONTAINS toLower($spec))")
                params["spec"] = spec
                
            if min_price is not None:
                where_clauses.append("p.price_num >= $min_price")
                params["min_price"] = min_price
            if max_price is not None:
                where_clauses.append("p.price_num <= $max_price")
                params["max_price"] = max_price
                
            if where_clauses:
                cypher_query += "WHERE " + " AND ".join(where_clauses) + "\n"
                
            cypher_query += """
            RETURN p.sku AS sku, p.name AS name, p.price_num AS price_num, 
                   p.regular_price AS regular_price, p.discount_percentage AS discount_percentage, 
                   p.image_url AS image_url, p.url AS url, p.rating_score AS rating, 
                   p.review_count AS review_count, p.tenant AS tenant, p.feature_descriptions AS feature_descriptions
            """
            
            if sort_by in ["price_asc", "price_low"]:
                cypher_query += "ORDER BY p.price_num ASC\n"
            elif sort_by in ["price_desc", "price", "price_high"]:
                cypher_query += "ORDER BY p.price_num DESC\n"
            elif sort_by in ["rating_desc", "rating", "highest_rated"]:
                cypher_query += "ORDER BY p.rating_score DESC\n"
            elif sort_by in ["reviews_desc", "reviews", "most_reviewed"]:
                cypher_query += "ORDER BY p.review_count DESC\n"
            elif tokens:
                cypher_query += "ORDER BY score DESC\n"
                
            cypher_query += "LIMIT $limit"
            
            logger.info(f"SearchProductsDatabase executing Cypher: {cypher_query} | Params: {params}")
            res = graph.query(cypher_query, params=params)
            
            if not res:
                return "[]"
                
            return json.dumps(res, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Error querying graph: {e}"

    def get_product_details_db(product_name: str):
        try:
            tokens = [t.strip() + "~" for t in product_name.split() if t.strip()]
            if not tokens:
                return "Please provide a valid product name."
            lucene_query = " AND ".join(tokens)
            
            cypher_query = """
            CALL db.index.fulltext.queryNodes("product_name_ft", $lucene_query) YIELD node AS p, score
            WITH p, score
            ORDER BY score DESC LIMIT 1
            OPTIONAL MATCH (p)-[:HAS_WARRANTY]->(w:Warranty)
            OPTIONAL MATCH (p)-[:HAS_SPEC]->(s:Spec)
            RETURN p.name AS name, p.price_num AS price, p.feature_descriptions AS feature_descriptions,
                   w.description AS warranty_info, w.duration_years AS warranty_duration,
                   collect(s.key + ': ' + s.value) AS specs
            """
            params = {"lucene_query": lucene_query}
            logger.info(f"ProductDetailsDatabase executing Cypher: {cypher_query} | Params: {params}")
            res = graph.query(cypher_query, params=params)
            
            if not res:
                return "Product not found."
                
            product = res[0]
            output = f"Product Name: {product.get('name')}\n"
            output += f"Price: {product.get('price')}\n"
            output += f"Features: {product.get('feature_descriptions')}\n"
            if product.get('warranty_info'):
                output += f"Warranty: {product.get('warranty_info')} ({product.get('warranty_duration')} years)\n"
            if product.get('specs'):
                output += f"Specifications: {', '.join(product.get('specs'))}\n"
                
            return output.encode("ascii", errors="ignore").decode("ascii")
        except Exception as e:
            return f"Error getting product details: {e}"

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

    from langchain_core.tools import StructuredTool

    tools = [
        StructuredTool.from_function(
            name="SearchProductsDatabase",
            func=search_products_db,
            description="ALWAYS use this to SEARCH, LIST, or FILTER products by specifications, prices, or categories. Returns a JSON array of products. If the user searches by a specification (e.g. IP65, wattage), pass it in the 'spec' parameter.",
            return_direct=True
        ),
        StructuredTool.from_function(
            name="ProductDetailsDatabase",
            func=get_product_details_db,
            description="Use this when the user asks a specific question about a product that requires a conversational sentence (e.g., 'What is the warranty of the Artoo light?', 'Tell me about its features.').",
            return_direct=False
        ),
        Tool(
            name="PolicyVectorDatabase",
            func=query_policies,
            description="Use this ONLY for general company-wide policies (e.g., general shipping, return, replacement, exchange, or warranty rules). DO NOT use this for finding product-specific features or warranties."
        ),
        Tool(
            name="ProductAdviceDatabase",
            func=query_product_faqs,
            description="Use this to answer conversational questions, FAQs, installation instructions, usage suitability (e.g. 'Is it suitable for commercial properties?'), or troubleshooting for a specific product."
        )
    ]

    global _system_prompt
    _system_prompt = """You are an e-commerce assistant for Inventaa. You have access to three tools and you MUST use them — you have NO general knowledge to offer.

ABSOLUTE RULES — NEVER BREAK THESE:
1. You MUST call a tool before giving ANY answer. Never answer directly from your own knowledge.
2. If you do not find relevant information in the first tool, try a SECOND tool before giving up.
3. If no tool returns relevant information, respond ONLY with: "I'm sorry, I don't have that information in our database."
4. NEVER suggest, guess, or recommend anything that was not returned by a tool.
5. NEVER say things like "I recommend contacting support" or "you should..." from your own reasoning.
6. If a tool returns content, use it to form your answer even if it is partial.

TOOL SELECTION RULES:
- Product search/listing/filtering (specs, category, price, wattage, IP rating) → SearchProductsDatabase
- Specific product details (warranty duration, warranty information, features, price, specs of ONE specific product) → ProductDetailsDatabase
- General policies: returns, refunds, shipping timelines, cancellations, general warranty claims → PolicyVectorDatabase
- Product-specific FAQs: installation, troubleshooting, usage tips → ProductAdviceDatabase FIRST, then PolicyVectorDatabase if needed
- Missing parts, damaged packages, order issues → PolicyVectorDatabase FIRST, then ProductAdviceDatabase if needed"""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        iterations: int

    tool_node = ToolNode(tools)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def evaluate_node(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        
        if state.get("iterations", 0) >= 3:
            return {"messages": []}
            
        user_query = ""
        for msg in messages:
            if isinstance(msg, HumanMessage) and not msg.content.startswith("Feedback:"):
                user_query = msg.content
                break

        eval_prompt = f"""You are a strict QA evaluator for Inventaa.
The user asked: "{user_query}"
The agent provided the following answer:
"{last_message.content}"

Evaluate if the agent's answer completely and accurately answers the user's question based on the tools it used.
If the agent says it doesn't have the information, check if it might have missed using a tool (e.g. using ProductAdviceDatabase when it should have used ProductDetailsDatabase to check product specs/warranty).
If the answer is fully satisfactory, or if you are completely certain the information truly doesn't exist in any database, output ONLY the word "VALID".
If the answer is unsatisfactory, output feedback on what the agent should do differently (e.g. "You didn't answer part two, try using the PolicyVectorDatabase" or "Try using ProductDetailsDatabase instead of ProductAdviceDatabase")."""

        eval_res = llm.invoke([SystemMessage(content=eval_prompt)])
        eval_content = eval_res.content.strip()
        
        if eval_content == "VALID":
            return {"messages": []}
        else:
            feedback_msg = HumanMessage(content=f"Feedback: {eval_content}\nPlease retry and provide a better answer.")
            return {"messages": [feedback_msg], "iterations": state.get("iterations", 0) + 1}

    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "evaluate"

    def after_tool_node(state: AgentState):
        from langchain_core.messages import ToolMessage
        messages = state["messages"]
        for msg in reversed(messages):
            if not isinstance(msg, ToolMessage):
                break
            if msg.name == "SearchProductsDatabase":
                return "end"
        return "agent"

    def should_loop(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if state.get("iterations", 0) >= 3:
            return "end"
        if isinstance(last_message, HumanMessage):
            return "agent"
        return "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("evaluate", evaluate_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "evaluate": "evaluate"})
    workflow.add_conditional_edges("tools", after_tool_node, {"end": END, "agent": "agent"})
    workflow.add_conditional_edges("evaluate", should_loop, {"agent": "agent", "end": END})

    _agent_executor = workflow.compile()
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
        from langchain_core.messages import SystemMessage, HumanMessage
        
        response = _agent_executor.invoke({
            "messages": [SystemMessage(content=_system_prompt), HumanMessage(content=query_text)],
            "iterations": 0
        })
        output = response["messages"][-1].content
        # If the agent returned a JSON string, parse it into a real object
        try:
            return json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return output
    except Exception as e:
        logger.error(f"Error invoking agent: {e}")
        return "I'm sorry, I encountered an error while processing your request. Please try again later."
