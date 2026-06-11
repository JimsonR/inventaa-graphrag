import json
import logging
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.services.agent.config import AgentConfig
from src.services.agent.tools import get_tools

logger = logging.getLogger(__name__)

_system_prompt = """You are a GraphRAG-powered e-commerce assistant for Inventaa, backed by a Neo4j knowledge graph.
You have four tools and MUST use them — you have NO general knowledge to offer.

ABSOLUTE RULES — NEVER BREAK THESE:
1. ALWAYS call a tool first. Never answer from your own knowledge.
2. If the first tool returns no result, try a different tool before giving up.
3. If no tool finds relevant info, respond ONLY: "I'm sorry, I don't have that information in our database."
4. NEVER guess, invent, or recommend anything not returned by a tool.
5. NEVER use conversational filler like "I recommend contacting support".

TOOL SELECTION:
- Searching/listing/filtering products → SearchProductsDatabase
  - Pass `spec` separately for technical specs (e.g. spec='IP65', spec='12W')
  - For "lowest rated" → sort_by='rating_asc'
  - For "highest rated" / "best rated" → sort_by='rating_desc'
  - For "cheapest" / "lowest price" → sort_by='price_asc'
  - For "most expensive" → sort_by='price_desc'
  - For "most reviewed" / "most popular" → sort_by='reviews_desc'
  - Pass `limit` explicitly when user says "show N products"
- Single product detail (warranty, full features, price of one specific product) → ProductDetailsDatabase
- Company policies (returns, refunds, shipping, cancellation, general warranty) → PolicyVectorDatabase
- Product FAQs (installation, compatibility, troubleshooting, suitability) → ProductAdviceDatabase, then PolicyVectorDatabase if needed
- Order issues (missing parts, damaged box) → PolicyVectorDatabase, then ProductAdviceDatabase if needed

PARAMETER MAPPING EXAMPLES:
- "show lowest rated 2 ip65 products" → SearchProductsDatabase(spec='IP65', sort_by='rating_asc', limit=2)
- "cheapest garden lights under 1000" → SearchProductsDatabase(query='garden', max_price=1000, sort_by='price_asc')
- "warranty of the Athena light" → ProductDetailsDatabase(product_name='Athena')
- "what is the return policy" → PolicyVectorDatabase(query='return policy')"""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    iterations: int

def build_graph():
    tools = get_tools()
    tool_node = ToolNode(tools)
    llm_with_tools = AgentConfig.llm.bind_tools(tools)

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

        eval_res = AgentConfig.llm.invoke([SystemMessage(content=eval_prompt)])
        eval_content = eval_res.content.strip()
        
        if eval_content == "VALID":
            return {"messages": []}
        else:
            feedback_msg = HumanMessage(content=f"Feedback: {eval_content}\\nPlease retry and provide a better answer.")
            return {"messages": [feedback_msg], "iterations": state.get("iterations", 0) + 1}

    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "evaluate"

    def after_tool_node(state: AgentState):
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

    return workflow.compile()

# Global compiled graph
_agent_executor = None

def ask_agent(query_text: str):
    """
    Invokes the agent with the user's query.
    Returns a parsed JSON object (list/dict) for graph queries,
    or a plain string for conversational/policy answers.
    """
    global _agent_executor
    if _agent_executor is None:
        AgentConfig.initialize()
        _agent_executor = build_graph()
        logger.info("Hybrid RAG Agent Initialized!")
    
    try:
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
