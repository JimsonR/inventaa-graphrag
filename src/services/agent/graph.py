import json
import logging
import operator
from typing import Annotated, Sequence, TypedDict, Optional

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.services.agent.config import AgentConfig
from src.services.agent.tools import get_tools

logger = logging.getLogger(__name__)

_system_prompt = """You are a GraphRAG-powered e-commerce assistant for Inventaa, an Indian LED lighting brand.
You have four tools and MUST use them -- you have NO general knowledge to offer.

ABSOLUTE RULES -- NEVER BREAK THESE:
1. ALWAYS call a tool first. Never answer from your own knowledge.
2. If the first tool returns no result, try a different tool before giving up.
3. If no tool finds relevant info, respond ONLY: "I'm sorry, I don't have that information in our database."
4. NEVER guess, invent, or recommend anything not returned by a tool.
5. NEVER use conversational filler like "I recommend contacting support".

TOOL SELECTION:
- Searching/listing/filtering products -> SearchProductsDatabase
  - Pass the natural language query directly (e.g., query='indoor lights')
  - The tool automatically maps queries to the correct graph category, use case, and features
  - For "lowest rated" -> sort_by='rating_asc'; For "highest rated" -> sort_by='rating_desc'
  - For "cheapest" -> sort_by='price_asc'; For "most expensive" -> sort_by='price_desc'
  - Pass `limit` explicitly when the user says "show N products"
- Single product detail (warranty, features, full specs of ONE specific product) -> ProductDetailsDatabase
- Company policies (returns, refunds, shipping, cancellation) -> PolicyVectorDatabase
- Product FAQs (installation, compatibility, troubleshooting, suitability) -> ProductAdviceDatabase

PRODUCT CATEGORIES IN THE DATABASE:
  Gate & Pillar Lights | Solar Lights | Outdoor Wall Lights | Bollard & Garden Lights
  Street Lights | Flood Lights | Indoor & Ceiling Lights | Panel Lights
  Pathway & Step Lights | Bulkhead Lights | Divine & Temple Lights | General Purpose Lights

PARAMETER MAPPING EXAMPLES:
- "show me indoor lights" -> SearchProductsDatabase(query='indoor lights', limit=10)
- "cheapest solar gate light" -> SearchProductsDatabase(query='solar gate', sort_by='price_asc')
- "best rated panel lights" -> SearchProductsDatabase(query='panel lights', sort_by='rating_desc')
- "show me bollard lights under 2000" -> SearchProductsDatabase(query='bollard', max_price=2000)
- "warranty of the Athena light" -> ProductDetailsDatabase(product_name='Athena')
- "what is the return policy" -> PolicyVectorDatabase(query='return policy')"""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    iterations: int
    last_product_search_result: Optional[str]


def build_graph():
    tools = get_tools()
    tool_node = ToolNode(tools)
    llm_with_tools = AgentConfig.llm.bind_tools(tools)

    def agent_node(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def capture_tool_output(state: AgentState):
        """After tools run, capture the raw SearchProductsDatabase output into state."""
        messages = state["messages"]
        last_product_result = state.get("last_product_search_result")
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.name == "SearchProductsDatabase":
                last_product_result = msg.content
                break
        return {"last_product_search_result": last_product_result}

    def judge_node(state: AgentState):
        """
        LLM Judge: evaluates the raw TOOL OUTPUT directly against user query.
        For product searches, the raw JSON from the tool is judged -- the LLM never synthesizes it.
        If VALID -> graph ends; caller returns tool output directly.
        If INVALID -> feedback is added so the agent retries.
        """
        messages = state["messages"]
        iterations = state.get("iterations", 0)

        if iterations >= 3:
            return {}

        user_query = ""
        for msg in messages:
            if isinstance(msg, HumanMessage) and not msg.content.startswith("Feedback:"):
                user_query = msg.content
                break

        product_search_result = state.get("last_product_search_result")
        last_ai_msg = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        last_tool_was_search = bool(product_search_result) and last_ai_msg and not last_ai_msg.tool_calls

        if last_tool_was_search and product_search_result:
            eval_prompt = f"""You are a strict product relevance judge for Inventaa, an Indian outdoor lighting brand.
The user asked: "{user_query}"

The product search tool returned the following JSON:
{product_search_result}

Decide if these products genuinely match what the user requested.
Rules:
- If the user asked for "indoor" products but ALL results are "outdoor", "exterior", or "gate" products -> REJECT.
- If the results are completely unrelated product categories -> REJECT.
- If the JSON is empty [] -> REJECT.
- If the results are reasonably relevant to the user's query -> ACCEPT.

Output ONLY one of:
- "VALID" if the results match
- A short feedback sentence if they do not (e.g. "Results are exterior gate lights but user wants indoor lights. Tell user we don't carry indoor lights.")"""
        else:
            last_ai_content = last_ai_msg.content if last_ai_msg else ""
            eval_prompt = f"""You are a strict QA evaluator for Inventaa.
The user asked: "{user_query}"
The agent responded: "{last_ai_content}"

Evaluate if the response completely and accurately answers the user's question.
If satisfactory, or if the agent correctly said it doesn't have the info, output ONLY: "VALID"
If unsatisfactory, output a short feedback sentence."""

        eval_res = AgentConfig.llm.invoke([SystemMessage(content=eval_prompt)])
        eval_content = eval_res.content.strip()
        logger.info(f"Judge decision: {eval_content[:120]}")

        if eval_content == "VALID":
            return {}
        else:
            feedback_msg = HumanMessage(
                content=f"Feedback: {eval_content}\nPlease retry and provide a better answer."
            )
            return {
                "messages": [feedback_msg],
                "iterations": iterations + 1,
                "last_product_search_result": None,
            }

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return "judge"

    def should_loop(state: AgentState):
        if state.get("iterations", 0) >= 3:
            return "end"
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            return "agent"
        return "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("capture", capture_tool_output)
    workflow.add_node("judge", judge_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "judge": "judge"})
    workflow.add_edge("tools", "capture")
    workflow.add_edge("capture", "agent")
    workflow.add_conditional_edges("judge", should_loop, {"agent": "agent", "end": END})

    return workflow.compile()


_agent_executor = None


def ask_agent(query_text: str, tenant_id: str = None):
    """
    Invokes the agent with the user's query.
    Returns:
      - A parsed JSON list/dict for product search queries (taken directly from tool output, not LLM synthesis).
      - A plain string for conversational/policy/detail answers.
    """
    global _agent_executor
    if _agent_executor is None:
        AgentConfig.initialize()
        _agent_executor = build_graph()
        logger.info("Hybrid RAG Agent Initialized!")

    try:
        final_state = _agent_executor.invoke({
            "messages": [SystemMessage(content=_system_prompt), HumanMessage(content=query_text)],
            "iterations": 0,
            "last_product_search_result": None,
        })

        # If there was a validated product search result, return IT directly (no LLM synthesis)
        product_result = final_state.get("last_product_search_result")
        if product_result:
            try:
                return json.loads(product_result)
            except (json.JSONDecodeError, TypeError):
                pass

        # Otherwise return the last AIMessage content (policy / FAQ / detail answer)
        last_ai = next(
            (m for m in reversed(final_state["messages"]) if isinstance(m, AIMessage) and not m.tool_calls),
            None
        )
        output = last_ai.content if last_ai else "I'm sorry, I encountered an error. Please try again."
        try:
            return json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return output

    except Exception as e:
        logger.error(f"Error invoking agent: {e}")
        return "I'm sorry, I encountered an error while processing your request. Please try again later."
