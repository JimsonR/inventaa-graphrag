import json
import logging
import operator
from typing import Annotated, Sequence, TypedDict, Optional

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.services.agent.config import AgentConfig
from src.services.agent.tools import get_tools

import sys

# Configure basic logging to ensure INFO statements show up in uvicorn
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_system_prompt = """You are an expert AI sales assistant for Inventaa, an Indian LED lighting brand.
You MUST use tools for every response — never answer from your own knowledge.

ABSOLUTE RULES:
1. ALWAYS call a tool first. Never answer from your own knowledge.
2. If the first tool returns no result, try a different tool before giving up.
3. If no tool finds relevant info, say ONLY: "I'm sorry, I don't have that information in our database."
4. NEVER guess, invent, or hallucinate product names, prices, specs, or policies.

─────────────────────────────────────
TOOL SELECTION GUIDE
─────────────────────────────────────

→ SearchProductsDatabase — Use for ANY of these:
  • Listing or browsing products ("show me gate lights", "what solar lights do you have")
  • Filtering by feature/spec ("waterproof outdoor lights", "warm white garden lights")
  • Budget-based ("lights under ₹2000", "products within ₹10,000")
  • Application/location ("lights for driveway", "lighting for villa entrance", "lights for garden")
  • Recommendation requests ("suggest lights for a hotel", "best lights for heavy rain area")
  • Comparison listings ("show me warm white AND cool white pathway lights")
  • Sorting ("cheapest gate lights", "highest rated bollard lights")

→ ProductDetailsDatabase — Use when the user asks about ONE specific NAMED product:
  • Technical specs: wattage, lumens, dimensions, beam angle, material, IP rating, colour temperature
  • "Is the Athena light available in warm white?"
  • "What material is the Tacita fixture?"
  • "What are the dimensions of the Mini Olivia?"
  • "Does the Artoo light come with mounting hardware?"
  • "What is the warranty on the Glenda light?"

→ ProductAdviceDatabase — Use for general how-to / suitability / comparison advice (no specific product name):
  • Installation: "Is installation easy?", "Can I install it myself?", "What is the mounting height?"
  • Energy: "How long do LEDs last?", "Will LED reduce my electricity bill?", "What is the lifespan?"
  • Comparison: "Which is better: warm white or cool white?", "What is the difference between bollard and pathway lights?"
  • Suitability: "Can this be used for coastal areas?", "Can this be used for commercial spaces?"
  • Smart/timer: "Can this be connected to a smart switch or timer?"
  • Maintenance: "Which outdoor light requires least maintenance?"

→ GeneralKnowledgeDatabase — Use for educational/comparison/concept questions about lighting:
  • 'Wave-Free LED Panel Lights vs Traditional LED Panel Lights'
  • 'What is the difference between bollard and pathway lights?'
  • 'How to choose outdoor lighting for my home?'
  • 'What is CRI (colour rendering index)?'
  • 'How many lumens do I need for outdoor spaces?'
  • 'Benefits of solar lights over wired lights'
  • 'LED vs fluorescent: which is better?'
  • 'What is IP rating?', 'How to read an LED spec sheet?'

→ PolicyVectorDatabase — Use ONLY for business/operational questions:
  • Returns, replacements, exchanges, damaged/wrong item
  • Delivery time, shipping charges, order tracking, express shipping
  • Warranty claim process, what is covered, replacement parts
  • Bulk/dealer/contractor/distributor pricing
  • Required documents for claims

─────────────────────────────────────
PARAMETER MAPPING EXAMPLES
─────────────────────────────────────
"show me indoor lights"               → SearchProductsDatabase(query='indoor lights', limit=10)
"cheapest solar gate light"           → SearchProductsDatabase(query='solar gate', sort_by='price_asc')
"best rated panel lights"             → SearchProductsDatabase(query='panel lights', sort_by='rating_desc')
"lights for garden under ₹2000"       → SearchProductsDatabase(query='garden', max_price=2000)
"waterproof warm white pathway lights"→ SearchProductsDatabase(query='pathway', feature='warm-white', spec='IP65')
"lights for villa entrance"           → SearchProductsDatabase(query='gate entrance villa')
"suggest for hotel landscape project" → SearchProductsDatabase(query='landscape garden bollard')
"energy efficient lights under ₹5000" → SearchProductsDatabase(feature='energy-efficient', max_price=5000, sort_by='rating_desc')
"IP65 street lights"                  → SearchProductsDatabase(query='street', spec='IP65')
"lights for heavy rain area"          → SearchProductsDatabase(feature='waterproof')
"Wave-Free vs Traditional LED panels"        → GeneralKnowledgeDatabase(query='wave-free LED panel traditional comparison')
"benefits of solar outdoor lights"            → GeneralKnowledgeDatabase(query='benefits solar outdoor lights')
"what is IP65 rating"                         → GeneralKnowledgeDatabase(query='IP65 rating waterproof outdoor')
"warranty of the Athena light"               → ProductDetailsDatabase(product_name='Athena')
"what material is the Tacita?"        → ProductDetailsDatabase(product_name='Tacita')
"what is the return policy?"          → PolicyVectorDatabase(query='return policy')
"how long does delivery take?"        → PolicyVectorDatabase(query='delivery time')
"is installation easy?"               → ProductAdviceDatabase(query='installation')
"can it be used near coastal areas?"  → ProductAdviceDatabase(query='coastal waterproof durability')

PRODUCT CATEGORIES IN THE DATABASE:
  Gate & Pillar Lights | Solar Lights | Outdoor Wall Lights | Bollard & Garden Lights
  Street Lights | Flood Lights | Indoor & Ceiling Lights | Panel Lights
  Pathway & Step Lights | Bulkhead Lights | Divine & Temple Lights | General Purpose Lights"""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    iterations: int
    last_product_search_result: Optional[str]


def build_graph():
    tools = get_tools()
    tool_node = ToolNode(tools)
    llm_with_tools = AgentConfig.llm.bind_tools(tools)

    def agent_node(state: AgentState):
        logger.info(f"[Agent Node] Invoking LLM with {len(state['messages'])} messages. Iteration: {state.get('iterations', 0)}")
        response = llm_with_tools.invoke(state["messages"])
        if response.tool_calls:
            logger.info(f"[Agent Node] LLM decided to call tools: {[tc['name'] for tc in response.tool_calls]}")
        else:
            logger.info(f"[Agent Node] LLM provided a direct response: {response.content[:100]}...")
        return {"messages": [response]}

    def capture_tool_output(state: AgentState):
        """After tools run, capture the raw SearchProductsDatabase output into state."""
        logger.info("[Capture Node] Capturing tool outputs...")
        messages = state["messages"]
        last_product_result = state.get("last_product_search_result")
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                logger.info(f"[Capture Node] Tool '{msg.name}' returned output (length {len(msg.content)})")
                if msg.name == "SearchProductsDatabase":
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
        logger.info(f"[Judge Node] Evaluating output at iteration {iterations}...")

        if iterations >= 3:
            logger.warning("[Judge Node] Max iterations reached (3). Forcing completion.")
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
            eval_prompt = f"""You are a lenient QA evaluator for Inventaa.
The user asked: "{user_query}"
The agent responded: "{last_ai_content}"

Output ONLY "VALID" unless the response has one of these CRITICAL failures:
- The agent made up information not supported by any tool (hallucinated)
- The agent said it doesn't know, but the response actually contains the answer
- The agent gave the WRONG product (completely wrong product name)
- The response is completely empty or a server error message

DO NOT reject for minor phrasing preferences (e.g. saying "available in 18W" vs "only 18W available").
DO NOT reject policy, FAQ, or product detail answers that contain the relevant facts.
If in doubt, output "VALID"."""

        eval_res = AgentConfig.llm.invoke([SystemMessage(content=eval_prompt)])
        eval_content = eval_res.content.strip()

        if eval_content == "VALID":
            logger.info(f"[Judge Node] Decision: VALID. Ending graph.")
            return {}
        else:
            logger.info(f"[Judge Node] Decision: INVALID. Sending feedback: {eval_content}")
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
        logger.info(f"--- STARTING GRAPH EXECUTION FOR QUERY: '{query_text}' ---")
        final_state = _agent_executor.invoke({
            "messages": [SystemMessage(content=_system_prompt), HumanMessage(content=query_text)],
            "iterations": 0,
            "last_product_search_result": None,
        })

        # If there was a validated product search result, return IT directly (no LLM synthesis)
        product_result = final_state.get("last_product_search_result")
        if product_result:
            try:
                parsed_json = json.loads(product_result)
                logger.info(f"--- GRAPH FINISHED: Returning raw JSON array from tool ({len(parsed_json)} items) ---")
                return parsed_json
            except (json.JSONDecodeError, TypeError):
                logger.warning("--- GRAPH FINISHED: Failed to parse tool JSON, falling back to conversational ---")
                pass

        # Otherwise return the last AIMessage content (policy / FAQ / detail answer)
        last_ai = next(
            (m for m in reversed(final_state["messages"]) if isinstance(m, AIMessage) and not m.tool_calls),
            None
        )
        output = last_ai.content if last_ai else "I'm sorry, I encountered an error. Please try again."
        
        try:
            parsed = json.loads(output)
            logger.info(f"--- GRAPH FINISHED: Returning JSON from agent ---")
            return parsed
        except (json.JSONDecodeError, TypeError):
            logger.info(f"--- GRAPH FINISHED: Returning text from agent ({len(output)} chars) ---")
            return output

    except Exception as e:
        logger.error(f"Error invoking agent: {e}")
        return "I'm sorry, I encountered an error while processing your request. Please try again later."
