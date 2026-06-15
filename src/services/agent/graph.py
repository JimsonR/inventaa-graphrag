import json
import logging
import operator
import sys
from typing import Annotated, Sequence, TypedDict, Optional

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.services.agent.config import AgentConfig
from src.services.agent.tools import get_tools
from src.services.agent.routing import get_intent_config

# Configure basic logging to ensure INFO statements show up in uvicorn
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    iterations: int
    last_product_search_result: Optional[str]


def build_graph(system_prompt: str, tools: list):
    """Build a LangGraph agent using the given focused prompt and tool subset."""
    tool_node = ToolNode(tools)
    llm_with_tools = AgentConfig.llm.bind_tools(tools)

    def agent_node(state: AgentState):
        logger.info(f"[Agent Node] Invoking LLM with {len(state['messages'])} messages. Iteration: {state.get('iterations', 0)}")
        response = llm_with_tools.invoke(state["messages"])

        # Guard: LangGraph crashes if the LLM returns neither text nor tool calls
        if not response.tool_calls and not response.content:
            logger.warning("[Agent Node] LLM returned empty response — injecting fallback message.")
            response.content = "I'm sorry, I'm only able to assist with LED lighting products and related queries from Inventaa."

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
        for msg in reversed(messages):
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
- If the JSON is empty [] -> ACCEPT. This is a valid response meaning we do not carry the requested product.
- If the results are reasonably relevant to the user's query -> ACCEPT.

Output ONLY one of:
- "VALID" if the results match (or if the JSON is empty [])
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


_initialized = False


def ask_agent(query_text: str, tenant_id: str = None, session_id: str = None, message_id: str = None):
    """
    Invokes the agent with the user's query.
    Returns:
      - A parsed JSON list/dict for product search queries (taken directly from tool output).
      - A plain string for conversational/policy/detail answers.
    """
    global _initialized
    if not _initialized:
        AgentConfig.initialize()
        _initialized = True
        logger.info("Hybrid RAG Agent Initialized!")

    try:
        from src.services.agent.context import tenant_context
        tenant_context.set(tenant_id)
        
        logger.info(f"--- STARTING GRAPH EXECUTION FOR QUERY: '{query_text}' ---")

        # Classify query intent → get focused prompt + only the relevant tools
        all_tools = get_tools()
        system_prompt, active_tools = get_intent_config(query_text, all_tools, llm=AgentConfig.llm)
        executor = build_graph(system_prompt, active_tools)

        # Load conversational memory
        from src.services.agent.memory import get_recent_messages
        history = get_recent_messages(session_id=session_id, exclude_message_id=message_id, limit=5)
        
        messages = [SystemMessage(content=system_prompt)] + history + [HumanMessage(content=query_text)]

        final_state = executor.invoke({
            "messages": messages,
            "iterations": 0,
            "last_product_search_result": None,
        })

        # If there was a validated product search result, return it directly (no LLM synthesis)
        product_result = final_state.get("last_product_search_result")
        if product_result:
            try:
                parsed_json = json.loads(product_result)
                logger.info(f"--- GRAPH FINISHED: Returning raw JSON array from tool ({len(parsed_json)} items) ---")
                return parsed_json
            except (json.JSONDecodeError, TypeError):
                logger.warning("--- GRAPH FINISHED: Failed to parse tool JSON, falling back to conversational ---")

        # Otherwise return the last AIMessage content (policy / FAQ / detail answer)
        last_ai = next(
            (m for m in reversed(final_state["messages"]) if isinstance(m, AIMessage) and not m.tool_calls),
            None
        )
        output = last_ai.content if last_ai else "I'm sorry, I encountered an error. Please try again."

        try:
            parsed = json.loads(output)
            logger.info("--- GRAPH FINISHED: Returning JSON from agent ---")
            return parsed
        except (json.JSONDecodeError, TypeError):
            logger.info(f"--- GRAPH FINISHED: Returning text from agent ({len(output)} chars) ---")
            return output

    except Exception as e:
        logger.error(f"Error invoking agent: {e}", exc_info=True)
        return "I'm sorry, I encountered an error while processing your request. Please try again later."
