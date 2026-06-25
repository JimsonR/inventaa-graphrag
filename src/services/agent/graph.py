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

    agent_logger = logging.getLogger("Node.Agent")
    
    def agent_node(state: AgentState):
        from src.services.agent.utils import track_time
        with track_time("Node: Agent", custom_logger=agent_logger):
            agent_logger.info(f"Invoking LLM with {len(state['messages'])} messages. Iteration: {state.get('iterations', 0)}")
            response = llm_with_tools.invoke(state["messages"])

        # Guard: LangGraph crashes if the LLM returns neither text nor tool calls
        if not response.tool_calls and not response.content:
            agent_logger.warning("LLM returned empty response — injecting fallback message.")
            fallback = AgentConfig.brain.get("prompts", {}).get(
                "fallback_error_message",
                "I'm sorry, I'm only able to assist with LED lighting products and related queries from Inventaa."
            )
            response.content = fallback

        if response.tool_calls:
            agent_logger.info(f"LLM decided to call tools: {[tc['name'] for tc in response.tool_calls]}")
        else:
            agent_logger.info(f"LLM provided a direct response: {response.content[:100]}...")
        return {"messages": [response]}

    capture_logger = logging.getLogger("Node.Capture")

    def capture_tool_output(state: AgentState):
        """After tools run, capture the raw SearchProductsDatabase output into state."""
        capture_logger.info("Capturing tool outputs...")
        messages = state["messages"]
        last_product_result = state.get("last_product_search_result")
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                capture_logger.info(f"Tool '{msg.name}' returned output (length {len(msg.content)})")
                if msg.name == "SearchProductsDatabase":
                    # If the tool returned a needs_clarification response,
                    # DON'T set it as product result — let the agent
                    # synthesize it into a conversational reply.
                    try:
                        parsed = json.loads(msg.content)
                        if isinstance(parsed, dict) and parsed.get("needs_clarification"):
                            capture_logger.info("Tool returned needs_clarification — routing to agent for synthesis.")
                            last_product_result = None
                        else:
                            last_product_result = msg.content
                    except (json.JSONDecodeError, TypeError):
                        last_product_result = msg.content
                break
        return {"last_product_search_result": last_product_result}

    judge_logger = logging.getLogger("Node.Judge")

    def judge_node(state: AgentState):
        from src.services.agent.utils import track_time
        with track_time("Node: Judge", custom_logger=judge_logger):
            """
            LLM Judge: evaluates the raw TOOL OUTPUT directly against user query.
            For product searches, the raw JSON from the tool is judged -- the LLM never synthesizes it.
            If VALID -> graph ends; caller returns tool output directly.
            If INVALID -> feedback is added so the agent retries.
            """
            messages = state["messages"]
            iterations = state.get("iterations", 0)
            judge_logger.info(f"Evaluating output at iteration {iterations}...")

        if iterations >= 3:
            judge_logger.warning("Max iterations reached (3). Forcing completion.")
            return {}

        user_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and not msg.content.startswith("Feedback:"):
                user_query = msg.content
                break

        product_search_result = state.get("last_product_search_result")
        
        # If the last message is a ToolMessage from SearchProductsDatabase, we bypassed the Agent synthesis.
        last_msg = messages[-1]
        is_direct_product_eval = isinstance(last_msg, ToolMessage) and last_msg.name == "SearchProductsDatabase"

        if is_direct_product_eval and product_search_result:
            brand_name = AgentConfig.brain.get("tenant", {}).get("name", "Inventaa")
            brand_desc = AgentConfig.brain.get("tenant", {}).get("description", "an Indian LED lighting brand")
            prompt_template = AgentConfig.brain.get("prompts", {}).get(
                "judge_system_product",
                f"You are a strict product relevance judge for {brand_name}.\n"
                "The user asked: \"{user_query}\"\n"
                "The product search tool returned the following JSON:\n{product_search_result}\n"
                "Output ONLY \"VALID\" or a short feedback sentence."
            )
            eval_prompt = prompt_template.format(
                brand_name=brand_name,
                brand_description=brand_desc,
                user_query=user_query,
                product_search_result=product_search_result
            )
        else:
            last_ai_msg = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
            last_ai_content = last_ai_msg.content if last_ai_msg else ""
            brand_name = AgentConfig.brain.get("tenant", {}).get("name", "Inventaa")
            
            prompt_template = AgentConfig.brain.get("prompts", {}).get(
                "judge_system_qa",
                f"You are a lenient QA evaluator for {brand_name}.\n"
                "The user asked: \"{user_query}\"\n"
                "The agent responded: \"{last_ai_content}\"\n"
                "Output ONLY \"VALID\" unless the response has CRITICAL failures."
            )
            eval_prompt = prompt_template.format(
                brand_name=brand_name,
                user_query=user_query,
                last_ai_content=last_ai_content
            )

        eval_res = AgentConfig.llm.invoke([SystemMessage(content=eval_prompt)])
        eval_content = eval_res.content.strip()

        if eval_content == "VALID":
            judge_logger.info(f"Decision: VALID. Ending graph.")
            return {}
        else:
            judge_logger.info(f"Decision: INVALID. Sending feedback: {eval_content}")
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

    def after_capture(state: AgentState):
        """Conditionally route after tools run."""
        last_msg = state["messages"][-1]
        # If we just searched for products, bypass the Agent and go straight to the Judge
        if isinstance(last_msg, ToolMessage) and last_msg.name == "SearchProductsDatabase":
            # EXCEPT when the tool returned needs_clarification — route to agent
            # so it can synthesize a conversational "which collection?" reply.
            try:
                parsed = json.loads(last_msg.content)
                if isinstance(parsed, dict) and parsed.get("needs_clarification"):
                    return "agent"
            except (json.JSONDecodeError, TypeError):
                pass
            return "judge"
        # For policy/faq tools, go to Agent to synthesize text
        return "agent"

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
    workflow.add_conditional_edges("capture", after_capture, {"judge": "judge", "agent": "agent"})
    workflow.add_conditional_edges("judge", should_loop, {"agent": "agent", "end": END})

    return workflow.compile()


_initialized = False


def ask_agent(query_text: str, tenant_id: str = None, session_id: str = None, message_id: str = None, user_id: str = None):
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

    from src.services.agent.utils import track_time
    request_logger = logging.getLogger("Task.Request")
    with track_time("Total Request Execution", custom_logger=request_logger):
        try:
            from src.services.agent.context import tenant_context
            tenant_context.set(tenant_id)
            
            logger.info(f"--- STARTING GRAPH EXECUTION FOR QUERY: '{query_text}' ---")

            # ─── Parallel Pre-computation ──────────────────────────────────────────
            # We run History+Routing, Embeddings, and Mem0 fetch simultaneously
            # to crush the pre-computation latency.
            from src.services.agent.memory import get_recent_messages
            from src.services.agent.mem0_client import fetch_long_term_context, store_long_term_context
            from src.services.agent.response_cache import cache_lookup, cache_store
            import concurrent.futures
            import os
            import threading

            all_tools = get_tools()
            task_logger = logging.getLogger("Task.Parallel")

            def task_history_and_intent():
                from src.services.agent.utils import track_time
                with track_time("Task: History & Intent", custom_logger=task_logger):
                    history = get_recent_messages(session_id=session_id, exclude_message_id=message_id, limit=5)
                    history_text = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}" for m in history[-2:]])
                    sys_prompt, active_tools, router_result = get_intent_config(query_text, all_tools, llm=AgentConfig.llm, history_context=history_text)
                    return history, sys_prompt, active_tools, router_result

            def task_embedding():
                from src.services.agent.utils import track_time
                with track_time("Task: Embedding Generation", custom_logger=task_logger):
                    return AgentConfig.embeddings.embed_query(query_text)

            def task_mem0():
                from src.services.agent.utils import track_time
                with track_time("Task: Mem0 Retrieval", custom_logger=task_logger):
                    return fetch_long_term_context(query_text, user_id)

            from src.services.agent.utils import track_time
            with track_time("Total Pre-computation Phase 1", custom_logger=task_logger):
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    future_intent = executor.submit(task_history_and_intent)
                    future_embedding = executor.submit(task_embedding)
                    future_mem0 = executor.submit(task_mem0)

                    history, system_prompt, active_tools, router_result = future_intent.result()
                    intent = router_result.intent
                    translation = router_result.english_translation
                    query_embedding = future_embedding.result()
                    long_term_context = future_mem0.result()

            forced_single_collection = None

            # ─── Agentic Broad Navigation Intercept ────────────────────────────────
            if router_result.is_broad_navigation:
                logger.info(f"Agentic Router intercepted broad query. Group: {router_result.broad_category_group}")
                if router_result.broad_category_group and router_result.broad_category_group.lower() != "all":
                    group_key = router_result.broad_category_group.lower()
                    
                    # Case-insensitive lookup in category_groups
                    matched_group = next((k for k in AgentConfig.category_groups.keys() if k.lower() == group_key), None) if AgentConfig.category_groups else None
                    
                    if matched_group:
                        cols = AgentConfig.category_groups[matched_group]
                        if len(cols) == 1:
                            logger.info(f"Top-level group '{group_key}' only has 1 collection ({cols[0]}). Bypassing clarification and forcing search.")
                            forced_single_collection = cols[0]
                        else:
                            reply = "Could you please specify which type you're interested in?\n\n"
                            reply += "\n".join(f"• {c}" for c in cols)
                            return reply
                
                if not forced_single_collection:
                    # Default "All" or unrecognized group
                    cols = AgentConfig.top_level_groups or []
                    reply = "Could you please specify which type you're interested in?\n\n"
                    if cols:
                        reply += "\n".join(f"• {c}" for c in cols)
                    else:
                        reply += AgentConfig.brain.get("prompts", {}).get(
                            "broad_query_fallback",
                            "For example, are you looking for Indoor, Outdoor, or Solar lights?"
                        )
                    return reply

            taxonomy_candidates_pending = None
            if forced_single_collection:
                system_prompt += (f"\n\nIMPORTANT: The user is asking for a category group that exactly maps to the "
                                  f"'{forced_single_collection}' collection. You MUST pass category='{forced_single_collection}' to SearchProductsDatabase.")
            else:
                with track_time("Total Pre-computation Phase 2 (Taxonomy Match)", custom_logger=task_logger):
                    # Only call Azure OpenAI to embed the translation if it actually changed (e.g. from Telugu to English)
                    # If it's the same, reuse the embedding from Phase 1 to save 1.5 - 15 seconds of latency!
                    if translation and translation != query_text:
                        translation_embedding = AgentConfig.embeddings.embed_query(translation)
                    else:
                        translation_embedding = query_embedding
                    
                    # Fetch raw candidates via vector similarity
                    from src.services.agent.taxonomy import fetch_taxonomy_candidates, extract_taxonomy_parameters
                    raw_candidates = fetch_taxonomy_candidates(translation_embedding)
                    
                    if raw_candidates and "SearchProductsDatabase" in [t.name for t in active_tools]:
                        taxonomy_candidates_pending = raw_candidates

            # ─── Sub-Agent: Taxonomy Parameter Extraction ──────────────────────────
            if taxonomy_candidates_pending:
                # Let the tiny Sub-Agent filter the messy vector candidates into exact tool parameters
                extracted_params = extract_taxonomy_parameters(query_text, taxonomy_candidates_pending)
                
                if extracted_params.clarify:
                    logger.info("Taxonomy Sub-Agent determined query is too broad. Returning clarification request.")
                    # Grab the candidate categories that confused the sub-agent
                    candidate_cols = taxonomy_candidates_pending.get("category", [])
                    reply = "There are a few different types of products that match your request. Could you specify which type you're interested in?\n\n"
                    if candidate_cols:
                        reply += "\n".join(f"• {c}" for c in candidate_cols[:5]) # limit to top 5
                    else:
                        reply += "For example, are you looking for Indoor, Outdoor, or Solar lights?"
                    return reply
                
                # Build clean parameter list for Main Agent
                clean_params = []
                if extracted_params.category: clean_params.append(f"category='{extracted_params.category}'")
                if extracted_params.use_case: clean_params.append(f"use_case='{extracted_params.use_case}'")
                if extracted_params.feature: clean_params.append(f"feature='{extracted_params.feature}'")
                
                if clean_params:
                    params_str = ", ".join(clean_params)
                    system_prompt += (f"\n\nSystem: Based on the user's request, you MUST use the following exact parameters "
                                      f"in your SearchProductsDatabase call: {params_str}.")
            # ─────────────────────────────────────────────────────────────────────

            # ─── Semantic Cache: Lookup ──────────────────────────────────────────
            cache_threshold = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.99"))
            cache_ttl = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24 hours
            cache_skip = ["detail", "search", "policy"]  # product, detail, and policy lookups should always be fresh
            
            cached = cache_lookup(
                embedding=query_embedding,
                tenant_id=tenant_id,
                intent=intent,
                threshold=cache_threshold,
                skip_intents=cache_skip,
            )
            if cached:
                logger.info(f"--- CACHE HIT: Returning cached response (original: {cached.get('query_text', '?')!r}) ---")
                # Still store in mem0 asynchronously
                if cached["response_type"] == "text":
                    threading.Thread(target=store_long_term_context, args=(query_text, cached["response"], user_id), daemon=True).start()
                return cached["response"]
            # ─── End Cache Lookup ────────────────────────────────────────────────
            
            executor_graph = build_graph(system_prompt, active_tools)
            
            # ─── Per-Intent Context Slimming ─────────────────────────────────────────
            # Reduce token overload by stripping irrelevant context before injection
            if intent in ("policy", "advice", "knowledge"):
                long_term_context = ""
                history = []
            elif intent == "detail":
                long_term_context = ""
                history = history[-1:] if history else []
            elif intent == "search":
                history = history[-2:] if len(history) >= 2 else history
            # ─────────────────────────────────────────────────────────────────────────
            
            if long_term_context:
                logger.info(f"Injected long term context for user {user_id}")
                system_prompt += long_term_context
            
            messages = [SystemMessage(content=system_prompt)] + history + [HumanMessage(content=query_text)]
            
            logger.info("--- PROMPT INJECTED TO AGENT LLM ---")
            for idx, m in enumerate(messages):
                role = "SYSTEM" if isinstance(m, SystemMessage) else "USER" if isinstance(m, HumanMessage) else "AGENT"
                logger.info(f"[{idx}] {role}:\n{m.content}\n")
            logger.info("------------------------------------")

            from src.services.agent.utils import track_time
            with track_time("Total Graph Execution"):
                final_state = executor_graph.invoke({
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
                    # Cache the product result
                    cache_store(
                        embedding=query_embedding, query_text=query_text,
                        response=parsed_json, tenant_id=tenant_id,
                        intent=intent, response_type="products",
                        ttl=cache_ttl, skip_intents=cache_skip,
                    )
                    return parsed_json
                except (json.JSONDecodeError, TypeError):
                    logger.info("--- GRAPH FINISHED: Returning plain text from SearchProductsDatabase ---")
                    return product_result

            # Otherwise return the last AIMessage content (policy / FAQ / detail answer)
            last_ai = next(
                (m for m in reversed(final_state["messages"]) if isinstance(m, AIMessage) and not m.tool_calls),
                None
            )
            output = last_ai.content if last_ai else "I'm sorry, I encountered an error. Please try again."

            # Helper to trigger async memory ingestion without blocking the response
            def run_background_storage(response_text):
                import threading
                threading.Thread(target=store_long_term_context, args=(query_text, response_text, user_id), daemon=True).start()

            try:
                parsed = json.loads(output)
                logger.info("--- GRAPH FINISHED: Returning JSON from agent ---")
                return parsed
            except (json.JSONDecodeError, TypeError):
                logger.info(f"--- GRAPH FINISHED: Returning text from agent ({len(output)} chars) ---")
                # Cache the text response
                cache_store(
                    embedding=query_embedding, query_text=query_text,
                    response=output, tenant_id=tenant_id,
                    intent=intent, response_type="text",
                    ttl=cache_ttl, skip_intents=cache_skip,
                )
                # Store the conversational answer in Mem0
                run_background_storage(output)
                return output

        except Exception as e:
            logger.error(f"Error invoking agent: {e}", exc_info=True)
            return "I'm sorry, I encountered an error while processing your request. Please try again later."

