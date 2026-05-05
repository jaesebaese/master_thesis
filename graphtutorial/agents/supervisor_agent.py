from langchain.chat_models import init_chat_model
from langchain.agents.middleware import wrap_tool_call
from deepagents.middleware.summarization import create_summarization_tool_middleware
from deepagents import create_deep_agent
from search_agent import search_agent
from config_agent import config_agent
from policy_agent import policy_agent
from cis_benchmark_agent import cis_benchmark_agent
from typing import Any
from langchain.agents import AgentState 
import asyncio
from agent_utils import stream_agent, format_messages, stream_agent_v2
import logging

OLLAMA_MODEL = "llama3.1:latest"

model = init_chat_model(model=OLLAMA_MODEL, model_provider="ollama", temperature=0.0)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    handlers=[
        logging.FileHandler("agent.log", mode='w'),  # overwrite log file on each run
    ],)


import time
from contextvars import ContextVar
from langchain.agents.middleware import before_model, after_model

_start = ContextVar("model_start", default=None)

@before_model
def log_before_model(state, runtime):
    _start.set(time.time())
    return None

@after_model
def log_after_model(state, runtime):
    started = _start.get()
    elapsed = time.time() - started if started else 0
    last_msg = state["messages"][-1]
    
    tool_calls = getattr(last_msg, "tool_calls", None) or []
    usage = getattr(last_msg, "usage_metadata", None) or {}
    
    logger.info(
        "← Model call done in %.2fs | tokens=%s | tool_calls=%s",
        elapsed,
        f"{usage.get('input_tokens', '?')}→{usage.get('output_tokens', '?')}",
        [tc["name"] for tc in tool_calls] if tool_calls else "none",
    )
    return None


def log_chunk(chunk: dict) -> None:
    if not isinstance(chunk, dict):
        logger.info("Chunk: %s", str(chunk)[:1000])
        return

    for node, payload in chunk.items():
        if payload is None:
            logger.info("[%s] (no output)", node)
            continue

        # Payload might be an Overwrite/Append wrapper, a dict, or something else
        if not isinstance(payload, dict):
            logger.info("[%s] %s", node, str(payload)[:1000])
            continue

        # Drop noisy keys
        payload = {k: v for k, v in payload.items() if k != "files"}

        messages = payload.get("messages")
        if messages is None:
            # No messages field — just log keys touched
            logger.info("[%s] keys=%s", node, list(payload.keys()))
            continue

        # messages might also be an Overwrite wrapper, not a list
        if not isinstance(messages, list):
            logger.info("[%s] messages=%s", node, str(messages)[:300])
            continue

        for msg in messages:
            role = type(msg).__name__
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                content = str(content)
            preview = content[:300] + ("..." if len(content) > 300 else "")
            logger.info("[%s] %s: %s", node, role, preview)


@wrap_tool_call
def tool_logger(request, handler):
    name = request.tool_call["name"]
    args = request.tool_call["args"]
    logger.info("Tool call: %s args=%s", name, args)
    result = handler(request)
    content = result.content if hasattr(result, "content") else str(result)
    truncated = content[:300] + ("..." if len(content) > 300 else "")
    logger.info("Tool result: %s", truncated)
    return result

agent = create_deep_agent(
    model=model,
    middleware=[tool_logger, log_before_model, log_after_model],
    subagents=[config_agent, policy_agent, search_agent, cis_benchmark_agent],
    system_prompt = """
You are a Microsoft Intune security supervisor. You orchestrate specialised subagents.

Delegate tasks to subagents using the `task` tool with this format:
{
  "description": "What the subagent should do",
  "subagent_type": "policy_agent" | "config_agent" | "cis_benchmark_agent" | "search_agent"
}

After each subagent returns, use the write_tool to save the result to memory.

PROCESS (follow all steps in order):
  Step 1: Delegate to policy_agent with security_policy.txt to find relevant policies for the topic.
  Step 2: Delegate to config_agent for current configured values, scoped to settings from Step 1.
  Step 3: Delegate to cis_benchmark_agent for CIS recommendations on those settings.
  Step 4: Delegate to search_agent for Microsoft recommendations on settings NOT covered by CIS.
  Step 5: Present results as a markdown table with EXACTLY these columns:
          | Setting | Configured | Recommended | Status | Dependencies |
          Status must be one of: COMPLIANT, NON-COMPLIANT, NOT CONFIGURED.
  Step 6: After the table, write:
          - A "Remediation" section listing each NON-COMPLIANT setting with its remediation path from CIS.
          - Any interdependency warnings from Step 4.
          - A note flagging which rows used web search (medium confidence).
          - A "Security Posture Summary" paragraph.

OUTPUT FORMAT:
Produce ONLY the markdown table from Step 5 followed by the sections from Step 6.
Do not wrap the output in JSON. Do not skip the table.
""",

)

query = input("Ask the supervisor agent: ")
# result = agent.invoke({"messages": [{"role": "user", "content": query}]})
#result = agent.invoke({"messages": [{"role": "user", "content": query}], "files": {"security_policy.txt": open("/home/hochuli/project/master_thesis/graphtutorial/agents/password_policy.txt").read()}})
#for chunk in agent.stream({"messages": [{"role": "user", "content": query}], "files": {"password_policy.txt": open("/home/hochuli/project/master_thesis/graphtutorial/agents/password_policy.txt").read()}}):
 #   log_chunk(chunk)
 #formatted_result = format_messages(final_state["messages"])
final_state = stream_agent_v2(agent, {"messages": [{"role": "user", "content": query}], "files": {"password_policy.txt": open("/home/hochuli/project/master_thesis/graphtutorial/agents/password_policy.txt").read()}})
