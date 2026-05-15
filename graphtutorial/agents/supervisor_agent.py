from langchain.chat_models import init_chat_model
from langchain.agents.middleware import wrap_tool_call
from deepagents.middleware.summarization import create_summarization_tool_middleware
from deepagents import create_deep_agent
from search_agent import search_agent
from config_agent import config_agent
from policy_agent import policy_agent
from cis_benchmark_agent import cis_benchmark_agent
from typing import Any
import json
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from agent_utils import stream_agent, format_messages, stream_agent_v2
from dotenv import load_dotenv
import logging
import os

load_dotenv()

OPENAI_MODEL = "gpt-5.4-mini-2026-03-17"

model = init_chat_model(model=OPENAI_MODEL, model_provider="openai", temperature=0.0)

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
    
    content = getattr(last_msg, "content", "") or ""
    logger.info(
        "← Model call done in %.2fs | tokens=%s | tool_calls=%s\n%s",
        elapsed,
        f"{usage.get('input_tokens', '?')}→{usage.get('output_tokens', '?')}",
        [tc["name"] for tc in tool_calls] if tool_calls else "none",
        content[:4000] + ("..." if len(content) > 4000 else ""),
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


def _extract_task_content(result) -> str:
    """Extract the subagent response text from whatever the task tool returns.

    Successful task calls return Command(update={"messages": [ToolMessage(...)], ...}).
    Failed arg-validation calls return a ToolMessage directly.
    """
    update = getattr(result, "update", None)
    if isinstance(update, dict):
        messages = update.get("messages", [])
        if messages:
            return getattr(messages[-1], "content", "") or ""
    if hasattr(result, "content"):
        return result.content or ""
    return str(result)


def _task_result_is_error(content: str) -> bool:
    """Return True when a subagent result looks like an error or empty response."""
    if "Error" in content or "error" in content[:50]:
        return True
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("settings") == []:
            return True
        if isinstance(data, dict) and data.get("found") == [] and "missing" in data:
            return True
    except (json.JSONDecodeError, TypeError):
        pass
    return False


@wrap_tool_call
def task_error_guard(request, handler):
    """After a task (subagent) call completes, pause for human review if the result looks like an error."""
    if request.tool_call["name"] != "task":
        return handler(request)
    result = handler(request)
    content = _extract_task_content(result)
    if _task_result_is_error(content):
        subagent = request.tool_call["args"].get("subagent_type", "?")
        human_decision = interrupt({
            "subagent": subagent,
            "error_preview": content[:500],
            "message": f"'{subagent}' returned a suspicious result. Continue pipeline?",
        })
        if not human_decision.get("continue", True):
            from langchain_core.messages import ToolMessage
            return ToolMessage(
                content=f"Pipeline aborted by human after '{subagent}' error.",
                tool_call_id=request.tool_call["id"],
            )
    return result

checkpointer = MemorySaver()

agent = create_deep_agent(
    model=model,
    middleware=[task_error_guard, tool_logger, log_before_model, log_after_model],
    subagents=[config_agent, policy_agent, search_agent, cis_benchmark_agent],
    checkpointer=checkpointer,
    system_prompt = """
You are a Microsoft Intune security supervisor. You orchestrate specialised subagents.

Delegate tasks to subagents using the `task` tool with this format:
{
  "description": "What the subagent should do",
  "subagent_type": "policy_agent" | "config_agent" | "cis_benchmark_agent" | "search_agent"
}

After each subagent returns, use the write_tool to save the result to memory.

PROCESS (follow all steps in order):
  Step 1: Delegate to policy_agent with password_policy.txt to find relevant policies for the topic.
  Step 2: Delegate to config_agent for current configured values, scoped to settings from Step 1.
          config_agent will automatically write its output to relevant_configs.json.
  Step 3: Delegate to cis_benchmark_agent. In the description tell it:
          "Read relevant_configs.json to get the settings list. Use compare_relevant_settings_to_cis_benchmark
          with those settings. Return CIS findings only."
  Step 4: Delegate to search_agent for Microsoft recommendations on settings NOT covered by CIS.
          In the description, tell it to read relevant_configs.json for the full settings list.
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

query = "What are the best practices for password configurations in Microsoft Intune for Windows 11 devices?"

def _file_data(path: str) -> dict:
    """Wrap a file's content in the FileData format deepagents expects."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with open(path) as f:
        lines = f.read().splitlines()
    return {"content": lines, "created_at": now, "modified_at": now}

run_config = {"configurable": {"thread_id": "1"}}
pending: Any = {
    "messages": [{"role": "user", "content": query}],
    "files": {"password_policy.txt": _file_data(os.path.join(os.path.dirname(__file__), "password_policy.txt"))},
}

print("Pending: ", pending)

def handle_interrupt(interrupt_values) -> Command:
    for iv in interrupt_values:
        info = iv.value
        print(f"\n--- ERROR DETECTED in '{info.get('subagent', '?')}' ---")
        print(info.get("error_preview", "")[:500])
        print(info.get("message", ""))
    decision = input("Continue pipeline? [y/n]: ").strip().lower()
    return Command(resume={"continue": decision != "n"})


stream_agent_v2(agent, pending, config=run_config, on_interrupt=handle_interrupt)
