from langchain.chat_models import init_chat_model
from langchain.agents.middleware import wrap_tool_call
from deepagents import create_deep_agent
from search_agent import search_agent
from config_agent import config_agent
from policy_agent import policy_agent
from cis_benchmark_agent import cis_benchmark_agent
from interdepedency_agent import interdependency_agent
from typing import Any
import json
from pathlib import Path
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from dotenv import load_dotenv
import logging
import os
import time
from activity_stream import astream_activity
from rich_renderer import RichRenderer
import asyncio



load_dotenv()

OPENAI_MODEL = "gpt-5.4-mini-2026-03-17"

model = init_chat_model(model=OPENAI_MODEL, model_provider="openai", temperature=0.0)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("agent.log", mode='w'),
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
    if not isinstance(content, str):
        content = str(content)

    logger.info(
        "← model %.2fs | tokens=%s | tool_calls=%s",
        elapsed,
        f"{usage.get('input_tokens', '?')}→{usage.get('output_tokens', '?')}",
        [tc["name"] for tc in tool_calls] if tool_calls else "none",
    )
    return None


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
async def task_error_guard(request, handler):
    """After a task (subagent) call completes, pause for human review if the result looks like an error."""
    if request.tool_call["name"] != "task":
        return await handler(request)
    result = await handler(request)
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

_run_dir = Path(__file__).parent / "runs" / str(int(time.time()))
_run_dir.mkdir(parents=True, exist_ok=True)

agent = create_deep_agent(
    model=model,
    middleware=[task_error_guard, log_before_model, log_after_model],
    subagents=[policy_agent, cis_benchmark_agent, config_agent, interdependency_agent],
    checkpointer=checkpointer,
    system_prompt = """
You are a Microsoft Intune security supervisor. You orchestrate specialised subagents.

Delegate tasks to subagents using the `task` tool with this format:
{
  "description": "What the subagent should do",
  "subagent_type": "policy_agent" | "config_agent" | "cis_benchmark_agent" | "interdependency_agent" 
}

After each subagent returns, use the write_tool to save the result to memory. Hence, write the result
of each subagent in a file "[subagent_type]_result.md" and save it as a file in memory.

PROCESS (follow all steps in order):
  Step 1: Delegate to policy_agent to extract all the requirements from the security policy.
  Step 2: Delegate to config_agent to find all security settings that match the requirements from the security policy.
  Step 3: Delegate to cis_benchmark_agent to check if the configurations are compliant to the CIS Benchmark.
  Step 4: Delegate to interdependency_agent. 
          Check whether there would be any conflicts or other interdependencies among the settings.
  Step 5: 
  Step 5: Present results as a markdown table with EXACTLY these columns:
          | Setting | Configured | Recommended | Status | Dependencies | Policy Name |
          Status must be one of: COMPLIANT, NON-COMPLIANT, NOT CONFIGURED.
  Step 6: After the table, write:
          - A "Remediation" section listing each NON-COMPLIANT setting with its remediation path from CIS.
          - Any interdependency warnings from Step 4.
          - A note flagging which rows used web search and what the recommendation from Microsoft is.
          - An overview of all security requirements and whether they are met with the settings in the tenant
          - A "Security Posture Summary" paragraph.
          Make sure to write this result into a file called "final_result.md"

Never call multiple subagents or tools at the same time. Each step must wait for the previous one to complete.

OUTPUT FORMAT:
Produce ONLY the markdown table from Step 5 followed by the sections from Step 6.
Do not wrap the output in JSON. Do not skip the table.
""",

)

prompt= """ Step 4: Delegate to search_agent for Microsoft recommendations on settings NOT covered by CIS.
          In the description, tell it to read tenant_configs_vs_benchmark.json.json for the full settings list."""

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
    "files": {"security_policy.txt": _file_data(os.path.join(os.path.dirname(__file__), "security_policy.txt"))},
}

def handle_interrupt(interrupt_values) -> Command:
    for iv in interrupt_values:
        info = iv.value
        print(f"\n--- ERROR DETECTED in '{info.get('subagent', '?')}' ---")
        print(info.get("error_preview", "")[:500])
        print(info.get("message", ""))
    decision = input("Continue pipeline? [y/n]: ").strip().lower()
    return Command(resume={"continue": decision != "n"})

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    renderer = RichRenderer(logger=logger)

    #result = stream_agent_v2(agent, pending, config=run_config, on_interrupt=handle_interrupt)

    final_state = asyncio.run(
        astream_activity(agent, agent_input=pending, config=run_config, render=False, on_event=renderer)
    )

    for vpath, entry in (final_state.get("files") or {}).items():
        name = vpath.lstrip("/")
        if not name:
            continue
        out = _run_dir / name
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = entry.get("content", []) if isinstance(entry, dict) else []
        out.write_text("\n".join(lines) if isinstance(lines, list) else str(lines))
    logger.info("Run files written to %s", _run_dir)