from langchain.chat_models import init_chat_model
from langchain.agents.middleware import wrap_tool_call
from deepagents import create_deep_agent
from search_agent import search_agent
from config_agent import config_agent
from policy_agent import policy_agent
from benchmark_agent import benchmark_agent
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
from contextvars import ContextVar
from langchain.agents.middleware import before_model, after_model



load_dotenv()

OPENAI_MODEL = "gpt-5.4-mini-2026-03-17"

model = init_chat_model(model=OPENAI_MODEL, model_provider="openai", temperature=0.0)

logger = logging.getLogger(__name__)
renderer = RichRenderer(logger=logger)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("agent.log", mode='w'),
    ],)

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
    try:
        result = await handler(request)
    except Exception as exc:
        subagent = request.tool_call["args"].get("subagent_type", "?")
        human_decision = interrupt({
            "subagent": subagent,
            "error_preview": str(exc)[:500],
            "message": f"'{subagent}' raised an exception. Continue pipeline?",
        })
        return ToolMessage(
            content=f"Pipeline aborted after exception in '{subagent}': {exc}",
            tool_call_id=request.tool_call["id"],
    )

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
    subagents=[policy_agent, benchmark_agent, config_agent, interdependency_agent, search_agent],
    checkpointer=checkpointer,
    system_prompt = """
You are a Microsoft Intune security supervisor. You orchestrate specialised subagents
and produce a compliance report. Your job is to faithfully present what the subagents return —
not to compute, infer, or adjudicate compliance yourself.

Delegate tasks to subagents using the `task` tool with this format:
{
  "description": "What the subagent should do",
  "subagent_type": "policy_agent" | "config_agent" | "benchmark_agent" | "interdependency_agent" | "search_agent"
}

After each subagent returns, write the result to "[subagent_type]_result.md" using
write_file. Run agents sequentially — never in parallel. Each step must wait for the
previous one to complete before proceeding.

If a subagent returns an error, stop the pipeline and ask for human review before proceeding. Use the interrupt tool for this.

# Workflow
  Step 1: Delegate to policy_agent to extract all the requirements from the security policy.
  Step 2: Delegate to config_agent to find all security settings that match the requirements from the security policy.
  Step 3: Delegate to benchmark_agent to check whether the configurations are compliant with the CIS Benchmark.
  Step 4: Delegate to interdependency_agent to check for conflicts or interdependencies among the settings.
  Step 5: Delegate to search_agent for Microsoft recommendations on settings NOT covered by CIS.
  Step 6: Only after all subagents have completed, produce the final report as a markdownfile written to "final_result.md". 
          The report must contain the following sections in order:
---

## 1. Security Requirements Compliance

Relay the summary table from the config_agent result unchanged.

---

## 2. Benchmark Compliance

Relay the compliance summary and table from the benchmark_agent result unchanged.

---

## 3. Remediation

List each NON-COMPLIANT and NOT CONFIGURED setting with its recommended value, rationale and the CIS Benchmark data
returned by benchmark_agent. Do not generate remediation steps from your own knowledge. 

---

## 4. Interdependencies

Use the structured output from interdependency_agent:
- List all entries from structural conflicts: setting name, conflict type, severity, and reason.
- For unmet_catalog_prerequisites entries, show the dependency chain:
  <setting_name> → depends on → <missing_parent_name> (NOT CONFIGURED IN TENANT)
- If both blocks are empty, write "No conflictinginterdependencies or conflicts identified."
- List all the parent-child relationships identified, even if compliant:
  <setting_name> → depends on → <parent_name> (COMPLIANT/NOT CONFIGURED)

---

## 5. Web Search Results

List each setting that used Microsoft documentation from the search_agent (Step 5) because it was not covered
by the CIS Benchmark. For each:
- Setting ID and name
- Description
- Microsoft recommendation
- Source URL

---

## 6. Security Posture Summary

A brief paragraph summarising the overall compliance posture, key risks, and recommended
priorities. Base this only on the data presented in sections 1–5.
""",

)

query = "How well is the tenant's security configuration aligned with the security policy? Produce a compliance report with detailed analysis and remediation steps."

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


def start_agent():
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


if __name__ == "__main__":


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