"""State management for deep agents with TODO tracking and virtual file systems.

from https://github.com/langchain-ai/deep-agents-from-scratch/blob/main/notebooks/utils.py

This module defines the extended agent state structure that supports:
- Task planning and progress tracking through TODO lists
- Context offloading through a virtual file system stored in state
- Efficient state merging with reducer functions
"""
"""Utility functions for displaying messages and prompts in Jupyter notebooks."""

import json
import logging
from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)

console = Console()


def _log_rich(renderable):
    """Extract plain text from a Rich Panel for the log file."""
    if isinstance(renderable, Panel):
        title = Text.from_markup(renderable.title).plain if renderable.title else ""
        content = renderable.renderable
        content_plain = (
            Text.from_markup(content).plain if isinstance(content, str) else str(content)
        )
        return f"[{title}]\n{content_plain}"
    buf = StringIO()
    Console(file=buf, highlight=False, no_color=True, width=120, force_terminal=False).print(renderable)
    return buf.getvalue().strip()

def stream_agent_v2(agent, query, config=None, on_interrupt=None):
    """Stream a deep agent run with subagent labeling, v2 format.

    Args:
        on_interrupt: Optional callable(interrupt_values) -> Command.
                      Called when the graph pauses at an interrupt; should
                      return a Command(resume=...) to continue.
    """
    active_subagents = {}  # tool_call_id -> {type, status}
    final_state = None
    pending = query

    while True:
        interrupted = False

        for chunk in agent.stream(
            pending,
            stream_mode=["updates", "values"],
            subgraphs=True,
            version="v2",
            config=config,
        ):
            ns = chunk["ns"]
            is_subagent = any(s.startswith("tools:") for s in ns)
            source = "main" if not is_subagent else next(
                s for s in ns if s.startswith("tools:")
            )

            if chunk["type"] == "values":
                final_state = chunk["data"]
                continue

            if chunk["type"] != "updates":
                continue

            if "__interrupt__" in chunk["data"]:
                interrupted = True
                if on_interrupt:
                    pending = on_interrupt(chunk["data"]["__interrupt__"])
                break

            for node_name, data in chunk["data"].items():
                # Track subagent lifecycle from the main agent's perspective
                if not is_subagent and node_name == "model_request":
                    for msg in (data or {}).get("messages", []):
                        for tc in getattr(msg, "tool_calls", []) or []:
                            if tc["name"] == "task":
                                sub_type = tc["args"].get("subagent_type", "unknown")
                                active_subagents[tc["id"]] = {
                                    "type": sub_type,
                                    "status": "pending",
                                }
                                console.print(
                                    f"[bold cyan]→ Delegating to {sub_type}[/]: "
                                    f"{tc['args'].get('description', '')[:120]}"
                                )
                                logger.info(
                                    "→ Delegating to %s: %s",
                                    sub_type,
                                    tc['args'].get('description', '')[:120],
                                )

                if not is_subagent and node_name == "tools":
                    for msg in (data or {}).get("messages", []):
                        if getattr(msg, "type", None) == "tool":
                            sub = active_subagents.get(getattr(msg, "tool_call_id", None))
                            if sub:
                                sub["status"] = "complete"
                                console.print(
                                    f"[bold green]✓ {sub['type']} returned[/]"
                                )
                                logger.info("✓ %s returned", sub['type'])

                # Print messages (full length) labeled by source
                if isinstance(data, dict):
                    for key, value in data.items():
                        if "messages" in key and isinstance(value, list) and value:
                            label = source if is_subagent else "main"
                            console.print(f"[dim]── {label} / {node_name} ──[/]")
                            logger.info("── %s / %s ──", label, node_name)
                            format_messages(value)

        if not interrupted:
            break

    return final_state


def format_message_content(message):
    """Convert message content to displayable string."""
    parts = []
    tool_calls_processed = False

    # Handle main content
    if isinstance(message.content, str):
        parts.append(message.content)
    elif isinstance(message.content, list):
        # Handle complex content like tool calls (Anthropic format)
        for item in message.content:
            if item.get("type") == "text":
                parts.append(item["text"])
            elif item.get("type") == "tool_use":
                parts.append(f"🔧 Tool Call: {item['name']}")
                parts.append(f"   Args: {json.dumps(item['input'], indent=2, ensure_ascii=False)}")
                parts.append(f"   ID: {item.get('id', 'N/A')}")
                tool_calls_processed = True
    else:
        parts.append(str(message.content))

    # Handle tool calls attached to the message (OpenAI format) - only if not already processed
    if (
        not tool_calls_processed
        and hasattr(message, "tool_calls")
        and message.tool_calls
    ):
        for tool_call in message.tool_calls:
            parts.append(f"🔧 Tool Call: {tool_call['name']}")
            parts.append(f"   Args: {json.dumps(tool_call['args'], indent=2, ensure_ascii=False)}")
            parts.append(f"   ID: {tool_call['id']}")

    return "\n".join(parts)


def _summarize_tool_content(raw: str) -> str:
    """Turn a raw tool result string into a readable summary."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Not JSON — truncate plain text
        return raw[:500] + ("..." if len(raw) > 500 else "")

    if isinstance(data, list):
        total = len(data)
        lines = [f"[{total} items]"]
        for item in data[:5]:
            if isinstance(item, dict):
                status = item.get("status", "")
                cis_id = item.get("cis_id", "")
                title = item.get("cis_title", item.get("name", str(item)))[:80]
                lines.append(f"  [{cis_id}] {title}  → {status}")
            else:
                lines.append(f"  {str(item)[:100]}")
        if total > 5:
            lines.append(f"  ... and {total - 5} more")
        return "\n".join(lines)

    if isinstance(data, dict):
        keys = list(data.keys())
        preview = json.dumps({k: data[k] for k in keys[:6]}, indent=2, ensure_ascii=False)
        suffix = f"\n  ... ({len(keys) - 6} more keys)" if len(keys) > 6 else ""
        return preview[:800] + suffix

    return str(data)[:500]


def format_messages(messages):
    """Format and display a list of messages with Rich formatting."""
    for m in messages:
        msg_type = m.__class__.__name__.replace("Message", "")

        if msg_type == "Tool":
            raw = m.content if isinstance(m.content, str) else str(m.content)
            tool_name = getattr(m, "name", "tool")
            summary = _summarize_tool_content(raw)
            panel = Panel(summary, title=f"🔧 Tool Result: {tool_name}", border_style="yellow")
            console.print(panel)
            logger.info(_log_rich(panel))
        else:
            content = format_message_content(m)
            if msg_type == "Human":
                panel = Panel(content, title="🧑 Human", border_style="blue")
            elif msg_type == "Ai":
                panel = Panel(content, title="🤖 Assistant", border_style="green")
            else:
                panel = Panel(content, title=f"📝 {msg_type}", border_style="white")
            console.print(panel)
            logger.info(_log_rich(panel))


