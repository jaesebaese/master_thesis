"""
rich_renderer.py
================

A pretty, Rich-based renderer for the ActivityEvent stream produced by
`activity_stream.stream_activity(..., on_event=...)`.

It is a drop-in `on_event` callback: it does not modify activity_stream.py.
Streaming deltas (agent text, reasoning, tool output) are buffered per source
and flushed into Rich panels when the message / tool / subagent completes, so
you get clean panels instead of token-by-token noise — while still showing
live progress via a status line.

Usage
-----
    from activity_stream import stream_activity
    from rich_renderer import RichRenderer

    renderer = RichRenderer()           # logs to console; optionally logger=...
    final_state = stream_activity(
        agent,
        pending,
        config=run_config,
        render=False,                   # turn OFF the built-in plain renderer
        on_event=renderer,              # use this one instead
    )

If you also want the structured events logged elsewhere, you can chain
callbacks — see `fan_out` at the bottom.
"""

from __future__ import annotations

import json
import logging
from io import StringIO
from typing import Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from activity_stream import ActivityEvent, EventType


# --------------------------------------------------------------------------- #
# Tool-result summarization (ported from your old utils.py)
# --------------------------------------------------------------------------- #
def summarize_tool_content(raw: str, *, limit: int = 800) -> str:
    """Turn a raw tool-result string into a compact, readable summary.

    JSON lists become "[N items]" with a few sampled rows; JSON dicts show the
    first handful of keys; anything else is truncated plain text.
    """
    raw = raw or ""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw[:500] + ("..." if len(raw) > 500 else "")

    if isinstance(data, list):
        total = len(data)
        lines = [f"[{total} items]"]
        for item in data[:5]:
            if isinstance(item, dict):
                status = item.get("status", "")
                cis_id = item.get("cis_id", "")
                title = str(item.get("cis_title", item.get("name", item)))[:80]
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
        return preview[:limit] + suffix

    return str(data)[:500]


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #
# Border colors cycle by nesting depth so each subagent level is distinct.
_DEPTH_COLORS = ["cyan", "magenta", "yellow", "green", "blue"]


def _depth_color(depth: int) -> str:
    return _DEPTH_COLORS[depth % len(_DEPTH_COLORS)]


class RichRenderer:
    """Stateful `on_event` callback that renders ActivityEvents as Rich panels.

    Parameters
    ----------
    console
        A `rich.console.Console`. One is created if omitted.
    logger
        Optional stdlib logger; if given, a plain-text version of every panel
        is also written to it (mirrors your old `_log_rich` behavior, so your
        agent.log file stays populated).
    show_reasoning
        Include model reasoning blocks in agent panels (dimmed).
    indent
        Spaces of left padding per nesting depth, applied via panel padding.
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        logger: Optional[logging.Logger] = None,
        *,
        show_reasoning: bool = True,
        indent: int = 2,
        full_content: bool = False,     # when True, never truncate
        tool_limit: int = 800,          # tool result body cap
        args_limit: int = 300,          # tool args cap
        task_limit: int = 160,          # subagent task description cap
    ) -> None:
        self.console = console or Console()
        self.logger = logger
        self.show_reasoning = show_reasoning
        self.indent = indent
        # Per-source buffers keyed by source label.
        self._text: dict[str, list[str]] = {}
        self._reasoning: dict[str, list[str]] = {}
        self._tool_out: dict[str, list[str]] = {}
        self._tool_meta: dict[str, dict] = {}   # source -> current tool call meta
        self._depth: dict[str, int] = {}        # source -> last seen depth
        self.full_content = full_content
        self.tool_limit = None if full_content else tool_limit
        self.args_limit = None if full_content else args_limit
        self.task_limit = None if full_content else task_limit

    # -- logging mirror ----------------------------------------------------- #
    def _log_panel(self, panel: Panel) -> None:
        if not self.logger:
            return
        buf = StringIO()
        Console(file=buf, highlight=False, no_color=True, width=120,
                force_terminal=False).print(panel)
        self.logger.info(buf.getvalue().rstrip())

    def _emit_panel(self, panel: Panel, depth: int) -> None:
        # Indent the whole panel by depth using padding on a wrapping group.
        pad = " " * (self.indent * depth)
        if pad:
            # Prepend padding to each rendered line via Padding-like wrap.
            self.console.print(Text(pad, end=""), panel)
        else:
            self.console.print(panel)
        self._log_panel(panel)

    # -- main callback ------------------------------------------------------ #
    def __call__(self, ev: ActivityEvent) -> None:
        self._depth[ev.source] = ev.depth
        t = ev.type

        if t is EventType.RUN_START:
            self.console.rule("[bold]agent run started")
            if self.logger:
                self.logger.info("=== agent run started ===")

        elif t is EventType.RUN_END:
            self._flush_text(ev.source, ev.depth)  # flush any trailing buffer
            self.console.rule("[bold]agent run finished")
            if self.logger:
                self.logger.info("=== agent run finished ===")

        elif t is EventType.AGENT_TEXT:
            self._text.setdefault(ev.source, []).append(ev.text)

        elif t is EventType.AGENT_REASONING:
            if self.show_reasoning:
                self._reasoning.setdefault(ev.source, []).append(ev.text)

        elif t is EventType.AGENT_MESSAGE_DONE:
            self._flush_text(ev.source, ev.depth)

        elif t is EventType.SUBAGENT_START:
            task = ""
            if ev.tool_input:
                task = _truncate(str(ev.tool_input), self.task_limit)
            body = Text()
            body.append("status: ", style="dim")
            body.append(str(ev.status))
            if task:
                body.append("\ntask: ", style="dim")
                body.append(task)
            panel = Panel(
                body,
                title=f"⊕ subagent · {ev.source}",
                border_style=_depth_color(ev.depth),
                title_align="left",
            )
            self._emit_panel(panel, ev.depth)

        elif t is EventType.SUBAGENT_STATUS:
            # lightweight inline note, no panel
            self.console.print(
                Text(f"   ↳ {ev.source}: {ev.status}", style="dim"))

        elif t is EventType.SUBAGENT_END:
            ok = ev.status == "completed"
            mark = "✓" if ok else "✗"
            style = _depth_color(ev.depth) if ok else "bold red"
            note = Text(f"{mark} {ev.source}: {ev.status}", style=style)
            if not ok and ev.extra.get("error"):
                note.append(f"\n  {_truncate(ev.extra['error'], 200)}", style="red")
            self.console.print(note)
            if self.logger:
                self.logger.info("%s %s: %s", mark, ev.source, ev.status)

        elif t is EventType.TOOL_START:
            self._tool_out[ev.source] = []
            self._tool_meta[ev.source] = {
                "name": ev.tool_name, "input": ev.tool_input, "depth": ev.depth,
            }

        elif t is EventType.TOOL_OUTPUT:
            self._tool_out.setdefault(ev.source, []).append(ev.text)

        elif t is EventType.TOOL_END:
            self._flush_tool(ev.source, ev.depth, final_output=ev.text,
                             tool_name=ev.tool_name)

        elif t is EventType.TOOL_ERROR:
            meta = self._tool_meta.pop(ev.source, {})
            self._tool_out.pop(ev.source, None)
            name = ev.tool_name or meta.get("name", "tool")
            panel = Panel(
                Text(_truncate(ev.text, 1000), style="red"),
                title=f"🔧 tool error · {name} ({ev.source})",
                border_style="red",
                title_align="left",
            )
            self._emit_panel(panel, ev.depth)

        elif t is EventType.ERROR:
            panel = Panel(
                Text(_truncate(ev.text, 1000), style="red"),
                title=f"!! error · {ev.source}",
                border_style="bold red",
                title_align="left",
            )
            self._emit_panel(panel, ev.depth)

    # -- flush helpers ------------------------------------------------------ #
    def _flush_text(self, source: str, depth: int) -> None:
        text = "".join(self._text.pop(source, []))
        reasoning = "".join(self._reasoning.pop(source, []))
        if not text and not reasoning:
            return

        renderables = []
        if reasoning:
            renderables.append(Text(reasoning.strip(), style="dim italic"))
        if text:
            renderables.append(Text(text.strip()))
        body = Group(*renderables) if len(renderables) > 1 else renderables[0]

        is_coord = source == "coordinator"
        title = "🤖 coordinator" if is_coord else f"🤖 {source}"
        border = "green" if is_coord else _depth_color(depth)
        panel = Panel(body, title=title, border_style=border, title_align="left")
        self._emit_panel(panel, depth)

    def _flush_tool(self, source: str, depth: int, *, final_output: str,
                    tool_name: str) -> None:
        meta = self._tool_meta.pop(source, {})
        deltas = "".join(self._tool_out.pop(source, []))
        name = tool_name or meta.get("name", "tool")
        raw = final_output or deltas

        parts = []
        inp = meta.get("input")
        if inp not in (None, "", {}):
            parts.append(Text(f"args: {_truncate(_fmt_args(inp), self.args_limit)}", style="dim"))
        parts.append(Text(summarize_tool_content(raw, limit=self.tool_limit)))
        body = Group(*parts) if len(parts) > 1 else parts[0]

        panel = Panel(
            body,
            title=f"🔧 tool result · {name}" + ("" if source == "coordinator" else f" ({source})"),
            border_style="yellow",
            title_align="left",
        )
        self._emit_panel(panel, depth)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _truncate(s: str, n) -> str:
    s = str(s)
    if n is None or len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _fmt_args(v) -> str:
    if isinstance(v, dict):
        try:
            return json.dumps(v, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(v)
    return str(v)


def fan_out(*callbacks):
    """Combine several on_event callbacks into one.

    Example:
        on_event = fan_out(RichRenderer(logger=logger), my_json_logger)
    """
    def _dispatch(ev: ActivityEvent) -> None:
        for cb in callbacks:
            cb(ev)
    return _dispatch


if __name__ == "__main__":
    # Exercise the renderer with the same fake agent used in activity_stream.
    from activity_stream import stream_activity
    from types import SimpleNamespace as NS

    class _Iter:
        def __init__(self, items): self._items = list(items)
        def __iter__(self): return iter(self._items)

    class _Msg:
        def __init__(self, text):
            self.text = _Iter(list(text)); self.reasoning = _Iter([]); self.node = "model"
        @property
        def output(self): return NS(usage_metadata=None)

    class _Call:
        def __init__(self, name, inp, out):
            self.tool_name = name; self.input = inp
            self.output_deltas = _Iter([]); self.output = out; self.error = None

    class _Sub:
        def __init__(self, name):
            self.name = name; self.path = (name,); self.status = "running"
            self.task_input = f"Extract requirements for {name}"
            self.messages = _Iter([_Msg(f"Analyzing for {name}. Here is my finding. ")])
            self.tool_calls = _Iter([
                _Call("search", {"q": name}, json.dumps(
                    [{"cis_id": "1.1", "cis_title": "Enforce password history", "status": "pass"},
                     {"cis_id": "1.2", "cis_title": "Minimum password length", "status": "fail"}]))
            ])
            self.subagents = _Iter([])
        @property
        def output(self): return {"done": True}

    class _Run:
        def __init__(self):
            self.messages = _Iter([_Msg("Delegating to subagents now. ")])
            self.tool_calls = _Iter([])
            self.subagents = _Iter([_Sub("policy_agent"), _Sub("config_agent")])
            self.output = {"messages": ["final answer"]}

    class _Agent:
        def stream_events(self, _i, version="v3", config=None): return _Run()

    logging.basicConfig(level=logging.INFO, handlers=[logging.NullHandler()])
    renderer = RichRenderer(logger=logging.getLogger("demo"))
    stream_activity(_Agent(), {"messages": []}, render=False,
                    on_event=renderer, interleave=False)