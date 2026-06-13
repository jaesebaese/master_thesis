"""
activity_stream.py
==================

Stream and render *all* activity from a Deep Agents run: the coordinator
agent, every delegated subagent (recursively, including nested subagents),
and every tool call (with live output deltas and errors).

Built on the Deep Agents event-streaming API documented at
https://docs.langchain.com/oss/python/deepagents/event-streaming

The public API surface this relies on:

    run = agent.stream_events(input, version="v3")          # sync

    run.messages          -> ChatModelStream handles (coordinator)
    run.tool_calls        -> ToolCall handles (coordinator)
    run.subagents         -> Subagent handles (one per `task` delegation)
    run.output            -> final coordinator state
    run.interleave(...)   -> sync, in-arrival-order over named projections

    subagent.name / .path / .status / .task_input
    subagent.messages / .tool_calls / .subagents / .output

    message.text          -> iterable of text deltas; str(message.text) = final
    message.reasoning     -> iterable of reasoning deltas (if model emits them)
    message.node          -> graph node that produced the message

    call.tool_name / .input / .output / .error / .completed
    call.output_deltas    -> iterable of streamed output chunks

Usage
-----
    from deepagents import create_deep_agent
    from activity_stream import stream_activity

    agent = create_deep_agent(tools=[...], subagents=[...])
    stream_activity(
        agent,
        {"messages": [{"role": "user", "content": "Research X and write a report"}]},
    )

You can pass `on_event=` to receive structured ActivityEvent objects (e.g.
to drive a web UI) instead of / in addition to console rendering.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Optional


# --------------------------------------------------------------------------- #
# Structured event model
# --------------------------------------------------------------------------- #
class EventType(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    AGENT_TEXT = "agent_text"           # text delta from coordinator/subagent
    AGENT_REASONING = "agent_reasoning"  # reasoning delta
    AGENT_MESSAGE_DONE = "agent_message_done"
    SUBAGENT_START = "subagent_start"
    SUBAGENT_STATUS = "subagent_status"
    SUBAGENT_END = "subagent_end"
    TOOL_START = "tool_start"
    TOOL_OUTPUT = "tool_output"         # output delta from a tool
    TOOL_END = "tool_end"
    TOOL_ERROR = "tool_error"
    ERROR = "error"


@dataclass
class ActivityEvent:
    """A single structured activity event.

    `source` is a human-readable label for who emitted it, e.g.
    "coordinator", "research-agent", or "research-agent > summarizer" for a
    nested subagent. `path` mirrors the subagent namespace path when available.
    """
    type: EventType
    source: str
    text: str = ""
    tool_name: str = ""
    tool_input: Any = None
    status: str = ""
    path: tuple = ()
    depth: int = 0
    extra: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


OnEvent = Callable[[ActivityEvent], None]


# --------------------------------------------------------------------------- #
# Console renderer
# --------------------------------------------------------------------------- #
class _Ansi:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    # palette cycled by depth so each nesting level is visually distinct
    COLORS = ["\033[36m", "\033[35m", "\033[33m", "\033[32m", "\033[34m"]

    @classmethod
    def color(cls, depth: int) -> str:
        return cls.COLORS[depth % len(cls.COLORS)]


class ConsoleRenderer:
    """Pretty, indentation-based console renderer for ActivityEvents.

    Thread-safe: a single lock guards stdout so concurrent subagent streams
    don't garble each other. Tracks whether we're mid-line (streaming deltas)
    so headers get their own line.
    """

    def __init__(self, *, color: bool = True, stream=None) -> None:
        self.color = color and (stream or sys.stdout).isatty()
        self.out = stream or sys.stdout
        self._lock = threading.Lock()
        self._open_line_source: Optional[str] = None  # who currently owns the line

    # -- low-level helpers -------------------------------------------------- #
    def _c(self, s: str, code: str) -> str:
        if not self.color:
            return s
        return f"{code}{s}{_Ansi.RESET}"

    def _indent(self, depth: int) -> str:
        return "  " * depth

    def _break_line(self) -> None:
        if self._open_line_source is not None:
            self.out.write("\n")
            self._open_line_source = None

    def _header(self, ev: ActivityEvent, label: str, body: str = "") -> None:
        self._break_line()
        pad = self._indent(ev.depth)
        tag = self._c(f"{label}", _Ansi.color(ev.depth) + _Ansi.BOLD)
        src = self._c(ev.source, _Ansi.DIM)
        line = f"{pad}{tag} {src}"
        if body:
            line += f" {body}"
        self.out.write(line + "\n")
        self.out.flush()

    def _delta(self, ev: ActivityEvent, text: str, *, code: str = "") -> None:
        """Append a streaming delta, continuing the same logical line."""
        pad = self._indent(ev.depth + 1)
        if self._open_line_source != ev.source:
            self._break_line()
            self.out.write(pad + (self._c("│ ", _Ansi.DIM)))
            self._open_line_source = ev.source
        chunk = self._c(text, code) if code else text
        # keep multi-line deltas indented under the gutter
        chunk = chunk.replace("\n", "\n" + pad + self._c("│ ", _Ansi.DIM))
        self.out.write(chunk)
        self.out.flush()

    # -- the on_event callback --------------------------------------------- #
    def __call__(self, ev: ActivityEvent) -> None:
        with self._lock:
            self._dispatch(ev)

    def _dispatch(self, ev: ActivityEvent) -> None:
        t = ev.type
        if t is EventType.RUN_START:
            self._header(ev, "▶ run", self._c("started", _Ansi.DIM))
        elif t is EventType.RUN_END:
            self._break_line()
            self.out.write(self._c("■ run finished\n", _Ansi.BOLD))
            self.out.flush()
        elif t is EventType.SUBAGENT_START:
            body = self._c(f"[{ev.status}]", _Ansi.DIM)
            if ev.tool_input is not None:
                body += " " + self._c(_truncate(str(ev.tool_input), 80), _Ansi.DIM)
            self._header(ev, "⊕ subagent", body)
        elif t is EventType.SUBAGENT_STATUS:
            self._header(ev, "  ↳ status", self._c(ev.status, _Ansi.DIM))
        elif t is EventType.SUBAGENT_END:
            mark = "✓" if ev.status == "completed" else "✗"
            code = _Ansi.color(ev.depth) if ev.status == "completed" else "\033[31m"
            self._header(ev, f"{mark} subagent", self._c(ev.status, code))
        elif t is EventType.AGENT_TEXT:
            self._delta(ev, ev.text)
        elif t is EventType.AGENT_REASONING:
            self._delta(ev, ev.text, code=_Ansi.DIM)
        elif t is EventType.AGENT_MESSAGE_DONE:
            self._break_line()
        elif t is EventType.TOOL_START:
            arg = _truncate(_fmt(ev.tool_input), 100)
            self._header(ev, "🔧 tool", f"{self._c(ev.tool_name, _Ansi.BOLD)}({arg})")
        elif t is EventType.TOOL_OUTPUT:
            self._delta(ev, ev.text, code=_Ansi.DIM)
        elif t is EventType.TOOL_END:
            self._break_line()
            body = ""
            if ev.text:
                body = self._c("→ " + _truncate(ev.text, 100), _Ansi.DIM)
            self._header(ev, "   tool done", body)
        elif t is EventType.TOOL_ERROR:
            self._header(ev, "   tool error", self._c(ev.text, "\033[31m"))
        elif t is EventType.ERROR:
            self._break_line()
            self.out.write(self._c(f"!! error in {ev.source}: {ev.text}\n", "\033[31m"))
            self.out.flush()


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _fmt(v: Any) -> str:
    if isinstance(v, dict):
        return ", ".join(f"{k}={_truncate(str(val), 40)}" for k, val in v.items())
    return str(v)


# --------------------------------------------------------------------------- #
# Core traversal
# --------------------------------------------------------------------------- #
def _label(parent: str, name: str) -> str:
    return name if parent in ("", "coordinator") else f"{parent} > {name}"


def _drain_message(message: Any, emit: OnEvent, source: str, depth: int) -> None:
    """Render one ChatModelStream handle: reasoning deltas then text deltas."""
    # Reasoning first (only present for models that emit it). Iterating an
    # absent projection should be cheap/empty; guard anyway.
    try:
        for delta in message.reasoning:
            if delta:
                emit(ActivityEvent(EventType.AGENT_REASONING, source, text=str(delta), depth=depth))
    except (AttributeError, TypeError):
        pass

    for delta in message.text:
        if delta:
            emit(ActivityEvent(EventType.AGENT_TEXT, source, text=str(delta), depth=depth))

    emit(ActivityEvent(EventType.AGENT_MESSAGE_DONE, source, depth=depth))


def _drain_tool_call(call: Any, emit: OnEvent, source: str, depth: int) -> None:
    """Render one ToolCall handle: start, streamed output deltas, end/error."""
    emit(ActivityEvent(
        EventType.TOOL_START, source,
        tool_name=getattr(call, "tool_name", "") or "",
        tool_input=getattr(call, "input", None),
        depth=depth,
    ))

    try:
        for delta in call.output_deltas:
            if delta:
                emit(ActivityEvent(EventType.TOOL_OUTPUT, source, text=str(delta), depth=depth))
    except (AttributeError, TypeError):
        pass

    err = getattr(call, "error", None)
    if err:
        emit(ActivityEvent(EventType.TOOL_ERROR, source, text=str(err),
                           tool_name=getattr(call, "tool_name", ""), depth=depth))
    else:
        out = getattr(call, "output", None)
        emit(ActivityEvent(
            EventType.TOOL_END, source,
            tool_name=getattr(call, "tool_name", ""),
            text="" if out is None else str(out),
            depth=depth,
        ))


def _drain_subagent(sub: Any, emit: OnEvent, parent: str, depth: int) -> None:
    """Render one subagent handle, recursing into nested subagents.

    Accessing `sub.output` is what signals the delegated task finished (per the
    lifecycle pattern in the docs); an exception there means it failed.
    """
    name = getattr(sub, "name", None) or getattr(sub, "graph_name", "subagent")
    source = _label(parent, name)
    path = tuple(getattr(sub, "path", ()) or ())
    status = getattr(sub, "status", "started") or "started"

    emit(ActivityEvent(
        EventType.SUBAGENT_START, source,
        status=status, path=path, depth=depth,
        tool_input=getattr(sub, "task_input", None),
    ))

    # Mirror the coordinator pattern: use sub.interleave so all projections are
    # subscribed (buffered) and consumed in arrival order. Skip "task" tool calls
    # for the same reason as at coordinator level — draining them blocks until
    # the nested subagent finishes, closing its stream before we can read it.
    # Nested subagents are handled by recursing into _drain_subagent.
    if hasattr(sub, "interleave"):
        try:
            for item_name, item in sub.interleave("messages", "tool_calls", "subagents"):
                if item_name == "messages":
                    _drain_message(item, emit, source, depth + 1)
                elif item_name == "tool_calls":
                    tool_name = getattr(item, "tool_name", "") or ""
                    if tool_name == "task":
                        emit(ActivityEvent(EventType.TOOL_START, source,
                                           tool_name=tool_name,
                                           tool_input=getattr(item, "input", None), depth=depth + 1))
                        emit(ActivityEvent(EventType.TOOL_END, source,
                                           tool_name=tool_name, text="", depth=depth + 1))
                    else:
                        _drain_tool_call(item, emit, source, depth + 1)
                elif item_name == "subagents":
                    _drain_subagent(item, emit, source, depth + 1)
        except Exception:  # noqa: BLE001
            pass
    else:
        # Sequential fallback for handles without interleave. Iterating messages
        # first drives the stream; tool_calls should still be buffered after.
        try:
            for message in sub.messages:
                _drain_message(message, emit, source, depth + 1)
        except Exception:  # noqa: BLE001
            pass
        try:
            for call in sub.tool_calls:
                _drain_tool_call(call, emit, source, depth + 1)
        except Exception:  # noqa: BLE001
            pass
        try:
            for nested in sub.subagents:
                _drain_subagent(nested, emit, source, depth + 1)
        except Exception:  # noqa: BLE001
            pass

    # Resolve final lifecycle status via .output.
    try:
        _ = sub.output
        final = getattr(sub, "status", "completed") or "completed"
        if final in ("started", "running"):
            final = "completed"
        emit(ActivityEvent(EventType.SUBAGENT_END, source, status=final, path=path, depth=depth))
    except Exception as exc:  # noqa: BLE001 - failure of a delegated task
        emit(ActivityEvent(EventType.SUBAGENT_END, source, status="failed",
                           path=path, depth=depth, extra={"error": str(exc)}))


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def stream_activity(
    agent: Any,
    agent_input: Any,
    *,
    version: str = "v3",
    on_event: Optional[OnEvent] = None,
    render: bool = True,
    config=None,                    
    color: bool = True,
    interleave: bool = True,
) -> Any:
    """Synchronously stream and render all activity of a Deep Agents run.

    Parameters
    ----------
    agent
        A Deep Agents agent exposing `stream_events(input, version=...)`.
    agent_input
        The input dict, e.g. ``{"messages": [{"role": "user", "content": "..."}]}``.
    on_event
        Optional callback receiving structured :class:`ActivityEvent`s. Called
        in addition to the console renderer (if ``render`` is True).
    render
        When True, print a live, indented console view of all activity.
    color
        Enable ANSI colors in the console renderer (auto-disabled if stdout
        is not a TTY).
    interleave
        When True (default), drive the stream via ``run.interleave(...)`` so events
        arrive in execution order. Deep Agents projection names ("tool_calls",
        "subagents") may not be recognised as interleave keys, so a fallback
        direct-attribute pass runs after the loop to catch any that were missed.

    Returns
    -------
    The final coordinator state (``run.output``), or None if unavailable.
    """
    sinks = []
    if render:
        sinks.append(ConsoleRenderer(color=color))
    if on_event:
        sinks.append(on_event)

    def emit(ev: ActivityEvent) -> None:
        for s in sinks:
            s(ev)

    run = agent.stream_events(agent_input, version=version, config=config)
    emit(ActivityEvent(EventType.RUN_START, "coordinator"))

    try:
        if interleave and hasattr(run, "interleave"):
            _consume_interleaved(run, emit)
        else:
            _consume_sequential(run, emit)
    except Exception as exc: 
        emit(ActivityEvent(EventType.ERROR, "coordinator", text=str(exc)))
        raise
    finally:
        emit(ActivityEvent(EventType.RUN_END, "coordinator"))

    output = getattr(run, "output", None)
    return output() if callable(output) else output


def _consume_interleaved(run: Any, emit: OnEvent) -> None:
    """Consume coordinator messages, tool calls, and subagents in arrival order.

    "tool_calls" must be in the interleave so the projection is buffered.
    However, "task" tool calls must NOT be drained normally — draining blocks
    on call.output until the subagent finishes, which closes the subagent
    stream before we can read it. Skip "task" entries entirely; their activity
    is captured via the "subagents" projection. All other tool calls are drained
    normally.
    """
    saw_subs = False
    for name, item in run.interleave("messages", "tool_calls", "subagents"):
        if name == "messages":
            _drain_message(item, emit, "coordinator", depth=1)
        elif name == "tool_calls":
            tool_name = getattr(item, "tool_name", "") or ""
            if tool_name == "task":
                emit(ActivityEvent(EventType.TOOL_START, "coordinator",
                                   tool_name=tool_name,
                                   tool_input=getattr(item, "input", None), depth=1))
                emit(ActivityEvent(EventType.TOOL_END, "coordinator",
                                   tool_name=tool_name, text="", depth=1))
            else:
                _drain_tool_call(item, emit, "coordinator", depth=1)
        elif name == "subagents":
            saw_subs = True
            _drain_subagent(item, emit, "coordinator", depth=1)

    if not saw_subs:
        try:
            for sub in run.subagents:
                _drain_subagent(sub, emit, "coordinator", depth=1)
                for call in sub.tool_calls:
                    print(f"[{sub.name} tool]", call.tool_name, call.input)
                    for delta in call.output_deltas:
                        print(delta, end="", flush=True)

                    if call.completed and call.error is None:
                        print(call.output)
                    elif call.error is not None:
                        print(call.error)
        except Exception:  # noqa: BLE001
            pass


def _consume_sequential(run: Any, emit: OnEvent) -> None:
    """Drain coordinator projections, then subagents."""
    for message in run.messages:
        _drain_message(message, emit, "coordinator", depth=1)
    for call in run.tool_calls:
        _drain_tool_call(call, emit, "coordinator", depth=1)
    for sub in run.subagents:
        _drain_subagent(sub, emit, "coordinator", depth=1)


# --------------------------------------------------------------------------- #
# Convenience: lifecycle-only tracker (mirrors the docs' counter example)
# --------------------------------------------------------------------------- #
@dataclass
class LifecycleCounts:
    running: int = 0
    completed: int = 0
    failed: int = 0


def track_subagent_lifecycle(agent: Any, agent_input: Any, *, version: str = "v3",
                             verbose: bool = True) -> LifecycleCounts:
    """Lightweight tracker that only reports which subagents started/finished.

    This is the minimal pattern from the "Track subagent lifecycle" section of
    the docs, wrapped up with counts returned at the end. It does not subscribe
    to message or tool projections.
    """
    run = agent.stream_events(agent_input, version=version)
    counts = LifecycleCounts()

    for sub in run.subagents:
        name = getattr(sub, "name", None) or getattr(sub, "graph_name", "subagent")
        counts.running += 1
        if verbose:
            print(f"{name}: started")
        try:
            _ = sub.output
            counts.running -= 1
            counts.completed += 1
            if verbose:
                print(f"{name}: completed")
        except Exception:  # noqa: BLE001
            counts.running -= 1
            counts.failed += 1
            if verbose:
                print(f"{name}: failed")

    return counts


if __name__ == "__main__":
    # Minimal smoke test against a fake agent so the renderer can be exercised
    # without a live model. Run: python activity_stream.py
    from types import SimpleNamespace as NS

    class _Iter:
        def __init__(self, items): self._items = list(items)
        def __iter__(self): return iter(self._items)

    class _Msg:
        def __init__(self, text): self.text = _Iter(list(text)); self.reasoning = _Iter([]); self.node = "model"
        @property
        def output(self): return NS(usage_metadata=None)

    class _Call:
        def __init__(self, name, inp, out): 
            self.tool_name = name; self.input = inp
            self.output_deltas = _Iter([out]); self.output = out; self.error = None

    class _Sub:
        def __init__(self, name):
            self.name = name; self.path = (name,); self.status = "running"
            self.task_input = f"do work for {name}"
            self.messages = _Iter([_Msg(f"Working on {name}... ")])
            self.tool_calls = _Iter([_Call("search", {"q": name}, f"results for {name}")])
            self.subagents = _Iter([])
        @property
        def output(self): return {"done": True}

    class _Run:
        def __init__(self):
            self.messages = _Iter([_Msg("Let me delegate this. ")])
            self.tool_calls = _Iter([])
            self.subagents = _Iter([_Sub("research-agent"), _Sub("writer-agent")])
            self.output = {"messages": ["final answer"]}

    class _Agent:
        def stream_events(self, _input, version="v3"): return _Run()

    print("=== smoke test (no interleave) ===")
    stream_activity(_Agent(), {"messages": []}, interleave=False, color=True)