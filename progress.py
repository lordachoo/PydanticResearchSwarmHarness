"""
Live progress tracking for swarm runs.

run_swarm() reports into a RunTracker: stage status, the current activity
(waiting on the model, which tool it called with what argument), search/fetch
counts, model requests and tokens. Tracking is in-process and costs nothing
when nobody is watching.

Consumers:
  - swarm_web.py serves it at /progress (live page), /progress/events (SSE)
    and /progress/state (JSON).
  - run_swarm() saves the final snapshot as reports/<run>/metrics.json.

Usage per stage comes from AgentRun.usage() after every graph node (see
run_agent_tracked), so tokens and request counts update while an agent is
still working, not only when it finishes.
"""

import asyncio
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import (PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta, ThinkingPart,
                                  ThinkingPartDelta, ToolCallPart)

from endpoints import POOL

MAX_RUNS_KEPT = 20      # finished runs kept in memory for the progress page
MAX_EVENTS = 300        # log lines kept per run


@dataclass
class Stage:
    key: str                        # "planner", "R1".."Rn", "writer"
    label: str                      # what it is working on
    status: str = "pending"         # pending | running | retrying | done | stopped | failed | timeout | aborted
    activity: str = ""              # latest thing it did
    activity_since: float | None = None   # when `activity` last changed (page shows how long)
    profile: str = ""               # endpoint profile serving this stage
    finalizing: bool = False        # answering from pages already read (no more research)
    started: float | None = None
    ended: float | None = None
    requests: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    searches: int = 0
    fetches: int = 0                # pages actually read
    fetch_failures: int = 0         # fetch attempts that returned an error
    attempts: int = 0               # researcher attempts (1 = first try)
    max_searches: int | None = None
    max_fetches: int | None = None
    error: str = ""


@dataclass
class RunState:
    run_id: str
    task: str
    started: float = field(default_factory=time.time)
    ended: float | None = None
    status: str = "running"         # running | done | failed | aborted
    report_dir: str = ""
    phase: str = ""                 # planning | researching | writing | saving | "" (idle)
    job: str = ""                   # post-run job in progress (re-run, rewrite, …)
    controllable: bool = False      # this process can still act on the run (controls shown)
    report_versions: int = 0
    stages: dict[str, Stage] = field(default_factory=dict)
    events: deque = field(default_factory=lambda: deque(maxlen=MAX_EVENTS))


class Tracker:
    """Registry of runs plus a change signal for live consumers."""

    def __init__(self) -> None:
        self.runs: dict[str, RunState] = {}
        self.version = 0
        self._changed = asyncio.Event()

    def _bump(self) -> None:
        self.version += 1
        self._changed.set()

    async def wait_change(self, seen: int, timeout: float) -> int:
        """Return once version moves past `seen` (or after `timeout`)."""
        if self.version == seen:
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), timeout)
            except asyncio.TimeoutError:
                pass
        return self.version

    def start_run(self, run_id: str, task: str) -> "RunTracker":
        self.runs[run_id] = RunState(run_id=run_id, task=task)
        finished = [r for r in self.runs.values() if r.status != "running"]
        for old in sorted(finished, key=lambda r: r.started)[: max(0, len(self.runs) - MAX_RUNS_KEPT)]:
            del self.runs[old.run_id]
        self._bump()
        return RunTracker(self, self.runs[run_id])

    def restore(self, saved: dict[str, Any]) -> "RunTracker | None":
        """Re-create a finished run from its saved metrics.json (after a
        server restart), so it shows on the page and post-run controls work."""
        run_id = saved.get("run_id")
        if not run_id or run_id in self.runs:
            return None
        fields = Stage.__dataclass_fields__
        stages = {s["key"]: Stage(**{k: v for k, v in s.items() if k in fields}) for s in saved.get("stages", [])}
        status = saved.get("status") if saved.get("status") in ("done", "failed", "aborted") else "done"
        run = RunState(run_id=run_id, task=saved.get("task", ""), started=saved.get("started") or time.time(),
                       ended=saved.get("ended") or time.time(), status=status,
                       report_dir=saved.get("report_dir", ""), stages=stages,
                       report_versions=saved.get("report_versions", 0))
        run.events.extend(saved.get("events", []))
        self.runs[run_id] = run
        self._bump()
        return RunTracker(self, run)

    def snapshot(self, logs_for: int = 3, log_lines: int = 150) -> dict[str, Any]:
        """All runs, newest first. Only the newest `logs_for` runs (and any
        still running) carry their event log, trimmed to `log_lines`, so the
        payload stays small when it is pushed on every change."""
        runs = sorted(self.runs.values(), key=lambda r: r.started, reverse=True)
        out = []
        for i, r in enumerate(runs):
            with_log = i < logs_for or r.status == "running"
            out.append(run_to_dict(r, log_lines if with_log else 0))
        return {"now": time.time(), "version": self.version, "runs": out, "endpoints": POOL.snapshot()}


def run_to_dict(run: RunState, log_lines: int | None = None) -> dict[str, Any]:
    stages = [asdict(s) for s in run.stages.values()]
    totals = {k: sum(s[k] for s in stages) for k in
              ("requests", "tool_calls", "input_tokens", "output_tokens", "searches", "fetches",
               "fetch_failures")}
    return {
        "run_id": run.run_id, "task": run.task, "status": run.status,
        "started": run.started, "ended": run.ended, "report_dir": run.report_dir,
        "phase": run.phase, "job": run.job, "controllable": run.controllable,
        "report_versions": run.report_versions,
        "totals": totals, "stages": stages,
        "events": list(run.events)[-log_lines:] if log_lines else ([] if log_lines == 0 else list(run.events)),
    }


class RunTracker:
    """Handle that one swarm run reports through."""

    def __init__(self, tracker: Tracker, run: RunState) -> None:
        self._t = tracker
        self.run = run

    def log(self, msg: str, stage: str | None = None) -> None:
        self.run.events.append({"t": time.time(), "stage": stage or "", "msg": msg})
        self._t._bump()

    def add_stage(self, key: str, label: str, **limits: Any) -> Stage:
        st = Stage(key=key, label=label, **limits)
        self.run.stages[key] = st
        self._t._bump()
        return st

    def update(self, key: str, **fields: Any) -> None:
        st = self.run.stages[key]
        if "activity" in fields and fields["activity"] != st.activity:
            st.activity_since = time.time()
        for k, v in fields.items():
            setattr(st, k, v)
        self._t._bump()

    def begin(self, key: str) -> None:
        self.update(key, status="running", started=time.time(), ended=None, error="", activity="starting")

    def end(self, key: str, status: str, error: str = "") -> None:
        self.update(key, status=status, ended=time.time(), error=error[:500],
                    activity="" if status == "done" else self.run.stages[key].activity)

    def finish(self, status: str, report_dir: str = "") -> None:
        self.run.status = status
        self.run.ended = time.time()
        self.run.report_dir = report_dir or self.run.report_dir
        self.run.phase = ""
        self._t._bump()

    def set_phase(self, phase: str, job: str | None = None) -> None:
        self.run.phase = phase
        if job is not None:
            self.run.job = job
        self._t._bump()

    def reopen(self, job: str) -> None:
        """A finished run is running again (post-run job)."""
        self.run.status = "running"
        self.run.ended = None
        self.run.job = job
        self._t._bump()

    def to_dict(self) -> dict[str, Any]:
        return run_to_dict(self.run)


TRACKER = Tracker()


def _short(value: Any, n: int = 140) -> str:
    s = str(value).replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


async def run_agent_tracked(agent: Agent, prompt: str, rt: RunTracker, key: str,
                            transcript: list | None = None, stream: bool = False, **run_kwargs: Any):
    """agent.run() equivalent that streams usage and activity into the tracker.

    Iterates the agent graph node by node: before each model request the stage
    shows "waiting for model"; after each model response it shows the tool
    calls the model made (or that it is answering), and usage is refreshed.

    Usage is added to whatever the stage already has, so a retried agent's
    counters accumulate across attempts. If `transcript` is given it is kept
    equal to the run's messages after every step, so a caller can still read
    the conversation when the run fails or is cancelled by a timeout.

    stream=True streams each model request and shows how much the model has
    produced so far ("thinking… 4,210 chars", "writing… 12,800 chars"), so a
    single long request (the writer's) isn't a silent multi-minute wait. It
    needs the endpoint to support streaming; agents without tools are the
    safe place for it.
    """
    st = rt.run.stages[key]
    base = (st.requests, st.tool_calls, st.input_tokens, st.output_tokens)
    async with agent.iter(prompt, **run_kwargs) as run:
        async for node in run:
            if Agent.is_model_request_node(node) and stream:
                rt.update(key, activity="waiting for model")
                since = rt.run.stages[key].activity_since
                chars = {"thinking": 0, "writing": 0}
                last = 0.0
                async with node.stream(run.ctx) as events:
                    async for ev in events:
                        kind, n = None, 0
                        if isinstance(ev, PartDeltaEvent):
                            d = ev.delta
                            if isinstance(d, ThinkingPartDelta) and d.content_delta:
                                kind, n = "thinking", len(d.content_delta)
                            elif isinstance(d, TextPartDelta):
                                kind, n = "writing", len(d.content_delta)
                        elif isinstance(ev, PartStartEvent):
                            if isinstance(ev.part, ThinkingPart):
                                kind, n = "thinking", len(ev.part.content or "")
                            elif isinstance(ev.part, TextPart):
                                kind, n = "writing", len(ev.part.content or "")
                        if kind:
                            chars[kind] += n
                            if time.monotonic() - last > 2:     # don't flood the page
                                last = time.monotonic()
                                rt.update(key, activity=", ".join(
                                    f"{k}… {v:,} chars" for k, v in chars.items() if v),
                                    activity_since=since)   # time the whole request, not each update
            elif Agent.is_model_request_node(node):
                rt.update(key, activity="waiting for model")
            elif Agent.is_call_tools_node(node):
                parts = node.model_response.parts
                calls = [p for p in parts if isinstance(p, ToolCallPart)]
                if calls:
                    desc = ", ".join(f"{c.tool_name}({_short(_first_arg(c), 80)})" for c in calls)
                    rt.update(key, activity=f"→ {desc}")
                elif any(isinstance(p, TextPart) and p.content.strip() for p in parts):
                    rt.update(key, activity="writing answer")
                elif any(isinstance(p, ThinkingPart) for p in parts):
                    rt.update(key, activity="thinking")
            u = run.usage   # a property in pydantic-ai 2.x
            rt.update(key, requests=base[0] + u.requests, tool_calls=base[1] + u.tool_calls,
                      input_tokens=base[2] + u.input_tokens, output_tokens=base[3] + u.output_tokens)
            if transcript is not None:
                msgs = run.all_messages()
                # A model-request node's request (carrying the latest tool
                # results) only enters the history once the model answers. If
                # the run is interrupted during that call, those results must
                # not be lost from the transcript: they are usually the newest
                # pages the agent read.
                pending = getattr(node, "request", None) if Agent.is_model_request_node(node) else None
                if pending is not None and not any(m is pending for m in msgs):
                    msgs = [*msgs, pending]
                transcript[:] = msgs
        return run.result


def _first_arg(call: ToolCallPart) -> Any:
    try:
        args = call.args_as_dict()
    except Exception:
        return call.args
    return next(iter(args.values()), "") if len(args) == 1 else args
