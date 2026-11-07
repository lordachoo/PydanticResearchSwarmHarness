"""
Web chat front end for the swarm in swarm_pydantic_ai.py.

Runs the same planner -> researchers -> writer pipeline, but behind a browser
chat UI instead of the command line. The swarm is exposed as a single tool on
an "orchestrator" agent, because Agent.to_web() serves one agent.

Run (from the same directory as swarm_pydantic_ai.py, venv active, env vars set):

    ./start_swarm                # or: uvicorn swarm_web:app --host 0.0.0.0 --port 8000

Then open http://<this-box-ip>:8000 from your browser.

to_web() is documented at https://ai.pydantic.dev/web/ — if the UI shows a
model picker with cloud models in it, that's the UI's default list; the
orchestrator below is bound to the local model regardless. Check the docs if
you want to change what the picker offers.
"""

import asyncio
import json
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route

from agent_config import load_agent
from progress import TRACKER

# Reuse everything from the CLI version: model, agents, tools, run_swarm().
from swarm_pydantic_ai import REPORTS_DIR, RUNS, ControlError, RunStopped, ServiceError, SwarmRun, model, run_swarm

# Prompt in prompts/orchestrator.md, model settings in agents/orchestrator.toml.
ORCHESTRATOR = load_agent("orchestrator")

orchestrator = Agent(
    model,
    instructions=ORCHESTRATOR["instructions"],
    model_settings=ORCHESTRATOR["model_settings"],
)


def _first_research_call_this_turn(ctx: RunContext) -> bool:
    """True if this is the first run_research call since the user's latest
    message. A swarm run costs up to hours and hundreds of thousands of tokens;
    the model must not re-run it on its own because a report has gaps."""
    calls: list[str] = []
    for msg in ctx.messages:
        if isinstance(msg, ModelRequest) and any(isinstance(p, UserPromptPart) for p in msg.parts):
            calls = []                              # a new user turn resets the count
        elif isinstance(msg, ModelResponse):
            calls += [p.tool_call_id for p in msg.parts
                      if isinstance(p, ToolCallPart) and p.tool_name == "run_research"]
    return not calls or calls[0] == ctx.tool_call_id


@orchestrator.tool
async def run_research(ctx: RunContext, task: str) -> str:
    """Run the full research swarm (planner -> parallel researchers -> writer)
    on the given task and return the final markdown report. Call it at most
    once per user message."""
    if not _first_research_call_this_turn(ctx):
        return (
            "NOT RUN: research already ran for this message. Do not call run_research "
            "again. Return the report you already have, exactly as it was returned. If "
            "it has gaps, the user can ask a follow-up question."
        )
    try:
        return await run_swarm(task)
    except RunStopped as e:
        return (
            f"RESEARCH STOPPED: {e}\n\nTell the user the research was stopped from the "
            "progress page and where the partial results are. Do not call run_research again."
        )
    except ServiceError as e:
        return (
            f"RESEARCH ABORTED before doing any useful work: {e}\n\n"
            "Tell the user exactly this, and that it must be fixed before research can "
            "run (`python doctor.py` checks every service). Do not call run_research again."
        )
    except Exception as e:  # anything unexpected: report it, don't let the model improvise a retry
        return (
            f"RESEARCH FAILED: {type(e).__name__}: {e}\n\nTell the user exactly this. "
            "Do not call run_research again."
        )


# --------------------------------------------------------------------------- #
# Live progress page
# --------------------------------------------------------------------------- #
# The chat UI only shows the run_research tool call as one long spinner, so the
# swarm's internals are served on a side page: /progress (HTML), pushed over
# Server-Sent Events from /progress/events, with /progress/state as JSON.

PROGRESS_HTML = Path(__file__).parent / "web" / "progress.html"


async def progress_page(request: Request) -> FileResponse:
    return FileResponse(PROGRESS_HTML, headers={"Cache-Control": "no-cache"})


async def progress_state(request: Request) -> JSONResponse:
    return JSONResponse(TRACKER.snapshot(logs_for=1000, log_lines=10_000))


async def progress_events(request: Request) -> StreamingResponse:
    async def stream():
        seen = -1
        while not await request.is_disconnected():
            version = await TRACKER.wait_change(seen, timeout=15)
            if version == seen:
                yield ": keepalive\n\n"          # holds proxies and the browser open
                continue
            seen = version
            yield f"data: {json.dumps(TRACKER.snapshot())}\n\n"
            await asyncio.sleep(0.5)               # coalesce bursts of updates

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------------- #
# Run controls (buttons on /progress)
# --------------------------------------------------------------------------- #
# POST /progress/api/{run_id}/{action}, JSON body:
#   stage    {"key": "R2", "cmd": "stop" | "finalize" | "restart"}   during a run
#   wrapup   {}        finalize running researchers, skip queued, write report
#   abort    {}        stop the run (or post-run job); finished work is saved
#   rerun    {"key": "R2", "subquestion": "..." (optional edit)}      after a run
#   add      {"subquestion": "..."}                                    after a run
#   rewrite  {}        run the writer again                            after a run
# Requests must carry X-Swarm-Control: 1 and a JSON body. Browsers can't send
# that cross-origin without a CORS preflight, which this app doesn't answer,
# so another website can't drive these buttons from a visitor's browser.

async def progress_control(request: Request) -> JSONResponse:
    if request.headers.get("x-swarm-control") != "1" or \
            not request.headers.get("content-type", "").startswith("application/json"):
        return JSONResponse({"error": "missing X-Swarm-Control header or JSON body"}, status_code=403)
    run_id, action = request.path_params["run_id"], request.path_params["action"]
    sr = RUNS.get(run_id) or (_restore(run_id) if _valid_run_id(run_id) else None)
    if sr is None:
        return JSONResponse({"error": "run not found (no saved metrics.json + findings.json for it)"}, status_code=404)
    try:
        body = await request.json()
    except Exception:
        body = {}
    body = body if isinstance(body, dict) else {}
    try:
        if action == "stage":
            msg = sr.control_stage(str(body.get("key", "")), str(body.get("cmd", "")))
        elif action == "wrapup":
            msg = sr.wrap_up()
        elif action == "abort":
            msg = sr.abort()
        elif action in ("rerun", "add", "rewrite"):
            msg = sr.start_job(action, body.get("key"), body.get("subquestion"))
        else:
            return JSONResponse({"error": f"unknown action {action!r}"}, status_code=404)
    except ControlError as e:
        return JSONResponse({"error": str(e)}, status_code=e.status)
    return JSONResponse({"ok": True, "message": msg})


def _valid_run_id(run_id: str) -> bool:
    return bool(run_id) and run_id.replace("-", "").isdigit()


def _restore(run_id: str):
    return SwarmRun.restore(REPORTS_DIR / run_id)


def restore_recent_runs(n: int = 10) -> int:
    """After a (re)start, put the last n finished runs back on /progress with
    their post-run controls (re-run, edit, add, rewrite) working."""
    if not REPORTS_DIR.is_dir():
        return 0
    dirs = sorted((d for d in REPORTS_DIR.iterdir() if d.is_dir() and _valid_run_id(d.name)), key=lambda d: d.name)
    return sum(1 for d in dirs[-n:] if _restore(d.name))


async def progress_report(request: Request) -> PlainTextResponse:
    """A run's saved report (or ?v=N for an earlier version), as Markdown text."""
    run_id = request.path_params["run_id"]
    v = request.query_params.get("v", "")
    if not _valid_run_id(run_id) or (v and not v.isdigit()):
        return PlainTextResponse("bad request", status_code=400)
    path = REPORTS_DIR / run_id / (f"report.v{v}.md" if v else "report.md")
    if not path.is_file():
        return PlainTextResponse("no report saved for this run yet", status_code=404)
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/plain; charset=utf-8")


# The ASGI app uvicorn serves. Routes: / (chat UI), /api/chat, /api/configure,
# /api/health, and the progress routes, which go first because the chat app
# has a catch-all /{id} route.
app = orchestrator.to_web()
restore_recent_runs()
app.router.routes[0:0] = [
    Route("/progress", progress_page),
    Route("/progress/state", progress_state),
    Route("/progress/events", progress_events),
    Route("/progress/api/{run_id}/{action}", progress_control, methods=["POST"]),
    Route("/progress/report/{run_id}", progress_report),
]
