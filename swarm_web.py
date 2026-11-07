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

from pydantic_ai import Agent
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from agent_config import load_agent
from progress import TRACKER

# Reuse everything from the CLI version: model, agents, tools, run_swarm().
from swarm_pydantic_ai import model, run_swarm

# Prompt in prompts/orchestrator.md, model settings in agents/orchestrator.toml.
ORCHESTRATOR = load_agent("orchestrator")

orchestrator = Agent(
    model,
    instructions=ORCHESTRATOR["instructions"],
    model_settings=ORCHESTRATOR["model_settings"],
)


@orchestrator.tool_plain
async def run_research(task: str) -> str:
    """Run the full research swarm (planner -> parallel researchers -> writer)
    on the given task and return the final markdown report."""
    return await run_swarm(task)


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


# The ASGI app uvicorn serves. Routes: / (chat UI), /api/chat, /api/configure,
# /api/health, and the progress routes, which go first because the chat app
# has a catch-all /{id} route.
app = orchestrator.to_web()
app.router.routes[0:0] = [
    Route("/progress", progress_page),
    Route("/progress/state", progress_state),
    Route("/progress/events", progress_events),
]
