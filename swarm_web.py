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

from pydantic_ai import Agent

from agent_config import load_agent

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


# The ASGI app uvicorn serves. Routes: / (chat UI), /api/chat, /api/configure, /api/health
app = orchestrator.to_web()
