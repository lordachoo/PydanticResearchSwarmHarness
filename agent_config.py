"""
Loads per-agent configuration for the swarm.

    agents/<name>.toml   limits, timeouts, and a [model_settings] table that is
                         passed straight to the model (temperature, max_tokens…)
    prompts/<name>.md    the agent's instructions; which file is named by the
                         toml's `prompt = "<name>"` key (defaults to the agent name)
    agents/swarm.toml    swarm-wide settings (max_workers, subquestion bounds,
                         output_retries)

load_agent(name) returns the toml as a dict with two extra keys:
    "instructions"   -> contents of the prompt file
    "model_settings" -> the [model_settings] table, or {} if absent
Every other toml key (request_limit, timeout_s, max_fetches, …) is returned
as-is, so the swarm reads them with RESEARCHER["max_fetches"] etc.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).parent
AGENTS_DIR = ROOT / "agents"
PROMPTS_DIR = ROOT / "prompts"


def load_agent(name: str) -> dict:
    path = AGENTS_DIR / f"{name}.toml"
    with path.open("rb") as f:
        cfg = tomllib.load(f)

    prompt_name = cfg.get("prompt", name)
    prompt_path = PROMPTS_DIR / f"{prompt_name}.md"
    cfg["instructions"] = prompt_path.read_text(encoding="utf-8").strip()
    cfg["model_settings"] = cfg.get("model_settings", {})
    return cfg


def load_swarm() -> dict:
    path = AGENTS_DIR / "swarm.toml"
    with path.open("rb") as f:
        return tomllib.load(f)
