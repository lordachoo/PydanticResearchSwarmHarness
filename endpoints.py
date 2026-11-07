"""
LLM endpoint profiles: which model each agent uses, and how many requests each
endpoint takes at once.

Profiles live in endpoints.toml (gitignored; copy endpoints.example.toml).
Without that file there is one profile, "default", built from LLM_BASE_URL /
LLM_MODEL / LLM_API_KEY, so a plain `source sourceMe` setup works unchanged.

    [profiles.<name>]
    provider        "openai" (any OpenAI-compatible server: vLLM, llama.cpp,
                    LM Studio, Ollama, strata, OpenRouter, …) or "anthropic"
    base_url        OpenAI-compatible base URL ending in /v1 (openai only)
    model           model id as the endpoint names it
    api_key         key, or "${ENV_VAR}" to read it from the environment
    max_concurrency requests this endpoint is given at once (default 4)
    output_mode     prompted | native | tool (default: the agent toml's)
    merge_system_messages  one system message per request (default true;
                    Qwen-style chat templates reject a second one)
    json_object_output     let prompted mode send response_format json_object
                    (default false; some servers refuse it alongside tools)

    [agents]
    orchestrator = "<profile>"            # web chat agent (read at server start)
    planner      = "<profile>"
    researcher   = ["<profile>", …]       # a pool: each researcher takes the
                                          # least-busy profile with a free slot
    writer       = "<profile>"

Any value may use ${VAR} or ${VAR:-default}. The file is re-read at the start
of every run, so edits apply to the next run without a restart (except the
orchestrator's, which needs one).
"""

from __future__ import annotations

import asyncio
import os
import re
import tomllib
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

ROOT = Path(__file__).parent
def endpoints_file() -> Path:
    """endpoints.toml next to this file, or $SWARM_ENDPOINTS."""
    return Path(os.environ.get("SWARM_ENDPOINTS") or ROOT / "endpoints.toml")
ROLES = ("orchestrator", "planner", "researcher", "writer")

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _VAR.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


class EndpointConfigError(ValueError):
    pass


@dataclass
class Profile:
    name: str
    provider: str = "openai"
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    max_concurrency: int = 4
    output_mode: str | None = None
    merge_system_messages: bool = True
    json_object_output: bool = False

    def describe(self) -> str:
        where = self.base_url if self.provider == "openai" else self.provider
        return f"{self.name}: {self.model} @ {where}"

    def masked(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        k = d.get("api_key") or ""
        d["api_key"] = "(unset)" if not k else "local" if k == "local" else f"…{k[-4:]}" if len(k) > 8 else "(set)"
        return d


@dataclass
class Endpoints:
    profiles: dict[str, Profile]
    agents: dict[str, list[str]]         # role -> profile names (pool; 1 for non-researchers)
    source: str                          # where this came from, for display

    def for_role(self, role: str) -> list[Profile]:
        return [self.profiles[n] for n in self.agents[role]]


def _default_profile() -> Profile:
    return Profile(
        name="default",
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1"),
        model=os.environ.get("LLM_MODEL", "qwen3.5-122b"),
        api_key=os.environ.get("LLM_API_KEY") or "local",
    )


def load(path: Path | None = None) -> Endpoints:
    """Read endpoints.toml (or the env fallback) and validate it."""
    path = path or endpoints_file()
    if not path.is_file():
        p = _default_profile()
        return Endpoints({"default": p}, {r: ["default"] for r in ROLES},
                         source="environment (LLM_BASE_URL / LLM_MODEL); no endpoints.toml")
    try:
        raw = _expand(tomllib.loads(path.read_text(encoding="utf-8")))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise EndpointConfigError(f"{path}: {e}") from e
    profiles: dict[str, Profile] = {}
    for name, cfg in (raw.get("profiles") or {}).items():
        unknown = set(cfg) - set(Profile.__dataclass_fields__)
        if unknown:
            raise EndpointConfigError(f"{path}: profile {name!r} has unknown keys {sorted(unknown)}")
        p = Profile(name=name, **cfg)
        if p.provider not in ("openai", "anthropic"):
            raise EndpointConfigError(f"{path}: profile {name!r}: provider must be 'openai' or 'anthropic'")
        if not p.model:
            raise EndpointConfigError(f"{path}: profile {name!r} has no model")
        if p.provider == "openai" and not p.base_url.rstrip("/").endswith("/v1"):
            raise EndpointConfigError(f"{path}: profile {name!r}: base_url must end in /v1 (got {p.base_url!r})")
        if p.provider == "anthropic" and not p.api_key:
            raise EndpointConfigError(f"{path}: profile {name!r}: api_key is empty (is its ${{VAR}} set?)")
        if p.output_mode and p.output_mode not in ("prompted", "native", "tool"):
            raise EndpointConfigError(f"{path}: profile {name!r}: output_mode must be prompted|native|tool")
        if p.max_concurrency < 1:
            raise EndpointConfigError(f"{path}: profile {name!r}: max_concurrency must be >= 1")
        p.api_key = p.api_key or "local"
        profiles[name] = p
    if not profiles:
        raise EndpointConfigError(f"{path}: no [profiles.<name>] defined")
    first = next(iter(profiles))
    agents: dict[str, list[str]] = {}
    for role in ROLES:
        v = (raw.get("agents") or {}).get(role, first)
        names = v if isinstance(v, list) else [v]
        if not names or any(n not in profiles for n in names):
            raise EndpointConfigError(f"{path}: [agents] {role} = {v!r}: unknown profile (have {sorted(profiles)})")
        if role != "researcher" and len(names) != 1:
            raise EndpointConfigError(f"{path}: [agents] {role} takes one profile (only researcher takes a list)")
        agents[role] = names
    return Endpoints(profiles, agents, source=str(path))


# --------------------------------------------------------------------------- #
# Model objects
# --------------------------------------------------------------------------- #

_models: dict[tuple, Any] = {}


def build_model(p: Profile, http_client=None):
    """A pydantic-ai model for the profile, cached by its settings."""
    key = (p.provider, p.base_url, p.model, p.api_key, p.merge_system_messages, p.json_object_output)
    if key in _models:
        return _models[key]
    if p.provider == "anthropic":
        try:
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider
        except ImportError as e:
            raise EndpointConfigError(f"profile {p.name!r} needs the Anthropic SDK: pip install anthropic ({e})")
        model = AnthropicModel(p.model, provider=AnthropicProvider(api_key=p.api_key))
    else:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.profiles import merge_profile
        from pydantic_ai.profiles.openai import OpenAIModelProfile
        from pydantic_ai.providers.openai import OpenAIProvider
        provider = OpenAIProvider(base_url=p.base_url, api_key=p.api_key, http_client=http_client)
        # Local-server fixes (both default on; see README "Structured output modes"):
        #   json_object_output=False  prompted mode never sends response_format,
        #                             which some servers refuse together with tools
        #   merge_system_messages     instructions go out as ONE system message;
        #                             Qwen chat templates reject a second one
        model = OpenAIChatModel(p.model, provider=provider, profile=merge_profile(
            provider.model_profile(p.model),
            OpenAIModelProfile(
                supports_json_object_output=p.json_object_output,
                openai_chat_supports_multiple_system_messages=not p.merge_system_messages,
            ),
        ))
    _models[key] = model
    return model


# --------------------------------------------------------------------------- #
# Concurrency: per-endpoint slots, least-busy choice from a pool
# --------------------------------------------------------------------------- #

@dataclass
class _Slots:
    in_use: int = 0
    limit: int = 4


class EndpointPool:
    """Tracks requests in flight per profile so no endpoint gets more than its
    max_concurrency, and spreads a role's pool by picking the least-busy
    profile (by in_use/limit) that has a free slot. Shared by every run and
    post-run job in the process."""

    def __init__(self) -> None:
        self.slots: dict[str, _Slots] = {}
        self._cond: asyncio.Condition | None = None
        self._loop = None

    def _condition(self) -> asyncio.Condition:
        loop = asyncio.get_running_loop()
        if self._cond is None or self._loop is not loop:
            self._cond, self._loop = asyncio.Condition(), loop
        return self._cond

    @asynccontextmanager
    async def slot(self, pool: list[Profile]) -> AsyncIterator[Profile]:
        for p in pool:
            s = self.slots.setdefault(p.name, _Slots())
            s.limit = p.max_concurrency   # follow edits to endpoints.toml
        cond = self._condition()
        async with cond:
            def free() -> list[Profile]:
                return [p for p in pool if self.slots[p.name].in_use < self.slots[p.name].limit]
            await cond.wait_for(lambda: bool(free()))
            chosen = min(free(), key=lambda p: self.slots[p.name].in_use / self.slots[p.name].limit)
            self.slots[chosen.name].in_use += 1
        try:
            yield chosen
        finally:
            async with cond:
                self.slots[chosen.name].in_use -= 1
                cond.notify_all()

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {n: {"in_use": s.in_use, "limit": s.limit} for n, s in self.slots.items()}


POOL = EndpointPool()
