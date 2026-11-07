"""
Minimal agent swarm on Pydantic AI against a local OpenAI-compatible server
(llama-swap -> vLLM on DGX Spark, or llama.cpp).

Topology:
    planner  -> splits a task into N independent sub-questions (structured output)
    N x researcher (fan-out, run concurrently) -> each has a fetch_page tool (Crawl4AI /md)
    writer   -> synthesizes the researchers' findings

Requirements:
    pip install pydantic-ai httpx

Structured output defaults to PromptedOutput: the JSON schema is put in the
instructions and the reply text is parsed and validated. Nothing beyond plain
chat + tool calling is required of the server, so it runs on vLLM, llama.cpp,
SGLang, Ollama, LM Studio and stricter proxies alike. See output_spec() for the
other modes and why each one breaks somewhere.

Written against pydantic-ai 2.60; if an import fails, check
https://ai.pydantic.dev for the current class names.

Agent prompts live in prompts/<name>.md; per-agent limits, timeouts and model
settings live in agents/<name>.toml; swarm-wide settings in agents/swarm.toml
(see agent_config.py). Which LLM endpoint each agent uses lives in
endpoints.toml (see endpoints.py); without it, one endpoint from LLM_* env.

Env:
    LLM_BASE_URL       e.g. http://<llm-host>:8080/v1   (must end in /v1)
    LLM_MODEL          model id as the server reports it in /v1/models
    LLM_API_KEY        only if the server requires one (default "local")
    LLM_OUTPUT_MODE    prompted | native | tool; overrides output_mode in every toml
    CRAWL4AI_URL       e.g. http://<services-host>:11235
    CRAWL4AI_API_TOKEN bearer token set on the container
    SEARXNG_URL        e.g. http://<services-host>:8888  (JSON API must be enabled)
    DOCLING_URL        e.g. http://<services-host>:5001  (docling-serve; PDFs go here)
    OTEL_EXPORTER_OTLP_ENDPOINT  e.g. http://localhost:6006 (Phoenix); enables tracing
    LLM_DEBUG=1        dump raw HTTP request/response bodies to stderr
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import re

import httpx
from pydantic import BaseModel, Field
import time

from pydantic_ai import Agent, ModelRetry, NativeOutput, PromptedOutput, RunContext, ToolOutput
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelRequest, ModelResponse, RetryPromptPart, TextPart, ToolCallPart

try:  # pydantic-ai 2.60 sends validation feedback as RetryFeedbackPart
    from pydantic_ai.messages import RetryFeedbackPart
    _RETRY_PARTS: tuple = (RetryPromptPart, RetryFeedbackPart)
except ImportError:
    _RETRY_PARTS = (RetryPromptPart,)
from pydantic_ai.usage import UsageLimits

from agent_config import load_agent, load_swarm
from endpoints import POOL, EndpointConfigError, Profile, build_model, load as load_endpoints
from progress import TRACKER, RunTracker, run_agent_tracked
from search import CACHE_PATH, SearchCache, SharedSearch, Throttle, normalize
from tracing import setup_tracing, swarm_span

setup_tracing()   # no-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
# LLM_DEBUG=1 prints every request/response body the OpenAI client sends and
# receives. That's the only reliable way to see what the model actually
# returned when pydantic-ai rejects an output. Off by default: it's very loud.

if os.environ.get("LLM_DEBUG") == "1":
    logging.basicConfig(
        level=logging.DEBUG,
        stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logging.getLogger("httpcore").setLevel(logging.DEBUG)
    logging.getLogger("openai").setLevel(logging.DEBUG)
    logging.getLogger("pydantic_ai").setLevel(logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

log = logging.getLogger("swarm")

# --------------------------------------------------------------------------- #
# Request body capture (LLM_DEBUG=1)
# --------------------------------------------------------------------------- #
# The openai client does not log request bodies. This hook prints the role
# order of every /chat/completions request plus the (truncated) body, so a
# server-side 400 like "System message must be at the beginning" can be read
# directly instead of inferred.

async def _dump_request(request: httpx.Request) -> None:
    if not request.url.path.endswith("/chat/completions"):
        return
    try:
        body = json.loads(request.content)
    except Exception:
        return
    roles = [m.get("role") for m in body.get("messages", [])]
    log.debug(
        "LLM request roles=%s tools=%s tool_choice=%s response_format=%s stream=%s",
        roles,
        bool(body.get("tools")),
        body.get("tool_choice"),
        body.get("response_format"),
        body.get("stream"),
    )
    log.debug("LLM request body=%s", json.dumps(body)[:6000])


_http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(600.0, connect=5.0),
    event_hooks={"request": [_dump_request]} if os.environ.get("LLM_DEBUG") == "1" else {},
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

CONFIG = {
    "crawl4ai_url": os.environ.get("CRAWL4AI_URL", "http://localhost:11235"),
    "crawl4ai_token": os.environ.get("CRAWL4AI_API_TOKEN", ""),
    "searxng_url": os.environ.get("SEARXNG_URL", "http://localhost:8888"),
    "docling_url": os.environ.get("DOCLING_URL", "http://localhost:5001"),
}

# Per-agent tunables (prompts, limits, temperature) come from agents/*.toml
PLANNER = load_agent("planner")
RESEARCHER = load_agent("researcher")
WRITER = load_agent("writer")
SWARM = load_swarm()

# How many times pydantic-ai re-prompts the model when its structured output
# fails validation before giving up. Kept at 1 while debugging so a hard
# server error fails fast instead of re-sending the same broken request;
# raise to 3 in agents/swarm.toml once the endpoint is behaving.
OUTPUT_RETRIES = SWARM.get("output_retries", 1)

# LLM endpoints. endpoints.toml is validated here, at startup, so a typo fails
# loudly; it is re-read at the start of every run so edits apply without a
# restart. The orchestrator (web chat) model is fixed at startup.
ENDPOINTS = load_endpoints()


def model_for(profile: Profile):
    return build_model(profile, _http_client)


model = model_for(ENDPOINTS.for_role("orchestrator")[0])


# --------------------------------------------------------------------------- #
# Shared dependencies (injected into tools via RunContext)
# --------------------------------------------------------------------------- #

@dataclass
class Deps:
    http: httpx.AsyncClient
    key: str = "researcher"             # stage key in the progress tracker ("R1", …)
    rt: RunTracker | None = None        # progress tracker for this run
    fetches: int = 0                    # fetch attempts (counts against max_fetches)
    searches: int = 0                   # searches this researcher has made
    seen: set[str] = field(default_factory=set)   # URLs attempted
    read: set[str] = field(default_factory=set)   # URLs that returned content
    fetch_failures: int = 0
    finalizing: bool = False            # answer-only mode: tools refuse, 1 page read suffices


FINALIZING_REFUSAL = ("FINALIZING: research is over for this sub-question. Do not call any more "
                      "tools. Write your final answer now, in the required JSON format, from the "
                      "pages you already read.")


class ServiceError(RuntimeError):
    """A support service is down or misconfigured (wrong token, unreachable).
    Raised out of a tool to abort the whole run immediately: no researcher can
    do useful work, and continuing only burns tokens."""


def _note(deps: Deps, msg: str) -> None:
    """Log a tool event to the terminal and the progress page, and refresh the
    stage's search/fetch counters."""
    print(f"[{deps.key}] {msg}")
    if deps.rt:
        deps.rt.update(deps.key, searches=deps.searches, fetches=len(deps.read),
                       fetch_failures=deps.fetch_failures)
        deps.rt.log(msg, deps.key)


_EMPTY_SEARCHES: dict[str, int] = {}   # run id -> consecutive empty searches with failing engines


def _engine_trouble(data: dict) -> str:
    """SearXNG's own report of upstream engines that failed for this query
    (rate-limited, suspended, timed out), as "brave: too many requests, …"."""
    bad = data.get("unresponsive_engines") or []
    return ", ".join(f"{e[0]}: {e[1]}" if isinstance(e, (list, tuple)) and len(e) > 1 else str(e) for e in bad)


def _fetch_failed(deps: Deps, msg: str) -> str:
    deps.fetch_failures += 1
    _note(deps, msg)
    return msg


# --------------------------------------------------------------------------- #
# Structured outputs
# --------------------------------------------------------------------------- #

class Plan(BaseModel):
    """What the planner must return. Forcing structure keeps small models honest."""
    key_terms: list[str] = Field(
        description=(
            "REQUIRED. Every proper noun, company, product, technology name and "
            "distinctive technical phrase from the user's task, copied VERBATIM "
            "(e.g. company names, product names, phrases in quotes). Include terms "
            "you do not recognize; never replace them with related concepts you "
            "know better. At least 3 entries."
        ),
        min_length=3,
        max_length=20,
    )
    subquestions: list[str] = Field(
        description=(
            "Independent sub-questions that together answer the task. Each must "
            "preserve the user's exact terminology and named entities."
        ),
        min_length=SWARM.get("min_subquestions", 1),
        max_length=SWARM.get("max_subquestions", 4),
    )


class Finding(BaseModel):
    subquestion: str
    answer: str
    sources: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Instruction context shared by every agent
# --------------------------------------------------------------------------- #
# The model's training data is older than today and it does not know that. Every
# agent gets the current date and a rule that fetched sources beat memory.
# The context is a function, so a long-running web server picks up the new date
# each day. pydantic-ai would send each instruction part as its own system
# message; the model profile above merges them into one, because Qwen's chat
# template rejects a system message that isn't first.

def _date_context() -> str:
    today = datetime.now().strftime("%A, %B %d, %Y")
    return (
        f"Today's date is {today}. Your training data ends well before this date, "
        "so your memory of products, companies, prices, and market conditions is "
        "stale. Never assume a name, product, or company does not exist because "
        "you have not heard of it. When you search, do not append past years to "
        "queries; search for current information. Content retrieved from the web "
        "is authoritative over your own recollection."
    )


def planner_context() -> str:
    return _date_context() + (
        "\n\nPlanning rules: sub-questions must preserve the user's exact words for "
        "every named entity, product, company, and technical concept. Do not "
        "substitute a concept you are more familiar with for the one the user "
        "named. If the task mentions a term you do not recognize, include a "
        "sub-question dedicated to establishing what it is. Fill key_terms with "
        "every such term verbatim."
    )


def researcher_context() -> str:
    return _date_context() + (
        "\n\nResearch rules: begin by searching for the key terms exactly as the "
        "user wrote them, before searching for anything else. Answer only the "
        "sub-question you were given, using pages you actually fetched. PDFs are "
        "fully supported by fetch_page; prefer primary sources (reports, filings, "
        "vendor documents, standards bodies) over summaries of them."
    )


def writer_context() -> str:
    return _date_context() + (
        "\n\nWriting rules: use the user's terminology, not your own synonyms. "
        "Where findings failed or lack verified sources, say so explicitly rather "
        "than filling the gap from memory."
    )

# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #
# Structured-output mode, per agent (`output_mode` in agents/<name>.toml), or
# for every agent at once with LLM_OUTPUT_MODE:
#   prompted -> (default) schema goes in the instructions, the reply text is
#               parsed as JSON (markdown fences stripped) and validated; a bad
#               reply is sent back for another try. Requires nothing of the
#               server beyond chat + tool calling.
#   native   -> response_format=json_schema (constrained decoding). Guaranteed
#               valid JSON, but needs server support, and some servers (e.g.
#               strata) refuse it together with tools:
#               400 "structured response_format with tools/MCP is not supported"
#   tool     -> pydantic-ai's final_result tool call. On vLLM + Qwen3.5 the
#               forced tool_choice makes xgrammar demand JSON while the model
#               emits XML tool calls, and the qwen3_xml parser drops the call
#               ("grammar_matcher.cc ... trying to accept new token").

_OUTPUT_MODES = {"native": NativeOutput, "prompted": PromptedOutput, "tool": ToolOutput}


def output_spec(cfg: dict, schema: type[BaseModel], profile: Profile | None = None):
    """Precedence: LLM_OUTPUT_MODE env > the endpoint profile's output_mode >
    the agent toml's output_mode > prompted."""
    mode = (os.environ.get("LLM_OUTPUT_MODE") or (profile.output_mode if profile else None)
            or cfg.get("output_mode") or "prompted").strip().lower()
    if mode not in _OUTPUT_MODES:
        raise ValueError(
            f"output mode must be one of {sorted(_OUTPUT_MODES)}, got {mode!r} "
            "(from LLM_OUTPUT_MODE, endpoints.toml or output_mode in agents/*.toml)"
        )
    return _OUTPUT_MODES[mode](schema)

# No model at construction: every run passes the model of the endpoint it was
# given (see SwarmRun._call).
planner = Agent(
    output_type=output_spec(PLANNER, Plan),
    retries=OUTPUT_RETRIES,
    instructions=[planner_context, PLANNER["instructions"]],
    model_settings=PLANNER["model_settings"],
)

# The researcher is built after its tools and validator are defined (below).

writer = Agent(
    instructions=[writer_context, WRITER["instructions"]],
    model_settings=WRITER["model_settings"],
)



# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

_SEARCH_URL = re.compile(r"/search[/?]|[?&]q=|[?&]query=|duckduckgo\.com|bing\.com|google\.com/search", re.I)




def _is_pdf(url: str) -> bool:
    return url.split("?", 1)[0].lower().endswith(".pdf")


async def _fetch_pdf_via_docling(http: httpx.AsyncClient, url: str) -> str:
    """Convert a PDF (or anything docling can read) to markdown via docling-serve.
    Sync endpoint; OCR on a long scanned document can take minutes, hence the
    long timeout. Pages are capped by researcher.toml pdf_page_limit."""
    page_limit = RESEARCHER.get("pdf_page_limit", 30)
    payload = {
        "sources": [{"kind": "http", "url": url}],
        "options": {
            "to_formats": ["md"],
            "do_ocr": True,
            "page_range": [1, page_limit],
            "include_images": False,
            "image_export_mode": "placeholder",
        },
    }
    r = await http.post(
        f"{CONFIG['docling_url']}/v1/convert/source",
        json=payload,
        timeout=900,
    )
    if r.status_code != 200:
        return f"PDF FETCH FAILED for {url}: HTTP {r.status_code}: {r.text[:500]}"
    data = r.json()
    doc = data.get("document") or {}
    md = doc.get("md_content")
    if not md:
        # Field name differs on this docling-serve version, or conversion failed.
        keys = sorted(doc.keys()) if isinstance(doc, dict) else type(doc).__name__
        errs = data.get("errors") or []
        return (
            f"PDF FETCH FAILED for {url}: no md_content in response. "
            f"status={data.get('status')} document_keys={keys} errors={errs[:3]}"
        )
    return md

def service_client() -> httpx.AsyncClient:
    """HTTP client for SearXNG / Crawl4AI / docling, one per run or job.
    (A function so tests can substitute a mock transport.)"""
    return httpx.AsyncClient()


# Shared by every researcher and run in this process (see search.py).
SEARCH = SharedSearch(
    SearchCache(CACHE_PATH, ttl_s=SWARM.get("search_cache_hours", 24) * 3600),
    Throttle(SWARM.get("search_min_interval_s", 1.5)),
)


async def _searxng(http: httpx.AsyncClient, query: str, categories: str) -> dict:
    """One SearXNG query. Raises ServiceError when search can't work at all;
    other failures come back as {"error": ...} for the model to work around."""
    try:
        r = await http.get(f"{CONFIG['searxng_url']}/search",
                           params={"q": query, "format": "json", "categories": categories}, timeout=30)
    except httpx.ConnectError as e:
        raise ServiceError(f"SearXNG is unreachable at {CONFIG['searxng_url']}: {e}")
    except httpx.HTTPError as e:
        return {"error": str(e) or type(e).__name__}
    if r.status_code == 403:
        raise ServiceError(f"SearXNG refused the JSON API (HTTP 403) at {CONFIG['searxng_url']}: "
                           "add `json` to search.formats in its settings.yml")
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    data = r.json()
    # Keep only what's used, so cache rows stay small.
    data["results"] = [{"title": x.get("title"), "url": x.get("url"), "content": x.get("content")}
                       for x in data.get("results", [])[:8]]
    return data


async def web_search(ctx: RunContext[Deps], query: str) -> str:
    """Search the web via SearXNG. Returns up to 8 results as title, URL and
    snippet. Use this first to find pages, then fetch_page the best ones."""
    deps = ctx.deps
    if deps.finalizing:
        return FINALIZING_REFUSAL
    max_searches = RESEARCHER.get("max_searches", 3)
    if deps.searches >= max_searches:
        return (
            "SEARCH BUDGET EXHAUSTED. Do not call web_search again. "
            "Fetch the best URLs you already have, or answer with what you have."
        )
    deps.searches += 1
    categories = SWARM.get("searxng_categories", "general")
    data, how = await SEARCH.search(f"{categories}|{normalize(query)}",
                                    lambda: _searxng(deps.http, query, categories))
    _note(deps, f"search{'' if how == 'network' else f' ({how})'}: {query}")
    if "error" in data:
        msg = f"SEARCH FAILED for {query!r}: {data['error']}"
        _note(deps, msg)
        return msg
    results = data.get("results", [])[:8]
    if not results:
        broken = _engine_trouble(data)
        if broken and deps.rt:
            # Empty because the upstream engines are failing, not because of
            # the query. A few in a row means search is down for this run.
            n = _EMPTY_SEARCHES[deps.rt.run.run_id] = _EMPTY_SEARCHES.get(deps.rt.run.run_id, 0) + 1
            _note(deps, f"NO RESULTS for {query!r}; SearXNG engines failing: {broken}")
            if n >= SWARM.get("searxng_dead_after", 3):
                raise ServiceError(f"SearXNG returns no results; its upstream engines are failing ({broken}). "
                                   "Wait for the rate limits to clear, or enable other engines in settings.yml")
        return f"NO RESULTS for {query!r}. Try different wording."
    if deps.rt:
        _EMPTY_SEARCHES[deps.rt.run.run_id] = 0
    lines = []
    for i, res in enumerate(results, 1):
        title = (res.get("title") or "").strip()
        url = res.get("url") or ""
        snippet = (res.get("content") or "").strip().replace("\n", " ")[:300]
        lines.append(f"{i}. {title}\n   {url}\n   {snippet}")
    return "\n".join(lines)

async def fetch_page(ctx: RunContext[Deps], url: str) -> str:
    """Fetch a web page and return its markdown. HTML goes through Crawl4AI;
    PDF URLs are converted (with OCR) through docling-serve, so PDFs are fine
    to fetch — reports, bulletins, datasheets, papers."""
    deps = ctx.deps
    if deps.finalizing:
        return FINALIZING_REFUSAL

    if deps.fetches >= RESEARCHER["max_fetches"]:
        return (
            "FETCH BUDGET EXHAUSTED. Do not call fetch_page again. "
            "Answer now using what you already have."
        )
    if url in deps.seen:
        return f"ALREADY FETCHED {url}. Do not fetch it again. Answer with what you have."
    if _SEARCH_URL.search(url):
        return (
            f"REFUSED {url}: search-result pages are not allowed. "
            "Fetch a specific article or page URL instead, or answer with what you have."
        )
    deps.seen.add(url)
    deps.fetches += 1

    if _is_pdf(url):
        _note(deps, f"fetch (pdf via docling): {url}")
        try:
            text = await _fetch_pdf_via_docling(deps.http, url)
        except httpx.HTTPError as e:
            text = f"PDF FETCH FAILED for {url}: {e}"   # docling only matters for PDFs; not fatal
        if text.startswith("PDF FETCH FAILED"):
            return _fetch_failed(deps, text[:300])
        deps.read.add(url)
        _note(deps, f"read {len(text):,} chars (pdf): {url}")
        return text[: RESEARCHER.get("pdf_char_limit", RESEARCHER["page_char_limit"] * 4)]

    headers = {"Content-Type": "application/json"}
    if CONFIG["crawl4ai_token"]:
        headers["Authorization"] = f"Bearer {CONFIG['crawl4ai_token']}"

    _note(deps, f"fetch: {url}")
    try:
        r = await ctx.deps.http.post(
            f"{CONFIG['crawl4ai_url']}/md",
            json={"url": url},
            headers=headers,
            timeout=90,
        )
    except httpx.ConnectError as e:
        raise ServiceError(f"Crawl4AI is unreachable at {CONFIG['crawl4ai_url']}: {e}")
    except httpx.HTTPError as e:
        return _fetch_failed(deps, f"FETCH FAILED for {url}: {e}")

    if r.status_code in (401, 403):
        # Crawl4AI's own auth, not the target site (site failures come back as
        # 500/502 with the browser error). Every fetch will fail the same way.
        raise ServiceError(
            f"Crawl4AI rejected CRAWL4AI_API_TOKEN (HTTP {r.status_code}) at "
            f"{CONFIG['crawl4ai_url']}: the token in sourceMe doesn't match the container's"
        )
    if r.status_code != 200:
        # Return the server's explanation to the model instead of crashing the
        # run; it can choose another URL or answer without this page.
        return _fetch_failed(deps, f"FETCH FAILED for {url}: HTTP {r.status_code}: {r.text[:300]}")

    data = r.json()
    text = data.get("markdown") or ""
    if not text.strip():
        return _fetch_failed(deps, f"FETCH FAILED for {url}: page had no readable text")
    deps.read.add(url)
    _note(deps, f"read {len(text):,} chars: {url}")
    return text[: RESEARCHER["page_char_limit"]]



# --------------------------------------------------------------------------- #
# Researcher output enforcement
# --------------------------------------------------------------------------- #
# The model will happily answer from memory and cite URLs it never opened.
# Reject any Finding that isn't backed by pages this researcher actually
# fetched; pydantic-ai sends the message back and the model retries.

async def require_fetched_sources(ctx: RunContext[Deps], finding: Finding) -> Finding:
    """Only pages that actually returned content count. Failed fetches don't:
    a 401 or a 502 is not a source.

    Unread URLs in `sources` are dropped rather than rejected, as long as at
    least one real source remains. Rejecting the whole answer over one bad
    citation sends the model into retry loops; dropping it keeps the answer
    and still guarantees every listed source was read."""
    deps = ctx.deps
    # When finalizing (user asked to finish, or recovering a failed attempt),
    # whatever was read is what there is: one page suffices.
    min_fetches = 1 if deps.finalizing else RESEARCHER.get("min_fetches", 1)
    if len(deps.read) < min_fetches:
        raise ModelRetry(
            f"You have read {len(deps.read)} pages successfully ({deps.fetch_failures} "
            f"fetches failed); at least {min_fetches} are required. Call web_search, "
            "then fetch_page the most relevant results (PDFs are fine), and base your "
            "answer on what those pages actually say."
        )
    kept = [u for u in finding.sources if u in deps.read]
    dropped = [u for u in finding.sources if u not in deps.read]
    if not kept:
        raise ModelRetry(
            "None of your sources are pages you successfully fetched"
            + (f" (not read: {', '.join(dropped[:5])})" if dropped else "")
            + ". List only URLs that fetch_page returned content for: "
            + ", ".join(sorted(deps.read)[:10])
        )
    if dropped and deps.rt:
        deps.rt.log(f"dropped {len(dropped)} unread source(s): {', '.join(dropped[:3])}", deps.key)
    return finding.model_copy(update={"sources": kept})

# --------------------------------------------------------------------------- #
# Researcher agents
# --------------------------------------------------------------------------- #
# pydantic-ai won't take a per-run output_type on an agent with an output
# validator, so there is one researcher agent per output mode (same tools,
# validator and instructions). `researcher` is the one for the default mode;
# an endpoint profile with its own output_mode gets the matching variant.

def _make_researcher(mode: str) -> Agent:
    agent = Agent(
        deps_type=Deps,
        output_type=_OUTPUT_MODES[mode](Finding),
        retries=OUTPUT_RETRIES,
        instructions=[researcher_context, RESEARCHER["instructions"]],
        model_settings=RESEARCHER["model_settings"],
        tools=[web_search, fetch_page],
    )
    agent.output_validator(require_fetched_sources)
    return agent


def _mode_for(cfg: dict, profile: "Profile | None") -> str:
    return type(output_spec(cfg, Finding, profile)).__name__.removesuffix("Output").lower()


researcher = _make_researcher(_mode_for(RESEARCHER, None))
_RESEARCHERS: dict[str, Agent] = {_mode_for(RESEARCHER, None): researcher}


def researcher_for(profile: Profile) -> Agent:
    mode = _mode_for(RESEARCHER, profile)
    if mode not in _RESEARCHERS:
        _RESEARCHERS[mode] = _make_researcher(mode)
    return _RESEARCHERS[mode]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def _new_run_id() -> str:
    """Timestamp id, also the report folder name; suffixed if one is taken."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id, n = stamp, 1
    while run_id in TRACKER.runs or (REPORTS_DIR / run_id).exists():
        n += 1
        run_id = f"{stamp}-{n}"
    return run_id


REPORTS_DIR = Path(__file__).parent / "reports"


def _reason(e: BaseException) -> str:
    return "timed out" if isinstance(e, asyncio.TimeoutError) else (str(e) or type(e).__name__)


_preflight_ok_until = 0.0


async def preflight(http: httpx.AsyncClient) -> None:
    """Check SearXNG and Crawl4AI before spending any tokens. A wrong token or a
    dead service otherwise shows up as an hour of failed fetches. A pass is
    cached for 5 minutes. docling is not checked: only PDFs need it."""
    global _preflight_ok_until
    if not SWARM.get("preflight", True) or time.time() < _preflight_ok_until:
        return
    try:
        r = await http.get(f"{CONFIG['searxng_url']}/search", params={"q": "wikipedia", "format": "json"}, timeout=30)
    except httpx.HTTPError as e:
        raise ServiceError(f"SearXNG is unreachable at {CONFIG['searxng_url']}: {e}")
    if r.status_code != 200:
        raise ServiceError(f"SearXNG returned HTTP {r.status_code} at {CONFIG['searxng_url']}"
                           + (" (enable `json` in search.formats)" if r.status_code == 403 else ""))
    if not r.json().get("results"):
        broken = _engine_trouble(r.json())
        raise ServiceError(
            f"SearXNG at {CONFIG['searxng_url']} returns no results even for a generic query"
            + (f"; its upstream engines are failing ({broken})" if broken else "")
            + ". Wait for rate limits to clear, or enable other engines in settings.yml")
    headers = {"Content-Type": "application/json"}
    if CONFIG["crawl4ai_token"]:
        headers["Authorization"] = f"Bearer {CONFIG['crawl4ai_token']}"
    try:
        r = await http.post(f"{CONFIG['crawl4ai_url']}/md", json={"url": "https://example.com"},
                            headers=headers, timeout=60)
    except httpx.HTTPError as e:
        raise ServiceError(f"Crawl4AI is unreachable at {CONFIG['crawl4ai_url']}: {e}")
    if r.status_code in (401, 403):
        raise ServiceError(f"Crawl4AI rejected CRAWL4AI_API_TOKEN (HTTP {r.status_code}) at "
                           f"{CONFIG['crawl4ai_url']}: the token in sourceMe doesn't match the container's")
    if r.status_code != 200:
        raise ServiceError(f"Crawl4AI returned HTTP {r.status_code} at {CONFIG['crawl4ai_url']}: {r.text[:200]}")
    _preflight_ok_until = time.time() + 300


def _service_error(e: BaseException) -> "ServiceError | None":
    """Find a ServiceError in e, its causes/contexts, or an exception group.
    Iterative with a visited set: exception chains can contain cycles."""
    stack, seen = [e], set()
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, ServiceError):
            return cur
        if isinstance(cur, BaseExceptionGroup):
            stack.extend(cur.exceptions)
        stack.extend((cur.__cause__, cur.__context__))
    return None


def _rejection_reason(messages: list) -> str:
    """The validator's (or parser's) last complaint, from a failed transcript."""
    for msg in reversed(messages):
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, _RETRY_PARTS):
                    c = part.content
                    text = c if isinstance(c, str) else "; ".join(
                        str(x.get("msg", x)) if isinstance(x, dict) else str(x) for x in c)
                    return " ".join(text.split())[:300]
    return ""


def _resumable(messages: list) -> list:
    """Trim a failed transcript so it can be sent again as message history:
    it must not end in a model response with tool calls that never got their
    results (the run was cut off mid-call)."""
    msgs = list(messages)
    while msgs and isinstance(msgs[-1], ModelResponse) and any(
        isinstance(p, ToolCallPart) for p in msgs[-1].parts
    ):
        msgs.pop()
    return msgs


FINALIZE_PROMPT = (
    "{lead} Stop researching. Using only the pages you successfully fetched, write "
    "your final answer now in the required JSON format. List as sources only URLs "
    "that fetch_page returned content for. If the pages don't fully answer the "
    "question, say what they do establish and what remains open, instead of guessing."
)


class RunStopped(Exception):
    """The user aborted the run from the progress page."""


class ControlError(Exception):
    """A control request that can't be applied (unknown run/stage, wrong state)."""

    def __init__(self, msg: str, status: int = 409) -> None:
        super().__init__(msg)
        self.status = status


@dataclass
class UserCmd:
    cmd: str        # stop | finalize | restart


_SEM: tuple | None = None


def _worker_sem() -> asyncio.Semaphore:
    """One researcher semaphore for the whole process, so concurrent runs and
    post-run jobs together never exceed max_workers (the server's slots)."""
    global _SEM
    loop = asyncio.get_running_loop()
    if _SEM is None or _SEM[0] is not loop:
        _SEM = (loop, asyncio.Semaphore(SWARM.get("max_workers", 4)))
    return _SEM[1]


RUNS: dict[str, "SwarmRun"] = {}
"""Runs this process can still control, by run id (pruned with the tracker)."""


class SwarmRun:
    """One research run: planner -> researchers -> writer, plus the state the
    progress page's controls act on.

    Every stage runs in its own asyncio task (stage_tasks). A control sets a
    command for the stage (stage_cmd) and cancels its task; the stage catches
    that cancellation in _cancellable(), sees the command and acts on it
    (finalize, restart, stop) instead of dying. Cancellations without a
    command (abort, shutdown) propagate normally."""

    def __init__(self, task: str, restored: tuple[str, RunTracker] | None = None) -> None:
        self.task = task
        if restored:
            self.run_id, self.rt = restored
        else:
            self.run_id = _new_run_id()
            self.rt = TRACKER.start_run(self.run_id, task)
        self.rt.run.controllable = True
        self.endpoints = ENDPOINTS
        self.key_terms: list[str] = []
        self.questions: dict[str, str] = {}       # "R1" -> sub-question
        self.findings: dict[str, Finding] = {}    # "R1" -> finding
        self.report = ""
        self.stage_tasks: dict[str, asyncio.Task] = {}
        self.stage_cmd: dict[str, str] = {}
        self.main_task: asyncio.Task | None = None  # the run, or a post-run job
        self.aborting = False
        self.http: httpx.AsyncClient | None = None
        for rid in [r for r in RUNS if r not in TRACKER.runs]:
            del RUNS[rid]
        RUNS[self.run_id] = self

    @classmethod
    def restore(cls, run_dir: Path) -> "SwarmRun | None":
        """Rebuild a finished run from reports/<run>/ (metrics.json +
        findings.json), e.g. after a server restart, so it reappears on the
        progress page and its post-run controls work."""
        try:
            saved = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            fj = json.loads((run_dir / "findings.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if saved.get("run_id") != run_dir.name or saved.get("status") == "running":
            return None
        rt = TRACKER.restore(saved)
        if rt is None:
            return RUNS.get(run_dir.name)
        sr = cls(fj.get("task", saved.get("task", "")), restored=(run_dir.name, rt))
        sr.key_terms = list(fj.get("key_terms") or [])
        for key, f in (fj.get("findings") or {}).items():
            sr.questions[key] = f.get("subquestion", "")
            if f.get("answer") is not None:
                sr.findings[key] = Finding(**f)
        for st in rt.run.stages.values():
            if st.status in ("pending", "running", "retrying"):
                st.status = "aborted"
        return sr

    # ------------------------------------------------------------------ #
    # Model calls: endpoint slot + timeout + progress
    # ------------------------------------------------------------------ #

    async def _call(self, role: str, key: str, agent: Agent, prompt: str, cfg: dict,
                    schema: type[BaseModel] | None = None, timeout: float | None = None, **kw):
        """Run `agent` on an endpoint from the role's pool. Waiting for a free
        endpoint slot doesn't count against the timeout; the run itself does."""
        pool = self.endpoints.for_role(role)
        self.rt.update(key, activity="waiting for a free endpoint")
        async with POOL.slot(pool) as prof:
            self.rt.update(key, profile=prof.name)
            extra = {}
            if role == "researcher":
                agent = researcher_for(prof)           # output mode is baked into the agent
            elif schema:
                extra["output_type"] = output_spec(cfg, schema, prof)
            return await asyncio.wait_for(
                run_agent_tracked(agent, prompt, self.rt, key, model=model_for(prof),
                                  stream=cfg.get("stream", False), **extra, **kw),
                timeout=timeout or cfg["timeout_s"])

    @property
    def run_dir(self) -> Path:
        return REPORTS_DIR / self.run_id

    def _keys(self) -> list[str]:
        return sorted(self.questions, key=lambda k: int(k[1:]))

    # ------------------------------------------------------------------ #
    # Cancellation plumbing
    # ------------------------------------------------------------------ #

    async def _cancellable(self, key: str, aw):
        """Await aw. If a user command for `key` cancelled it, swallow the
        cancellation and return UserCmd; any other cancellation propagates."""
        try:
            return await aw
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if self.aborting or task is not self.stage_tasks.get(key) or key not in self.stage_cmd:
                raise
            task.uncancel()   # we handled it; pydantic-ai must not see a pending cancel
            return UserCmd(self.stage_cmd.pop(key))

    async def _in_stage_task(self, key: str, coro):
        """Run coro as stage `key`'s own task (the unit controls cancel)."""
        t = asyncio.create_task(coro)
        self.stage_tasks[key] = t
        try:
            return await t
        finally:
            self.stage_cmd.pop(key, None)

    # ------------------------------------------------------------------ #
    # The run
    # ------------------------------------------------------------------ #

    async def run(self) -> str:
        rt = self.rt
        try:
            self.endpoints = load_endpoints()     # pick up endpoints.toml edits
            rt.log("endpoints: " + "; ".join(
                f"{role} → {', '.join(self.endpoints.agents[role])}" for role in ("planner", "researcher", "writer")))
            async with service_client() as http:
                self.http = http
                # 0. Fail fast if search or fetch can't work, before any tokens are spent.
                await preflight(http)
                rt.log("preflight ok: SearXNG and Crawl4AI reachable")
                # 1. Plan
                rt.set_phase("planning")
                plan = await self._in_stage_task("planner", self._plan())
                self.key_terms = list(plan.key_terms)
                self.questions = {f"R{i}": q for i, q in enumerate(plan.subquestions, 1)}
                for key, q in self.questions.items():
                    rt.add_stage(key, q, max_searches=RESEARCHER.get("max_searches", 3),
                                 max_fetches=RESEARCHER["max_fetches"])
                    rt.update(key, activity="queued")
                # 2. Research
                rt.set_phase("researching")
                await self._research_all(self._keys())
                # 3. Write
                rt.set_phase("writing")
                self.report = await self._in_stage_task("writer", self._write())
                # 4. Save
                rt.set_phase("saving")
                self._save("done")
                return self.report
        except asyncio.CancelledError:
            if self.aborting:
                self._abort_cleanup("stopped by the user from the progress page")
            raise
        except BaseException as e:
            se = _service_error(e)
            if se:
                print(f"[swarm] ABORTED: {se}")
                self._abort_cleanup(str(se))
                raise se from None
            rt.log(f"run failed: {_reason(e)}")
            rt.finish("failed")
            raise
        finally:
            self.http = None

    def _abort_cleanup(self, why: str) -> None:
        rt = self.rt
        rt.log(f"run aborted: {why}")
        for st in rt.run.stages.values():
            if st.status in ("pending", "running", "retrying"):
                rt.update(st.key, status="aborted", ended=time.time())
        # Keep whatever finished: findings so far plus the raw text as the report.
        if self.questions:
            self.report = (f"> Run aborted ({why}). These are the findings that had "
                           f"finished; there is no synthesized report.\n\n{self._findings_text()}")
            self._save("aborted")
        else:
            rt.finish("aborted")

    # ---------------------------- planner ----------------------------- #

    async def _plan(self) -> Plan:
        rt = self.rt
        rt.add_stage("planner", "split the task into sub-questions")
        rt.begin("planner")
        fallback = Plan.model_construct(key_terms=[], subquestions=[self.task])
        try:
            r = await self._cancellable("planner", self._call(
                "planner", "planner", planner, self.task, PLANNER, schema=Plan,
                usage_limits=UsageLimits(request_limit=PLANNER["request_limit"])))
        except Exception as e:
            if _service_error(e) or isinstance(e, EndpointConfigError):
                raise
            reason = _reason(e)
            print(f"[planner] FAILED ({reason}); falling back to one researcher on the whole task")
            log.debug("planner exception", exc_info=e)
            rt.end("planner", "timeout" if isinstance(e, asyncio.TimeoutError) else "failed", reason)
            rt.log(f"planner failed ({reason}); one researcher will take the whole task", "planner")
            return fallback
        if isinstance(r, UserCmd):
            rt.end("planner", "stopped", "stopped by user")
            rt.log("planner stopped by user; one researcher will take the whole task", "planner")
            return fallback
        plan = r.output
        rt.end("planner", "done")
        rt.update("planner", activity=f"{len(plan.subquestions)} sub-questions")
        print(f"[planner] key terms: {', '.join(plan.key_terms) or '(none)'}")
        print(f"[planner] {len(plan.subquestions)} sub-questions")
        for q in plan.subquestions:
            print(f"   - {q}")
        rt.log(f"plan: {len(plan.subquestions)} sub-questions; key terms: "
               f"{', '.join(plan.key_terms) or '(none)'}", "planner")
        return plan

    # --------------------------- researchers -------------------------- #

    async def _research_all(self, keys: list[str]) -> None:
        """Run researchers concurrently. A ServiceError in one cancels the rest
        at once. (Explicit cancel rather than asyncio.TaskGroup: on Python
        3.12.3 a failed TaskGroup leaves the calling task marked as
        cancelling, and pydantic-ai then refuses to run further steps in it.)"""
        tasks = []
        for key in keys:
            t = asyncio.create_task(self._research(key))
            self.stage_tasks[key] = t
            tasks.append(t)
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for t in done:
                if not t.cancelled() and t.exception() is not None:
                    raise t.exception()
        finally:
            pending = [t for t in tasks if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            for key in keys:
                self.stage_cmd.pop(key, None)

    async def _attempt(self, key: str, prompt: str, rdeps: Deps, transcript: list,
                       request_limit: int, timeout: float, history: list | None = None) -> Finding:
        result = await self._call(
            "researcher", key, researcher, prompt, RESEARCHER, schema=Finding, timeout=timeout,
            transcript=transcript, deps=rdeps, message_history=history,
            usage_limits=UsageLimits(request_limit=request_limit))
        return result.output

    async def _research(self, key: str) -> Finding:
        """One researcher, with an automatic retry ladder plus user commands.

        Automatic, so a single failure costs one researcher's work, not the run:
          1. the normal attempt;
          2. on failure, if it read any pages: finalize, continuing its own
             conversation (pages already in context) and asking only for the
             answer, with a fresh output-retry budget;
          3. if that fails too: up to `researcher_retries` fresh attempts,
             each told why the last one failed.
        User commands (progress page): finalize now, restart, stop.
        A ServiceError (dead service, bad token) aborts the run: no attempt
        could succeed."""
        rt, q = self.rt, self.questions[key]
        prompt = (
            f"Overall task from the user:\n{self.task}\n\n"
            f"Key terms to search for verbatim: {', '.join(self.key_terms) or '(none identified)'}\n\n"
            f"Your sub-question:\n{q}\n\n"
            "Answer only your sub-question, but honor any instructions "
            "in the overall task (e.g. which kinds of sources to read)."
        )
        retries = SWARM.get("researcher_retries", 1)
        auto_finalize = SWARM.get("finalize_on_failure", True)
        rt.update(key, label=q, status="pending", activity="queued", error="", ended=None)

        sem = _worker_sem()
        got = await self._cancellable(key, sem.acquire())
        if isinstance(got, UserCmd):
            return self._stopped(key, "stopped by user before it started")
        try:
            print(f"[{key}] start: {q[:60]}")
            rt.begin(key)
            rt.log("started", key)
            failures, reason = 0, ""
            while True:
                # Fresh Deps per attempt so budgets don't bleed across attempts.
                rdeps = Deps(http=self.http, key=key, rt=rt)
                transcript: list = []
                n = rt.run.stages[key].attempts + 1
                rt.update(key, attempts=n, status="running", searches=0, fetches=0, fetch_failures=0)
                p = prompt if not reason else (
                    prompt + f"\n\nA previous attempt at this sub-question failed ({reason}). "
                    "Use different searches and sources if needed.")
                try:
                    r = await self._cancellable(key, self._attempt(
                        key, p, rdeps, transcript, RESEARCHER["request_limit"], RESEARCHER["timeout_s"]))
                except Exception as e:
                    if _service_error(e) or isinstance(e, EndpointConfigError):
                        raise
                    why = _rejection_reason(transcript)
                    reason = (f"timed out after {RESEARCHER['timeout_s']}s" if isinstance(e, asyncio.TimeoutError)
                              else _reason(e)) + (f"; last rejection: {why}" if why else "")
                    print(f"[{key}] attempt {n} failed: {reason}")
                    log.debug("researcher exception", exc_info=e)
                    rt.log(f"attempt {n} failed: {reason}", key)
                    r = None

                if isinstance(r, Finding):
                    return self._done(key, r, rdeps)
                if isinstance(r, UserCmd):
                    if r.cmd == "restart":
                        rt.log("restarted by user", key)
                        reason = ""
                        continue
                    if r.cmd == "finalize":
                        if not rdeps.read:
                            return self._stopped(key, "finalize requested, but no pages had been read yet")
                        f = await self._finalize(key, rdeps, transcript, "You have been asked to finish now.")
                        if isinstance(f, Finding):
                            return f
                        if isinstance(f, UserCmd) and f.cmd == "restart":
                            continue
                        return self._stopped(key, "finalize failed" if f is None else "stopped by user")
                    return self._stopped(key, "stopped by user")

                # The attempt failed: automatic ladder.
                # (a) It already wrote an answer, just not as JSON: keep it.
                f = self._salvage(key, rdeps, transcript)
                if f:
                    return f
                # (b) Finalize: ask for the answer from the pages already read.
                if auto_finalize and rdeps.read and transcript:
                    why = _rejection_reason(transcript)
                    f = await self._finalize(key, rdeps, transcript,
                                             f"Your answer was not accepted{': ' + why if why else ''}.")
                    if isinstance(f, Finding):
                        return f
                    if isinstance(f, UserCmd):
                        if f.cmd == "restart":
                            reason = ""
                            continue
                        return self._stopped(key, "stopped by user")
                failures += 1
                if failures > retries:
                    break
                rt.update(key, status="retrying", activity=f"retry {failures} of {retries}: starting over")
                rt.log(f"retrying from scratch ({failures} of {retries})", key)

            rt.end(key, "timeout" if reason.startswith("timed out") else "failed", reason)
            f = Finding(subquestion=q, sources=[],
                        answer=f"RESEARCHER FAILED after {failures} failed attempt(s) (no verified sources): {reason}")
            self.findings[key] = f
            return f
        finally:
            sem.release()

    async def _finalize(self, key: str, rdeps: Deps, transcript: list, lead: str):
        """Continue the researcher's conversation and ask only for the answer.
        Returns a Finding, a UserCmd (stop/restart by the user) or None (failed).
        A "finalize" command arriving while finalizing (a click racing the
        automatic finalize) just lets it carry on."""
        rt = self.rt
        rt.update(key, status="retrying", finalizing=True, activity="finalizing from pages already read")
        rt.log(f"finalizing from {len(rdeps.read)} pages already read", key)
        rdeps.finalizing = True
        f_finding = FINALIZE_PROMPT.format(lead=lead)
        f_transcript: list = []
        try:
            for _ in range(2):
                try:
                    r = await self._cancellable(key, self._attempt(
                        key, f_finding, rdeps, f_transcript,
                        request_limit=4, timeout=min(600, RESEARCHER["timeout_s"]),
                        history=_resumable(transcript)))
                except Exception as e:
                    if _service_error(e):
                        raise
                    rt.log(f"finalize failed: {_reason(e)[:200]}", key)
                    return self._salvage(key, rdeps, f_transcript)
                if isinstance(r, Finding):
                    return self._done(key, r, rdeps, finalized=True)
                if r.cmd != "finalize":
                    return r
            return None
        finally:
            rt.update(key, finalizing=False)

    def _salvage(self, key: str, rdeps: Deps, transcript: list) -> Finding | None:
        """The researcher wrote its answer as prose (Markdown) instead of the
        required JSON, then got rejected or ran out of time. That prose is the
        answer: keep it, with the URLs it cites that it actually read as
        sources (or every page it read, if it cites none). No model call."""
        if not rdeps.read:
            return None
        for msg in reversed(transcript):
            if not isinstance(msg, ModelResponse):
                continue
            if any(isinstance(p, ToolCallPart) for p in msg.parts):
                return None          # its latest move was more research, not an answer
            text = "\n".join(p.content for p in msg.parts if isinstance(p, TextPart)).strip()
            if len(text) < 400 or text.lstrip("`json \n").startswith("{"):
                return None          # too short, or JSON (finalize handles that)
            cited = [u for u in sorted(rdeps.read) if u in text]
            f = Finding(subquestion=self.questions[key], answer=text, sources=cited or sorted(rdeps.read))
            self.rt.log(f"kept its prose answer ({len(text):,} chars) instead of re-asking for JSON", key)
            return self._done(key, f, rdeps, finalized=True)
        return None

    def _done(self, key: str, out: Finding, rdeps: Deps, finalized: bool = False) -> Finding:
        n = self.rt.run.stages[key].attempts
        how = (" (finalized)" if finalized else "") + (f" on attempt {n}" if n > 1 else "")
        print(f"[{key}] done{how}: {self.questions[key][:60]} ({len(rdeps.read)} pages read)")
        self.rt.end(key, "done")
        self.rt.log(f"done{how}: {len(rdeps.read)} pages read, {rdeps.fetch_failures} failed, "
                    f"{len(out.sources)} sources", key)
        out = out.model_copy(update={"subquestion": self.questions[key]})
        self.findings[key] = out
        return out

    def _stopped(self, key: str, why: str) -> Finding:
        print(f"[{key}] {why}")
        self.rt.end(key, "stopped", why)
        self.rt.log(why, key)
        f = Finding(subquestion=self.questions[key], sources=[],
                    answer=f"NOT ANSWERED: {why} (no verified sources).")
        self.findings[key] = f
        return f

    # ----------------------------- writer ----------------------------- #

    def _findings_text(self) -> str:
        parts = []
        for key in self._keys():
            f = self.findings.get(key)
            if f is None:
                parts.append(f"### {self.questions[key]}\nNOT ANSWERED: research did not finish.\nSources: none")
            else:
                parts.append(f"### {f.subquestion}\n{f.answer}\nSources: {', '.join(f.sources) or 'none'}")
        return "\n\n".join(parts)

    async def _write(self) -> str:
        """Synthesize. If the writer fails or is stopped, the raw findings
        become the report so the research is still saved and returned."""
        rt = self.rt
        if "writer" not in rt.run.stages:
            rt.add_stage("writer", "synthesize the findings into the report")
        rt.begin("writer")
        findings_text = self._findings_text()
        answered = sum(1 for k in self._keys() if k in self.findings and self.findings[k].sources)
        rt.log(f"writing the report from {answered}/{len(self._keys())} answered sub-questions "
               f"({len(findings_text):,} chars of findings)", "writer")
        t0 = time.time()
        try:
            r = await self._cancellable("writer", self._call(
                "writer", "writer", writer, f"Task: {self.task}\n\nFindings:\n{findings_text}", WRITER,
                usage_limits=UsageLimits(request_limit=WRITER["request_limit"])))
        except Exception as e:
            reason = _reason(e)
            print(f"[writer] FAILED ({reason}); returning the raw findings instead")
            log.debug("writer exception", exc_info=e)
            rt.end("writer", "timeout" if isinstance(e, asyncio.TimeoutError) else "failed", reason)
            rt.log(f"writer failed ({reason}); returning raw findings", "writer")
            return (f"> The writer failed ({reason}), so these are the researchers' "
                    f"raw findings, unsynthesized.\n\n{findings_text}")
        if isinstance(r, UserCmd):
            rt.end("writer", "stopped", "stopped by user")
            rt.log("writer stopped by user; returning raw findings", "writer")
            return ("> The writer was stopped, so these are the researchers' raw findings, "
                    f"unsynthesized.\n\n{findings_text}")
        rt.end("writer", "done")
        rt.log(f"report written: {len(r.output):,} chars in {(time.time() - t0) / 60:.1f} min", "writer")
        return r.output

    # ------------------------------ save ------------------------------ #

    def _save(self, status: str) -> None:
        """Write every stage of the run so nothing is lost if the browser
        session dies or the orchestrator paraphrases the report. A previous
        report.md (from before a re-run or rewrite) is kept as report.vN.md."""
        rt, run_dir = self.rt, self.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "task.md").write_text(f"# Task\n\n{self.task}\n", encoding="utf-8")
        (run_dir / "plan.md").write_text(
            "# Plan\n\n## Key terms\n\n" + ("\n".join(f"- {t}" for t in self.key_terms) or "none")
            + "\n\n## Sub-questions\n\n" + "\n".join(f"- {k}: {self.questions[k]}" for k in self._keys()) + "\n",
            encoding="utf-8")
        for key in self._keys():
            f = self.findings.get(key)
            (run_dir / f"finding-{key[1:]}.md").write_text(
                f"# {self.questions[key]}\n\n{f.answer if f else 'NOT ANSWERED: research did not finish.'}"
                "\n\n## Sources\n\n" + ("\n".join(f"- {u}" for u in f.sources) if f and f.sources else "none") + "\n",
                encoding="utf-8")
        (run_dir / "findings.json").write_text(json.dumps({
            "task": self.task, "key_terms": self.key_terms,
            "findings": {k: (self.findings[k].model_dump() if k in self.findings
                             else {"subquestion": self.questions[k], "answer": None, "sources": []})
                         for k in self._keys()},
        }, indent=2) + "\n", encoding="utf-8")
        report_path = run_dir / "report.md"
        if report_path.exists():
            n = 1
            while (run_dir / f"report.v{n}.md").exists():
                n += 1
            report_path.rename(run_dir / f"report.v{n}.md")
            rt.run.report_versions = n
            rt.log(f"previous report kept as report.v{n}.md")
        report_path.write_text(self.report + "\n", encoding="utf-8")
        (run_dir / "full.md").write_text(
            f"# {self.task}\n\n{self.report}\n\n---\n\n## Raw findings\n\n{self._findings_text()}\n",
            encoding="utf-8")
        rt.log(f"saved reports/{self.run_id}/")
        rt.finish(status, report_dir=f"reports/{self.run_id}")
        rt.run.job = ""
        # metrics.json: per-stage timing, requests, tokens, searches, fetches, event log.
        (run_dir / "metrics.json").write_text(json.dumps(rt.to_dict(), indent=2) + "\n", encoding="utf-8")
        t = rt.to_dict()["totals"]
        print(f"[swarm] saved {run_dir}/ | {t['requests']} requests, {t['input_tokens']:,} in / "
              f"{t['output_tokens']:,} out tokens, {t['searches']} searches, {t['fetches']} pages read")

    # ------------------------------------------------------------------ #
    # Controls (called from the progress page)
    # ------------------------------------------------------------------ #

    def busy(self) -> bool:
        return self.main_task is not None and not self.main_task.done()

    def control_stage(self, key: str, cmd: str) -> str:
        st = self.rt.run.stages.get(key)
        if st is None:
            raise ControlError(f"no stage {key!r} in this run", 404)
        if cmd not in ("stop", "finalize", "restart"):
            raise ControlError(f"unknown command {cmd!r}", 400)
        task = self.stage_tasks.get(key)
        if task is None or task.done() or st.status not in ("pending", "running", "retrying"):
            raise ControlError(f"{key} is not running")
        if key in ("planner", "writer") and cmd != "stop":
            raise ControlError(f"{key} can only be stopped")
        if st.status == "pending" and cmd != "stop":
            raise ControlError(f"{key} hasn't started yet; it can only be stopped")
        if cmd == "finalize" and st.finalizing:
            raise ControlError(f"{key} is already finalizing")
        self.stage_cmd[key] = cmd
        task.cancel()
        self.rt.log(f"user: {cmd}", key)
        return f"{key}: {cmd} requested"

    def wrap_up(self) -> str:
        if self.rt.run.phase != "researching" or not self.busy():
            raise ControlError("wrap up works while researchers are running")
        n = 0
        for key in self._keys():
            st, task = self.rt.run.stages[key], self.stage_tasks.get(key)
            if task is None or task.done() or st.status not in ("pending", "running", "retrying"):
                continue
            if st.finalizing:
                continue        # already producing its answer; let it finish
            self.stage_cmd[key] = "finalize" if st.status in ("running", "retrying") else "stop"
            task.cancel()
            n += 1
        self.rt.log(f"user: wrap up ({n} researcher(s) finalizing or skipped; writer runs next)")
        return f"wrapping up {n} researcher(s)"

    def abort(self) -> str:
        if not self.busy():
            raise ControlError("nothing is running")
        self.aborting = True
        self.main_task.cancel()
        return "aborting"

    def start_job(self, kind: str, key: str | None = None, subquestion: str | None = None) -> str:
        """Post-run work on a finished run: re-run one researcher (optionally
        with an edited sub-question), add a sub-question, or rewrite the report.
        The writer then runs again and the report is replaced (old one kept)."""
        if self.busy():
            raise ControlError("this run is busy; wait for it or abort it first")
        if not self.questions:
            raise ControlError("this run has no plan to work from")
        subquestion = (subquestion or "").strip() or None
        if kind == "rerun":
            if key not in self.questions:
                raise ControlError(f"no researcher {key!r} in this run", 404)
            desc = f"re-run {key}" + (" (edited)" if subquestion else "")
        elif kind == "add":
            if not subquestion:
                raise ControlError("a sub-question is required", 400)
            key = f"R{max(int(k[1:]) for k in self.questions) + 1}"
            desc = f"add {key}"
        elif kind == "rewrite":
            desc = "rewrite report"
        else:
            raise ControlError(f"unknown job {kind!r}", 400)
        self.aborting = False
        self.main_task = asyncio.create_task(self._job(kind, key, subquestion, desc))
        self.main_task.add_done_callback(_consume)
        return desc

    async def _job(self, kind: str, key: str | None, subquestion: str | None, desc: str) -> None:
        rt = self.rt
        rt.reopen(desc)
        rt.log(f"user: {desc}")
        try:
            self.endpoints = load_endpoints()
        except EndpointConfigError as e:
            rt.log(f"{desc} not started: {e}")
            rt.finish("done")
            rt.run.job = ""
            return
        with swarm_span(self.run_id, f"[{desc}] {self.task}"):
            try:
                async with service_client() as http:
                    self.http = http
                    if kind in ("rerun", "add"):
                        await preflight(http)
                        if kind == "add":
                            self.questions[key] = subquestion
                            rt.add_stage(key, subquestion, max_searches=RESEARCHER.get("max_searches", 3),
                                         max_fetches=RESEARCHER["max_fetches"])
                        elif subquestion:
                            self.questions[key] = subquestion
                        rt.set_phase("researching")
                        self.findings.pop(key, None)
                        await self._research_all([key])
                    rt.set_phase("writing")
                    self.report = await self._in_stage_task("writer", self._write())
                    rt.set_phase("saving")
                    self._save("done")
            except asyncio.CancelledError:
                rt.log(f"{desc} stopped by user; the previous report is unchanged")
                for st in rt.run.stages.values():
                    if st.status in ("pending", "running", "retrying"):
                        rt.update(st.key, status="stopped", ended=time.time())
                rt.finish("done")
                rt.run.job = ""
            except Exception as e:
                se = _service_error(e)
                rt.log(f"{desc} failed: {se or _reason(e)}; the previous report is unchanged")
                for st in rt.run.stages.values():
                    if st.status in ("pending", "running", "retrying"):
                        rt.update(st.key, status="aborted", ended=time.time())
                rt.finish("done")
                rt.run.job = ""
            finally:
                self.http = None


def _consume(task: asyncio.Task) -> None:
    """Retrieve a background task's exception so asyncio doesn't warn about it
    (it has already been logged to the run)."""
    if not task.cancelled():
        task.exception()


async def run_swarm(task: str) -> str:
    """Run the swarm on `task` and return the report.

    The run executes in its own task, shielded from the caller: if the caller
    goes away (chat tab closed, request cancelled) the run keeps going and is
    saved; follow it on /progress. Raises RunStopped if the user aborts it,
    ServiceError if a support service is down or misconfigured."""
    sr = SwarmRun(task)
    sr.rt.log("run started")
    with swarm_span(sr.run_id, task):
        sr.main_task = asyncio.create_task(sr.run())
        sr.main_task.add_done_callback(_consume)
        try:
            return await asyncio.shield(sr.main_task)
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling() or not sr.aborting:
                raise   # our caller was cancelled; the run itself carries on
            raise RunStopped(
                f"stopped by the user from the progress page; finished findings are in "
                f"reports/{sr.run_id}/") from None


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "Compare the current state of Hermes Agent and OpenClaw."
    print(asyncio.run(run_swarm(task)))
