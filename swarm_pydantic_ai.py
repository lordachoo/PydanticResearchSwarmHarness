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
(see agent_config.py). Edit those, restart.

Env (or edit CONFIG):
    LLM_BASE_URL       e.g. http://<llm-host>:8080/v1   (must end in /v1)
    LLM_MODEL          model id as the server reports it in /v1/models
    LLM_API_KEY        only if the server requires one (default "local")
    LLM_OUTPUT_MODE    prompted | native | tool; overrides output_mode in every toml
    CRAWL4AI_URL       e.g. http://<services-host>:11235
    CRAWL4AI_API_TOKEN bearer token set on the container
    SEARXNG_URL        e.g. http://<services-host>:8888  (JSON API must be enabled)
    DOCLING_URL        e.g. http://<services-host>:5001  (docling-serve; PDFs go here)
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
from pydantic_ai import Agent, ModelRetry, NativeOutput, PromptedOutput, RunContext, ToolOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from agent_config import load_agent, load_swarm

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
    "llm_base_url": os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1"),
    "llm_model": os.environ.get("LLM_MODEL", "qwen3.5-122b"),
    "llm_api_key": os.environ.get("LLM_API_KEY") or "local",
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

# One model object shared by every agent. Most local servers ignore the API
# key, but the OpenAI client insists on having one; set LLM_API_KEY if yours
# checks it (llama.cpp --api-key, LM Studio, hosted proxies).
#
# Profile overrides for local servers:
#   supports_json_object_output=False   PromptedOutput must not send
#       response_format={"type":"json_object"}; servers that refuse
#       response_format + tools reject that too.
#   openai_chat_supports_multiple_system_messages=False   merge instructions
#       (incl. PromptedOutput's schema block) into ONE system message; Qwen's
#       chat template rejects a system message that isn't first.
_provider = OpenAIProvider(
    base_url=CONFIG["llm_base_url"],
    api_key=CONFIG["llm_api_key"],
    http_client=_http_client,
)
model = OpenAIChatModel(
    CONFIG["llm_model"],
    provider=_provider,
    profile=merge_profile(
        _provider.model_profile(CONFIG["llm_model"]),
        OpenAIModelProfile(
            supports_json_object_output=False,
            openai_chat_supports_multiple_system_messages=False,
        ),
    ),
)


# --------------------------------------------------------------------------- #
# Shared dependencies (injected into tools via RunContext)
# --------------------------------------------------------------------------- #

@dataclass
class Deps:
    http: httpx.AsyncClient
    fetches: int = 0                    # fetches this researcher has made
    searches: int = 0                   # searches this researcher has made
    seen: set[str] = field(default_factory=set)


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


def output_spec(cfg: dict, schema: type[BaseModel]):
    mode = (os.environ.get("LLM_OUTPUT_MODE") or cfg.get("output_mode") or "prompted").strip().lower()
    if mode not in _OUTPUT_MODES:
        raise ValueError(
            f"output mode must be one of {sorted(_OUTPUT_MODES)}, got {mode!r} "
            "(from LLM_OUTPUT_MODE or output_mode in agents/*.toml)"
        )
    return _OUTPUT_MODES[mode](schema)

planner = Agent(
    model,
    output_type=output_spec(PLANNER, Plan),
    retries=OUTPUT_RETRIES,
    instructions=[planner_context, PLANNER["instructions"]],
    model_settings=PLANNER["model_settings"],
)

researcher = Agent(
    model,
    deps_type=Deps,
    output_type=output_spec(RESEARCHER, Finding),
    retries=OUTPUT_RETRIES,
    instructions=[researcher_context, RESEARCHER["instructions"]],
    model_settings=RESEARCHER["model_settings"],
)

writer = Agent(
    model,
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

@researcher.tool
async def web_search(ctx: RunContext[Deps], query: str) -> str:
    """Search the web via SearXNG. Returns up to 8 results as title, URL and
    snippet. Use this first to find pages, then fetch_page the best ones."""
    deps = ctx.deps
    max_searches = RESEARCHER.get("max_searches", 3)
    if deps.searches >= max_searches:
        return (
            "SEARCH BUDGET EXHAUSTED. Do not call web_search again. "
            "Fetch the best URLs you already have, or answer with what you have."
        )
    deps.searches += 1
    print(f"[web_search] {query}")
    try:
        r = await deps.http.get(
            f"{CONFIG['searxng_url']}/search",
            params={"q": query, "format": "json"},
            timeout=30,
        )
    except httpx.HTTPError as e:
        msg = f"SEARCH FAILED for {query!r}: {e}"
        print(f"[web_search] {msg}")
        return msg
    if r.status_code != 200:
        msg = f"SEARCH FAILED for {query!r}: HTTP {r.status_code}: {r.text[:300]}"
        print(f"[web_search] {msg}")
        return msg
    results = r.json().get("results", [])[:8]
    if not results:
        return f"NO RESULTS for {query!r}. Try different wording."
    lines = []
    for i, res in enumerate(results, 1):
        title = (res.get("title") or "").strip()
        url = res.get("url") or ""
        snippet = (res.get("content") or "").strip().replace("\n", " ")[:300]
        lines.append(f"{i}. {title}\n   {url}\n   {snippet}")
    return "\n".join(lines)

@researcher.tool
async def fetch_page(ctx: RunContext[Deps], url: str) -> str:
    """Fetch a web page and return its markdown. HTML goes through Crawl4AI;
    PDF URLs are converted (with OCR) through docling-serve, so PDFs are fine
    to fetch — reports, bulletins, datasheets, papers."""
    deps = ctx.deps

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
        print(f"[fetch_page] (pdf via docling) {url}")
        try:
            text = await _fetch_pdf_via_docling(deps.http, url)
        except httpx.HTTPError as e:
            text = f"PDF FETCH FAILED for {url}: {e}"
        if text.startswith("PDF FETCH FAILED"):
            print(f"[fetch_page] {text[:200]}")
            return text
        return text[: RESEARCHER.get("pdf_char_limit", RESEARCHER["page_char_limit"] * 4)]

    headers = {"Content-Type": "application/json"}
    if CONFIG["crawl4ai_token"]:
        headers["Authorization"] = f"Bearer {CONFIG['crawl4ai_token']}"

    print(f"[fetch_page] {url}")
    try:
        r = await ctx.deps.http.post(
            f"{CONFIG['crawl4ai_url']}/md",
            json={"url": url},
            headers=headers,
            timeout=90,
        )
    except httpx.HTTPError as e:
        msg = f"FETCH FAILED for {url}: {e}"
        print(f"[fetch_page] {msg}")
        return msg

    if r.status_code != 200:
        # Return the server's explanation to the model instead of crashing the
        # run; it can choose another URL or answer without this page.
        body = r.text[:500]
        msg = f"FETCH FAILED for {url}: HTTP {r.status_code}: {body}"
        print(f"[fetch_page] {msg}")
        return msg

    data = r.json()
    # 0.7-era docs show the markdown under "markdown". Verify against your 0.9.3
    # response and adjust this line if the key differs.
    text = data.get("markdown") or str(data)
    return text[: RESEARCHER["page_char_limit"]]



# --------------------------------------------------------------------------- #
# Researcher output enforcement
# --------------------------------------------------------------------------- #
# The model will happily answer from memory and cite URLs it never opened.
# Reject any Finding that isn't backed by pages this researcher actually
# fetched; pydantic-ai sends the message back and the model retries.

@researcher.output_validator
async def require_fetched_sources(ctx: RunContext[Deps], finding: Finding) -> Finding:
    deps = ctx.deps
    min_fetches = RESEARCHER.get("min_fetches", 1)
    if deps.fetches < min_fetches:
        raise ModelRetry(
            f"You answered without fetching any pages ({deps.fetches} fetched, "
            f"{min_fetches} required). Call web_search, then fetch_page the most "
            "relevant results (PDFs are fine), and base your answer on what those "
            "pages actually say."
        )
    fetched = {u for u in deps.seen}
    bogus = [u for u in finding.sources if u not in fetched]
    if bogus:
        raise ModelRetry(
            "Your sources list includes URLs you did not fetch: "
            + ", ".join(bogus[:5])
            + ". Only list URLs you retrieved with fetch_page. Remove the others "
            "(or fetch them first if you need them)."
        )
    if not finding.sources:
        raise ModelRetry(
            "You fetched pages but listed no sources. List the URLs you fetched "
            "and actually used."
        )
    return finding

# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

async def run_swarm(task: str) -> str:
    async with httpx.AsyncClient() as http:
        # 1. Plan. If planning fails (timeout, bad JSON after retries, server
        #    error), degrade to a single researcher on the whole task rather
        #    than losing the run.
        try:
            plan = (
                await asyncio.wait_for(
                    planner.run(task, usage_limits=UsageLimits(request_limit=PLANNER["request_limit"])),
                    timeout=PLANNER["timeout_s"],
                )
            ).output
        except Exception as e:
            reason = "timed out" if isinstance(e, asyncio.TimeoutError) else str(e)
            print(f"[planner] FAILED ({reason}); falling back to one researcher on the whole task")
            log.debug("planner exception", exc_info=e)
            # model_construct skips the min_length validation on key_terms.
            plan = Plan.model_construct(key_terms=[], subquestions=[task])
        print(f"[planner] key terms: {', '.join(plan.key_terms) or '(none)'}")
        print(f"[planner] {len(plan.subquestions)} sub-questions")
        for q in plan.subquestions:
            print(f"   - {q}")

        # 2. Fan out researchers, bounded by a semaphore so we don't exceed
        #    the server's concurrency (vLLM --max-num-seqs / llama.cpp --parallel).
        sem = asyncio.Semaphore(SWARM.get("max_workers", 4))

        async def research(q: str) -> Finding:
            async with sem:
                print(f"[researcher] start: {q[:60]}")
                # Fresh Deps per researcher so fetch budgets don't bleed across workers.
                rdeps = Deps(http=http)
                try:
                    key_terms = ", ".join(plan.key_terms) or "(none identified)"
                    prompt = (
                        f"Overall task from the user:\n{task}\n\n"
                        f"Key terms to search for verbatim: {key_terms}\n\n"
                        f"Your sub-question:\n{q}\n\n"
                        "Answer only your sub-question, but honor any instructions "
                        "in the overall task (e.g. which kinds of sources to read)."
                    )
                    result = await asyncio.wait_for(
                        researcher.run(
                            prompt,
                            deps=rdeps,
                            usage_limits=UsageLimits(
                                request_limit=RESEARCHER["request_limit"]
                            ),
                        ),
                        timeout=RESEARCHER["timeout_s"],
                    )
                    print(f"[researcher] done:  {q[:60]} ({rdeps.fetches} fetches)")
                    return result.output
                except asyncio.TimeoutError:
                    print(f"[researcher] TIMEOUT: {q[:60]}")
                    return Finding(subquestion=q, answer="Researcher timed out.", sources=[])
                except Exception as e:  # request limit, validation, model errors
                    print(f"[researcher] FAILED: {q[:60]}: {e}")
                    log.debug("researcher exception", exc_info=e)
                    return Finding(
                        subquestion=q,
                        answer=f"RESEARCHER FAILED (no verified sources): {e}",
                        sources=[],
                    )

        findings = await asyncio.gather(*(research(q) for q in plan.subquestions))

        # 3. Synthesize
        findings_text = "\n\n".join(
            f"### {f.subquestion}\n{f.answer}\nSources: {', '.join(f.sources) or 'none'}"
            for f in findings
        )
        # If the writer fails, the raw findings become the report so the
        # research is still saved and returned.
        try:
            report = (
                await asyncio.wait_for(
                    writer.run(
                        f"Task: {task}\n\nFindings:\n{findings_text}",
                        usage_limits=UsageLimits(request_limit=WRITER["request_limit"]),
                    ),
                    timeout=WRITER["timeout_s"],
                )
            ).output
        except Exception as e:
            reason = "timed out" if isinstance(e, asyncio.TimeoutError) else str(e)
            print(f"[writer] FAILED ({reason}); returning the raw findings instead")
            log.debug("writer exception", exc_info=e)
            report = (
                f"> The writer failed ({reason}), so these are the researchers' "
                f"raw findings, unsynthesized.\n\n{findings_text}"
            )

        # 4. Persist every stage of the run so nothing is lost if the browser
        #    session dies or the orchestrator paraphrases the report.
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(__file__).parent / "reports" / stamp
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "task.md").write_text(f"# Task\n\n{task}\n", encoding="utf-8")
        (run_dir / "plan.md").write_text(
            "# Plan\n\n## Key terms\n\n"
            + ("\n".join(f"- {t}" for t in plan.key_terms) or "none")
            + "\n\n## Sub-questions\n\n"
            + "\n".join(f"- {q}" for q in plan.subquestions) + "\n",
            encoding="utf-8",
        )
        for i, f in enumerate(findings, 1):
            (run_dir / f"finding-{i}.md").write_text(
                f"# {f.subquestion}\n\n{f.answer}\n\n## Sources\n\n"
                + ("\n".join(f"- {u}" for u in f.sources) if f.sources else "none")
                + "\n",
                encoding="utf-8",
            )
        # report.md is the writer's output: the canonical final report.
        (run_dir / "report.md").write_text(report + "\n", encoding="utf-8")
        (run_dir / "full.md").write_text(
            f"# {task}\n\n{report}\n\n---\n\n## Raw findings\n\n{findings_text}\n",
            encoding="utf-8",
        )
        print(f"[swarm] saved {run_dir}/ (task, plan, {len(findings)} findings, report, full)")

        return report


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "Compare the current state of Hermes Agent and OpenClaw."
    print(asyncio.run(run_swarm(task)))
