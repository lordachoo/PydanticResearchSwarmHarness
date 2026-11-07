"""
Preflight check: is every service the swarm depends on reachable, and what does
the LLM endpoint support?  Run after `source sourceMe`:

    python doctor.py

Exit code 0 if the swarm can run, 1 otherwise. It reads the same environment
variables as swarm_pydantic_ai.py, so it checks exactly what the swarm will use.
"""

import json
import os
import sys
import urllib.parse

import httpx

LLM = os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "qwen3.5-122b")
KEY = os.environ.get("LLM_API_KEY") or "local"
SEARXNG = os.environ.get("SEARXNG_URL", "http://localhost:8888").rstrip("/")
CRAWL4AI = os.environ.get("CRAWL4AI_URL", "http://localhost:11235").rstrip("/")
CRAWL4AI_TOKEN = os.environ.get("CRAWL4AI_API_TOKEN", "")
DOCLING = os.environ.get("DOCLING_URL", "http://localhost:5001").rstrip("/")
OTLP = (os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "").rstrip("/")

failed = False


def report(ok: bool | None, name: str, detail: str = "") -> None:
    """ok=True pass, False fail (blocks the swarm), None informational."""
    global failed
    tag = {True: "  OK  ", False: " FAIL ", None: " info "}[ok]
    print(f"[{tag}] {name}" + (f": {detail}" if detail else ""))
    if ok is False:
        failed = True


def chat(http: httpx.Client, extra: dict, max_tokens: int = 2048) -> httpx.Response:
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "You are a test harness. Be brief."},
            {"role": "user", "content": "Search the web for: NVIDIA Rubin"},
        ],
        **extra,
    }
    return http.post(f"{LLM}/chat/completions", json=body,
                     headers={"Authorization": f"Bearer {KEY}"}, timeout=600)


TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]},
    },
}]
SCHEMA = {"type": "json_schema", "json_schema": {"name": "F", "schema": {
    "type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}}


def check_llm(http: httpx.Client) -> None:
    if not LLM.endswith("/v1"):
        report(False, "LLM_BASE_URL", f"{LLM} should end in /v1")
    try:
        r = http.get(f"{LLM}/models", headers={"Authorization": f"Bearer {KEY}"}, timeout=10)
    except httpx.HTTPError as e:
        report(False, "LLM endpoint reachable", f"{LLM}: {e}")
        return
    if r.status_code != 200:
        report(False, "LLM /v1/models", f"HTTP {r.status_code}: {r.text[:200]}")
        return
    data = r.json().get("data", [])
    ids = set()
    for m in data:
        ids.add(m.get("id"))
        ids.update((m.get("meta") or {}).get("aliases") or [])
        ids.update(m.get("aliases") or [])
    if MODEL in ids:
        report(True, "LLM model listed", MODEL)
    else:
        # llama-swap aliases and llama.cpp (single model) may still accept it.
        report(None, "LLM model listed", f"{MODEL!r} not in {sorted(i for i in ids if i)}; "
               "trying it anyway (aliases are not always listed)")

    # Required: plain tool calling. Every researcher depends on it.
    print("         (asking the model for a tool call; a cold model load can take minutes)")
    try:
        r = chat(http, {"tools": TOOLS, "tool_choice": "auto"})
    except httpx.HTTPError as e:
        report(False, "LLM tool calling", str(e))
        return
    if r.status_code != 200:
        report(False, "LLM tool calling", f"HTTP {r.status_code}: {r.text[:300]}")
        return
    choice = r.json()["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    if calls:
        report(True, "LLM tool calling", f"{calls[0]['function']['name']}({calls[0]['function']['arguments']})")
    else:
        content = (choice["message"].get("content") or "")[:200]
        report(False, "LLM tool calling",
               f"no tool_calls returned (finish_reason={choice.get('finish_reason')}). "
               "Server tool parsing is off or wrong: vLLM needs --enable-auto-tool-choice "
               "--tool-call-parser <parser>, llama.cpp needs --jinja. "
               f"content={content!r}")

    # Informational: does output_mode = "native" work here?
    try:
        alone = chat(http, {"response_format": SCHEMA}, max_tokens=4096)
        both = chat(http, {"response_format": SCHEMA, "tools": TOOLS}, max_tokens=4096)
    except httpx.HTTPError as e:
        report(None, "native structured output", f"not tested: {e}")
        return
    if alone.status_code != 200:
        report(None, "native structured output", "unsupported; keep output_mode = \"prompted\"")
    elif both.status_code != 200:
        msg = both.json().get("error", {}).get("message", both.text[:150]) if both.text else ""
        report(None, "native structured output",
               f"works without tools, refused with tools ({msg}). "
               "native is OK for the planner only; keep the researcher on prompted")
    else:
        report(None, "native structured output", "supported, including with tools; "
               "LLM_OUTPUT_MODE=native is an option")


def check_searxng(http: httpx.Client) -> None:
    try:
        r = http.get(f"{SEARXNG}/search", params={"q": "test", "format": "json"}, timeout=30)
    except httpx.HTTPError as e:
        report(False, "SearXNG", f"{SEARXNG}: {e}")
        return
    if r.status_code == 403:
        report(False, "SearXNG", "HTTP 403: add `json` to search.formats in settings.yml")
    elif r.status_code != 200:
        report(False, "SearXNG", f"HTTP {r.status_code}")
    else:
        data = r.json()
        n = len(data.get("results", []))
        bad = ", ".join(f"{e[0]}: {e[1]}" for e in data.get("unresponsive_engines", []) if len(e) > 1)
        if n:
            report(True, "SearXNG", f"{n} results for 'test'" + (f" (some engines failing: {bad})" if bad else ""))
        else:
            report(False, "SearXNG", "0 results for a generic query; the swarm will refuse to run. "
                   + (f"Upstream engines failing: {bad}. " if bad else "")
                   + "Wait for rate limits to clear or enable other engines (see /stats)")


def check_crawl4ai(http: httpx.Client) -> None:
    headers = {"Content-Type": "application/json"}
    if CRAWL4AI_TOKEN:
        headers["Authorization"] = f"Bearer {CRAWL4AI_TOKEN}"
    try:
        r = http.post(f"{CRAWL4AI}/md", json={"url": "https://example.com"}, headers=headers, timeout=90)
    except httpx.HTTPError as e:
        report(False, "Crawl4AI", f"{CRAWL4AI}: {e}")
        return
    if r.status_code in (401, 403):
        report(False, "Crawl4AI", f"HTTP {r.status_code}: check CRAWL4AI_API_TOKEN")
    elif r.status_code != 200:
        report(False, "Crawl4AI", f"HTTP {r.status_code}: {r.text[:200]}")
    elif not r.json().get("markdown"):
        report(False, "Crawl4AI", f"no `markdown` key in response; keys={sorted(r.json())}")
    else:
        report(True, "Crawl4AI", f"{len(r.json()['markdown'])} chars from example.com")


def check_docling(http: httpx.Client) -> None:
    try:
        r = http.get(f"{DOCLING}/health", timeout=10)
    except httpx.HTTPError as e:
        report(False, "docling-serve", f"{DOCLING}: {e} (only PDF fetches need it)")
        return
    report(r.status_code == 200, "docling-serve", f"HTTP {r.status_code}")


def check_tracing(http: httpx.Client) -> None:
    if not OTLP:
        report(None, "tracing", "off (set OTEL_EXPORTER_OTLP_ENDPOINT to enable)")
        return
    url = OTLP if OTLP.endswith("/v1/traces") else f"{OTLP}/v1/traces"
    try:
        # An empty protobuf export request is valid OTLP: proves the collector accepts traces.
        headers = {"Content-Type": "application/x-protobuf"}
        # Same header variable the OTLP exporter reads: "k=v,k2=v2", values URL-encoded.
        for item in filter(None, (os.environ.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS")
                                  or os.environ.get("OTEL_EXPORTER_OTLP_HEADERS") or "").split(",")):
            k, _, v = item.partition("=")
            headers[k.strip()] = urllib.parse.unquote(v.strip())
        r = http.post(url, content=b"", headers=headers, timeout=10)
    except httpx.HTTPError as e:
        report(False, "tracing collector", f"{url}: {e} (start it: ./services up phoenix, or unset the variable)")
        return
    if r.status_code in (401, 403):
        report(False, "tracing collector", f"{url}: HTTP {r.status_code}; collector wants auth. "
               "For ./services Phoenix run ./services write-env (sets OTEL_EXPORTER_OTLP_HEADERS)")
        return
    report(r.status_code < 300, "tracing collector", f"{url}: HTTP {r.status_code}")
    try:
        import opentelemetry.sdk  # noqa: F401
        import opentelemetry.exporter.otlp.proto.http  # noqa: F401
    except ImportError:
        report(False, "tracing packages", "missing; pip install -r requirements.txt")


def check_anthropic(http: httpx.Client, key: str, model: str) -> None:
    try:
        r = http.get("https://api.anthropic.com/v1/models", timeout=20,
                     headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
    except httpx.HTTPError as e:
        report(False, "Anthropic API reachable", str(e))
        return
    if r.status_code in (401, 403):
        report(False, "Anthropic API key", f"HTTP {r.status_code}: key rejected")
        return
    if r.status_code != 200:
        report(False, "Anthropic API", f"HTTP {r.status_code}: {r.text[:200]}")
        return
    ids = {m.get("id") for m in r.json().get("data", [])}
    report(True if model in ids else None, "Anthropic model",
           model if model in ids else f"{model!r} not in the key's model list (aliases may still work)")
    try:
        import anthropic  # noqa: F401
    except ImportError:
        report(False, "Anthropic SDK", "missing; pip install anthropic")


def check_endpoints(http: httpx.Client) -> None:
    """Every profile some agent uses (endpoints.toml, or LLM_* env)."""
    global LLM, MODEL, KEY
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import endpoints
        eps = endpoints.load()
    except Exception as e:
        report(False, "endpoints.toml", str(e))
        return
    used = {n for names in eps.agents.values() for n in names}
    print(f"\nLLM endpoints ({eps.source}):")
    for role, names in eps.agents.items():
        print(f"  {role:<12} → {', '.join(names)}")
    for name in sorted(used):
        p = eps.profiles[name]
        print(f"\n--- profile {name}: {p.model} @ {p.base_url if p.provider == 'openai' else p.provider}"
              f" (max_concurrency {p.max_concurrency}{', output_mode ' + p.output_mode if p.output_mode else ''})")
        if p.provider == "anthropic":
            check_anthropic(http, p.api_key, p.model)
            continue
        LLM, MODEL, KEY = p.base_url.rstrip("/"), p.model, p.api_key
        check_llm(http)


def main() -> int:
    print(f"SearXNG  {SEARXNG}\nCrawl4AI {CRAWL4AI}\ndocling  {DOCLING}")
    print(f"LLM_OUTPUT_MODE={os.environ.get('LLM_OUTPUT_MODE') or '(unset: endpoints/tomls decide)'}\n")
    with httpx.Client() as http:
        check_searxng(http)
        check_crawl4ai(http)
        check_docling(http)
        check_tracing(http)
        check_endpoints(http)
    print("\n" + ("Problems found; fix the FAIL lines above." if failed else "Ready."))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
