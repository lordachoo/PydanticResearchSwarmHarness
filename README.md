# Local Research Swarm

A self-hosted, multi-agent research system built on
[Pydantic AI](https://ai.pydantic.dev). A planner splits a question into
sub-questions, parallel researchers search the web and read pages and PDFs, and
a writer synthesizes a report with sources. It runs against any
OpenAI-compatible endpoint (vLLM, llama.cpp, SGLang, Ollama, LM Studio, …) plus
three self-hosted services. No cloud API keys are needed.

- **Grounded answers:** a researcher's answer is rejected unless it fetched
  pages and cited only URLs it actually opened.
- **Portable across endpoints:** by default it needs nothing beyond chat
  completions and tool calling.
- **Survives failures:** a failed planner, researcher or writer degrades the
  run instead of aborting it, and every stage of every run is saved to disk.
- **Configured in files:** prompts are Markdown and limits are TOML; the code
  doesn't need editing for tuning.

```
                     ┌──────────────────────────────────────────────────┐
  Browser ─:8000─►   │  Swarm host (any Linux box with Python + Docker) │
                     │                                                  │
                     │  swarm_web.py  (Pydantic AI web chat, uvicorn)   │
                     │    └─ orchestrator ─► run_research tool          │
                     │         └─ planner ─► N researchers ─► writer    │
                     │                         │          │             │
                     │              web_search ┘          └ fetch_page  │
                     │                  │                   │       │   │
                     │            SearXNG :8888    Crawl4AI :11235  │   │
                     │                             (HTML)  docling :5001│
                     │                                     (PDF + OCR)  │
                     └────────────────────────┬─────────────────────────┘
                                              │ OpenAI-compatible API
                                              ▼
                     ┌──────────────────────────────────────────────────┐
                     │  LLM endpoint: vLLM / llama.cpp / SGLang / …     │
                     │  (any OpenAI-compatible server with tool calls)  │
                     └──────────────────────────────────────────────────┘
```

Everything is reached over HTTP and every address is an environment variable,
so the services can share one machine or be spread across several.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Requirements](#2-requirements)
3. [Configuration](#3-configuration)
4. [How it works](#4-how-it-works)
5. [Setting up the services](#5-setting-up-the-services)
6. [Setting up an LLM endpoint](#6-setting-up-an-llm-endpoint)
7. [Troubleshooting](#7-troubleshooting)
8. [Project layout](#8-project-layout)

---

## 1. Quick start

Assumes an LLM endpoint and the three services are already running (see
[§5](#5-setting-up-the-services) and [§6](#6-setting-up-an-llm-endpoint)).

```bash
git clone <this repo> research-swarm && cd research-swarm
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt

cp sourceMe.example sourceMe     # gitignored; holds your endpoints and token
$EDITOR sourceMe
source sourceMe

python doctor.py                 # preflight: services reachable, tool calling works
./start_swarm                    # web chat UI on http://<host>:8000
```

To run from the command line without the web UI:

```bash
source sourceMe
python swarm_pydantic_ai.py "When does NVIDIA Rubin ship, and what are its specs?"
```

Results print to the terminal and are saved under `reports/<timestamp>/`.

---

## 2. Requirements

| Component | Purpose | Requirement |
|---|---|---|
| Python | runs the swarm | 3.12+ (uses `tomllib`) |
| pydantic-ai | agent framework | ≥ 2.60 (`requirements.txt`) |
| LLM endpoint | all agents | OpenAI-compatible `/v1/chat/completions` with **tool calling** |
| [SearXNG](https://github.com/searxng/searxng) | `web_search` | JSON output format enabled |
| [Crawl4AI](https://github.com/unclecode/crawl4ai) | `fetch_page` for HTML | `/md` endpoint |
| [docling-serve](https://github.com/docling-project/docling-serve) | `fetch_page` for PDFs | `/v1/convert/source`; only needed for PDFs |

### LLM endpoint compatibility

By default the swarm needs only two things from the server: chat completions and
tool calling. It sends one system message per request and never sends
`response_format` (see [§3.3](#33-structured-output-modes)).

| Server | Status | Server-side requirement |
|---|---|---|
| vLLM 0.19 (Qwen3.5-122B-A10B) | tested | `--enable-auto-tool-choice --tool-call-parser <parser for your model>` |
| strata (Qwen3.8-Flash-Next) | tested | none; it rejects `response_format` + tools, so stay on `prompted` |
| llama.cpp `llama-server` | expected to work, untested | `--jinja` (tool calling); a context size (`-c`) larger than `max_tokens` plus prompt |
| SGLang, Ollama, LM Studio | expected to work, untested | tool calling enabled for the loaded model |

The model must be good at tool calling. Thinking models are fine, but give
them a large `max_tokens` (see [§7](#7-troubleshooting)).

`python doctor.py` checks this for you. It asks the model for a real tool call
and reports whether `native` structured output would also work on this server.

---

## 3. Configuration

### 3.1 Environment (`sourceMe`)

Copy `sourceMe.example` to `sourceMe` (gitignored), fill it in, and
`source sourceMe` before running anything. Don't execute it.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible base URL; must end in `/v1` |
| `LLM_MODEL` | `qwen3.5-122b` | Model id as `/v1/models` reports it (llama-swap aliases work) |
| `LLM_API_KEY` | `local` | Only if the server checks keys (llama.cpp `--api-key`, LM Studio, proxies) |
| `LLM_OUTPUT_MODE` | *(unset)* | `prompted` / `native` / `tool`; overrides `output_mode` in every toml |
| `SEARXNG_URL` | `http://localhost:8888` | SearXNG base URL |
| `CRAWL4AI_URL` | `http://localhost:11235` | Crawl4AI base URL |
| `CRAWL4AI_API_TOKEN` | *(empty)* | Bearer token set on the Crawl4AI container |
| `DOCLING_URL` | `http://localhost:5001` | docling-serve base URL |
| `LLM_DEBUG` | *(unset)* | `1` dumps every LLM request/response to stderr |

### 3.2 Agents (`agents/*.toml`, `prompts/*.md`)

Each agent has a toml file for limits and model settings, and a Markdown prompt
named by its `prompt = "…"` key. `./show_config` prints all of it plus the
current environment. Restart after editing. `start_swarm`'s `--reload` only
watches `.py` files.

Keys above the `[model_settings]` header are read by the swarm. Everything under
`[model_settings]` is sent to the model as request parameters (`temperature`,
`max_tokens`, `top_p`, …), so a limit placed under that header goes to the
server instead of being enforced.

**`agents/swarm.toml`**

| Key | Value | Meaning |
|---|---|---|
| `max_workers` | 4 | Concurrent researchers. Keep ≤ the server's parallel slots (vLLM `--max-num-seqs`, llama.cpp `--parallel`) |
| `min_subquestions` / `max_subquestions` | 1 / 5 | Bounds on the planner's plan |
| `output_retries` | 3 | Retries when structured output fails to parse or validate |

**`agents/researcher.toml`**

| Key | Value | Meaning |
|---|---|---|
| `output_mode` | `prompted` | See [§3.3](#33-structured-output-modes) |
| `request_limit` | 55 | Model calls per researcher (each tool round costs 1) |
| `timeout_s` | 900 | Wall-clock cap per researcher |
| `max_searches` | 25 | `web_search` budget |
| `max_fetches` / `min_fetches` | 10 / 2 | `fetch_page` budget / minimum before an answer is accepted |
| `page_char_limit` | 10000 | Characters kept per HTML page |
| `pdf_page_limit` / `pdf_char_limit` | 100 / 40000 | Pages converted / characters kept per PDF |
| `[model_settings]` | `temperature = 0.3`, `max_tokens = 16384` | Sent to the model |

**`agents/planner.toml`**: `output_mode = "prompted"`, `request_limit = 5`,
`timeout_s = 900`. **`agents/writer.toml`**: `request_limit = 5`,
`timeout_s = 900`. **`agents/orchestrator.toml`** (web UI only): prompt and
model settings.

### 3.3 Structured output modes

The planner returns a `Plan` and each researcher returns a `Finding`, both
Pydantic models. pydantic-ai can obtain structured output from the model in
three ways. Each has failed on at least one real server, which is why this is a
setting:

| Mode | What is sent | Pros | Where it breaks |
|---|---|---|---|
| `prompted` **(default)** | JSON schema in the system prompt; the reply text is parsed (code fences stripped) and validated | Needs nothing beyond chat + tools; works everywhere | A weak model can emit bad JSON; it is sent back to retry, up to `output_retries` times |
| `native` | `response_format: json_schema` (constrained decoding) | Output is always valid JSON | Needs server support. Some servers (e.g. strata) refuse it combined with tools: `400 structured response_format with tools/MCP is not supported` |
| `tool` | A forced `final_result` tool call | pydantic-ai's default | vLLM + Qwen3.5: the forced `tool_choice` makes xgrammar expect JSON while the model emits XML tool calls, and the call is dropped (`grammar_matcher.cc … trying to accept new token`) |

Set the mode per agent in its toml, or set `LLM_OUTPUT_MODE` to override every
agent. On a server where `doctor.py` reports that `native` works with tools
(e.g. vLLM), `LLM_OUTPUT_MODE=native` gives the strongest guarantees.

The shared model profile in `swarm_pydantic_ai.py` also overrides two
pydantic-ai defaults that break local servers:

- `supports_json_object_output=False`. Otherwise `prompted` sends
  `response_format: {"type": "json_object"}`, which brings back the
  `response_format` + tools conflict.
- `openai_chat_supports_multiple_system_messages=False`. All instruction
  parts (date context, prompt file, prompted schema) are merged into **one**
  system message, because Qwen's chat template rejects a system message that
  isn't first (`System message must be at the beginning`).

---

## 4. How it works

### 4.1 Pipeline

1. **Orchestrator** (`swarm_web.py`, web UI only): a chat agent with a single
   tool, `run_research`. Small talk is answered directly; research questions go
   to the swarm, and the report is returned verbatim.
2. **Planner** → `Plan{key_terms[3..20], subquestions[min..max]}`. `key_terms`
   must copy every proper noun, product and distinctive phrase from the task
   verbatim, including terms the model doesn't recognize. This stops it from
   swapping in a concept it knows better for the one you asked about.
3. **Researchers**: one per sub-question, up to `max_workers` in parallel. Each
   gets the full task, the key terms and its sub-question, plus two tools:
   - `web_search(query)` → SearXNG, top 8 results with snippets
   - `fetch_page(url)` → Crawl4AI for HTML, docling-serve for `.pdf` (with OCR)

   It returns `Finding{subquestion, answer, sources}`.
4. **Writer**: writes one Markdown answer from all findings, with inline
   citations and a Sources section.
5. **Save**: `reports/<YYYYMMDD-HHMMSS>/` gets `task.md`, `plan.md`,
   `finding-N.md`, `report.md` (the final answer) and `full.md` (report plus
   raw findings).

### 4.2 Grounding guards

The model will happily answer from memory and cite URLs it never opened. These
guards stop that:

- **Output validator.** A `Finding` is rejected and sent back if fewer than
  `min_fetches` pages were fetched, if any listed source wasn't fetched by
  that researcher, or if sources are empty.
- **Tool guards.** `fetch_page` refuses search-result URLs and repeat fetches,
  and both tools enforce their budgets. Each refusal is returned to the model
  as an instruction, so it changes course instead of crashing.
- **Date context.** Every agent's system prompt opens with today's date
  (re-evaluated per run, so a long-running server stays current) and states
  that the model's training data is stale, that unfamiliar names aren't
  evidence of non-existence, and that fetched content beats memory. Without
  this, models add stale years to searches and dismiss products released after
  their cutoff.

### 4.3 Failure handling

| Failure | What happens |
|---|---|
| Planner fails (timeout, invalid output after retries, server error) | One researcher is run on the whole task |
| A researcher fails or times out | Its finding says so; the writer reports that sub-question as unanswered |
| Writer fails | The raw findings become the report, with a note at the top |
| A fetch or search fails | The error text goes back to the model, which picks another source |

The run is always saved to `reports/`, even if every agent fails.

---

## 5. Setting up the services

Each service is a single Docker container. Use any host the swarm can reach and
put the URLs in `sourceMe`.

### 5.1 SearXNG (web search)

```bash
mkdir -p ~/searxng
docker run -d --name searxng --restart unless-stopped \
  -p 8888:8080 -v ~/searxng:/etc/searxng \
  -e SEARXNG_BASE_URL=http://<host>:8888/ \
  searxng/searxng:latest
```

The first start writes `~/searxng/settings.yml`. Make two edits there, then run
`docker restart searxng`:

```yaml
search:
  formats:
    - html
    - json          # the JSON API returns 403 without this
server:
  limiter: false    # the rate limiter blocks non-browser clients
```

```bash
curl -s 'http://<host>:8888/search?q=test&format=json' | jq '.results[0].url'
```

Engine health is at `/stats`.

### 5.2 Crawl4AI (HTML → Markdown)

```bash
docker run -d --name crawl4ai --restart unless-stopped \
  -p 11235:11235 --shm-size=1g \
  -e CRAWL4AI_API_TOKEN=<pick-a-token> \
  unclecode/crawl4ai:latest
```

```bash
curl -s -X POST http://<host>:11235/md -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" -d '{"url":"https://example.com"}' | jq '.markdown | length'
```

The swarm reads the `markdown` key of the `/md` response; `doctor.py` flags it
if your version names it differently. Crawl4AI drives headless Chromium, so it
can't read PDFs (that's what docling is for). Sites behind bot protection return
502; the researcher moves on.

### 5.3 docling-serve (PDF → Markdown, with OCR)

```bash
mkdir -p ~/docling
docker run -d --name docling --restart unless-stopped \
  -p 5001:5001 -e DOCLING_SERVE_ENABLE_UI=1 \
  -v ~/docling:/root/.cache \
  ghcr.io/docling-project/docling-serve:latest
```

This is the CPU image; CUDA-tagged images speed up OCR. The volume caches OCR
and layout models (a few hundred MB on first run). A test UI is at `/ui`.
"RapidOCR returned empty result" warnings on born-digital PDFs are harmless.

```bash
curl -s http://<host>:5001/health
```

---

## 6. Setting up an LLM endpoint

Any server from the [compatibility table](#llm-endpoint-compatibility) works.
Minimal examples:

```bash
# vLLM (pick the tool parser for your model family: hermes, qwen3_xml, llama3_json, …)
vllm serve <model> --enable-auto-tool-choice --tool-call-parser <parser> \
  --max-num-seqs 4 --served-model-name my-model

# llama.cpp
llama-server -m model.gguf --jinja -c 65536 --parallel 4 --host 0.0.0.0 --port 8080
```

Then set `LLM_BASE_URL=http://<host>:<port>/v1` and `LLM_MODEL` to the served
model id, and run `python doctor.py`.

[llama-swap](https://github.com/mostlygeek/llama-swap) in front of the server
gives you one stable port and on-demand model loading.

Runs are slow by design. On a ~50 tok/s endpoint, a 4-researcher run with a
few fetches each takes 15–40 minutes, because every tool round re-sends the
growing conversation. Parallelism is capped by the server's concurrent slots.

---

## 7. Troubleshooting

Start with `python doctor.py`. Most problems show up as a FAIL line there.

| Symptom | Cause / fix |
|---|---|
| `404 page not found` | `LLM_BASE_URL` must end in `/v1`, with nothing after it |
| `model_name: X` 404 | `LLM_MODEL` must match an id from `/v1/models` |
| `401` / `403` from the LLM | Server checks keys; set `LLM_API_KEY` |
| `structured response_format with tools/MCP is not supported` | Something is in `native` mode on a server that refuses `response_format` + tools. Use `prompted` (the default); check `LLM_OUTPUT_MODE` |
| `System message must be at the beginning` | A second system message is being sent. The model profile merges them; check you haven't added `@agent.instructions` / `system_prompt` on a separately built model |
| `Exceeded maximum output retries` | Check which agent in the terminal. Usually a thinking model hitting `max_tokens` (`content: null`, `finish_reason: length`): raise `max_tokens`. Otherwise bad JSON in `prompted` mode, or a researcher refusing to fetch (validator rejection) |
| `grammar_matcher.cc … accept new token` (vLLM log) | `tool` output mode on vLLM + Qwen; use `prompted` or `native` |
| Researcher finishes with no tool calls / prints tool-call text | Server isn't parsing tool calls: vLLM `--enable-auto-tool-choice --tool-call-parser …`, llama.cpp `--jinja` |
| `[planner] FAILED …; falling back` | Planner couldn't produce a valid plan; the run continues with one researcher. Look at the error; often `max_tokens` or a weak model in `prompted` mode |
| Context-length errors from llama.cpp | Raise `-c`, or lower `max_tokens` / `page_char_limit` |
| Researchers are slow or time out together | `max_workers` exceeds the server's parallel slots; requests queue |
| SearXNG 403 on `format=json` | Add `json` to `search.formats` in `settings.yml` |
| SearXNG returns 0 results | Upstream engines are rate-limiting you; check `/stats` |
| Crawl4AI `401` | `CRAWL4AI_API_TOKEN` doesn't match the container's |
| Crawl4AI 502 "Blocked by anti-bot protection" | Site-side; the researcher picks another source |
| `PDF FETCH FAILED … no md_content` | docling-serve's response shape differs; the error lists the keys it got |

**Inspecting requests:** `LLM_DEBUG=1 python swarm_pydantic_ai.py "…" 2> debug.log`,
then `grep 'LLM request roles' debug.log` shows each request's message roles,
whether tools were sent and the `response_format`, if any. Check the server's
own logs for the other side.

---

## 8. Project layout

```
├── swarm_pydantic_ai.py   # model, agents, tools, validator, run_swarm(); CLI entry point
├── swarm_web.py           # web chat UI (orchestrator agent + Agent.to_web())
├── agent_config.py        # loads agents/*.toml and prompts/*.md
├── doctor.py              # preflight checks for every dependency
├── agents/                # per-agent limits and model settings; swarm.toml for fan-out
├── prompts/               # per-agent instructions (Markdown)
├── sourceMe.example       # environment template → copy to sourceMe (gitignored)
├── start_swarm            # uvicorn launcher for the web UI (port 8000)
├── show_config            # print all agent config, prompts and the current env
├── requirements.txt
└── reports/               # one folder per run (gitignored, created automatically)
```
