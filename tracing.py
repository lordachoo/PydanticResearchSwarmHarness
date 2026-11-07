"""
Optional OpenTelemetry tracing for the swarm (Arize Phoenix or any OTLP/HTTP
collector).

Off unless OTEL_EXPORTER_OTLP_ENDPOINT (or OTEL_EXPORTER_OTLP_TRACES_ENDPOINT)
is set, e.g.:

    export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:6006"   # Phoenix
    ./services up phoenix                                         # starts one

When on, every agent run, model request (prompt, response, tokens, latency)
and tool call is exported. Each swarm run is one trace: a "research swarm"
root span, with the planner, every researcher and the writer nested under it.
In the web UI the orchestrator's run is the parent of that span.

The OpenInference span processor rewrites pydantic-ai's GenAI attributes into
the conventions Phoenix uses for its LLM views; without it (package missing)
spans still export and show up as plain traces.

Tracing problems never stop the swarm: missing packages or an unreachable
collector only print a warning.
"""

import os
from contextlib import contextmanager
from typing import Iterator

from opentelemetry import trace

_tracer = trace.get_tracer("research-swarm")
_enabled = False


def setup_tracing() -> bool:
    global _enabled
    if _enabled:
        return True
    if not (os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")):
        return False
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from pydantic_ai import Agent
        from pydantic_ai.agent import InstrumentationSettings
    except ImportError as e:
        print(f"[tracing] disabled: {e}. Install: pip install -r requirements.txt")
        return False

    project = os.environ.get("PHOENIX_PROJECT_NAME", "research-swarm")
    provider = TracerProvider(resource=Resource.create({
        "service.name": os.environ.get("OTEL_SERVICE_NAME", "research-swarm"),
        "openinference.project.name": project,   # Phoenix groups traces by this
    }))
    try:
        from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
        # Must run before the exporter so exported spans carry the rewritten attributes.
        provider.add_span_processor(OpenInferenceSpanProcessor())
    except ImportError:
        print("[tracing] openinference-instrumentation-pydantic-ai not installed; "
              "exporting plain OTel spans (Phoenix LLM views will be sparse)")
    # The exporter reads OTEL_EXPORTER_OTLP_(TRACES_)ENDPOINT itself and appends
    # /v1/traces to the generic one.
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    Agent.instrument_all(InstrumentationSettings(tracer_provider=provider, include_content=True))

    global _tracer
    _tracer = provider.get_tracer("research-swarm")
    _enabled = True
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
    print(f"[tracing] exporting traces to {endpoint} (project {project!r})")
    return True


@contextmanager
def swarm_span(run_id: str, task: str) -> Iterator[None]:
    """Root span for one swarm run. A no-op span when tracing is off."""
    with _tracer.start_as_current_span(
        "research swarm",
        attributes={"swarm.run_id": run_id, "swarm.task": task[:2000],
                    "openinference.span.kind": "CHAIN", "input.value": task[:2000]},
    ):
        yield
