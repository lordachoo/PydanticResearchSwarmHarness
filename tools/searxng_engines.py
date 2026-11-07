#!/usr/bin/env python3
"""
Inspect, test and enable SearXNG search engines.

Useful for any self-hosted SearXNG that is used as a search API (AI agents,
scripts) and keeps returning empty results because its few default web
engines are rate-limited or suspended by upstream ("too many requests",
"access denied", CAPTCHA). Spreading queries over more engines fixes that.

    status       enabled engines, their recent error rates, and a live test query
    probe        query engines one at a time (enabled or not) and report which answer
    recommend    probe a list of well-known web engines that are disabled by default;
                 with --apply, enable the ones that work
    enable       enable engines in settings.yml
    disable      disable engines in settings.yml
    suspend-times  set how long SearXNG benches an engine after an error

Examples:
    tools/searxng_engines.py --url http://localhost:8888 status
    tools/searxng_engines.py --url http://localhost:8888 probe bing yahoo qwant
    tools/searxng_engines.py --url http://localhost:8888 --settings ~/searxng/settings.yml \\
        recommend --apply --restart searxng
    tools/searxng_engines.py --url http://localhost:8888 --settings ~/searxng/settings.yml \\
        enable bing yahoo --restart searxng

--url is the SearXNG base URL (default: $SEARXNG_URL, else http://localhost:8888).
--settings is the settings.yml SearXNG reads (the file you mount at
/etc/searxng/settings.yml). Edits keep a timestamped backup next to it (or in
~/.local/state/searxng_engines/ if that directory isn't writable); YAML
comments are not preserved in the rewritten file (they are in the backup).
--restart CONTAINER runs `docker restart CONTAINER` afterwards; SearXNG only
reads settings.yml at startup.

Reading needs only the Python standard library. Editing needs PyYAML
(pip install pyyaml). The JSON API must be enabled on the instance
(search.formats: [html, json]).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Web search engines that ship disabled in SearXNG's defaults but work without
# API keys. Only the ones present in the instance's version are probed.
CANDIDATES = ["bing", "yahoo", "google", "startpage", "mojeek", "qwant", "presearch",
              "mwmbl", "yep", "yandex", "duckduckgo", "brave"]
PROBE_QUERY = "open source software"


# --------------------------------------------------------------------------- #
# SearXNG HTTP API
# --------------------------------------------------------------------------- #

def get_json(url: str, timeout: float = 30) -> dict | list:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "searxng-engines-tool"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def config(base: str) -> dict:
    try:
        return get_json(f"{base}/config")
    except urllib.error.URLError as e:
        sys.exit(f"can't reach SearXNG at {base}: {e}")


def search(base: str, query: str, engines: str | None = None, categories: str | None = None) -> dict:
    params = {"q": query, "format": "json"}
    if engines:
        params["engines"] = engines
    if categories:
        params["categories"] = categories
    url = f"{base}/search?{urllib.parse.urlencode(params)}"
    try:
        return get_json(url)
    except urllib.error.HTTPError as e:
        if e.code == 403:
            sys.exit(f"{base} refused format=json (HTTP 403): add `json` to search.formats in settings.yml")
        return {"error": f"HTTP {e.code}"}
    except (urllib.error.URLError, TimeoutError) as e:
        return {"error": str(e)}


def engine_errors(base: str) -> dict:
    try:
        data = get_json(f"{base}/stats/errors")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def short_error(entry: dict) -> str:
    cls = (entry.get("exception_classname") or "").rsplit(".", 1)[-1]
    params = entry.get("log_parameters") or []
    detail = params[0] if params else ""
    names = {"SearxEngineTooManyRequestsException": "too many requests",
             "SearxEngineAccessDeniedException": "access denied",
             "SearxEngineCaptchaException": "CAPTCHA",
             "Timeout": "timeout", "ReadTimeout": "timeout", "ConnectTimeout": "timeout"}
    label = names.get(cls, cls or "error")
    return f"{label} ({detail})" if detail and label not in detail else label


# --------------------------------------------------------------------------- #
# Commands: read-only
# --------------------------------------------------------------------------- #

def cmd_status(a) -> None:
    cfg = config(a.url)
    cats = set(a.categories.split(","))
    engines = [e for e in cfg["engines"] if cats & set(e.get("categories", []))]
    errors = engine_errors(a.url)
    print(f"SearXNG {cfg.get('version', '?')} at {a.url}  |  categories: {a.categories}\n")
    enabled = [e for e in engines if e.get("enabled")]
    print(f"Enabled engines ({len(enabled)}):")
    for e in sorted(enabled, key=lambda e: e["name"]):
        errs = errors.get(e["name"]) or []
        worst = max(errs, key=lambda x: x.get("percentage", 0)) if errs else None
        health = f"{worst.get('percentage', 0):>3}% of requests failing: {short_error(worst)}" if worst else "no recent errors"
        print(f"  {e['name']:<22} !{e.get('shortcut', ''):<6} {health}")
    disabled = sorted(e["name"] for e in engines if not e.get("enabled"))
    print(f"\nDisabled ({len(disabled)}): {', '.join(disabled)}")
    print(f"\nTest query {PROBE_QUERY!r}:")
    d = search(a.url, PROBE_QUERY, categories=a.categories)
    if "error" in d:
        print(f"  failed: {d['error']}")
        return
    by_engine: dict[str, int] = {}
    for r in d.get("results", []):
        for eng in r.get("engines") or [r.get("engine")]:
            by_engine[eng] = by_engine.get(eng, 0) + 1
    print(f"  {len(d.get('results', []))} results" + (f" from {by_engine}" if by_engine else ""))
    for name, why in d.get("unresponsive_engines", []):
        print(f"  unresponsive: {name}: {why}")
    if not d.get("results"):
        print("\n  No results. Run `recommend` to find engines that work from this host.")


def probe_engines(base: str, names: list[str], delay: float) -> dict[str, tuple[int, str]]:
    """name -> (result count, problem or "")."""
    out = {}
    for i, name in enumerate(names):
        if i:
            time.sleep(delay)   # be gentle: the point is to avoid rate limits
        d = search(base, PROBE_QUERY, engines=name)
        if "error" in d:
            out[name] = (0, d["error"])
            continue
        n = sum(1 for r in d.get("results", []) if name in (r.get("engines") or [r.get("engine")]))
        why = "; ".join(w for e, w in d.get("unresponsive_engines", []) if e == name)
        out[name] = (n, why or ("" if n else "no results"))
        status = f"{n:>3} results" if n else f"  - {out[name][1]}"
        print(f"  {name:<22} {status}", flush=True)
    return out


def cmd_probe(a) -> None:
    cfg = config(a.url)
    known = {e["name"]: e for e in cfg["engines"]}
    names = a.engines or sorted(e["name"] for e in cfg["engines"]
                                if e.get("enabled") and set(a.categories.split(",")) & set(e.get("categories", [])))
    unknown = [n for n in names if n not in known]
    for n in unknown:
        print(f"  {n:<22}   - not an engine in this SearXNG version")
    names = [n for n in names if n in known]
    print(f"Probing {len(names)} engine(s), {a.delay}s apart, query {PROBE_QUERY!r}:")
    probe_engines(a.url, names, a.delay)


# --------------------------------------------------------------------------- #
# Commands: editing settings.yml
# --------------------------------------------------------------------------- #

def load_settings(path: Path):
    try:
        import yaml
    except ImportError:
        sys.exit("editing settings.yml needs PyYAML: pip install pyyaml")
    if not path.is_file():
        sys.exit(f"{path}: not found")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        sys.exit(f"{path}: not a SearXNG settings file")
    return yaml, data


def check_writable(path: Path) -> None:
    """Fail before doing anything slow if the file can't be rewritten in place."""
    if not path.is_file():
        sys.exit(f"{path}: not found")
    if not os.access(path, os.W_OK):
        sys.exit(f"{path} is not writable by {os.environ.get('USER', 'you')}. SearXNG's container often "
                 f"chowns it to uid 977. Re-run with sudo, or: sudo chown $USER {path}")


BACKUP_FALLBACK = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "searxng_engines"


def save_settings(path: Path, yaml, data: dict) -> None:
    check_writable(path)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # Backup next to the file if the directory allows it. A config directory
    # mounted into SearXNG's container is usually owned by uid 977, so the file
    # can be writable (after a chown) while the directory is not.
    if os.access(path.parent, os.W_OK):
        backup = path.with_name(f"{path.name}.bak-{stamp}")
    else:
        BACKUP_FALLBACK.mkdir(parents=True, exist_ok=True)
        flat = str(path.resolve()).strip("/").replace("/", "_")
        backup = BACKUP_FALLBACK / f"{flat}.bak-{stamp}"
    shutil.copyfile(path, backup)
    # Rewrite in place (no temp file + rename): that needs only the file's own
    # write permission, and keeps its owner and mode, which the container needs.
    with open(path, "w", encoding="utf-8") as f:
        f.write(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    print(f"wrote {path} (backup: {backup})")


def set_disabled(a, names: list[str], disabled: bool) -> list[str]:
    path = Path(os.path.expanduser(a.settings or "")) if a.settings else None
    if not path:
        sys.exit("--settings PATH is required to change engines")
    known = {e["name"] for e in config(a.url)["engines"]}
    bad = [n for n in names if n not in known]
    if bad:
        # An unknown name would be added as a new engine with no module and
        # SearXNG would refuse to start.
        sys.exit(f"not engines in this SearXNG version: {', '.join(bad)} (see `status` / `probe`)")
    check_writable(path)
    yaml, data = load_settings(path)
    merged = bool(data.get("use_default_settings"))
    engines = data.setdefault("engines", []) if merged else data.get("engines")
    if engines is None:
        sys.exit(f"{path} has no `engines:` list and no `use_default_settings: true`; can't edit safely")
    changed = []
    for name in names:
        entry = next((e for e in engines if isinstance(e, dict) and e.get("name") == name), None)
        if entry is None:
            if not merged:
                print(f"  {name}: not listed in this full settings file; skipped")
                continue
            # With use_default_settings, SearXNG merges this into the default
            # definition of the engine with the same name.
            entry = {"name": name}
            engines.append(entry)
        if entry.get("disabled", False) != disabled or "disabled" not in entry:
            entry["disabled"] = disabled
            changed.append(name)
    if changed:
        save_settings(path, yaml, data)
        print(f"{'disabled' if disabled else 'enabled'}: {', '.join(changed)}")
    else:
        print("nothing to change")
    return changed


def restart(a) -> None:
    if not a.restart:
        print("SearXNG reads settings.yml at startup: restart it to apply "
              "(docker restart <container>, or pass --restart <container>).")
        return
    r = subprocess.run(["docker", "restart", a.restart], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"docker restart {a.restart} failed: {r.stderr.strip()}")
    print(f"restarted {a.restart}; waiting for it to answer…")
    for _ in range(30):
        time.sleep(2)
        try:
            get_json(f"{a.url}/config", timeout=5)
            print("SearXNG is back up.")
            return
        except Exception:
            pass
    print("SearXNG hasn't come back after 60 s; check `docker logs` for a settings error "
          "(the backup next to settings.yml is the previous version).")


def cmd_enable(a) -> None:
    if set_disabled(a, a.engines, False):
        restart(a)


def cmd_disable(a) -> None:
    if set_disabled(a, a.engines, True):
        restart(a)


def cmd_recommend(a) -> None:
    if a.apply:
        if not a.settings:
            sys.exit("--apply needs --settings PATH (the settings.yml SearXNG reads)")
        check_writable(Path(os.path.expanduser(a.settings)))
    cfg = config(a.url)
    known = {e["name"]: e for e in cfg["engines"]}
    candidates = [n for n in CANDIDATES if n in known and not known[n].get("enabled")]
    if not candidates:
        print("All candidate web engines this SearXNG version has are already enabled.")
        return
    print(f"Probing {len(candidates)} disabled web engine(s), {a.delay}s apart:")
    res = probe_engines(a.url, candidates, a.delay)
    working = [n for n, (count, _) in res.items() if count > 0]
    print()
    if not working:
        print("None of them returned results from this host right now.")
        return
    print(f"Working: {', '.join(working)}")
    if not a.apply:
        print("Re-run with --apply (and --settings, optionally --restart) to enable them.")
        return
    if set_disabled(a, working, False):
        restart(a)


def cmd_suspend_times(a) -> None:
    path = Path(os.path.expanduser(a.settings or "")) if a.settings else None
    if not path:
        sys.exit("--settings PATH is required")
    yaml, data = load_settings(path)
    st = data.setdefault("search", {}).setdefault("suspended_times", {})
    for key, val in (("SearxEngineTooManyRequests", a.too_many_requests),
                     ("SearxEngineAccessDenied", a.access_denied),
                     ("SearxEngineCaptcha", a.captcha)):
        if val is not None:
            st[key] = val
    if not st:
        sys.exit("pass at least one of --too-many-requests / --access-denied / --captcha (seconds)")
    save_settings(path, yaml, data)
    print("suspended_times:", st)
    restart(a)


# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="See the module docstring (head of this file) for examples.")
    p.add_argument("--url", default=os.environ.get("SEARXNG_URL", "http://localhost:8888"), help="SearXNG base URL")
    p.add_argument("--settings", help="settings.yml to edit (the file mounted at /etc/searxng/settings.yml)")
    p.add_argument("--restart", metavar="CONTAINER", help="docker restart this container after an edit")
    p.add_argument("--categories", default="general", help="categories to show/probe (default: general)")
    p.add_argument("--delay", type=float, default=2.0, help="seconds between probe queries (default 2)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="enabled engines, error rates, a live test query")
    sp = sub.add_parser("probe", help="query engines one at a time"); sp.add_argument("engines", nargs="*")
    sp = sub.add_parser("recommend", help="probe disabled web engines; --apply enables the working ones")
    sp.add_argument("--apply", action="store_true")
    sp = sub.add_parser("enable", help="enable engines"); sp.add_argument("engines", nargs="+")
    sp = sub.add_parser("disable", help="disable engines"); sp.add_argument("engines", nargs="+")
    sp = sub.add_parser("suspend-times", help="seconds an engine is benched after an error")
    sp.add_argument("--too-many-requests", type=int); sp.add_argument("--access-denied", type=int)
    sp.add_argument("--captcha", type=int)
    a = p.parse_args()
    a.url = a.url.rstrip("/")
    {"status": cmd_status, "probe": cmd_probe, "recommend": cmd_recommend, "enable": cmd_enable,
     "disable": cmd_disable, "suspend-times": cmd_suspend_times}[a.cmd](a)


if __name__ == "__main__":
    main()
