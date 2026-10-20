#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fake ``ais_bench`` CLI for the ``aisbench-sla`` offline test-suite.

It mimics the two things the SLA search pipeline actually depends on:

1. **The per-point model config is the source of truth.**  Before every point
   the pipeline rewrites
   ``<workspace>/ais_bench/benchmark/configs/models/vllm_api/vllm_api_chat_temp.py``
   (``batch_size=<concurrency>``, ``max_out_len=<output length>`` ...).  This
   stub reads that file back - exactly like the real ``ais_bench`` - so the
   concurrency the pipeline *thinks* it is testing is the concurrency the stub
   reports.  Nothing is passed on the command line for it.

2. **A deterministic SLA model plus prefix-cache traffic.**  The stub prints an
   ``ais_bench``-style performance summary on stdout (legacy line format and/or
   the ``│`` table format) and, before printing, POSTs the counters a real run
   would have produced to the mock vLLM server's simulation hook::

       ttft_p90_ms = 100 + 20 * concurrency
       tpot_p90_ms =  10 +  0.5 * concurrency

   With ``--ttft-p90 340 --tpot-p90 19`` that makes every point up to
   concurrency 12 pass and everything above fail, so a correct search must
   converge to 12.

Only the standard library is used (``urllib.request`` instead of ``requests``)
so the stub keeps working inside a bare Linux container.

Self-check without any pipeline::

    python tests/stub_aisbench.py --self-check --concurrency 4,8,12,16

Useful environment variables (all optional):

``STUB_WORK_DIR``       same as ``--work-dir``
``STUB_MODEL_CFG``      explicit path to ``vllm_api_chat_temp.py``
``STUB_TRAFFIC_URL``    fallback mock base URL, e.g. ``http://127.0.0.1:8099``
``STUB_CONCURRENCY``    fallback concurrency when no config can be found
``STUB_DATA_NUM`` / ``STUB_INPUT_LEN`` / ``STUB_REPEAT_RATE``  traffic hints
``STUB_LOG_FORMAT``     ``legacy`` (default), ``table`` or ``both``
``STUB_FAIL_AT``        comma separated concurrency values that must exit 1
``AISBENCH_PT_WORK``    alt workspace env used by the reference tool
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
CFG_REL = "ais_bench/benchmark/configs/models/vllm_api/vllm_api_chat_temp.py"
DEFAULT_CONCURRENCY = 1
DEFAULT_OUTPUT_LEN = 32
DEFAULT_DATA_NUM = 64
DEFAULT_INPUT_LEN = 2048
DEFAULT_REPEAT_RATE = 0.9

_RE_BATCH = re.compile(r"batch_size\s*[=:]\s*['\"]?(\d+(?:\.\d+)?)")
_RE_OUTLEN = re.compile(r"max_out_len\s*[=:]\s*['\"]?(\d+(?:\.\d+)?)")
_RE_HOST = re.compile(r"host_ip\s*[=:]\s*['\"]?([^'\"\s,)]+)")
_RE_PORT = re.compile(r"host_port\s*[=:]\s*['\"]?(\d+)")
_RE_URL = re.compile(r"\burl\s*[=:]\s*['\"]?([^'\"\s,)]+)")
_RE_MODEL = re.compile(r"\bmodel\s*[=:]\s*['\"]?([^'\"\s,)]+)")
_RE_RATE = re.compile(r"request_rate\s*[=:]\s*['\"]?([^'\",\s)]+)")
_PLACEHOLDER = re.compile(r"\w+_for_replace")


# --------------------------------------------------------------------------- #
# discovery: model config / dataset / run parameters
# --------------------------------------------------------------------------- #
def _warn(msg: str) -> None:
    sys.stderr.write(f"[stub-aisbench] {msg}\n")


def _candidate_roots(work_dir: str, cwd: str) -> list:
    """Directories to search, nearest first: work dir, its parents, then cwd."""
    roots: list = []
    bases = [work_dir, os.environ.get("STUB_WORK_DIR"), os.environ.get("AISBENCH_PT_WORK"), cwd]
    for base in bases:
        if not base:
            continue
        try:
            p = Path(base).expanduser().resolve()
        except OSError:
            continue
        for _ in range(4):  # work dir + up to 3 parents
            if p not in roots:
                roots.append(p)
            if p.parent == p:
                break
            p = p.parent
    return roots


def find_model_config(work_dir: str, cwd: str, explicit: str = "") -> "Path | None":
    """Newest ``vllm_api_chat_temp.py`` reachable from work_dir/cwd, or None.

    The pipeline rewrites this file before every concurrency point, so when
    several copies exist the most recently modified one is the live config.
    """
    found: list = []
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
    for root in _candidate_roots(work_dir, cwd):
        for rel in (CFG_REL,
                    "benchmark/configs/models/vllm_api/vllm_api_chat_temp.py",
                    "configs/models/vllm_api/vllm_api_chat_temp.py"):
            p = root / rel
            if p.is_file():
                found.append(p)
    if not found:
        # bounded shallow glob (never recursive: work dirs live in temp trees):
        # covers "<out>/../<ws>/ais_bench/..." layouts where the workspace is
        # not an ancestor of --work-dir.
        for root in _candidate_roots(work_dir, cwd)[:2]:
            for pattern in (f"*/{CFG_REL}", f"*/*/{CFG_REL}"):
                try:
                    found.extend(Path(h) for h in glob.glob(str(root / pattern)))
                except OSError:
                    pass
            if found:
                break
    found = [p for p in found if p.is_file()]
    if not found:
        return None
    uniq = sorted({str(p): p for p in found}.values(), key=lambda q: q.stat().st_mtime,
                  reverse=True)
    return uniq[0]


def parse_model_config(path: "Path | None") -> dict:
    """Pull the parameters the pipeline substituted into the model config."""
    cfg: dict = {}
    if path is None:
        return cfg
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        _warn(f"cannot read model config {path}: {exc}")
        return cfg
    cfg["_path"] = str(path)
    for key, rx, cast in (("concurrency", _RE_BATCH, float),
                          ("output_len", _RE_OUTLEN, float),
                          ("host_port", _RE_PORT, int)):
        m = rx.search(text)
        if m:
            try:
                cfg[key] = cast(m.group(1))
            except ValueError:
                pass
    for key, rx in (("host_ip", _RE_HOST), ("url", _RE_URL),
                    ("model", _RE_MODEL), ("request_rate", _RE_RATE)):
        m = rx.search(text)
        if m:
            value = m.group(1).strip()
            if value and not _PLACEHOLDER.search(value):
                cfg[key] = value
    return cfg


def find_dataset(work_dir: str, cwd: str) -> "Path | None":
    for root in _candidate_roots(work_dir, cwd):
        for rel in ("ais_bench/datasets/gsm8k/test.jsonl",
                    "datasets/gsm8k/test.jsonl",
                    "ais_bench/datasets/gsm8k/train.jsonl"):
            p = root / rel
            if p.is_file():
                return p
        hits = sorted(glob.glob(str(root / "ais_bench/datasets/*/test.jsonl")))
        if hits:
            return Path(hits[0])
    return None


def dataset_stats(path: "Path | None", max_sample: int = 64) -> dict:
    """Line count (= data_num) and a cheap token estimate for input_len."""
    stats: dict = {}
    if path is None:
        return stats
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = [ln for ln in (ln.strip() for ln in fh) if ln]
    except OSError as exc:
        _warn(f"cannot read dataset {path}: {exc}")
        return stats
    stats["data_num"] = len(lines)
    lengths: list = []
    for line in lines[:max_sample]:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        for key in ("prompt", "question", "input", "text", "messages", "content"):
            value = obj.get(key)
            if isinstance(value, list) and value:
                if all(isinstance(x, int) for x in value[:16]):
                    lengths.append(len(value))
                else:
                    lengths.append(max(1, sum(len(str(x)) for x in value) // 4))
                break
            if isinstance(value, str) and value:
                lengths.append(max(1, len(value) // 4))
                break
    if lengths:
        stats["input_len"] = max(1, int(round(sum(lengths) / len(lengths))))
    return stats


def load_param_hints(work_dir: str) -> dict:
    """Best effort: a run-params json the pipeline may have dropped in work_dir."""
    hints: dict = {}
    if not work_dir:
        return hints
    try:
        files = sorted(Path(work_dir).glob("*.json"))[:25]
    except OSError:
        return hints
    wanted = ("input_len", "data_num", "repeat_rate", "repeat-rate", "output_len",
              "prefix_num", "concurrency", "batch_size")
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            lkey = str(key).lower()
            if lkey in wanted and lkey not in hints:
                try:
                    hints[lkey] = float(value)
                except (TypeError, ValueError):
                    continue
    return hints


def _env_float(name: str) -> "float | None":
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return float(str(raw).strip().rstrip("%"))
    except ValueError:
        _warn(f"ignoring non-numeric {name}={raw!r}")
        return None


def _env_int(name: str) -> "int | None":
    value = _env_float(name)
    return None if value is None else int(value)


def resolve_params(args, cfg: dict, work_dir: str) -> dict:
    """Effective parameters, in priority order: CLI > env > hints > files > default."""
    hints = load_param_hints(work_dir)
    dataset = find_dataset(work_dir, os.getcwd())
    dstats = dataset_stats(dataset)

    def pick(cli, env_name, *fallbacks):
        if cli is not None:
            return cli
        env = _env_float(env_name)
        if env is not None:
            return env
        for value in fallbacks:
            if value is not None:
                return value
        return None

    concurrency = pick(args.concurrency, "STUB_CONCURRENCY", cfg.get("concurrency"),
                       hints.get("concurrency"), hints.get("batch_size"), DEFAULT_CONCURRENCY)
    output_len = pick(args.output_len, "STUB_OUTPUT_LEN", cfg.get("output_len"),
                      hints.get("output_len"), DEFAULT_OUTPUT_LEN)
    data_num = pick(args.data_num, "STUB_DATA_NUM", hints.get("data_num"),
                    dstats.get("data_num"), DEFAULT_DATA_NUM)
    input_len = pick(args.input_len, "STUB_INPUT_LEN", hints.get("input_len"),
                     dstats.get("input_len"), DEFAULT_INPUT_LEN)
    repeat_rate = pick(args.repeat_rate, "STUB_REPEAT_RATE", hints.get("repeat_rate"),
                       hints.get("repeat-rate"), DEFAULT_REPEAT_RATE)
    if repeat_rate > 1.0:  # "90" was passed instead of "0.9"
        repeat_rate = repeat_rate / 100.0

    return {
        "concurrency": max(1.0, float(concurrency)),
        "output_len": max(1.0, float(output_len)),
        "data_num": max(1, int(round(float(data_num)))),
        "input_len": max(1, int(round(float(input_len)))),
        "repeat_rate": min(1.0, max(0.0, float(repeat_rate))),
        "request_rate": max(0.0, float(pick(None, "STUB_REQUEST_RATE",
                                           cfg.get("request_rate"), 0.0) or 0.0)),
        "dataset": str(dataset) if dataset else "",
        "_cfg": cfg,
    }


def traffic_base(cfg: dict, explicit: str) -> str:
    """Mock base URL: --traffic-url > config url/host:port > STUB_TRAFFIC_URL."""
    for candidate in (explicit, cfg.get("url")):
        if candidate:
            return str(candidate).rstrip("/")
    if cfg.get("host_ip") and cfg.get("host_port"):
        return f"http://{cfg['host_ip']}:{cfg['host_port']}"
    env = os.environ.get("STUB_TRAFFIC_URL", "")
    return env.rstrip("/") if env else ""


# --------------------------------------------------------------------------- #
# the deterministic SLA model
# --------------------------------------------------------------------------- #
_TTFT_BASE_MS = 100.0
_TTFT_PER_CONC_MS = 20.0


def effective_concurrency(concurrency: float, request_rate: float = 0.0) -> float:
    """模拟 ais_bench 的 ``request_rate``（RPS 到达上限）。

    在途并发受「到达速率 × 服务时延」约束，即 ``min(并发, rate × 时延)``；
    时延本身又与在途并发相关，故做定点迭代。``rate < 0.001`` 视为不限速。
    """
    c = max(1.0, float(concurrency))
    rate = float(request_rate or 0.0)
    if rate < 0.001:
        return c
    eff = c
    for _ in range(16):
        latency_s = (_TTFT_BASE_MS + _TTFT_PER_CONC_MS * eff) / 1000.0
        new = min(c, max(1.0, rate * latency_s))
        if abs(new - eff) < 1e-3:
            return new
        eff = new
    return max(1.0, min(c, eff))


def sla_metrics(concurrency: float) -> dict:
    """The contract the end-to-end test asserts against."""
    c = float(concurrency)
    ttft_p90 = _TTFT_BASE_MS + _TTFT_PER_CONC_MS * c
    tpot_p90 = 10.0 + 0.5 * c
    ttft_avg = 60.0 + 14.0 * c
    tpot_avg = 8.0 + 0.4 * c
    return {
        "ttft": {"avg": ttft_avg, "min": ttft_avg * 0.5, "max": ttft_p90 * 1.8,
                 "median": ttft_avg * 1.05, "p75": ttft_avg * 1.2,
                 "p90": ttft_p90, "p99": ttft_p90 * 1.15},
        "tpot": {"avg": tpot_avg, "min": tpot_avg * 0.7, "max": tpot_p90 * 1.3,
                 "median": tpot_avg * 1.02, "p75": tpot_avg * 1.1,
                 "p90": tpot_p90, "p99": tpot_p90 * 1.1},
    }


def benchmark_duration_ms(concurrency: float, data_num: int, output_len: int, metrics: dict) -> float:
    per_request_ms = metrics["ttft"]["avg"] + metrics["tpot"]["avg"] * output_len
    return max(1.0, per_request_ms * data_num / max(1.0, concurrency))


# --------------------------------------------------------------------------- #
# log rendering - parsed by sla_tool.parse.parse_aisbench_log
# --------------------------------------------------------------------------- #
def legacy_lines(metrics: dict) -> list:
    """``<stat> <avg> <min> <max> <median> <p75> <p90> <p99>`` (p90 = 6th number).

    The parser reads index 0 as ``avg`` and index 5 as ``p90`` for TTFT/TPOT,
    so the column order below is load bearing.  Never mention
    "Time To First Token" here: lines containing it are skipped by the parser.
    """
    out = []
    for name in ("TTFT", "TPOT"):
        s = metrics[name.lower()]
        out.append(
            f"{name} (ms)   {s['avg']:.2f}   {s['min']:.2f}   {s['max']:.2f}   "
            f"{s['median']:.2f}   {s['p75']:.2f}   {s['p90']:.2f}   {s['p99']:.2f}"
        )
    # E2EL keeps the same 7-column layout so the parser's index 5 stays p90.
    e2el_avg = metrics["ttft"]["avg"] + metrics["tpot"]["avg"] * 1.0
    out.append(
        f"E2EL (ms)   {e2el_avg:.2f}   {e2el_avg * 0.6:.2f}   {e2el_avg * 2.0:.2f}   "
        f"{e2el_avg * 0.9:.2f}   {e2el_avg * 1.1:.2f}   {e2el_avg * 1.5:.2f}   {e2el_avg * 1.8:.2f}"
    )
    return out


def table_lines(metrics: dict, data_num: int) -> list:
    """AISBench >=3.x ``│`` table; the real header layout is reproduced verbatim."""
    header = ("│ Performance Parameters   │ Stage   │ Average        │ Min            │ "
              "Max            │ Median         │ P75            │ P90            │ P99            │  N  │")
    rows = [header]
    for name in ("TTFT", "TPOT"):
        s = metrics[name.lower()]
        rows.append(
            f"│ {name:<24} │ total   │ {s['avg']:>10.2f} ms  │ {s['min']:>10.2f} ms  │ "
            f"{s['max']:>10.2f} ms  │ {s['median']:>10.2f} ms  │ {s['p75']:>10.2f} ms  │ "
            f"{s['p90']:>10.2f} ms  │ {s['p99']:>10.2f} ms  │ {data_num:>3}  │"
        )
    return rows


def build_summary(c: float, params: dict, fmt: str, exp_folder: str) -> str:
    """The complete fake ais_bench stdout."""
    eff = effective_concurrency(c, params.get("request_rate", 0.0))
    metrics = sla_metrics(eff)
    data_num = params["data_num"]
    output_len = int(round(params["output_len"]))
    duration_ms = benchmark_duration_ms(eff, data_num, output_len, metrics)
    duration_s = max(1e-6, duration_ms / 1000.0)
    total_in = data_num * params["input_len"]
    total_out = data_num * output_len

    lines = [f"Current exp folder: {exp_folder}"]
    if fmt in ("legacy", "both"):
        lines.extend(legacy_lines(metrics))
    if fmt in ("table", "both"):
        lines.append("")
        lines.extend(table_lines(metrics, data_num))
    lines.extend([
        f"Total Requests: {data_num}",
        f"Max Concurrency: {int(round(c))}",
        f"Concurrency (measured): {eff:.2f}",
        f"Benchmark Duration (ms): {duration_ms:.2f}",
        f"Output Token Throughput: {total_out / duration_s:.2f} tok/s",
        f"Input Token Throughput: {total_in / duration_s:.2f} tok/s",
        f"Total Token Throughput: {(total_in + total_out) / duration_s:.2f} tok/s",
        f"Request Throughput: {data_num / duration_s:.2f} req/s",
        f"InputTokens: {total_in}",
        f"OutputTokens: {total_out}",
    ])
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# traffic simulation (drives the mock server's prefix-cache counters)
# --------------------------------------------------------------------------- #
def push_traffic(base: str, params: dict, timeout: float = 5.0) -> bool:
    """POST the counter deltas a real benchmark run would have produced."""
    if not base:
        _warn("no mock base URL (config had none and STUB_TRAFFIC_URL is unset); "
              "skipping traffic simulation")
        return False
    queries = int(params["data_num"]) * int(params["input_len"])
    hits = int(round(queries * float(params["repeat_rate"])))
    ext_queries = max(0, queries // 20)
    ext_hits = int(round(ext_queries * 0.25))
    url = (f"{base}/__sim__/traffic?queries={queries}&hits={hits}"
           f"&ext_queries={ext_queries}&ext_hits={ext_hits}")
    req = urlrequest.Request(url, data=b"", method="POST")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local mock
            body = resp.read().decode("utf-8", errors="replace")
        _warn(f"traffic pushed: queries={queries} hits={hits} "
              f"ext_queries={ext_queries} ext_hits={ext_hits} -> {body[:160]}")
        return True
    except (urlerror.URLError, OSError, ValueError) as exc:
        _warn(f"traffic push to {base} failed ({type(exc).__name__}: {exc}); "
              "continuing - the log is still valid")
        return False


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _fail_at_set() -> set:
    raw = os.environ.get("STUB_FAIL_AT", "").strip()
    if not raw:
        return set()
    values = set()
    for part in re.split(r"[,\s]+", raw):
        if not part:
            continue
        if part.lower() == "all":
            values.add(-1)
            continue
        try:
            values.add(int(float(part)))
        except ValueError:
            _warn(f"ignoring non-numeric STUB_FAIL_AT entry {part!r}")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ais_bench(mock)",
        description="Deterministic offline stand-in for the real ais_bench CLI.")
    # arguments the real ais_bench receives (values are ignored on purpose)
    ap.add_argument("--models", default="")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--mode", default="perf")
    ap.add_argument("--summarizer", default="default_perf")
    ap.add_argument("--work-dir", dest="work_dir", default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--num-warmups", dest="num_warmups", type=int, default=0)
    # stub-only conveniences
    ap.add_argument("--concurrency", default=None,
                    help="concurrency, or a comma separated list with --self-check")
    ap.add_argument("--output-len", dest="output_len", type=float, default=None)
    ap.add_argument("--data-num", dest="data_num", type=float, default=None)
    ap.add_argument("--input-len", dest="input_len", type=float, default=None)
    ap.add_argument("--repeat-rate", dest="repeat_rate", type=float, default=None)
    ap.add_argument("--traffic-url", dest="traffic_url", default="")
    ap.add_argument("--no-traffic", dest="no_traffic", action="store_true")
    ap.add_argument("--log-format", dest="log_format",
                    choices=("legacy", "table", "both"), default=None)
    ap.add_argument("--self-check", dest="self_check", action="store_true",
                    help="ignore the pipeline config and print a sample log")
    ap.add_argument("--self-check-conc", dest="self_check_conc", default="4,8,12,16",
                    help="comma separated concurrency list for --self-check")
    return ap


def _parse_conc_list(raw: str) -> list:
    points = []
    for part in re.split(r"[,\s]+", str(raw)):
        if not part:
            continue
        try:
            points.append(float(part))
        except ValueError:
            _warn(f"ignoring non-numeric concurrency {part!r}")
    return points or [DEFAULT_CONCURRENCY]


def main(argv=None) -> int:
    ap = build_arg_parser()
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        _warn(f"ignoring unknown args: {' '.join(unknown[:8])}")

    work_dir = (args.work_dir or os.environ.get("STUB_WORK_DIR")
                or os.environ.get("AISBENCH_PT_WORK") or os.getcwd())
    cli_points = _parse_conc_list(args.concurrency) if args.concurrency else []
    if len(cli_points) > 1 and not args.self_check:
        _warn(f"--concurrency got {len(cli_points)} values outside --self-check; "
              f"using the first ({cli_points[0]:g})")
    args.concurrency = cli_points[0] if cli_points else None
    explicit_cfg = os.environ.get("STUB_MODEL_CFG", "")
    cfg_path = find_model_config(work_dir, os.getcwd(), explicit_cfg)
    cfg = parse_model_config(cfg_path)
    if cfg_path is None and not args.self_check:
        _warn("no vllm_api_chat_temp.py found - searched (and their parents) from "
              f"work_dir={work_dir!r} and cwd={os.getcwd()!r}. Falling back to "
              "STUB_CONCURRENCY / defaults; set STUB_MODEL_CFG to pin it.")
    params = resolve_params(args, cfg, work_dir)
    fmt = args.log_format or os.environ.get("STUB_LOG_FORMAT") or "legacy"
    if fmt not in ("legacy", "table", "both"):
        _warn(f"unknown STUB_LOG_FORMAT={fmt!r}; using legacy")
        fmt = "legacy"
    fail_at = _fail_at_set()
    base = traffic_base(cfg, args.traffic_url)

    # STUB_SLEEP_S：模拟「单点很慢」，用于测试工具的 --point-timeout 终止路径
    sleep_s = 0.0
    try:
        sleep_s = float(os.environ.get("STUB_SLEEP_S", "0") or 0)
    except ValueError:
        _warn("STUB_SLEEP_S 不是数字，忽略")
    if sleep_s > 0:
        _warn(f"sleeping {sleep_s:g}s (STUB_SLEEP_S) to simulate a slow point")
        time.sleep(sleep_s)

    if args.self_check:
        points = cli_points or _parse_conc_list(args.self_check_conc)
        out = []
        for c in points:
            if int(round(c)) in fail_at or -1 in fail_at:
                _warn(f"simulated failure at concurrency={int(round(c))} (STUB_FAIL_AT)")
                return 1
            if not args.no_traffic:
                push_traffic(base, params)
            tr = traffic_base(cfg, args.traffic_url)
            _warn(f"self-check conc={c:g} out_len={params['output_len']:g} "
                  f"rows={params['data_num']} in_len={params['input_len']} "
                  f"rr={params['repeat_rate']:.2f} fmt={fmt} traffic={tr or '-'}")
            out.append(build_summary(c, params, fmt,
                                     os.path.join(os.path.abspath(work_dir), f"selfcheck_c{int(round(c))}")))
        sys.stdout.write("\n".join(out))
        sys.stdout.write("ais_bench perf run finished\n")
        sys.stdout.flush()
        return 0

    c = params["concurrency"]
    _warn(f"conc={c:g} out_len={params['output_len']:g} rows={params['data_num']} "
          f"in_len={params['input_len']} rr={params['repeat_rate']:.2f} "
          f"cfg={cfg_path} dataset={params['dataset'] or '-'} fmt={fmt}")

    if int(round(c)) in fail_at or -1 in fail_at:
        _warn(f"simulated failure at concurrency={int(round(c))} (STUB_FAIL_AT)")
        return 1

    if not args.no_traffic:
        push_traffic(base, params)

    exp_folder = os.path.join(os.path.abspath(work_dir),
                              time.strftime("%Y%m%d_%H%M%S"))
    text = build_summary(c, params, fmt, exp_folder)
    sys.stdout.write(text)
    sys.stdout.write("ais_bench perf run finished\n")
    sys.stdout.flush()

    try:
        os.makedirs(exp_folder, exist_ok=True)
        with open(os.path.join(exp_folder, "log.txt"), "w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError as exc:
        _warn(f"could not persist log to {exp_folder}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
