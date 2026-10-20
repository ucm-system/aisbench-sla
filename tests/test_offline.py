#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline test-suite for the ``aisbench-sla`` SLA search CLI.

Runs with the standard library only::

    py -3.11 tests/test_offline.py             # from the project root
    py -3.11 -m unittest discover -s tests -v

Coverage
--------
``MockVllmTests``      mock vLLM: /metrics contract, /__sim__/traffic, OpenAI surface.
``StubAisbenchTests``  fake ais_bench: reads the generated model config, prints an
                       ais_bench-style summary, drives the mock counters, exit codes.
``ParseTests``         ``sla_tool.parse.parse_aisbench_log`` on legacy + ``│`` table
                       fixture logs, and on the stub's real stdout (differential
                       against an independent oracle of the documented rules).
``EndToEndSearchTests`` full ``sla_search.py search`` run against mock + stub.

The stub/mock tests need no product code at all; the parse and end-to-end tests
skip themselves with an explicit reason while ``sla_tool`` / ``sla_search.py``
are still being written.

Env knobs: ``SLA_E2E_TOKENIZER``, ``SLA_E2E_TIMEOUT`` (s), ``SLA_E2E_KEEP=1``,
``SLA_E2E_LOOSE=1`` (downgrade the "search must converge to 12" check to a
warning), ``SLA_E2E_EXTRA_ARGS`` (extra flags appended to the search command).
"""
from __future__ import annotations

import csv
import inspect
import json
import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MOCK_VLLM = HERE / "mock_vllm.py"
STUB_AISBENCH = HERE / "stub_aisbench.py"
SLA_SEARCH = ROOT / "sla_search.py"
TOKENIZER = Path(os.environ.get(
    "SLA_E2E_TOKENIZER",
    r"D:\Vibe_Workspace\AISBench_PrefixTest_Tools\assets\model\Qwen3-0.6B"))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MOCK_MODEL = "mock-model"
TTFT_P90_LIMIT = 340.0          # -> concurrency <= 12
TPOT_P90_LIMIT = 19.0           # -> concurrency <= 18; TTFT is the binding limit
EXPECTED_MAX_OK = 12


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _quote_path(path) -> str:
    """Quoted only when needed: works with shlex, str.split and cmd.exe alike."""
    text = str(path).replace("\\", "/")
    return f'"{text}"' if " " in text else text


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(match.group(0)) if match else None


def _pick(mapping, *names):
    for name in names:
        if isinstance(mapping, dict) and name in mapping:
            return mapping[name]
    return None


# --------------------------------------------------------------------------- #
# independent oracle for the documented log contract
# --------------------------------------------------------------------------- #
# A straight re-implementation of the rules documented for
# sla_tool.parse.parse_aisbench_log (the port of sidecar/app/result_parse.py).
# It is used to prove the stub emits a *parseable* log even while the product
# parser does not exist yet, and later as a differential check against it.
_ORACLE_STAT_ALIASES = {"average": "avg", "median": "p50", "p75": "p75",
                        "p90": "p90", "p99": "p99", "min": "min", "max": "max"}
_ORACLE_KEYS = (
    "ttft_avg_ms", "ttft_p90_ms", "ttft_min_ms", "ttft_max_ms",
    "ttft_p50_ms", "ttft_p75_ms", "ttft_p99_ms",
    "tpot_avg_ms", "tpot_p90_ms", "tpot_min_ms", "tpot_max_ms",
    "tpot_p50_ms", "tpot_p75_ms", "tpot_p99_ms",
    "e2el_avg_ms", "e2el_p90_ms",
    "measured_concurrency", "max_concurrency", "benchmark_duration_s",
    "output_token_throughput", "single_output_throughput",
    "input_token_throughput", "total_token_throughput", "single_total_throughput",
    "prefill_token_throughput", "request_throughput_qps", "request_throughput_qpm",
    "total_requests", "total_input_tokens", "total_output_tokens", "log_dir",
)


def oracle_parse(text: str) -> dict:
    """Reference implementation of the documented parsing rules."""
    out = {key: None for key in _ORACLE_KEYS}
    header: list = []
    seps = "\u2502"  # '│'
    for line in text.splitlines():
        match = re.search(r"Current exp folder:\s*(.+)$", line)
        if match:
            out["log_dir"] = match.group(1).strip()
        if "Performance Parameters" in line:
            header = [c.strip() for c in line.split(seps) if c.strip()]
            continue
        if seps in line and header:
            for row_name, prefix in (("TTFT", "ttft"), ("TPOT", "tpot"), ("E2EL", "e2el")):
                if line.startswith(seps + " " + row_name):
                    cells = [c.strip() for c in line.split(seps) if c.strip()]
                    if cells and cells[0] == row_name:
                        for cname, cval in zip(header[2:], cells[2:]):
                            stat = _ORACLE_STAT_ALIASES.get(cname.lower())
                            num = re.match(r"([0-9.]+)", cval)
                            if stat and num:
                                out[f"{prefix}_{stat}_ms"] = float(num.group(1))
                    break
        for row_name, prefix in (("TTFT", "ttft"), ("TPOT", "tpot"), ("E2EL", "e2el")):
            if row_name in line and "Time To First Token" not in line \
                    and "Time Per Output Token" not in line:
                values = [float(x) for x in re.findall(r"(\d+\.\d+)", line)]
                if values:
                    out[f"{prefix}_avg_ms"] = values[0]
                    out[f"{prefix}_p90_ms"] = values[5] if len(values) > 5 else None
        if "Benchmark Duration" in line:
            nums = re.findall(r"(\d+\.\d+)", line)
            if nums:
                out["benchmark_duration_s"] = float(nums[0]) / 1000.0
        if "Concurrency" in line and "Max Concurrency" not in line:
            nums = re.findall(r"(\d+\.\d+)", line)
            if nums:
                out["measured_concurrency"] = float(nums[0])
        if "Max Concurrency" in line:
            parts = re.findall(r"[\w']+", line)
            if parts:
                out["max_concurrency"] = parts[-1]
        if "Output Token Throughput" in line:
            nums = re.findall(r"(\d+\.\d+)", line)
            if nums:
                out["output_token_throughput"] = float(nums[0])
        if "Input Token Throughput" in line:
            nums = re.findall(r"(\d+\.\d+)", line)
            if nums:
                out["input_token_throughput"] = float(nums[0])
        if "Total Token Throughput" in line:
            nums = re.findall(r"(\d+\.\d+)", line)
            if nums:
                out["total_token_throughput"] = float(nums[0])
        if "InputTokens" in line:
            nums = re.findall(r"(\d+\.?\d*)", line)
            if nums:
                out["total_input_tokens"] = float(nums[0])
        if "OutputTokens" in line:
            nums = re.findall(r"(\d+\.?\d*)", line)
            if nums:
                out["total_output_tokens"] = float(nums[0])
        if "Total Requests" in line and "Request Throughput" not in line:
            nums = re.findall(r"(\d+\.?\d*)", line)
            if nums:
                out["total_requests"] = float(nums[0])
        if "Request Throughput" in line:
            nums = re.findall(r"(\d+\.\d+)", line)
            if nums:
                out["request_throughput_qps"] = float(nums[0])
                out["request_throughput_qpm"] = float(nums[0]) * 60.0
        if "Prefill Token Throughput" in line:
            nums = re.findall(r"(\d+\.\d+)", line)
            if nums:
                out["prefill_token_throughput"] = float(nums[0])
    return out


def load_core_parser():
    """Return (parse_fn, error_message). parse_fn is None when unavailable."""
    try:
        from sla_tool import parse as parse_module  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return None, f"cannot import sla_tool.parse ({type(exc).__name__}: {exc})"
    fn = getattr(parse_module, "parse_aisbench_log", None)
    if fn is None:
        return None, "sla_tool.parse.parse_aisbench_log is missing"
    return fn, None


_CORE_PARSER: dict = {}


def core_parser():
    """Cached (parse_fn, error). Kept off the TestCase class on purpose: a bare
    function stored as a class attribute would become a bound method."""
    if "fn" not in _CORE_PARSER:
        _CORE_PARSER["fn"], _CORE_PARSER["error"] = load_core_parser()
    return _CORE_PARSER["fn"], _CORE_PARSER["error"]


def call_core_parser(fn, path) -> dict:
    """Tolerant adapter: matches the parser's optional kwargs by name and
    accepts either ``(metrics, log_dir)`` or a bare metrics dict."""
    kwargs = {}
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    for name in params:
        if name in ("request_rate", "rr", "rate"):
            kwargs[name] = "0"
        elif name in ("npu_num", "npu", "dp_num", "num_npu"):
            kwargs[name] = 1
    try:
        result = fn(str(path), **kwargs)
    except TypeError:
        result = fn(str(path))
    if isinstance(result, tuple):
        return result[0]
    return result


# --------------------------------------------------------------------------- #
# mock vLLM process helper
# --------------------------------------------------------------------------- #
class MockServer:
    def __init__(self, model_name: str = MOCK_MODEL):
        self.model_name = model_name
        self.port = None
        self.proc = None
        self._log_path = None
        self._log_fh = None

    # -- lifecycle -------------------------------------------------------
    def start(self, timeout: float = 25.0) -> "MockServer":
        self.port = _free_port()
        handle = tempfile.NamedTemporaryFile(prefix="mock_vllm_", suffix=".log", delete=False)
        handle.close()
        self._log_path = handle.name
        self._log_fh = open(self._log_path, "ab")
        self.proc = subprocess.Popen(
            [sys.executable, str(MOCK_VLLM), "--port", str(self.port),
             "--model-name", self.model_name],
            stdout=self._log_fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        deadline = time.time() + timeout
        last_error = None
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"mock_vllm exited early rc={self.proc.returncode}\n{self.tail()}")
            try:
                with urlrequest.urlopen(self.url("/metrics"), timeout=1.0) as resp:
                    if resp.status == 200:
                        return self
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(0.15)
        raise RuntimeError(f"mock_vllm not ready within {timeout}s ({last_error})\n{self.tail()}")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None
        if self._log_path and os.path.exists(self._log_path):
            try:
                os.unlink(self._log_path)
            except OSError:
                pass

    def tail(self, limit: int = 4000) -> str:
        if not self._log_path or not os.path.exists(self._log_path):
            return "<no mock log>"
        try:
            text = Path(self._log_path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"<unreadable mock log: {exc}>"
        return text[-limit:]

    # -- http ------------------------------------------------------------
    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def url(self, path: str) -> str:
        return self.base + path

    def get(self, path: str, timeout: float = 10.0) -> str:
        with urlrequest.urlopen(self.url(path), timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def post(self, path: str, payload=None, timeout: float = 10.0) -> str:
        if payload is None:
            data, headers = b"", {}
        else:
            data = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
        req = urlrequest.Request(self.url(path), data=data, headers=headers, method="POST")
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def metrics(self) -> str:
        return self.get("/metrics")

    def state(self) -> dict:
        return json.loads(self.get("/__sim__/state"))

    def push_traffic(self, queries=0, hits=0, ext_queries=0, ext_hits=0, engine=0) -> dict:
        path = (f"/__sim__/traffic?engine={engine}&queries={queries}&hits={hits}"
                f"&ext_queries={ext_queries}&ext_hits={ext_hits}")
        return json.loads(self.post(path))


def metrics_values(text: str) -> dict:
    """{metric_name: value} for the counter lines the SLA pipeline scrapes."""
    values = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if 'model_name="' not in line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{")[0]
        try:
            values[name] = float(parts[-1])
        except ValueError:
            continue
    return values


# --------------------------------------------------------------------------- #
# stub-ais_bench fixtures
# --------------------------------------------------------------------------- #
# Stand-in for the config the pipeline regenerates before every point; the
# placeholder names mirror the real template (default_api.py in the reference
# tool) after the pipeline substituted its own values.
CONFIG_TEMPLATE = '''from ais_bench.benchmark.models import VLLMCustomAPIChatStream

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm_api_chat_temp',
        path="{model_path}",
        model="{model}",
        api_key="EMPTY",
        request_rate=0,
        retry=2,
        host_ip="{host}",
        host_port={port},
        url="",
        max_out_len={out_len},
        batch_size={concurrency},
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True,
        )
    )
]
'''

CFG_REL = Path("ais_bench/benchmark/configs/models/vllm_api/vllm_api_chat_temp.py")
DATASET_REL = Path("ais_bench/datasets/gsm8k/test.jsonl")


def write_workspace(work_dir: Path, concurrency: int, out_len: int, host: str, port: int,
                    data_num: int = 64, model: str = MOCK_MODEL) -> Path:
    cfg = work_dir / CFG_REL
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(CONFIG_TEMPLATE.format(
        model_path=str(TOKENIZER).replace("\\", "/"), model=model,
        host=host, port=port, out_len=out_len, concurrency=concurrency), encoding="utf-8")
    dataset = work_dir / DATASET_REL
    dataset.parent.mkdir(parents=True, exist_ok=True)
    # 400 chars -> the stub's chars/4 estimator yields a predictable 100 tokens
    with dataset.open("w", encoding="utf-8") as fh:
        for _ in range(data_num):
            fh.write(json.dumps({"question": "q" * 400}) + "\n")
    (dataset.parent / "train.jsonl").write_text("", encoding="utf-8")
    return cfg


def run_stub(work_dir: Path, extra=(), env_extra=None, timeout: float = 90.0,
             with_real_args: bool = True):
    """Invoke the fake ais_bench the way the pipeline would."""
    args = [sys.executable, str(STUB_AISBENCH)]
    if with_real_args:
        args += ["--models", "vllm_api_chat_temp",
                 "--datasets", "gsm8k_gen_0_shot_cot_str_perf",
                 "--mode", "perf", "--summarizer", "default_perf",
                 "--work-dir", str(work_dir), "--debug", "--num-warmups", "0"]
    args += list(extra)
    env = dict(os.environ)
    env.pop("STUB_FAIL_AT", None)
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})
    return subprocess.run(args, cwd=str(ROOT), env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)


# --------------------------------------------------------------------------- #
# 1. mock vLLM
# --------------------------------------------------------------------------- #
class MockVllmTests(unittest.TestCase):
    """The mock must expose exactly the surface the SLA pipeline scrapes."""

    def setUp(self):
        self.server = MockServer().start()
        self.addCleanup(self.server.stop)

    def test_metrics_contract(self):
        text = self.server.metrics()
        for name in ("vllm:prefix_cache_queries_total",
                     "vllm:prefix_cache_hits_total",
                     "external_prefix_cache_queries_total",
                     "external_prefix_cache_hits_total",
                     "vllm:num_requests_running"):
            self.assertIn(name, text, f"missing metric {name}")
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            self.assertIn('engine="0"', line, f"engine label missing: {line}")
            self.assertIn('model_name="mock-model"', line, f"model label missing: {line}")
        # the reference scraper's filter must match our counter lines
        scraped = [ln for ln in text.splitlines()
                   if "model_name" in ln and "prefix_cache_queries_total" in ln]
        self.assertEqual(len(scraped), 2, scraped)  # normal + external queries
        for line in scraped:
            self.assertRegex(line, r'engine="(\d+)"')

    def test_traffic_hook_accumulates_counters(self):
        self.server.push_traffic(queries=1000, hits=900, ext_queries=50, ext_hits=12)
        self.server.push_traffic(queries=500, hits=450, ext_queries=25, ext_hits=6)
        values = metrics_values(self.server.metrics())
        self.assertEqual(values["vllm:prefix_cache_queries_total"], 1500.0)
        self.assertEqual(values["vllm:prefix_cache_hits_total"], 1350.0)
        self.assertEqual(values["external_prefix_cache_queries_total"], 75.0)
        self.assertEqual(values["external_prefix_cache_hits_total"], 18.0)
        state = self.server.state()
        self.assertEqual(state["queries"], [1500])
        self.assertEqual(state["hits"], [1350])

    def test_openai_surface_and_running_gauge(self):
        models = json.loads(self.server.get("/v1/models"))
        self.assertEqual([m["id"] for m in models["data"]], [MOCK_MODEL])
        body = json.loads(self.server.post("/v1/chat/completions", {
            "model": MOCK_MODEL, "messages": [{"role": "user", "content": "hi"}]}))
        self.assertEqual(body["choices"][0]["message"]["role"], "assistant")
        self.assertIn("usage", body)
        # running gauge must return to zero after the request
        self.assertEqual(self.server.state()["running"], [0])
        # streaming variant must terminate with [DONE]
        stream = self.server.post("/v1/chat/completions",
                                  {"model": MOCK_MODEL, "stream": True,
                                   "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn("[DONE]", stream)
        self.assertEqual(self.server.state()["running"], [0])

    def test_reset_clears_counters(self):
        self.server.push_traffic(queries=10, hits=5)
        self.server.post("/v1/models")  # unrelated call must not change counters
        self.assertEqual(metrics_values(self.server.metrics())["vllm:prefix_cache_queries_total"], 10.0)
        self.server.get("/__sim__/reset")
        self.assertEqual(metrics_values(self.server.metrics())["vllm:prefix_cache_queries_total"], 0.0)

    def test_unknown_path_is_404(self):
        with self.assertRaises(urlerror.HTTPError) as ctx:
            self.server.get("/nope")
        self.assertEqual(ctx.exception.code, 404)


# --------------------------------------------------------------------------- #
# 2. fake ais_bench
# --------------------------------------------------------------------------- #
class StubAisbenchTests(unittest.TestCase):
    """The stub must read the pipeline config and print a parseable summary."""

    def setUp(self):
        self.server = MockServer().start()
        self.addCleanup(self.server.stop)
        self.tmp = Path(tempfile.mkdtemp(prefix="stub_aisbench_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.work = self.tmp / "work"
        self.work.mkdir(parents=True)

    def _write(self, concurrency=12, out_len=32, data_num=64):
        return write_workspace(self.work, concurrency, out_len,
                               "127.0.0.1", self.server.port, data_num=data_num)

    def test_reads_generated_config_and_emits_parseable_legacy_log(self):
        cfg = self._write(concurrency=12)
        self.assertTrue(cfg.is_file())
        before = metrics_values(self.server.metrics())
        proc = run_stub(self.work)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        parsed = oracle_parse(proc.stdout)
        self.assertEqual(parsed["ttft_avg_ms"], 228.0)
        self.assertEqual(parsed["ttft_p90_ms"], 100.0 + 20.0 * 12)      # 340
        self.assertEqual(parsed["tpot_avg_ms"], 12.8)
        self.assertEqual(parsed["tpot_p90_ms"], 10.0 + 0.5 * 12)        # 16
        self.assertEqual(parsed["measured_concurrency"], 12.0)
        self.assertEqual(parsed["max_concurrency"], "12")
        self.assertEqual(parsed["total_requests"], 64.0)
        # 400-char questions -> the stub's chars/4 estimate is exactly 100 tokens
        self.assertEqual(parsed["total_input_tokens"], 64.0 * 100)
        self.assertGreater(parsed["benchmark_duration_s"], 0.0)
        self.assertGreater(parsed["output_token_throughput"], 0.0)
        self.assertTrue(parsed["log_dir"], "Current exp folder line missing")
        # the summary must be parseable by the documented p90 column rule
        ttft_line = [ln for ln in proc.stdout.splitlines() if ln.startswith("TTFT")][0]
        self.assertEqual(len(re.findall(r"\d+\.\d+", ttft_line)), 7, ttft_line)
        # and it must have driven the mock's prefix-cache counters
        after = metrics_values(self.server.metrics())
        delta_q = after["vllm:prefix_cache_queries_total"] - before["vllm:prefix_cache_queries_total"]
        delta_h = after["vllm:prefix_cache_hits_total"] - before["vllm:prefix_cache_hits_total"]
        self.assertGreater(delta_q, 0.0, "traffic hook was not called")
        self.assertAlmostEqual(delta_h / delta_q, 0.9, places=2)

    def test_config_concurrency_is_picked_up_per_point(self):
        """The same command must report a different point when the config changes."""
        seen = {}
        for concurrency in (4, 12, 16):
            self._write(concurrency=concurrency)
            proc = run_stub(self.work)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            parsed = oracle_parse(proc.stdout)
            seen[concurrency] = (parsed["ttft_p90_ms"], parsed["tpot_p90_ms"],
                                 parsed["max_concurrency"])
            self.assertEqual(parsed["ttft_p90_ms"], 100.0 + 20.0 * concurrency)
            self.assertEqual(parsed["tpot_p90_ms"], 10.0 + 0.5 * concurrency)
            self.assertEqual(parsed["max_concurrency"], str(concurrency))
        # 4/12 pass the documented SLA, 16 does not
        self.assertLessEqual(seen[4][0], TTFT_P90_LIMIT)
        self.assertLessEqual(seen[12][0], TTFT_P90_LIMIT)
        self.assertGreater(seen[16][0], TTFT_P90_LIMIT)

    def test_table_and_legacy_formats_agree(self):
        self._write(concurrency=12)
        legacy = oracle_parse(run_stub(self.work, ["--log-format", "legacy"]).stdout)
        table_proc = run_stub(self.work, ["--log-format", "table"])
        table_text = table_proc.stdout
        self.assertIn("Performance Parameters", table_text)
        self.assertIn("\u2502", table_text)
        table = oracle_parse(table_text)
        for key in ("ttft_avg_ms", "ttft_p90_ms", "tpot_avg_ms", "tpot_p90_ms"):
            self.assertEqual(legacy[key], table[key], key)
        # table-only format also carries the extra percentiles
        self.assertEqual(table["ttft_min_ms"], 114.0)
        self.assertEqual(table["ttft_max_ms"], 612.0)
        self.assertEqual(table["ttft_p99_ms"], 391.0)

    def test_unknown_arguments_are_tolerated(self):
        self._write(concurrency=8)
        proc = run_stub(self.work, ["--some-future-flag", "value", "--another=1"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(oracle_parse(proc.stdout)["ttft_p90_ms"], 260.0)

    def test_fail_at_marks_point_as_failed(self):
        self._write(concurrency=16)
        before = metrics_values(self.server.metrics())
        proc = run_stub(self.work, env_extra={"STUB_FAIL_AT": "16"})
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertNotIn("TTFT", proc.stdout)
        self.assertIn("simulated failure", proc.stderr)
        after = metrics_values(self.server.metrics())
        self.assertEqual(after["vllm:prefix_cache_queries_total"],
                         before["vllm:prefix_cache_queries_total"],
                         "a failed point must not push traffic")

    def test_self_check_runs_without_any_pipeline(self):
        proc = run_stub(self.tmp / "nowhere", ["--self-check", "--no-traffic",
                                               "--concurrency", "12"],
                        with_real_args=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        parsed = oracle_parse(proc.stdout)
        self.assertEqual(parsed["ttft_p90_ms"], 340.0)
        self.assertEqual(parsed["tpot_p90_ms"], 16.0)


# --------------------------------------------------------------------------- #
# 3. product parser (sla_tool.parse)
# --------------------------------------------------------------------------- #
LEGACY_FIXTURE = """ais_bench perf run
Current exp folder: /tmp/ais_bench/20261020_113000
TTFT (ms)   200.00   100.00   400.00   180.00   220.00   340.00   380.00
TPOT (ms)   12.50   9.00   20.00   12.00   13.00   16.00   18.00
E2EL (ms)   450.00   300.00   900.00   420.00   500.00   700.00   850.00
Total Requests: 160
Max Concurrency: 40
Concurrency (measured): 38.50
Benchmark Duration (ms): 752000.00
Output Token Throughput: 3214.50 tok/s
Input Token Throughput: 99644.50 tok/s
Total Token Throughput: 102859.00 tok/s
Request Throughput: 0.21 req/s
InputTokens: 5242880
OutputTokens: 81920
"""

TABLE_FIXTURE = """ais_bench perf run
[2026-09-20 16:09:47] Current exp folder: D:\\ws\\results\\20260920_160947
\u2502 Performance Parameters   \u2502 Stage   \u2502 Average        \u2502 Min            \u2502 Max            \u2502 Median         \u2502 P75            \u2502 P90            \u2502 P99            \u2502  N  \u2502
\u2502 TTFT                     \u2502 total   \u2502 343.1 ms       \u2502 166.3 ms       \u2502 638.0 ms       \u2502 304.0 ms       \u2502 336.9 ms       \u2502 635.0 ms       \u2502 637.7 ms       \u2502 32  \u2502
\u2502 TPOT                     \u2502 total   \u2502 20.5 ms        \u2502 18.0 ms        \u2502 25.0 ms        \u2502 20.0 ms        \u2502 21.0 ms        \u2502 24.0 ms        \u2502 24.9 ms        \u2502 32  \u2502
\u2502 E2EL                     \u2502 total   \u2502 123.4 ms       \u2502 100.0 ms       \u2502 200.0 ms       \u2502 120.0 ms       \u2502 130.0 ms       \u2502 150.0 ms       \u2502 190.0 ms       \u2502 32  \u2502
Request Throughput (req/s): 11.8726
Prefill Token Throughput: 12554.3482 token/s
Output Token Throughput: 379.9217 token/s
Total Requests: 32
"""


class ParseTests(unittest.TestCase):
    """parse_aisbench_log must accept the legacy rows and the │ table."""

    def setUp(self):
        self.parser, self.error = core_parser()
        if self.parser is None:
            self.skipTest(f"core parser not available yet - {self.error}")

    def _write(self, text: str) -> Path:
        path = Path(tempfile.mkdtemp(prefix="sla_log_")) / "aisbench.log"
        path.write_text(text, encoding="utf-8")
        self.addCleanup(shutil.rmtree, path.parent, True)
        return path

    def _assert_close(self, actual, expected, label):
        self.assertIsNotNone(actual, f"{label} not parsed")
        self.assertAlmostEqual(float(actual), float(expected), places=4, msg=label)

    def test_legacy_lines(self):
        metrics = call_core_parser(self.parser, self._write(LEGACY_FIXTURE))
        self._assert_close(_pick(metrics, "ttft_avg_ms", "ttft_avg"), 200.0, "ttft_avg")
        self._assert_close(_pick(metrics, "ttft_p90_ms", "ttft_p90"), 340.0, "ttft_p90")
        self._assert_close(_pick(metrics, "tpot_avg_ms", "tpot_avg"), 12.5, "tpot_avg")
        self._assert_close(_pick(metrics, "tpot_p90_ms", "tpot_p90"), 16.0, "tpot_p90")
        self._assert_close(_pick(metrics, "measured_concurrency"), 38.5, "measured_concurrency")
        self.assertEqual(str(_pick(metrics, "max_concurrency")), "40")
        self._assert_close(_pick(metrics, "benchmark_duration_s"), 752.0, "duration_s")
        self._assert_close(_pick(metrics, "output_token_throughput"), 3214.50, "output_tp")
        self._assert_close(_pick(metrics, "input_token_throughput"), 99644.50, "input_tp")
        self._assert_close(_pick(metrics, "total_token_throughput"), 102859.00, "total_tp")
        self._assert_close(_pick(metrics, "request_throughput_qps"), 0.21, "qps")
        self._assert_close(_pick(metrics, "total_requests"), 160.0, "total_requests")
        self._assert_close(_pick(metrics, "total_input_tokens"), 5242880.0, "input_tokens")
        self._assert_close(_pick(metrics, "total_output_tokens"), 81920.0, "output_tokens")

    def test_table_rows(self):
        metrics = call_core_parser(self.parser, self._write(TABLE_FIXTURE))
        self._assert_close(_pick(metrics, "ttft_avg_ms", "ttft_avg"), 343.1, "ttft_avg")
        self._assert_close(_pick(metrics, "ttft_p90_ms", "ttft_p90"), 635.0, "ttft_p90")
        self._assert_close(_pick(metrics, "ttft_min_ms"), 166.3, "ttft_min")
        self._assert_close(_pick(metrics, "ttft_max_ms"), 638.0, "ttft_max")
        self._assert_close(_pick(metrics, "ttft_p75_ms"), 336.9, "ttft_p75")
        self._assert_close(_pick(metrics, "ttft_p99_ms"), 637.7, "ttft_p99")
        self._assert_close(_pick(metrics, "tpot_avg_ms", "tpot_avg"), 20.5, "tpot_avg")
        self._assert_close(_pick(metrics, "e2el_avg_ms"), 123.4, "e2el_avg")
        self._assert_close(_pick(metrics, "output_token_throughput"), 379.9217, "output_tp")
        self._assert_close(_pick(metrics, "request_throughput_qps"), 11.8726, "qps")
        self._assert_close(_pick(metrics, "total_requests"), 32.0, "total_requests")

    def test_missing_file_returns_defaults(self):
        missing = Path(tempfile.mkdtemp(prefix="sla_log_missing_")) / "nope.log"
        self.addCleanup(shutil.rmtree, missing.parent, True)
        target = logging.getLogger("sla_tool.parse")
        previous = target.level
        target.setLevel(logging.CRITICAL)  # the core logs an error on purpose
        try:
            metrics = call_core_parser(self.parser, missing)
        finally:
            target.setLevel(previous)
        self.assertIsInstance(metrics, dict)
        sentinel = _pick(metrics, "ttft_p90_ms", "ttft_p90")
        self.assertTrue(sentinel in (-1, None, 0) or float(sentinel) < 0,
                        f"expected a default marker for a missing log, got {sentinel!r}")

    def test_stub_output_matches_oracle(self):
        """Differential: the real parser and the documented rules must agree."""
        tmp = Path(tempfile.mkdtemp(prefix="sla_log_stub_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        for fmt in ("legacy", "table", "both"):
            proc = run_stub(tmp, ["--self-check", "--no-traffic", "--concurrency", "12",
                                  "--log-format", fmt], with_real_args=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            path = tmp / f"stub_{fmt}.log"
            path.write_text(proc.stdout, encoding="utf-8")
            actual = call_core_parser(self.parser, path)
            expected = oracle_parse(proc.stdout)
            for key in ("ttft_avg_ms", "ttft_p90_ms", "tpot_avg_ms", "tpot_p90_ms",
                        "measured_concurrency", "benchmark_duration_s",
                        "total_requests", "output_token_throughput",
                        "input_token_throughput", "total_token_throughput"):
                got = _to_float(_pick(actual, key))
                self.assertIsNotNone(got, f"[{fmt}] parser returned nothing for {key}: "
                                          f"keys={sorted(actual)[:40]}")
                self.assertAlmostEqual(got, float(expected[key]), places=4,
                                       msg=f"[{fmt}] {key}")


# --------------------------------------------------------------------------- #
# 4. end to end: sla_search.py search against mock + stub
# --------------------------------------------------------------------------- #
CORE_READY = SLA_SEARCH.is_file()


@unittest.skipUnless(CORE_READY, "sla_search.py does not exist yet (core still being written)")
class EndToEndSearchTests(unittest.TestCase):
    """Full search run: mock vLLM + fake ais_bench, no NPU, no network."""

    timeout = float(os.environ.get("SLA_E2E_TIMEOUT", "900"))

    # -- shared runner -----------------------------------------------------
    def _run_search(self, extra_env=None, tag="run"):
        """Run the CLI once against mock + stub and load its artifacts."""
        parser, error = core_parser()
        if parser is None:
            self.skipTest(f"core not usable yet - {error}")
        if not TOKENIZER.is_dir():
            self.skipTest(f"tokenizer directory not found: {TOKENIZER}")

        tmp = Path(tempfile.mkdtemp(prefix=f"sla_e2e_{tag}_"))
        keep = os.environ.get("SLA_E2E_KEEP") == "1"
        if keep:
            print(f"\n[e2e] keeping temp workspace: {tmp}")
        else:
            self.addCleanup(shutil.rmtree, tmp, True)
        work, dataset, out = tmp / "work", tmp / "dataset", tmp / "out"
        for folder in (work, dataset, out):
            folder.mkdir(parents=True)

        server = MockServer().start()
        self.addCleanup(server.stop)

        aisbench_cmd = f"{_quote_path(sys.executable)} {_quote_path(STUB_AISBENCH)}"
        cmd = [sys.executable, str(SLA_SEARCH), "search",
               "--host", "127.0.0.1", "--port", str(server.port),
               "--model", MOCK_MODEL, "--model-path", str(TOKENIZER),
               "--work-path", str(work), "--dataset-path", str(dataset),
               "--input-len", "2048", "--output-len", "32", "--data-num", "64",
               "--prefix-num", "1", "--repeat-rate", "90%", "--seed", "1", "--dp", "1",
               "--ttft-p90", str(int(TTFT_P90_LIMIT)), "--tpot-p90", str(int(TPOT_P90_LIMIT)),
               "--start", "4", "--max", "128", "--output-dir", str(out),
               "--aisbench-cmd", aisbench_cmd]
        extra = os.environ.get("SLA_E2E_EXTRA_ARGS", "").strip()
        if extra:
            cmd.extend(shlex.split(extra))

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["AISBENCH_PT_WORK"] = str(work)     # hint used by the reference tool
        env["STUB_TRAFFIC_URL"] = server.base   # only a fallback for the stub
        if extra_env:
            env.update({k: str(v) for k, v in extra_env.items()})

        started = time.time()
        try:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace",
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            self.fail(f"search timed out after {self.timeout}s\n"
                      f"cmd: {' '.join(cmd)}\n{(exc.stdout or '')[-3000:]}")
        elapsed = time.time() - started
        (tmp / "search.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (tmp / "search.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
        print(f"\n[e2e:{tag}] search finished in {elapsed:.1f}s rc={proc.returncode} "
              f"logs: {tmp}")

        self.assertEqual(proc.returncode, 0,
                         f"search failed rc={proc.returncode}\n"
                         f"--- stdout tail ---\n{(proc.stdout or '')[-2000:]}\n"
                         f"--- stderr tail ---\n{(proc.stderr or '')[-3000:]}\n"
                         f"--- mock log tail ---\n{server.tail(1500)}")

        points_csv = out / "sla_points.csv"
        summary_json = out / "sla_summary.json"
        self.assertTrue(points_csv.is_file(),
                        f"{points_csv} missing; out dir contains: "
                        f"{sorted(p.name for p in out.iterdir()) if out.is_dir() else 'n/a'}")
        self.assertTrue(summary_json.is_file(),
                        f"{summary_json} missing; out dir contains: "
                        f"{sorted(p.name for p in out.iterdir()) if out.is_dir() else 'n/a'}")
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        with points_csv.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertTrue(rows, f"sla_points.csv has no probe rows: {points_csv}")
        return types.SimpleNamespace(tmp=tmp, work=work, out=out, proc=proc, server=server,
                                     summary=summary, rows=rows, elapsed=elapsed, cmd=cmd)

    def test_search_converges_to_sla_boundary(self):
        run = self._run_search(tag="boundary")
        summary, rows, out = run.summary, run.rows, run.out

        # -- the SLA boundary -------------------------------------------------
        max_ok = _pick(summary, "max_ok_concurrency", "max_ok_conc",
                       "best_concurrency", "max_passing_concurrency")
        self.assertIsNotNone(
            max_ok, f"summary has no max_ok_concurrency key: {sorted(summary)[:40]}")
        max_ok = int(max_ok)
        self.assertGreaterEqual(max_ok, 8,
                                "search stopped below concurrency 8, which the SLA model passes")
        self.assertLessEqual(max_ok, EXPECTED_MAX_OK,
                             "SLA model cannot let any concurrency above 12 pass "
                             "(ttft_p90 = 100 + 20*c <= 340)")
        if os.environ.get("SLA_E2E_LOOSE") == "1":
            if max_ok != EXPECTED_MAX_OK:
                print(f"[e2e] WARNING: max_ok_concurrency={max_ok}, expected "
                      f"{EXPECTED_MAX_OK} (SLA_E2E_LOOSE=1)")
        else:
            self.assertEqual(max_ok, EXPECTED_MAX_OK,
                             f"expected the search to converge to the largest passing "
                             f"point ({EXPECTED_MAX_OK}): ttft_p90(12)=340<=340 passes, "
                             f"ttft_p90(16)=420>340 fails")

        # -- metrics snapshot diffing actually worked -------------------------
        hit_cols = [c for c in rows[0] if "hit_rate" in c.lower() and "hbm" in c.lower()]
        hit_cols = hit_cols or [c for c in rows[0] if "hit_rate" in c.lower()]
        self.assertTrue(hit_cols,
                        f"no hbm hit-rate column in sla_points.csv: {list(rows[0])}")
        for row in rows:
            values = [v for v in (_to_float(row.get(c)) for c in hit_cols) if v is not None]
            self.assertTrue(values, f"no numeric hit rate in row {row}")
            self.assertGreater(max(values), 0.0,
                               f"probe concurrency={row.get('concurrency')} reports a "
                               f"zero HBM hit rate: the metrics snapshot/diff chain is "
                               f"broken ({ {c: row.get(c) for c in hit_cols} })")

        # -- probe budget ------------------------------------------------------
        budget = _to_float(_pick(summary, "max_probes", "max_probes_used", "probe_budget"))
        if budget is None and isinstance(summary.get("search"), dict):
            budget = _to_float(summary["search"].get("max_probes"))
        kind_col = "kind" if "kind" in rows[0] else None
        probe_rows = [r for r in rows
                      if not kind_col or str(r.get(kind_col) or "").strip().lower() == "probe"]
        probe_rows = probe_rows or rows
        self.assertLessEqual(len(probe_rows), int(budget) if budget else 20,
                             f"{len(probe_rows)} probes recorded, budget={budget}")

        # -- search bookkeeping vs raw measurements ---------------------------
        concurrency_col = next((c for c in rows[0] if c.strip().lower() == "concurrency"),
                               next((c for c in rows[0] if "concurrency" in c.lower()), None))
        distinct = set()
        if concurrency_col:
            distinct = {int(float(r[concurrency_col])) for r in probe_rows
                        if _to_float(r.get(concurrency_col)) is not None}
            self.assertGreater(len(distinct), 1,
                               f"every probe ran at the same concurrency {distinct}: the "
                               f"per-point config rewrite is not reaching the stub")
        ttft_col = next((c for c in rows[0] if "ttft" in c.lower() and "p90" in c.lower()), None)
        tpot_col = next((c for c in rows[0] if "tpot" in c.lower() and "p90" in c.lower()), None)
        if concurrency_col and ttft_col and tpot_col:
            passing = []
            for row in probe_rows:
                conc = _to_float(row.get(concurrency_col))
                ttft = _to_float(row.get(ttft_col))
                tpot = _to_float(row.get(tpot_col))
                if None in (conc, ttft, tpot):
                    continue
                if ttft <= TTFT_P90_LIMIT and tpot <= TPOT_P90_LIMIT:
                    passing.append(int(conc))
            if passing:
                self.assertEqual(max(passing), max_ok,
                                 "max_ok_concurrency disagrees with the recorded points")
                self.assertGreater(max(distinct), max_ok,
                                   "no probe above the boundary was ever run")

    def test_failing_point_is_recorded_without_crashing(self):
        """A point whose ais_bench run exits non-zero must fail the point, not the search."""
        run = self._run_search(tag="failpoint", extra_env={"STUB_FAIL_AT": "16"})
        summary, rows = run.summary, run.rows
        max_ok = _pick(summary, "max_ok_concurrency", "max_ok_conc")
        self.assertIsNotNone(max_ok, f"summary has no max_ok_concurrency: {sorted(summary)[:40]}")
        self.assertEqual(int(max_ok), EXPECTED_MAX_OK,
                         "a hard-failing point at concurrency 16 must not change the "
                         "SLA boundary, which the model puts at 12")

        concurrency_col = next((c for c in rows[0] if "concurrency" in c.lower()), None)
        self.assertIsNotNone(concurrency_col, f"no concurrency column in {list(rows[0])}")
        failing = [r for r in rows if _to_float(r.get(concurrency_col)) == 16.0]
        self.assertTrue(failing, "concurrency 16 was never probed")
        verdict_col = "verdict" if "verdict" in rows[0] else None
        if verdict_col:
            self.assertNotEqual(str(failing[0].get(verdict_col)).strip().upper(), "PASS",
                                f"concurrency 16 was reported as passing: {failing[0]}")
        else:
            self.assertIn("-1", json.dumps(failing[0]),
                          f"failing point shows no missing-data marker: {failing[0]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
