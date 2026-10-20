#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline mock of a vLLM OpenAI-compatible HTTP server (stdlib only).

Part of the ``aisbench-sla`` offline test-suite. It exposes the *minimum*
surface the SLA search pipeline needs, so the whole benchmark chain can run
with no NPU, no real vLLM and no network:

======================  =====================================================
``GET  /metrics``       Prometheus text with the vLLM prefix-cache counters
                        (``engine`` / ``model_name`` labels), exactly the
                        metric names ``cal_prefix_hit_rate.py`` scrapes.
``POST /__sim__/traffic`` test-only hook: bump the counters by the given
                        deltas, the way a finished benchmark run would.
``GET  /__sim__/state`` JSON snapshot of every counter (debug / assertions).
``GET  /__sim__/reset`` zero every counter.
``GET  /v1/models``     OpenAI-style model list (liveness / model probe).
``POST /v1/chat/completions``  minimal completion (stream and non-stream).
======================  =====================================================

Counters are guarded by a lock; the server is threaded so parallel probes work.

Usage::

    python tests/mock_vllm.py --port 8099
    python tests/mock_vllm.py --port 0          # OS picks a free port
                                                  # (prints MOCK_VLLM_READY port=N)
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DEFAULT_MODEL_NAME = "mock-model"

# Metric names are the ones the reference scraper looks for; do not rename.
M_PREFIX_QUERIES = "vllm:prefix_cache_queries_total"
M_PREFIX_HITS = "vllm:prefix_cache_hits_total"
M_EXT_QUERIES = "external_prefix_cache_queries_total"
M_EXT_HITS = "external_prefix_cache_hits_total"
M_RUNNING = "vllm:num_requests_running"


class MockState:
    """Thread-safe counter bank, one slot per simulated DP engine."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, engines: int = 1) -> None:
        self.model_name = model_name
        self.engines = max(1, int(engines))
        self.lock = threading.Lock()
        n = self.engines
        self.queries = [0] * n
        self.hits = [0] * n
        self.ext_queries = [0] * n
        self.ext_hits = [0] * n
        self.running = [0] * n
        self.completions = 0

    # -- helpers ---------------------------------------------------------
    def _slot(self, engine) -> int:
        try:
            idx = int(engine)
        except (TypeError, ValueError):
            idx = 0
        if idx < 0 or idx >= self.engines:
            raise ValueError(f"engine {engine!r} out of range (engines={self.engines})")
        return idx

    def add_traffic(self, engine=0, queries=0, hits=0, ext_queries=0, ext_hits=0) -> dict:
        with self.lock:
            i = self._slot(engine)
            self.queries[i] += int(queries)
            self.hits[i] += int(hits)
            self.ext_queries[i] += int(ext_queries)
            self.ext_hits[i] += int(ext_hits)
            return self._snapshot_locked()

    def reset(self) -> dict:
        with self.lock:
            n = self.engines
            self.queries = [0] * n
            self.hits = [0] * n
            self.ext_queries = [0] * n
            self.ext_hits = [0] * n
            self.running = [0] * n
            self.completions = 0
            return self._snapshot_locked()

    def enter(self, engine=0) -> None:
        with self.lock:
            self.running[self._slot(engine)] += 1

    def leave(self, engine=0) -> None:
        with self.lock:
            self.running[self._slot(engine)] = max(0, self.running[self._slot(engine)] - 1)

    def note_completion(self) -> None:
        with self.lock:
            self.completions += 1

    def _snapshot_locked(self) -> dict:
        return {
            "model_name": self.model_name,
            "engines": self.engines,
            "queries": list(self.queries),
            "hits": list(self.hits),
            "ext_queries": list(self.ext_queries),
            "ext_hits": list(self.ext_hits),
            "running": list(self.running),
            "completions": self.completions,
        }

    def snapshot(self) -> dict:
        with self.lock:
            return self._snapshot_locked()


def render_metrics(state: MockState) -> str:
    """Prometheus exposition text for the prefix-cache counters."""
    snap = state.snapshot()
    model = snap["model_name"]
    out: list = []

    def series(name: str, mtype: str, help_text: str, values, suffix: str = "") -> None:
        out.append(f"# HELP {name} {help_text}")
        out.append(f"# TYPE {name} {mtype}")
        for eng, value in enumerate(values):
            out.append(f'{name}{{engine="{eng}",model_name="{model}"{suffix}}} {value}')

    series(M_PREFIX_QUERIES, "counter",
           "Number of prefix cache queries (tokens) processed by the local HBM cache.",
           snap["queries"])
    series(M_PREFIX_HITS, "counter",
           "Number of prefix cache hits (tokens) served from the local HBM cache.",
           snap["hits"])
    # NOTE: no "vllm:" prefix on the external counters - the reference scraper
    # classifies a line as *external* when it contains this exact substring.
    series(M_EXT_QUERIES, "counter",
           "Number of queries (tokens) handled by the external prefix cache.",
           snap["ext_queries"])
    series(M_EXT_HITS, "counter",
           "Number of hits (tokens) served by the external prefix cache.",
           snap["ext_hits"])
    series(M_RUNNING, "gauge",
           "Number of requests currently running.", snap["running"])
    return "\n".join(out) + "\n"


class _Handler(BaseHTTPRequestHandler):
    server_version = "mock-vllm/1.0"
    protocol_version = "HTTP/1.0"  # no keep-alive: connection close ends the body

    # -- plumbing --------------------------------------------------------
    @property
    def state(self) -> MockState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):  # noqa: D102 - silence by default
        if getattr(self.server, "verbose", False):  # type: ignore[attr-defined]
            sys.stderr.write("[mock-vllm] " + (fmt % args) + "\n")

    def _body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        return self.rfile.read(length) if length > 0 else b""

    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, code: int, payload) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    # -- routing ---------------------------------------------------------
    def do_GET(self):  # noqa: N802
        self._route()

    def do_POST(self):  # noqa: N802
        self._route()

    def _route(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        try:
            if path == "/metrics":
                self._send(200, render_metrics(self.state).encode("utf-8"),
                           "text/plain; version=0.0.4; charset=utf-8")
            elif path == "/__sim__/traffic":
                self._handle_traffic(query)
            elif path == "/__sim__/state":
                self._send_json(200, self.state.snapshot())
            elif path == "/__sim__/reset":
                self._send_json(200, self.state.reset())
            elif path in ("/v1/models", "/models"):
                self._send_json(200, {
                    "object": "list",
                    "data": [{"id": self.state.model_name, "object": "model",
                              "created": 0, "owned_by": "mock"}],
                })
            elif path == "/health" or path == "/":
                self._send_json(200, {"status": "ok", "model": self.state.model_name})
            elif path == "/v1/chat/completions":
                self._handle_chat()
            else:
                self._send_json(404, {"error": {"message": f"unknown path {path}",
                                                "type": "invalid_request_error"}})
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001 - never kill the server thread
            try:
                self._send_json(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            except Exception:  # noqa: BLE001
                pass

    # -- endpoints -------------------------------------------------------
    def _handle_traffic(self, query) -> None:
        def num(name: str) -> int:
            values = query.get(name) or []
            if not values:
                return 0
            try:
                return max(0, int(float(values[0])))
            except (TypeError, ValueError):
                return 0

        engine = (query.get("engine") or ["0"])[0]
        try:
            snap = self.state.add_traffic(
                engine=engine,
                queries=num("queries"),
                hits=num("hits"),
                ext_queries=num("ext_queries"),
                ext_hits=num("ext_hits"),
            )
        except ValueError as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
            return
        self._send_json(200, {"ok": True, "engine": str(engine), **snap})

    def _handle_chat(self) -> None:
        raw = self._body()
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        model = payload.get("model") or self.state.model_name
        stream = bool(payload.get("stream"))
        self.state.enter(0)
        try:
            # Small transient so the running gauge is observable as >0.
            time.sleep(0.02)
            created = int(time.time())
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                def chunk(delta: dict, finish=None) -> None:
                    frame = {
                        "id": "chatcmpl-mock", "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                    self.wfile.write(b"data: " + json.dumps(frame).encode("utf-8") + b"\n\n")
                    self.wfile.flush()

                chunk({"role": "assistant", "content": ""})
                chunk({"content": "mock"})
                chunk({}, finish="stop")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                self._send_json(200, {
                    "id": "chatcmpl-mock", "object": "chat.completion",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "mock"}}],
                    "usage": {"prompt_tokens": 16, "completion_tokens": 4, "total_tokens": 20},
                })
        finally:
            self.state.leave(0)
            self.state.note_completion()


def build_server(host: str, port: int, model_name: str = DEFAULT_MODEL_NAME,
                 engines: int = 1, verbose: bool = False) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _Handler)
    server.daemon_threads = True
    server.state = MockState(model_name=model_name, engines=engines)  # type: ignore[attr-defined]
    server.verbose = verbose  # type: ignore[attr-defined]
    return server


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Offline mock vLLM OpenAI server (stdlib only).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8099, help="0 lets the OS pick a free port")
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--engines", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    server = build_server(args.host, args.port, args.model_name, args.engines, args.verbose)
    host, port = server.server_address[0], server.server_address[1]
    print(f"MOCK_VLLM_READY host={host} port={port} model={args.model_name}", flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
