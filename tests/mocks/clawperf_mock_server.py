"""Mock LLM API server for ClawPerf testing.

Simulates realistic LLM serving behavior:
- Token-by-token streaming with configurable inter-token delay (TPOT)
- Configurable first-token delay (TTFT)
- Respects max_tokens and ignore_eos from request
- OpenAI-compatible API format
- vLLM-compatible /metrics Prometheus endpoint with prefix cache counters
- Trie-based prefix cache simulation for realistic multi-turn hit rates

--- vendored copy note (aisbench-sla) -----------------------------------------
来源：ClawPerf 的 src/clawperf/mock_server.py（Apache-2.0，全文见同目录
LICENSE.ClawPerf）。相对上游**唯一改动**：新增「负载相关 TTFT 模型」
（--free-concurrency / --queue-ms，见 _effective_ttft）——上游各请求 TTFT 恒定，
真实压测中跑不出延迟拐点，无法验证 SLA 搜索的阶梯失败与二分分支；
本副本可按在途请求数线性抬高 TTFT，制造可控拐点。其余行为与上游一致。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
import uuid
from collections import OrderedDict
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

app = FastAPI(title="ClawPerf Mock LLM", version="2.0.0")

SUPPORTED_MODELS = ["gpt-3.5-turbo", "qwen3-32b", "claude-3-opus", "deepseek-r1"]

# ── Defaults ──
TTFT_MS = 500
TPOT_MS = 10
CHARS_PER_TOKEN = 4

# ── 负载相关 TTFT 模型（本副本新增；默认 0 = 关闭，行为同上游）──
# 在途请求不超过 LOAD_FREE 时 TTFT 为基准值；超出的每个请求按 LOAD_QUEUE_MS
# 线性抬高 TTFT，模拟服务端排队，用于制造可复现的 SLA 拐点。
LOAD_FREE = 0
LOAD_QUEUE_MS = 0.0


def _effective_ttft(base_ms: float) -> float:
    """按当前在途请求数抬高 TTFT（未开启负载模型时原样返回）。"""
    if LOAD_FREE <= 0 or LOAD_QUEUE_MS <= 0:
        return base_ms
    with _metrics_lock:
        running = _metrics_counters.get("requests_running", 0)
    over = max(0, running - LOAD_FREE)
    return base_ms + LOAD_QUEUE_MS * over

_defaults = {
    "ttft_ms": TTFT_MS,
    "tpot_ms": TPOT_MS,
}

FILLER_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In a world of constant change, the ability to adapt and learn becomes paramount. "
    "Technology continues to reshape our understanding of what is possible, "
    "pushing the boundaries of human achievement further than ever before. "
    "Each new discovery opens doors to possibilities previously unimagined, "
    "creating a cascade of innovation that transforms entire industries. "
    "The integration of artificial intelligence into daily workflows "
    "represents one of the most significant shifts in how we approach problem-solving. "
    "As these systems become more sophisticated, they enable us to tackle "
    "increasingly complex challenges with greater efficiency and precision. "
    "The future holds immense promise for those who embrace continuous learning "
    "and remain open to the transformative potential of emerging technologies. "
)


class PrefixCacheTrie:
    """Trie-based prefix cache simulation for realistic multi-turn hit rates.

    Each message content is hashed into a "chunk". Walking the trie with a
    request's message sequence finds the longest cached prefix — matching how
    vLLM's KV-block prefix cache works at the semantic level.

    Eviction is LRU over *complete request paths*: every inserted sequence is
    tracked with a reference count per node, so shared prefixes survive until no
    live sequence references them. This keeps the hit-rate simulation correct
    even as the cache turns over.
    """

    def __init__(self, max_prefixes: int = 200):
        self._root: dict = {}
        self._token_counts: dict = {}  # chunk_hash -> estimated tokens
        self._max_prefixes = max_prefixes
        # Inserted sequences keyed by their chunk-hash tuple, in insertion order
        # (the head is the least-recently-inserted / eviction candidate).
        self._sequences: "OrderedDict[tuple, list]" = OrderedDict()

    def query(self, chunks: list[tuple[str, int]]) -> int:
        """Walk the trie and return the number of matched tokens."""
        node = self._root
        matched_tokens = 0
        for chunk_hash, tok_count in chunks:
            child = node.get(chunk_hash)
            if child is None:
                break
            node = child
            matched_tokens += tok_count
        return matched_tokens

    def insert(self, chunks: list[tuple[str, int]]) -> bool:
        """Insert a full prompt sequence. Returns True if eviction occurred."""
        seq_key = tuple(h for h, _ in chunks)

        # Build / extend the path, bumping the per-node reference count so shared
        # prefixes stay alive while any sequence still references them.
        node = self._root
        for chunk_hash, tok_count in chunks:
            self._token_counts[chunk_hash] = tok_count
            child = node.get(chunk_hash)
            if child is None:
                child = {"_refs": 0}
                node[chunk_hash] = child
            child["_refs"] += 1
            node = child

        # (Re)inserting the same sequence refreshes its recency.
        if seq_key in self._sequences:
            self._sequences.move_to_end(seq_key)
        else:
            self._sequences[seq_key] = [h for h, _ in chunks]

        evicted = False
        while len(self._sequences) > self._max_prefixes:
            self._evict_oldest()
            evicted = True
        return evicted

    def _evict_oldest(self):
        """Drop the oldest inserted sequence, pruning nodes whose refcount hits 0."""
        seq_key, chunk_hashes = self._sequences.popitem(last=False)
        node = self._root
        path = []  # (parent_dict, chunk_hash, child_node)
        for chunk_hash in chunk_hashes:
            child = node.get(chunk_hash)
            if child is None:
                break  # already partially gone
            path.append((node, chunk_hash, child))
            node = child
        # Decrement refcounts in reverse so we can delete emptied nodes.
        for parent, chunk_hash, child in reversed(path):
            child["_refs"] -= 1
            if child["_refs"] <= 0:
                del parent[chunk_hash]


# ── Metrics counters (thread-safe) ──
_metrics_lock = threading.Lock()
_metrics_counters = {
    "prefix_cache_queries": 0,
    "prefix_cache_evictions": 0,
    "prefix_cache_hit_tokens": 0,
    "prefix_cache_query_tokens": 0,
    "external_prefix_cache_queries": 0,
    "external_prefix_cache_hit_tokens": 0,
    "external_prefix_cache_query_tokens": 0,
    "requests_total": 0,
    "requests_running": 0,
    "requests_waiting": 0,
    "prompt_tokens_total": 0,
    "generation_tokens_total": 0,
}
_hbm_trie = PrefixCacheTrie()
_ext_trie = PrefixCacheTrie()


def _content_to_text(content) -> str:
    """Normalize a message's content to a string.

    OpenAI allows content to be null or an array of content parts
    (e.g. [{"type":"text","text":"hi"}]); both would otherwise crash the
    trie's hash()/len() logic.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Concatenate the text field of each part (multimodal-safe).
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(str(p.get("text", "")))
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


BLOCK_CHARS = 64  # ~16 tokens at 4 chars/token, matching vLLM's default block_size


def _messages_to_chunks(messages: list) -> list[tuple[str, int]]:
    """Convert a messages list into trie chunks: (block_hash, estimated_tokens).

    Each message's text is split into BLOCK_CHARS-sized sub-chunks so that a
    measure request sharing only a leading prefix with a prefill request still
    partially matches (mimics vLLM's block-level KV cache, not message-level).
    Small messages (< one block) collapse to a single chunk, preserving prior
    behavior for short test fixtures.
    """
    chunks = []
    for m in messages:
        text = _content_to_text(m.get("content", ""))
        if not text:
            chunks.append((hash(""), 1))
            continue
        for i in range(0, len(text), BLOCK_CHARS):
            block = text[i : i + BLOCK_CHARS]
            chunks.append((hash(block), _estimate_tokens(block)))
    return chunks


def _generate_content(messages: list, max_tokens: int) -> str:
    """Generate filler content targeting ~max_tokens tokens.

    Always caps at the requested length: max_tokens=0 yields "", and small
    max_tokens no longer over-generates a 50-char seed.
    """
    target_chars = max(0, max_tokens) * CHARS_PER_TOKEN
    if target_chars == 0:
        return ""
    # Seed with a slice of the last user message (mimics echo), then pad with
    # filler text until the target character count is reached.
    seed = (_content_to_text(messages[-1].get("content", ""))[:50]) if messages else "Hello"
    if len(seed) >= target_chars:
        return seed[:target_chars]
    filler_needed = target_chars - len(seed)
    cycles = (filler_needed // len(FILLER_TEXT)) + 1
    return (seed + " " + (FILLER_TEXT * cycles))[:target_chars]


def _update_metrics_on_request(messages: list, model: str):
    """Simulate HBM + external prefix cache using trie-based prefix matching.

    Also marks the request as in-flight (``requests_running``) — the caller must
    pair this with :func:`_release_request_slot` once streaming completes.
    """
    with _metrics_lock:
        _metrics_counters["requests_total"] += 1
        _metrics_counters["requests_running"] += 1
        prompt_tokens = sum(_estimate_tokens(_content_to_text(m.get("content", ""))) for m in messages)
        _metrics_counters["prompt_tokens_total"] += prompt_tokens

        chunks = _messages_to_chunks(messages)

        # HBM prefix cache: query trie for longest matching prefix
        _metrics_counters["prefix_cache_queries"] += 1
        _metrics_counters["prefix_cache_query_tokens"] += prompt_tokens
        hit_tokens = _hbm_trie.query(chunks)
        if hit_tokens > 0:
            _metrics_counters["prefix_cache_hit_tokens"] += hit_tokens
        else:
            # External prefix cache: query trie on HBM miss
            _metrics_counters["external_prefix_cache_queries"] += 1
            _metrics_counters["external_prefix_cache_query_tokens"] += prompt_tokens
            ext_hit_tokens = _ext_trie.query(chunks)
            if ext_hit_tokens > 0:
                _metrics_counters["external_prefix_cache_hit_tokens"] += ext_hit_tokens

        # Always insert into both trie instances (simulates vLLM storing all KV blocks after request)
        _ext_trie.insert(chunks)
        evicted = _hbm_trie.insert(chunks)
        if evicted:
            _metrics_counters["prefix_cache_evictions"] += 1


def _release_request_slot():
    """Decrement the in-flight request counter (call when a request finishes)."""
    with _metrics_lock:
        if _metrics_counters["requests_running"] > 0:
            _metrics_counters["requests_running"] -= 1


def _update_metrics_on_complete(output_tokens: int):
    """Update metrics when a request completes."""
    with _metrics_lock:
        _metrics_counters["generation_tokens_total"] += output_tokens


def make_chunk(
    id: str, model: str, content: str = "", finish: bool = False, usage: dict | None = None
) -> str:
    choice = {
        "index": 0,
        "delta": {"content": content} if content else {},
        "finish_reason": "stop" if finish else None,
    }
    data = {
        "id": id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
    }
    if usage is not None:
        data["usage"] = usage
    return f"data: {json.dumps(data)}\n\n"


async def stream_response(
    messages: list,
    model: str,
    max_tokens: int,
    ignore_eos: bool = True,
    ttft_ms: float = TTFT_MS,
    tpot_ms: float = TPOT_MS,
) -> AsyncGenerator[str, None]:
    """Stream tokens one-by-one with realistic TTFT + TPOT delays.

    Wraps the whole body in try/finally so the in-flight request slot is released
    even if the client disconnects mid-stream (Starlette closes the generator,
    triggering the finally via GeneratorExit).
    """
    try:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        content = _generate_content(messages, max_tokens)
        input_tokens = sum(_estimate_tokens(_content_to_text(m.get("content", ""))) for m in messages)
        output_tokens = _estimate_tokens(content)

        # TTFT: wait before first token（负载模型会按在途请求数抬高它）
        await asyncio.sleep(_effective_ttft(ttft_ms) / 1000)

        # Role chunk
        yield make_chunk(chunk_id, model)

        # Token-by-token streaming (1 token ≈ CHARS_PER_TOKEN chars per chunk)
        for i in range(0, len(content), CHARS_PER_TOKEN):
            chunk_content = content[i : i + CHARS_PER_TOKEN]
            yield make_chunk(chunk_id, model, content=chunk_content)
            # TPOT delay between tokens
            await asyncio.sleep(tpot_ms / 1000)

        # Final chunk with finish_reason (no usage here due to evalscope's elif bug)
        yield make_chunk(chunk_id, model, finish=True)

        # Separate usage chunk (evalscope processes usage only when no choices)
        usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        usage_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "usage": usage,
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n"
        yield "data: [DONE]\n\n"

        _update_metrics_on_complete(output_tokens)
    finally:
        _release_request_slot()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    model = data.get("model", "gpt-3.5-turbo")
    stream = data.get("stream", False)
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 10)
    ignore_eos = data.get("ignore_eos", True)
    # Allow per-request override of latency via extra_body
    extra = data.get("extra_body", {}) or {}
    req_ttft = extra.get("mock_ttft_ms", _defaults["ttft_ms"])
    req_tpot = extra.get("mock_tpot_ms", _defaults["tpot_ms"])

    if model not in SUPPORTED_MODELS:
        model = "test-model"

    _update_metrics_on_request(messages, model)

    if stream:
        return StreamingResponse(
            stream_response(messages, model, max_tokens, ignore_eos, req_ttft, req_tpot),
            media_type="text/event-stream",
        )

    # Non-streaming: still simulate TTFT + total time
    try:
        await asyncio.sleep(_effective_ttft(req_ttft) / 1000 + req_tpot * max_tokens / 1000)
        content = _generate_content(messages, max_tokens)
        input_tokens = sum(_estimate_tokens(_content_to_text(m.get("content", ""))) for m in messages)
        output_tokens = _estimate_tokens(content)
        _update_metrics_on_complete(output_tokens)

        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        })
    finally:
        _release_request_slot()


@app.get("/metrics")
async def metrics():
    """vLLM-compatible Prometheus metrics endpoint."""
    with _metrics_lock:
        c = dict(_metrics_counters)

    lines = [
        "# HELP vllm:num_requests_running Number of requests currently running.",
        "# TYPE vllm:num_requests_running gauge",
        f"vllm:num_requests_running{{model_name=\"mock\"}} {c['requests_running']}",
        "",
        "# HELP vllm:num_requests_waiting Number of requests waiting to be processed.",
        "# TYPE vllm:num_requests_waiting gauge",
        f"vllm:num_requests_waiting{{model_name=\"mock\"}} {c['requests_waiting']}",
        "",
        "# HELP vllm:kv_cache_usage_perc Fraction of used KV cache blocks.",
        "# TYPE vllm:kv_cache_usage_perc gauge",
        "vllm:kv_cache_usage_perc{model_name=\"mock\"} 0.0",
        "",
        "# HELP vllm:prefix_cache_queries_total Total prompt tokens queried against the prefix cache.",
        "# TYPE vllm:prefix_cache_queries_total counter",
        f"vllm:prefix_cache_queries_total{{model_name=\"mock\",engine=\"0\"}} {c['prefix_cache_query_tokens']}",
        "# TYPE vllm:prefix_cache_queries_created gauge",
        "vllm:prefix_cache_queries_created{model_name=\"mock\",engine=\"0\"} 1782374641.0006",
        "",
        "# HELP vllm:prefix_cache_hits_total Total prompt tokens reused via the prefix cache.",
        "# TYPE vllm:prefix_cache_hits_total counter",
        f"vllm:prefix_cache_hits_total{{model_name=\"mock\",engine=\"0\"}} {c['prefix_cache_hit_tokens']}",
        "# TYPE vllm:prefix_cache_hits_created gauge",
        "vllm:prefix_cache_hits_created{model_name=\"mock\",engine=\"0\"} 1782374641.0007",
        "",
        "# HELP vllm:prefix_cache_evictions_total Total number of prefix cache evictions.",
        "# TYPE vllm:prefix_cache_evictions_total counter",
        f"vllm:prefix_cache_evictions_total{{model_name=\"mock\",engine=\"0\"}} {c['prefix_cache_evictions']}",
        "",
        "# HELP external_prefix_cache_queries_total Total prompt tokens queried against the external prefix cache.",
        "# TYPE external_prefix_cache_queries_total counter",
        (
            f'external_prefix_cache_queries_total{{model_name="mock",engine="0"}} '
            f"{c['external_prefix_cache_query_tokens']}"
        ),
        "",
        "# HELP external_prefix_cache_hits_total Total prompt tokens reused via the external prefix cache.",
        "# TYPE external_prefix_cache_hits_total counter",
        f"external_prefix_cache_hits_total{{model_name=\"mock\",engine=\"0\"}} {c['external_prefix_cache_hit_tokens']}",
        "",
        "# HELP vllm:num_requests Total number of requests received.",
        "# TYPE vllm:num_requests counter",
        f"vllm:num_requests{{model_name=\"mock\"}} {c['requests_total']}",
        "",
        "# HELP vllm:request_prompt_tokens Total prompt tokens processed.",
        "# TYPE vllm:request_prompt_tokens counter",
        f"vllm:request_prompt_tokens{{model_name=\"mock\"}} {c['prompt_tokens_total']}",
        "",
        "# HELP vllm:request_generation_tokens Total generation tokens produced.",
        "# TYPE vllm:request_generation_tokens counter",
        f"vllm:request_generation_tokens{{model_name=\"mock\"}} {c['generation_tokens_total']}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n")


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": int(time.time()), "owned_by": "mock"}
            for m in SUPPORTED_MODELS
        ],
    }


# ---- 测试用仿真端点（本副本新增）---------------------------------------------
# 用于「无 Docker / 无真实压测」时构造前缀缓存命中率：直接累加与 trie 相同的
# 计数器，供 tests/stub_aisbench.py 这类假压测器推动指标。
@app.post("/__sim__/traffic")
async def sim_traffic(request: Request):
    params = request.query_params

    def _int(name: str) -> int:
        try:
            return int(float(params.get(name, 0) or 0))
        except (TypeError, ValueError):
            return 0

    delta = {
        "prefix_cache_query_tokens": _int("queries"),
        "prefix_cache_hit_tokens": _int("hits"),
        "external_prefix_cache_query_tokens": _int("ext_queries"),
        "external_prefix_cache_hit_tokens": _int("ext_hits"),
    }
    with _metrics_lock:
        for key, value in delta.items():
            _metrics_counters[key] = _metrics_counters.get(key, 0) + value
    return {"ok": True, "applied": delta}


@app.post("/__sim__/reset")
async def sim_reset():
    """清零全部计数器（测试用）。"""
    with _metrics_lock:
        for key in list(_metrics_counters):
            _metrics_counters[key] = 0
    return {"ok": True}


@app.get("/health")
async def health():
    return {"status": "ok"}


def main():
    parser = argparse.ArgumentParser(
        prog="clawperf-mock-server",
        description="Mock LLM API server with realistic latency simulation and vLLM metrics",
    )
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--ttft", type=float, default=500,
                        help="First-token delay in ms (default: 500)")
    parser.add_argument("--tpot", type=float, default=10,
                        help="Per-token output delay in ms (default: 10)")
    parser.add_argument("--free-concurrency", type=int, default=0,
                        help="负载模型：不超过该在途请求数时 TTFT 不抬高（0=关闭）")
    parser.add_argument("--queue-ms", type=float, default=0.0,
                        help="负载模型：超出 free-concurrency 的每个在途请求抬高的 TTFT 毫秒数")
    parser.add_argument("--ready-file", type=str, default="",
                        help="就绪后写入该文件（含 host/port），便于测试脚本等待")
    args = parser.parse_args()

    _defaults["ttft_ms"] = args.ttft
    _defaults["tpot_ms"] = args.tpot
    global LOAD_FREE, LOAD_QUEUE_MS
    LOAD_FREE = max(0, args.free_concurrency)
    LOAD_QUEUE_MS = max(0.0, args.queue_ms)

    if args.ready_file:
        with open(args.ready_file, "w", encoding="utf-8") as handle:
            handle.write(f"{args.host}:{args.port}\n")
        print(f"MOCK_READY {args.host}:{args.port} ttft={args.ttft} tpot={args.tpot} "
              f"free={LOAD_FREE} queue_ms={LOAD_QUEUE_MS}", flush=True)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
