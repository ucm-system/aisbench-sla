"""ais_bench 汇总日志解析。

移植自 AISBench_PrefixTest_Tools/sidecar/app/result_parse.py（本项目作者，已验证）
与参考工具 aisbench_auto_tools_prefix/save_file.py（MIT）的解析规则，支持两种输出：

1. 传统行格式（老版本 ais_bench / default_perf）：
       TTFT    123.45 50.00 900.00 100.00 120.00 150.00 800.00
   —— 第 1 个数为 avg，第 6 个数（index 5）为 P90。
2. 表格格式（AISBench >= 3.x）：
       │ Performance Parameters │ average │ p90 │ ...
       │ TTFT                   │ 123.45 ms │ ...
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

METRIC_DEFAULTS: Dict[str, float] = {
    "ttft_avg_ms": -1, "ttft_p90_ms": -1,
    "tpot_avg_ms": -1, "tpot_p90_ms": -1,
    "e2el_avg_ms": -1, "e2el_p90_ms": -1,
    "ttft_min_ms": -1, "ttft_max_ms": -1, "ttft_median_ms": -1,
    "ttft_p50_ms": -1, "ttft_p75_ms": -1, "ttft_p99_ms": -1,
    "tpot_min_ms": -1, "tpot_max_ms": -1, "tpot_median_ms": -1,
    "tpot_p50_ms": -1, "tpot_p75_ms": -1, "tpot_p99_ms": -1,
    "e2el_p99_ms": -1,
    "benchmark_duration_s": -1,
    "measured_concurrency": -1,
    "max_concurrency": -1,
    "output_token_throughput": -1,
    "single_output_throughput": -1,
    "input_token_throughput": -1,
    "total_token_throughput": -1,
    "single_total_throughput": -1,
    "prefill_token_throughput": -1,
    "request_throughput_qps": -1,
    "request_throughput_qpm": -1,
    "total_requests": -1,
    "total_input_tokens": -1,
    "total_output_tokens": -1,
}

_HEADER_ALIASES = {"average": "avg", "median": "p50", "p75": "p75",
                   "p90": "p90", "p99": "p99", "min": "min", "max": "max"}
_FLOAT_RE = re.compile(r"(\d+\.\d+)")
_NUMBER_RE = re.compile(r"(\d+\.?\d*)")


def parse_aisbench_log(log_path: str, npu_num: int = 1) -> Tuple[Dict[str, float], str]:
    """→ (指标字典, ais_bench 输出目录)。解析失败时字段保持默认值 -1。"""
    result = dict(METRIC_DEFAULTS)
    log_dir = ""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        logger.error("找不到 ais_bench 日志：%s", log_path)
        return result, log_dir
    except OSError as exc:
        logger.error("读取 ais_bench 日志失败：%s", exc)
        return result, log_dir

    header_cells: list[str] = []
    for line in lines:
        match = re.search(r"Current exp folder:\s*(.+)$", line)
        if match:
            log_dir = match.group(1).strip()

        if "Performance Parameters" in line:
            header_cells = [c.strip() for c in line.split("│") if c.strip()]
            continue
        if header_cells:
            for row_name, prefix in (("TTFT", "ttft"), ("TPOT", "tpot"), ("E2EL", "e2el")):
                if line.lstrip().startswith("│ " + row_name):
                    cells = [c.strip() for c in line.split("│") if c.strip()]
                    if cells and cells[0] == row_name:
                        for cname, cval in zip(header_cells[2:], cells[2:]):
                            stat = _HEADER_ALIASES.get(cname.lower())
                            number = _NUMBER_RE.match(cval)
                            if stat and number:
                                result[f"{prefix}_{stat}_ms"] = float(number.group(1))
                    break

        if "TTFT" in line and "Time To First Token" not in line:
            values = [float(v) for v in _FLOAT_RE.findall(line)]
            if values:
                result["ttft_avg_ms"] = values[0]
                if len(values) > 5:
                    result["ttft_p90_ms"] = values[5]
        if "TPOT" in line and "Time Per Output Token" not in line:
            values = [float(v) for v in _FLOAT_RE.findall(line)]
            if values:
                result["tpot_avg_ms"] = values[0]
                if len(values) > 5:
                    result["tpot_p90_ms"] = values[5]
        if "E2EL" in line and "End-to-End Latency" not in line:
            values = [float(v) for v in _FLOAT_RE.findall(line)]
            if values:
                result["e2el_avg_ms"] = values[0]
                if len(values) > 5:
                    result["e2el_p90_ms"] = values[5]

        if "Benchmark Duration" in line:
            numbers = _FLOAT_RE.findall(line)
            if numbers:
                result["benchmark_duration_s"] = float(numbers[0]) / 1000.0
        if "Max Concurrency" in line:
            parts = re.findall(r"[\w']+", line)
            if parts:
                result["max_concurrency"] = parts[-1]
        if "Concurrency" in line and "Max Concurrency" not in line:
            numbers = _FLOAT_RE.findall(line)
            if numbers:
                result["measured_concurrency"] = float(numbers[0])
        if "Output Token Throughput" in line:
            numbers = _FLOAT_RE.findall(line)
            if numbers:
                result["output_token_throughput"] = float(numbers[0])
                result["single_output_throughput"] = float(numbers[0]) / max(1, npu_num)
        if "Input Token Throughput" in line:
            numbers = _FLOAT_RE.findall(line)
            if numbers:
                result["input_token_throughput"] = float(numbers[0])
        if "Prefill Token Throughput" in line:
            numbers = _FLOAT_RE.findall(line)
            if numbers:
                result["prefill_token_throughput"] = float(numbers[0])
        if "Total Token Throughput" in line:
            numbers = _FLOAT_RE.findall(line)
            if numbers:
                result["total_token_throughput"] = float(numbers[0])
                result["single_total_throughput"] = float(numbers[0]) / max(1, npu_num)
        if "InputTokens" in line:
            numbers = _NUMBER_RE.findall(line)
            if numbers:
                result["total_input_tokens"] = float(numbers[0])
        if "OutputTokens" in line:
            numbers = _NUMBER_RE.findall(line)
            if numbers:
                result["total_output_tokens"] = float(numbers[0])
        if "Total Requests" in line and "Request Throughput" not in line:
            numbers = _NUMBER_RE.findall(line)
            if numbers:
                result["total_requests"] = float(numbers[0])
        if "Request Throughput" in line:
            numbers = _FLOAT_RE.findall(line)
            if numbers:
                result["request_throughput_qps"] = float(numbers[0])
                result["request_throughput_qpm"] = float(numbers[0]) * 60
    return result, log_dir


def looks_like_bad_num_warmups(log_path: str) -> bool:
    """日志里是否出现 '--num-warmups' 不被识别（老版本 ais_bench）。"""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return False
    return "unrecognized arguments" in text and "num-warmups" in text
