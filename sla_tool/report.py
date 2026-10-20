"""结果输出：控制台表格 + CSV（可追加）+ JSON 摘要。"""
from __future__ import annotations

import csv
import json
import os
import time
from typing import Dict, Iterable, List

from . import __version__
from .search import PointResult, SearchResult, dry_run_points


def _pct(value) -> str:
    if value is None:
        return "  -  "
    return f"{float(value) * 100:5.2f}%"


def _num(value, digits: int = 0) -> str:
    if value is None or (isinstance(value, (int, float)) and value < 0):
        return "  -  "
    return f"{float(value):.{digits}f}"


def point_line(point: PointResult) -> str:
    hit = point.hit_rate or {}
    return ("  {mark} c={c:<4} TTFT P90 {ttft:>8}  TPOT P90 {tpot:>7}  "
            "out {tput:>8} tok/s  HBM {hbm}  ext {ext}  {reason}").format(
        mark="PASS" if point.ok else "FAIL",
        c=point.concurrency,
        ttft=_num(point.metrics.get("ttft_p90_ms")),
        tpot=_num(point.metrics.get("tpot_p90_ms"), 1),
        tput=_num(point.metrics.get("output_token_throughput"), 1),
        hbm=_pct(hit.get("hbm_hit_rate")),
        ext=_pct(hit.get("ext_hit_rate")),
        reason=point.reason,
    )


HEADERS = ["并发", "判定", "TTFT avg", "TTFT P90", "TPOT avg", "TPOT P90",
           "输出吞吐", "QPS", "请求数", "实测并发", "HBM命中率", "ext命中率",
           "预热命中率", "峰值在途", "耗时(s)"]


def print_table(points: Iterable[PointResult]) -> None:
    rows = []
    for point in points:
        hit = point.hit_rate or {}
        rows.append([
            str(point.concurrency),
            "PASS" if point.ok else "FAIL",
            _num(point.metrics.get("ttft_avg_ms")),
            _num(point.metrics.get("ttft_p90_ms")),
            _num(point.metrics.get("tpot_avg_ms"), 1),
            _num(point.metrics.get("tpot_p90_ms"), 1),
            _num(point.metrics.get("output_token_throughput"), 1),
            _num(point.metrics.get("request_throughput_qps"), 2),
            _num(point.metrics.get("total_requests")),
            _num(point.metrics.get("measured_concurrency"), 1),
            _pct(hit.get("hbm_hit_rate")),
            _pct(hit.get("ext_hit_rate")),
            _pct(point.warmup_hit_rate),
            _num(point.gauges.get("max_running")),
            _num(point.elapsed_s, 1),
        ])
    print("\n" + "=" * 118)
    print("探测结果明细")
    print("=" * 118)
    print("".join(f"{h:<11}" for h in HEADERS))
    print("-" * 118)
    for row in rows:
        print("".join(f"{cell:<11}" for cell in row))
    print("=" * 118)


def throughput_summary(points: Iterable[PointResult]) -> Dict[str, object]:
    """挑出「最高输出吞吐」与「达标点里的最高吞吐」。

    需求场景是「找最大并发 + 最高吞吐」，二者可能不在同一个点：
    吞吐通常随并发单调上升，但越过 SLA 后的点不再可用，故分开报告。
    """
    probes = [p for p in points if p.kind != "warmup"]
    peak = best_ok = None
    for point in probes:
        value = point.metrics.get("output_token_throughput", -1)
        if value is None or value < 0:
            continue
        record = {"value": float(value), "concurrency": point.concurrency,
                  "ok": bool(point.ok),
                  "ttft_p90_ms": point.metrics.get("ttft_p90_ms", -1),
                  "tpot_p90_ms": point.metrics.get("tpot_p90_ms", -1)}
        if peak is None or record["value"] > peak["value"]:
            peak = record
        if point.ok and (best_ok is None or record["value"] > best_ok["value"]):
            best_ok = record
    summary: Dict[str, object] = {}
    if peak:
        summary["peak"] = peak
    if best_ok:
        summary["best_within_sla"] = best_ok
    return summary


def print_summary(result: SearchResult, cfg) -> None:
    print()
    print("=" * 118)
    print(f"结论：{result.note}")
    if result.max_ok is not None:
        print(f"  SLA 内最大并发：{result.max_ok}")
    throughput = throughput_summary(result.points)
    peak = throughput.get("peak") or {}
    best_ok = throughput.get("best_within_sla") or {}
    if peak:
        flag = "达标" if peak.get("ok") else "不达标"
        print(f"  最高输出吞吐：{peak['value']:.1f} tok/s @ 并发={peak['concurrency']}"
              f"（{flag}：TTFT P90 {_num(peak.get('ttft_p90_ms'))}ms，"
              f"TPOT P90 {_num(peak.get('tpot_p90_ms'), 1)}ms）")
    if best_ok and (not peak or best_ok["concurrency"] != peak["concurrency"]):
        print(f"  SLA 内最高吞吐：{best_ok['value']:.1f} tok/s @ 并发={best_ok['concurrency']}")
    print(f"  状态：{result.status}（退出码 {result.exit_code}）")
    for warning in result.warnings:
        print(f"  注意：{warning}")
    if result.dataset_info:
        print(f"  数据集：{result.dataset_info.get('dataset_path', '')}")
    print("=" * 118)


def print_dry_run(cfg) -> None:
    points = dry_run_points(cfg)
    print("dry-run：将按以下阶梯探测（二分区间由实测结果决定）")
    print(f"  并发点：{' -> '.join(str(p) for p in points)}"
          + (" -> …" if points and points[-1] * 2 <= cfg.max_concurrency else ""))
    print(f"  预检：{'开（并发=1）' if cfg.start > 1 else '关（start=1 时首个探针即预检）'}")
    print(f"  前缀预热：{cfg.prefix_warmup}")
    print(f"  单点：数据集={cfg.dataset or '(需生成)'} work_path={cfg.work_path}")
    print(f"  输出目录：{cfg.output_dir}")


# --------------------------------------------------------------------- 落盘
CSV_FIELDS = [
    "kind", "concurrency", "seed", "output_len", "verdict", "reason",
    "ttft_avg_ms", "ttft_p90_ms", "tpot_avg_ms", "tpot_p90_ms", "e2el_p90_ms",
    "output_token_throughput", "request_throughput_qps", "total_requests",
    "measured_concurrency", "hbm_hit_rate", "ext_hit_rate", "hit_rate_note",
    "warmup_hit_rate", "max_running", "elapsed_s", "log_path",
]


def write_csv(out_dir: str, points: List[PointResult]) -> str:
    path = os.path.join(out_dir, "sla_points.csv")
    rows = [{key: point.as_row().get(key) for key in CSV_FIELDS} for point in points]
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
        columns = list(CSV_FIELDS)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in existing:
                writer.writerow(row)
            for row in rows:
                writer.writerow(row)
    else:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return path


def write_summary(out_dir: str, cfg, result: SearchResult) -> str:
    path = os.path.join(out_dir, "sla_summary.json")
    payload: Dict[str, object] = {
        "tool": f"aisbench-sla {__version__}",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": result.status,
        "exit_code": result.exit_code,
        "note": result.note,
        "error": result.error,
        "max_ok_concurrency": result.max_ok,
        "sla": {"ttft_p90_ms": cfg.ttft_p90, "tpot_p90_ms": cfg.tpot_p90},
        "search": {"start": cfg.start, "max": cfg.max_concurrency,
                   "tolerance": cfg.tolerance, "max_probes": cfg.max_probes,
                   "max_bisect": cfg.max_bisect,
                   "prefix_warmup": cfg.prefix_warmup,
                   "seed_per_point": cfg.seed_per_point,
                   "reset_cache": cfg.reset_cache},
        "target": {"base_url": cfg.base_url(), "model": cfg.model,
                   "pods": cfg.metrics_pods()},
        "workload": {"input_len": cfg.input_len, "output_len": cfg.output_len,
                     "data_num": cfg.data_num, "request_rate": cfg.request_rate,
                     "prefix_num": cfg.prefix_num, "repeat_rate": cfg.repeat_rate,
                     "seed": cfg.seed, "dp": cfg.dp,
                     "dataset_type": cfg.dataset_type},
        "dataset": result.dataset_info,
        "duration_s": round(result.finished_at - result.started_at, 1),
        "warnings": list(result.warnings),
        "throughput": throughput_summary(result.points),
        "points": [point.as_row() for point in result.points],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    return path
