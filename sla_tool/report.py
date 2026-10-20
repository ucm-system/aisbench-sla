"""结果输出：控制台表格 + CSV（可追加）+ JSON 摘要。"""
from __future__ import annotations

import csv
import json
import os
import time
from typing import Dict, Iterable, List

from . import __version__, console
from .search import ALL_METRIC_KEYS, PointResult, SearchResult, dry_run_points


def _pct(value) -> str:
    if value is None:
        return "  -  "
    return f"{float(value) * 100:5.2f}%"


def _num(value, digits: int = 0) -> str:
    if value is None or (isinstance(value, (int, float)) and value < 0):
        return "  -  "
    return f"{float(value):.{digits}f}"


def sla_columns(cfg) -> List[tuple]:
    """配置里参与判定的 (指标键, 表头, 是否整数) 列表 —— 表格按此动态出列。"""
    from .sla_spec import METRIC_LABELS
    columns = []
    for rule in cfg.sla_rules():
        unit_ms = rule.metric != "output_token_throughput"
        header = f"{METRIC_LABELS.get(rule.metric, rule.metric)} {rule.agg.upper()}"
        columns.append((rule.metrics_key, header, unit_ms))
    return columns


def point_line(point: PointResult, cfg=None) -> str:
    hit = point.hit_rate or {}
    gated = []
    if cfg is not None:
        for key, header, unit_ms in sla_columns(cfg):
            gated.append(f"{header} {_num(point.metrics.get(key), 0 if unit_ms else 1)}")
    else:
        gated = [f"TTFT P90 {_num(point.metrics.get('ttft_p90_ms'))}",
                 f"TPOT P90 {_num(point.metrics.get('tpot_p90_ms'), 1)}"]
    mark = console.verdict(point.ok, "PASS" if point.ok else "FAIL")
    reason = point.reason if point.ok else console.c(point.reason, console.YELLOW)
    return ("  {mark} c={c:<4} {gated}  out {tput:>8} tok/s  HBM {hbm}  ext {ext}  {reason}").format(
        mark=mark,
        c=point.concurrency,
        gated="  ".join(gated),
        tput=_num(point.metrics.get("output_token_throughput"), 1),
        hbm=_pct(hit.get("hbm_hit_rate")),
        ext=_pct(hit.get("ext_hit_rate")),
        reason=reason,
    )


def print_table(points: Iterable[PointResult], cfg=None) -> None:
    gated = sla_columns(cfg) if cfg is not None else [
        ("ttft_p90_ms", "TTFT P90", True), ("tpot_p90_ms", "TPOT P90", True)]
    headers = ["并发", "判定"] + [header for _, header, _ in gated] + [
        "输出吞吐", "QPS", "请求数", "实测并发", "HBM命中率", "ext命中率",
        "预热命中率", "峰值在途", "耗时(s)"]
    rows = []
    for point in points:
        hit = point.hit_rate or {}
        row = [str(point.concurrency), "PASS" if point.ok else "FAIL"]
        for key, _, unit_ms in gated:
            row.append(_num(point.metrics.get(key), 0 if unit_ms else 1))
        row += [
            _num(point.metrics.get("output_token_throughput"), 1),
            _num(point.metrics.get("request_throughput_qps"), 2),
            _num(point.metrics.get("total_requests")),
            _num(point.metrics.get("measured_concurrency"), 1),
            _pct(hit.get("hbm_hit_rate")),
            _pct(hit.get("ext_hit_rate")),
            _pct(point.warmup_hit_rate),
            _num(point.gauges.get("max_running")),
            _num(point.elapsed_s, 1),
        ]
        rows.append(row)
    width = max(118, 13 * len(headers))
    console.rule("探测结果明细", char="=", width=width, codes=(console.BOLD, console.BLUE))
    print(console.c("".join(console.pad(h, 13) for h in headers), console.BOLD))
    print(console.c("-" * width, console.DIM))
    for row in rows:
        line = "".join(console.pad(cell, 13) for cell in row)
        if row[1] == "PASS":
            line = line.replace("PASS", console.verdict(True, "PASS"), 1)
        elif row[1] == "FAIL":
            line = line.replace("FAIL", console.verdict(False, "FAIL"), 1)
        print(line)
    print(console.c("=" * width, console.BOLD))


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
    ok = result.exit_code == 0
    title = "结论" + ("（找到可用并发）" if ok else "（未找到可用并发）")
    console.banner(title, char="=", codes=((console.BOLD, console.GREEN) if ok
                                          else (console.BOLD, console.YELLOW)))
    if result.error:
        print(console.c(f"错误：{result.error}", console.BOLD, console.RED))
    print(f"结论：{result.note}")
    if cfg is not None and cfg.sla:
        print("  判定口径：" + "；".join(rule.label() for rule in cfg.sla_rules()))
    if result.max_ok is not None:
        print(f"  SLA 内最大并发：{result.max_ok}")
    throughput = throughput_summary(result.points)
    peak = throughput.get("peak") or {}
    best_ok = throughput.get("best_within_sla") or {}
    if peak:
        flag = "达标" if peak.get("ok") else "不达标"
        print(f"  最高输出吞吐：{peak['value']:.1f} tok/s @ 并发={peak['concurrency']}"
              f"（{flag}）")
    if best_ok and (not peak or best_ok["concurrency"] != peak["concurrency"]):
        print(f"  SLA 内最高吞吐：{best_ok['value']:.1f} tok/s @ 并发={best_ok['concurrency']}")
    print(f"  状态：{result.status}（退出码 {result.exit_code}）")
    for warning in result.warnings:
        print(console.c(f"  注意：{warning}", console.YELLOW))
    if result.dataset_info:
        print(f"  数据集：{result.dataset_info.get('dataset_path', '')}")
    console.rule(char="=", codes=((console.BOLD, console.GREEN) if ok
                                  else (console.BOLD, console.YELLOW)))


def print_dry_run(cfg) -> None:
    points = dry_run_points(cfg)
    print("dry-run：将按以下阶梯探测（二分区间由实测结果决定）")
    print(f"  并发点：{' -> '.join(str(p) for p in points)}"
          + (" -> …" if points and points[-1] * 2 <= cfg.max_concurrency else ""))
    print("  SLA 口径：" + "；".join(rule.label() for rule in cfg.sla_rules()))
    for rule in cfg.sla_rules():
        print(f"    {rule.key} -> {rule.metrics_key}")
    print(f"  预检：{'开（并发=1）' if cfg.start > 1 else '关（start=1 时首个探针即预检）'}")
    print(f"  前缀预热：{cfg.prefix_warmup}（每点独立数据集：{'是' if cfg.seed_per_point else '否'}）")
    print(f"  单点：数据集={cfg.dataset or '(需生成)'} work_path={cfg.work_path}")
    print(f"  输出目录：{cfg.output_dir}")


# --------------------------------------------------------------------- 落盘
CSV_FIELDS = [
    "kind", "concurrency", "seed", "output_len", "verdict", "reason",
    "request_rate",
    *ALL_METRIC_KEYS,
    "hbm_hit_rate", "ext_hit_rate", "hit_rate_note",
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
        "sla": dict(cfg.sla),
        "sla_labels": [rule.label() for rule in cfg.sla_rules()],
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
