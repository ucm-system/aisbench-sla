#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""完整真实链路测试：vendored mock（含负载模型）+ 官方 ais_bench 镜像 + 本工具。

与 tests/test_offline.py（纯离线、确定性桩）互补：本脚本跑**真 ais_bench**，
并用 mock 的负载模型制造真实延迟拐点，从而验证 SLA 搜索的
「阶梯通过 → 阶梯失败 → 二分收敛」整条路径。

用法：
    py -3.11 tests/e2e_real.py                 # 默认 Docker 模式
    py -3.11 tests/e2e_real.py --no-docker     # 无 Docker：用确定性桩跑同样的搜索

预期（默认参数：free=8, queue-ms=250, ttft=500, ttft-p90=1200）：
    c≤8   TTFT≈500ms                       -> 阶梯通过
    c>8   每超一个在途请求 +250ms           -> 阶梯在 c=16 失败
    二分在 (8,16) 内收敛到 c=10（P90≈1000ms 通过，c=11≈1250ms 失败）
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOCK = ROOT / "tests" / "mocks" / "clawperf_mock_server.py"
STUB = ROOT / "tests" / "stub_aisbench.py"
DEFAULT_IMAGE = "ghcr.io/aisbench/aisbench_benchmark:v3.1-20260330-master_x86_64_py_310"

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, title: str, detail: str = "") -> bool:
    _results.append((bool(ok), title, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {title}" + (f"  [{detail}]" if detail else ""))
    return bool(ok)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def http_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - 本地测试
        return json.loads(response.read().decode("utf-8"))


def wait_ready(url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            http_json(url, timeout=3)
            return True
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(0.5)
    return False


def start_mock(args) -> subprocess.Popen:
    ready_file = Path(args.out) / "mock.ready"
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    if ready_file.exists():
        ready_file.unlink()
    cmd = [sys.executable, str(MOCK),
           "--host", "127.0.0.1", "--port", str(args.port),
           "--ttft", str(args.ttft), "--tpot", str(args.tpot),
           "--free-concurrency", str(args.free), "--queue-ms", str(args.queue_ms),
           "--ready-file", str(ready_file)]
    print("启动 mock：", " ".join(cmd))
    # 注意：必须把 mock 的输出重定向到文件而不是 PIPE —— uvicorn 的 access log
    # 会把管道缓冲写满，没人读就会阻塞整个服务端（实测导致后续 /metrics 全部超时）
    mock_log = open(Path(args.out) / "mock.log", "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=mock_log, stderr=subprocess.STDOUT, text=True)
    if not wait_ready(f"http://127.0.0.1:{args.port}/v1/models", timeout=60):
        proc.terminate()
        raise RuntimeError("mock 启动失败")
    models = [m["id"] for m in http_json(f"http://127.0.0.1:{args.port}/v1/models")["data"]]
    print(f"mock 就绪：{args.port}，模型 {models}")
    args.model = models[1] if len(models) > 1 else models[0]
    return proc


def build_command(args) -> list[str]:
    out = str(Path(args.out).resolve())
    common = [
        "search",
        "--target", f"host.docker.internal:{args.port}" if not args.no_docker
                    else f"127.0.0.1:{args.port}",
        "--model", args.model,
        "--input-len", str(args.input_len), "--output-len", str(args.output_len),
        "--data-num", str(args.data_num),
        "--dataset-type", "prefix_cache", "--prefix-num", "1",
        "--repeat-rate", "90%", "--seed", "1", "--dp", "1",
        "--ttft-p90", str(args.ttft_p90), "--tpot-p90", str(args.tpot_p90),
        "--start", str(args.start), "--max", str(args.max),
        "--max-probes", str(args.max_probes), "--max-bisect", str(args.max_bisect),
        "--tolerance", str(args.tolerance),
        "--point-timeout", str(args.point_timeout),
    ]
    if args.reset_cache:
        common.append("--reset-cache")
    if args.seed_per_point:
        common.append("--seed-per-point")
    if args.no_docker:
        work = str(Path(out, "work").resolve())
        os.makedirs(work, exist_ok=True)
        if not Path(args.tokenizer).is_dir():
            raise RuntimeError(f"--tokenizer 目录不存在：{args.tokenizer}")
        return [sys.executable, str(ROOT / "sla_search.py"), *common,
                "--model-path", args.tokenizer,
                "--dataset-path", str(Path(out, "ds")),
                "--output-dir", str(Path(out, "out")),
                "--work-path", work,
                "--aisbench-cmd", f'{sys.executable} {STUB}']
    return ["docker", "run", "--rm",
            "--add-host=host.docker.internal:host-gateway",
            "-v", f"{ROOT}:/tool",
            "-v", f"{out}:/rtest",
            "-v", f"{Path(args.tokenizer).resolve()}:/tok:ro",
            args.image,
            "python3", "/tool/sla_search.py", *common,
            "--model-path", "/tok",
            "--work-path", "/benchmark",
            "--dataset-path", "/rtest/ds", "--output-dir", "/rtest/out"]


def expected_ladder(args) -> list[int]:
    points, current = [], args.start
    while current <= args.max:
        points.append(current)
        current *= 2
    return points


def evaluate(args, summary: dict, exit_code: int) -> None:
    points = summary.get("points", [])
    probes = [p for p in points if p.get("kind") != "warmup"]
    ladder = expected_ladder(args)
    max_ok = summary.get("max_ok_concurrency") or 0
    check(exit_code == 0, "搜索退出码为 0", f"实际 {exit_code}")
    check(any(p["verdict"] == "PASS" for p in probes), "存在通过点")
    check(any(p["verdict"] == "FAIL" for p in probes), "存在失败点（阶梯确实撞到拐点）")
    bisect_points = [p["concurrency"] for p in probes if p["concurrency"] not in ladder]
    check(bool(bisect_points), "二分确实执行了", f"二分点 {bisect_points}")

    # 单调性：SLA 搜索的前提是被测系统随并发单调变差，最终结果必须自洽
    ordered = sorted(probes, key=lambda p: p["concurrency"])
    bad_pairs = [(a["concurrency"], a["verdict"], b["concurrency"], b["verdict"])
                 for a, b in zip(ordered, ordered[1:])
                 if a["verdict"] == "FAIL" and b["verdict"] == "PASS"]
    check(not bad_pairs, "判定随并发单调（低并发 PASS 高于高并发 FAIL 才正确）", str(bad_pairs))
    passing = [p for p in probes if p["verdict"] == "PASS"]
    failing = [p for p in probes if p["verdict"] == "FAIL"]
    check(max_ok == max((p["concurrency"] for p in passing), default=0),
          "最大并发 = 通过点里的最大并发", f"max_ok={max_ok}")
    check(args.start <= max_ok <= args.max, "最大并发落在 [start, max] 区间",
          f"{args.start} <= {max_ok} <= {args.max}")

    # 二分收敛：边界缺口必须落在容差内（tolerance 与 lo//8 的较大者）
    if passing and failing:
        first_fail = min(p["concurrency"] for p in failing if p["concurrency"] > max_ok)
        gap = first_fail - max_ok
        allowed = max(args.tolerance, max_ok // 8) + 1
        check(gap <= allowed, "二分收敛在容差内",
              f"边界缺口 {gap} <= 允许 {allowed}（最大通过 {max_ok}，最小失败 {first_fail}）")
        best_pass = max(p["ttft_p90_ms"] for p in passing)
        worst_fail = max(p["ttft_p90_ms"] for p in failing)
        check(best_pass <= args.ttft_p90 < worst_fail, "拐点两侧 TTFT 分居阈值两边",
              f"通过侧最大 {best_pass:.0f}ms <= 阈值 {args.ttft_p90:.0f}ms < 失败侧 {worst_fail:.0f}ms")
        if args.expect_max_ok:
            check(max_ok == args.expect_max_ok, f"最大并发 = {args.expect_max_ok}（指定值）",
                  f"实际 {max_ok}")
    check(all((p.get("hbm_hit_rate") or 0) > 0 for p in probes),
          "每个探测点都算出了命中率（快照差分链路通）",
          f"min={min((p.get('hbm_hit_rate') or 0) for p in probes):.2%}" if probes else "无点")
    rates = [p["hbm_hit_rate"] for p in probes if p.get("hbm_hit_rate") is not None]
    if args.assert_no_saturation and rates:
        check(max(rates) < 0.99, "命中率未饱和（各点 <99%）",
              f"max={max(rates):.2%}（若饱和说明缓存未复位/数据集未更换）")
        check(not summary.get("warnings"), "无「命中率不可比」告警",
              "；".join(summary.get("warnings") or []) or "无")
    peak = (summary.get("throughput") or {}).get("peak") or {}
    check(bool(peak) and peak.get("value", 0) > 0, "报告了最高输出吞吐",
          f"{peak.get('value', 0):.1f} tok/s @ c={peak.get('concurrency')}")
    csv_path = Path(args.out, "out", "sla_points.csv")
    check(csv_path.is_file(), "生成 sla_points.csv", str(csv_path))
    check((Path(args.out) / "out" / "sla_summary.json").is_file(), "生成 sla_summary.json")


def stop_mock(proc: subprocess.Popen) -> None:
    """稳妥停止 mock：terminate 在 Windows 上偶尔不能及时退出，兜底 kill。"""
    try:
        proc.terminate()
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=8)
        except Exception:  # noqa: BLE001
            pass
    print("mock 已停止")


def main() -> int:
    parser = argparse.ArgumentParser(description="aisbench-sla 完整真实链路测试")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--tokenizer", default=r"D:\Models\Qwen3-0.6B")
    parser.add_argument("--out", default=str(ROOT / ".rtest" / "e2e_real"))
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ttft", type=float, default=500)
    parser.add_argument("--tpot", type=float, default=10)
    parser.add_argument("--free", type=int, default=8, help="负载模型 free-concurrency")
    parser.add_argument("--queue-ms", type=float, default=250.0, help="负载模型每超一档抬高的 TTFT")
    parser.add_argument("--input-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--data-num", type=int, default=16)
    parser.add_argument("--ttft-p90", type=float, default=1200)
    parser.add_argument("--tpot-p90", type=float, default=40)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--max", type=int, default=16)
    parser.add_argument("--max-probes", type=int, default=12)
    parser.add_argument("--max-bisect", type=int, default=5)
    parser.add_argument("--tolerance", type=int, default=1, help="二分收敛容差（并发数）")
    parser.add_argument("--point-timeout", type=float, default=900)
    parser.add_argument("--expect-max-ok", type=int, default=0,
                        help="指定则严格断言最大并发（0=只断言搜索性质，如单调性与二分收敛）")
    parser.add_argument("--no-docker", action="store_true", help="不依赖 Docker，用确定性桩")
    parser.add_argument("--reset-cache", action="store_true",
                        help="透传 --reset-cache：每点前 POST /reset_prefix_cache")
    parser.add_argument("--seed-per-point", action="store_true",
                        help="透传 --seed-per-point：每点换 seed 生成独立数据集")
    parser.add_argument("--assert-no-saturation", action="store_true",
                        help="断言各点命中率未饱和（<99%）且无「不可比」告警，"
                             "配合 --reset-cache / --seed-per-point 使用")
    parser.add_argument("--keep", action="store_true", help="保留 mock 进程与输出目录")
    args = parser.parse_args()
    if args.no_docker:
        # 桩内置确定性 SLA 模型：ttft_p90 = 100 + 20c、tpot_p90 = 10 + 0.5c
        # 阈值 340 -> c<=12 通过、c=16 起失败，二分收敛到 12
        if args.ttft_p90 == 1200:
            args.ttft_p90 = 340
        if not args.expect_max_ok:
            args.expect_max_ok = 12
    args.port = args.port or free_port()
    args.model = ""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    print(f"模式：{'离线桩（无 Docker）' if args.no_docker else '真实 ais_bench（Docker）'}")
    print(f"输出目录：{args.out}")
    if not args.keep and Path(args.out).exists():
        shutil.rmtree(args.out, ignore_errors=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if not MOCK.is_file():
        print(f"缺少 mock：{MOCK}")
        return 1
    if not args.no_docker:
        if not shutil.which("docker"):
            print("找不到 docker（可用 --no-docker）")
            return 1
        probe = subprocess.run(["docker", "image", "inspect", args.image],
                               capture_output=True, text=True)
        check(probe.returncode == 0, "ais_bench 镜像存在", args.image)
        if probe.returncode != 0:
            return 1
    else:
        if not Path(args.tokenizer).is_dir():
            print(f"--tokenizer 不存在：{args.tokenizer}")
            return 1

    mock = start_mock(args)
    exit_code, summary = 1, {}
    try:
        command = build_command(args)
        log_path = Path(args.out) / "search.log"
        print("\n执行搜索：\n  " + " ".join(command) + f"\n  （stdout -> {log_path}）")
        started = time.time()
        env = dict(os.environ)
        if args.no_docker:
            # 桩按 mtime 最新查找模型配置，显式告知工作区更稳
            env["STUB_WORK_DIR"] = str(Path(args.out, "work").resolve())
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                  text=True, cwd=str(ROOT), env=env)
        exit_code = proc.returncode
        print(f"搜索结束：rc={exit_code}，耗时 {time.time() - started:.1f}s")
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]:
            print("  |", line)
        summary_path = Path(args.out) / "out" / "sla_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            check(False, "生成 sla_summary.json", "文件缺失")
        try:
            evaluate(args, summary, exit_code)
        except Exception as exc:  # noqa: BLE001 - 断言自身异常也要报出来
            import traceback
            traceback.print_exc()
            check(False, "断言过程无异常", f"{type(exc).__name__}: {exc}")
    finally:
        if not args.keep:
            stop_mock(mock)

    failed = [r for r in _results if not r[0]]
    print("\n" + "=" * 78)
    print(f"E2E REAL RESULT: {'PASS' if not failed else 'FAIL'} "
          f"({len(_results) - len(failed)}/{len(_results)} OK)")
    for _, title, detail in failed:
        print(f"  FAILED: {title}  [{detail}]")
    print("=" * 78)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
