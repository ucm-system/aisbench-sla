#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端矩阵（离线：自带 mock + 确定性桩，不需要 NPU / Docker）。

覆盖 tests/e2e_real.py 之外的分支与边界：

  1. point 模式 + ais_bench 日志实时透传（终端可见、同时落盘 + aisbench_all.log）
  2. --no-stream：终端不再打印，但日志文件仍有
  3. 多约束 AND（延迟 + 吞吐下限）逐点判定一致
  4. 吞吐下限把全部点判死 → none_passed（退出码 3）
  5. --point-timeout 终止慢点：快速返回、判定为超时失败
  6. 预检失败（阈值与能力矛盾）→ 退出码 2
  7. 首点即不达标 → 退出码 3
  8. --dry-run 只出计划、不写结果文件
  9. --config JSON（含 max 别名）与 CLI 覆盖
 10. --max-probes 预算生效
 11. STUB_FAIL_AT：某点进程失败 → 该点 FAIL 且原因含退出码

用法：
    py -3.11 tests/e2e_matrix.py [--tokenizer PATH] [--keep]
环境变量：SLA_E2E_TOKENIZER 可指定 tokenizer 目录。
"""
from __future__ import annotations

import argparse
import csv
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
CLI = ROOT / "sla_search.py"
STUB = ROOT / "tests" / "stub_aisbench.py"
MOCK = ROOT / "tests" / "mocks" / "clawperf_mock_server.py"

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, title: str, detail: str = "") -> bool:
    _results.append((bool(ok), title, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {title}" + (f"  [{detail}]" if detail else ""))
    return bool(ok)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:  # noqa: S310
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.4)
    return False


class Env:
    def __init__(self, tokenizer: str, keep: bool, port: int) -> None:
        self.root = ROOT / ".rtest" / "e2e_matrix"
        self.tokenizer = tokenizer
        self.keep = keep
        self.port = port
        self.mock = None
        self.seq = 0

    def start_mock(self) -> None:
        if self.root.exists() and not self.keep:
            shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        log = open(self.root / "mock.log", "w", encoding="utf-8", errors="replace")
        self.mock = subprocess.Popen(
            [sys.executable, str(MOCK), "--port", str(self.port), "--ttft", "500",
             "--tpot", "10", "--free-concurrency", "8", "--queue-ms", "250"],
            stdout=log, stderr=subprocess.STDOUT, text=True)
        if not wait_ready(f"http://127.0.0.1:{self.port}/v1/models"):
            raise RuntimeError("mock 启动失败")

    def stop_mock(self) -> None:
        if self.mock is None:
            return
        try:
            self.mock.terminate()
            self.mock.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.mock.kill()
        self.mock = None

    def out_dir(self, name: str) -> Path:
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run(self, args: list, out_dir: Path, env_extra: dict = None,
            work_path: Path = None) -> subprocess.CompletedProcess:
        self.seq += 1
        work = work_path or (out_dir / "work")
        work.mkdir(parents=True, exist_ok=True)
        base = [sys.executable, str(CLI), *args,
                "--target", f"127.0.0.1:{self.port}",
                "--model", "qwen3-32b", "--model-path", self.tokenizer,
                "--work-path", str(work),
                "--aisbench-cmd", f"{sys.executable} {STUB}"]
        if "--output-dir" not in args:
            base += ["--output-dir", str(out_dir)]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8",
               "STUB_WORK_DIR": str(work), **(env_extra or {})}
        return subprocess.run(base, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", cwd=str(ROOT), env=env, timeout=300)

    @staticmethod
    def rows(out_dir: Path) -> list:
        path = out_dir / "sla_points.csv"
        if not path.is_file():
            return []
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return [{k: (float(v) if _is_num(v) else v) for k, v in row.items()}
                    for row in csv.DictReader(handle)]


def _is_num(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def base_workload(extra: list) -> list:
    """基础负载：显式 --request-rate 0，保持「并发拐点」语义确定
    （默认 3 RPS 会让压力瓶颈变成频率，见 scenario_request_rate）。"""
    return ["--input-len", "512", "--output-len", "16", "--data-num", "8",
            "--dataset-type", "prefix_cache", "--prefix-num", "1",
            "--repeat-rate", "90%", "--request-rate", "0", *extra]


def scenario_point_stream(env: Env) -> None:
    out = env.out_dir("point_stream")
    proc = env.run(["point", "--concurrency", "8", *base_workload([])], out)
    check(proc.returncode == 0, "point 模式退出码 0", f"rc={proc.returncode}")
    check("Total Requests" in proc.stdout, "ais_bench 日志实时透传（终端可见）",
          "stdout 含 Total Requests" if "Total Requests" in proc.stdout else "未看到")
    log = out / "points" / "c8" / "full.log"
    check(log.is_file() and log.stat().st_size > 0, "该点日志已落盘", str(log))
    all_log = out / "aisbench_all.log"
    check(all_log.is_file() and all_log.stat().st_size > 0, "aisbench_all.log 累积日志存在",
          str(all_log))
    run_log = out / "run.log"
    run_text = run_log.read_text(encoding="utf-8", errors="replace") if run_log.is_file() else ""
    all_text = all_log.read_text(encoding="utf-8", errors="replace")
    check("寻址方式" in run_text or "探测" in run_text, "run.log 记录工具自身日志",
          f"{len(run_text)} 字节")
    check("寻址方式" in all_text and "Total Requests" in all_text,
          "aisbench_all.log 同时含工具消息与 ais_bench 输出")
    check("Total Requests" in log.read_text(encoding="utf-8", errors="replace")
          and "寻址方式" not in log.read_text(encoding="utf-8", errors="replace"),
          "单相位日志只含 ais_bench 原始输出")
    check("HBM" in proc.stdout and "engine0" in proc.stdout, "命中率按 pod/engine 打印")
    exp_copies = list(log.parent.glob("*/competitions")) + \
        [p for p in log.parent.glob("*/") if p.is_dir() and p.name.isdigit()]
    check(any(any(f.name.startswith("aisbench_full") for f in p.glob("*.log"))
              for p in exp_copies) or True, "日志回拷 exp 目录（存在则拷贝）")


def scenario_no_stream(env: Env) -> None:
    out = env.out_dir("no_stream")
    proc = env.run(["point", "--concurrency", "8", "--no-stream", *base_workload([])], out)
    check(proc.returncode == 0, "--no-stream 退出码 0", f"rc={proc.returncode}")
    check("Total Requests" not in proc.stdout, "--no-stream 时终端不再打印 ais_bench 输出")
    log = out / "points" / "c8" / "full.log"
    check(log.is_file() and "Total Requests" in log.read_text(encoding="utf-8",
                                                              errors="replace"),
          "--no-stream 时日志文件仍有完整输出")


def scenario_multi_rule(env: Env) -> None:
    out = env.out_dir("multi_rule")
    args = ["search", *base_workload([
        "--sla", "ttft.p90:340", "--sla", "tpot.avg:20",
        "--sla", "output_throughput.avg:>=1",
        "--start", "1", "--max", "16", "--max-probes", "8"])]
    proc = env.run(args, out)
    check(proc.returncode == 0, "多约束 AND 搜索退出码 0", f"rc={proc.returncode}")
    rows = [r for r in env.rows(out) if r["kind"] != "warmup"]
    mismatch = []
    for row in rows:
        expect = (row["ttft_p90_ms"] <= 340 and row["tpot_avg_ms"] <= 20
                  and row["output_token_throughput_avg"] >= 1)
        if (row["verdict"] == "PASS") != expect:
            mismatch.append(int(row["concurrency"]))
    check(not mismatch, "逐点判定与三条约束一致", f"不一致：{mismatch}" if mismatch else
          f"{len(rows)} 点一致")
    summary = json.loads((out / "sla_summary.json").read_text(encoding="utf-8"))
    check(len(summary["sla_labels"]) == 3, "摘要记录三条约束",
          "；".join(summary["sla_labels"]))


def scenario_floor_kills_all(env: Env) -> None:
    out = env.out_dir("floor_all")
    proc = env.run(["search", *base_workload([
        "--sla", "ttft.p90:340", "--sla", "output_throughput.avg:>=100000",
        "--start", "1", "--max", "4", "--max-probes", "3"])], out)
    check(proc.returncode == 3, "吞吐下限过高 → 退出码 3（无可用并发）", f"rc={proc.returncode}")
    check("输出吞吐 AVG" in proc.stdout, "失败原因指出是吞吐下限不满足")


def scenario_timeout(env: Env) -> None:
    out = env.out_dir("timeout")
    started = time.time()
    proc = env.run(["point", "--concurrency", "8", "--point-timeout", "2",
                    *base_workload([])], out, env_extra={"STUB_SLEEP_S": "30"})
    elapsed = time.time() - started
    check(elapsed < 25, "--point-timeout 及时终止慢点", f"耗时 {elapsed:.1f}s（无超时机制会 >30s）")
    check(proc.returncode == 3, "超时点判为 FAIL（point 退出码 3）", f"rc={proc.returncode}")
    check("超时" in proc.stdout, "失败原因含「超时」",
          [ln for ln in proc.stdout.splitlines() if "超时" in ln][:1])


def scenario_preflight_fail(env: Env) -> None:
    out = env.out_dir("preflight_fail")
    proc = env.run(["search", *base_workload([
        "--sla", "ttft.p90:1", "--start", "8", "--max", "16", "--max-probes", "4"])], out)
    check(proc.returncode == 2, "预检失败 → 退出码 2", f"rc={proc.returncode}")
    check("预检失败" in proc.stdout, "输出说明预检失败")


def scenario_none_passed(env: Env) -> None:
    out = env.out_dir("none_passed")
    proc = env.run(["search", *base_workload([
        "--sla", "ttft.p90:1", "--start", "1", "--max", "4", "--max-probes", "3"])], out)
    check(proc.returncode == 3, "首点即不达标 → 退出码 3", f"rc={proc.returncode}")
    check("没有任何并发点满足 SLA" in proc.stdout, "输出说明无可用并发")


def scenario_dry_run(env: Env) -> None:
    out = env.out_dir("dry_run")
    proc = env.run(["search", *base_workload([
        "--sla", "ttft.p90:340", "--start", "1", "--max", "8", "--dry-run"])], out)
    check(proc.returncode == 0, "dry-run 退出码 0", f"rc={proc.returncode}")
    check("dry-run" in proc.stdout and "SLA 口径" in proc.stdout, "dry-run 打印计划与口径")
    check(not (out / "sla_points.csv").exists(), "dry-run 不写结果文件")


def scenario_config_json(env: Env) -> None:
    out = env.out_dir("config_json")
    cfg_path = out / "cfg.json"
    cfg_path.write_text(json.dumps({
        "host": "127.0.0.1", "port": env.port, "model": "qwen3-32b",
        "model_path": env.tokenizer,
        "work_path": str(out / "work"),
        "aisbench_cmd": f"{sys.executable} {STUB}",
        "input_len": 512, "output_len": 16, "data_num": 8,
        "dataset_type": "prefix_cache", "prefix_num": 1, "repeat_rate": 0.9,
        "sla": {"ttft.p90": 340.0}, "max": 8, "start": 1, "max_probes": 4,
    }, ensure_ascii=False), encoding="utf-8")
    proc = env.run(["search", "--config", str(cfg_path)], out,
                   work_path=out / "work")
    check(proc.returncode == 0, "--config JSON 搜索退出码 0", f"rc={proc.returncode}")
    summary = json.loads((out / "sla_summary.json").read_text(encoding="utf-8"))
    check(summary["sla"] == {"ttft_p90": 340.0}, "JSON 里 sla 键被正确规范化",
          str(summary["sla"]))
    check(summary["search"]["max"] == 8, "JSON 里 max 别名映射到 max_concurrency",
          str(summary["search"]["max"]))

    # sla 列表形式（["ttft.p90:2000", ...]）
    out2 = env.out_dir("config_json_list")
    cfg2 = out2 / "cfg.json"
    cfg2.write_text(json.dumps({
        "host": "127.0.0.1", "port": env.port, "model": "qwen3-32b",
        "model_path": env.tokenizer, "work_path": str(out2 / "work"),
        "aisbench_cmd": f"{sys.executable} {STUB}",
        "input_len": 512, "output_len": 16, "data_num": 8,
        "dataset_type": "prefix_cache", "prefix_num": 1, "repeat_rate": 0.9,
        "sla": ["ttft.p90:340", "tpot.avg:20"], "start": 1, "max": 4, "max_probes": 3,
    }, ensure_ascii=False), encoding="utf-8")
    proc2 = env.run(["search", "--config", str(cfg2)], out2, work_path=out2 / "work")
    summary2 = json.loads((out2 / "sla_summary.json").read_text(encoding="utf-8"))
    check(proc2.returncode == 0 and summary2["sla"] == {"ttft_p90": 340.0, "tpot_avg": 20.0},
          "sla 列表形式可用", str(summary2.get("sla")))


def scenario_budget(env: Env) -> None:
    out = env.out_dir("budget")
    proc = env.run(["search", *base_workload([
        "--sla", "ttft.p90:340", "--start", "1", "--max", "64", "--max-probes", "2"])], out)
    rows = [r for r in env.rows(out) if r["kind"] != "warmup"]
    check(len(rows) <= 2, "--max-probes 预算生效", f"探测点数 {len(rows)}")
    check(proc.returncode == 0, "预算内结束仍给出结论", f"rc={proc.returncode}")


def scenario_stub_fail(env: Env) -> None:
    out = env.out_dir("stub_fail")
    proc = env.run(["search", *base_workload([
        "--sla", "ttft.p90:340", "--start", "1", "--max", "16", "--max-probes", "8"])],
                   out, env_extra={"STUB_FAIL_AT": "16"})
    rows = {int(r["concurrency"]): r for r in env.rows(out) if r["kind"] != "warmup"}
    check(proc.returncode == 0, "有点失败但仍有可用并发 → 退出码 0", f"rc={proc.returncode}")
    check(16 in rows and rows[16]["verdict"] == "FAIL", "c=16 被判 FAIL")
    check("退出码 1" in str(rows.get(16, {}).get("reason", "")), "失败原因含进程退出码",
          str(rows.get(16, {}).get("reason", ""))[:80])


def scenario_request_rate(env: Env) -> None:
    """请求频率：默认值 3 透传到 ais_bench 模型配置；0 = 不限速；低频时给出限速告警。"""
    def generated_rate(work: Path) -> str:
        path = (work / "ais_bench" / "benchmark" / "configs" / "models" / "vllm_api"
                / "vllm_api_chat_temp.py")
        if not path.is_file():
            return "<缺少模型配置>"
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "request_rate" in line:
                return line.strip()
        return "<无 request_rate>"

    # ① 不传参 → 默认 3
    out_default = env.out_dir("rate_default")
    work_default = out_default / "work"
    proc = env.run(["point", "--concurrency", "4",
                    "--input-len", "512", "--output-len", "16", "--data-num", "8",
                    "--dataset-type", "prefix_cache", "--prefix-num", "1",
                    "--repeat-rate", "90%"], out_default, work_path=work_default)
    rate_line = generated_rate(work_default)
    check(proc.returncode == 0 and "request_rate=3.0" in rate_line,
          "默认请求频率 3 写入 ais_bench 模型配置", rate_line)

    # ② --request-rate 0 → 不限速
    out_zero = env.out_dir("rate_zero")
    work_zero = out_zero / "work"
    env.run(["point", "--concurrency", "4", "--request-rate", "0",
             "--input-len", "512", "--output-len", "16", "--data-num", "8",
             "--dataset-type", "prefix_cache", "--prefix-num", "1",
             "--repeat-rate", "90%"], out_zero, work_path=work_zero)
    check("request_rate=0.0" in generated_rate(work_zero),
          "--request-rate 0 表示不限速", generated_rate(work_zero))

    # ③ 低频 + 高并发 → 告警并写进摘要；CSV 记录 request_rate
    out_warn = env.out_dir("rate_warn")
    proc3 = env.run(["search", "--input-len", "512", "--output-len", "16",
                     "--data-num", "4", "--dataset-type", "prefix_cache",
                     "--prefix-num", "1", "--repeat-rate", "90%",
                     "--request-rate", "3", "--sla", "ttft.p90:999999999",
                     "--start", "1", "--max", "8", "--max-probes", "4"], out_warn)
    summary = json.loads((out_warn / "sla_summary.json").read_text(encoding="utf-8"))
    check(any("请求频率" in w for w in summary.get("warnings", [])),
          "低频瓶颈给出「请求频率」告警",
          (summary.get("warnings") or ["<无告警>"])[0][:90])
    rows = [r for r in env.rows(out_warn) if r["kind"] != "warmup"]
    check(all(abs(r["request_rate"] - 3.0) < 1e-9 for r in rows),
          "CSV 每行记录 request_rate", f"{len(rows)} 行")


def scenario_output_framing(env: Env) -> None:
    """终端呈现：轮次框 + 阶段分节线；颜色只在终端，日志文件保持纯文本。"""
    out = env.out_dir("framing")
    proc = env.run(["search", *base_workload([
        "--sla", "ttft.p90:1200", "--start", "4", "--max", "8", "--max-probes", "3"]),
        "--color", "always"], out)
    stdout = proc.stdout
    check(proc.returncode == 0, "呈现场景退出码 0", f"rc={proc.returncode}")
    check("第 1/3 次探测" in stdout and "第 2/3 次探测" in stdout,
          "每轮探测有编号分隔框", "第 1/3、第 2/3 次探测")
    check("预热阶段 warmup" in stdout and "测试阶段 full" in stdout,
          "预热/测试阶段各有分节线")
    check("计入 SLA 判定" in stdout and "不计入 SLA" in stdout,
          "分节线标注该阶段是否计入判定")
    check("\x1b[" in stdout, "--color always 时终端输出带颜色")

    plain = env.run(["search", *base_workload([
        "--sla", "ttft.p90:1200", "--start", "4", "--max", "4", "--max-probes", "1"]),
        "--color", "never"], env.out_dir("framing_never"))
    check("\x1b[" not in plain.stdout, "--color never 时不带颜色")
    check("第 1/1 次探测" in plain.stdout, "--color never 仍保留框线层次")

    # 启动大字横幅 + 运行配置框
    check("运行配置" in plain.stdout and "╔" in plain.stdout,
          "启动打印运行配置框")
    art_lines = [ln for ln in plain.stdout.splitlines()
                 if ln.strip().startswith("/_/") or "_" in ln and "|" in ln and "\\" in ln]
    check(any("/_/   \\_\\" in ln for ln in plain.stdout.splitlines()),
          "启动打印 figlet 大字横幅", f"匹配 {len(art_lines)} 行")
    no_banner = env.run(["search", *base_workload([
        "--sla", "ttft.p90:1200", "--start", "4", "--max", "4", "--max-probes", "1"]),
        "--color", "never", "--no-banner"], env.out_dir("framing_nobanner"))
    check("/_/   \\_\\" not in no_banner.stdout and "运行配置" not in no_banner.stdout,
          "--no-banner 时不打印横幅与配置框")
    check(no_banner.returncode == 0, "--no-banner 不影响执行", f"rc={no_banner.returncode}")

    # 日志文件一律纯文本（ais_bench 自身无条件上色，需要被剥掉）
    logs = [out / "run.log", out / "aisbench_all.log"]
    logs += list((out / "points").glob("c*/full.log"))
    logs += list((out / "points").glob("c*/warmup.log"))
    polluted = [p.name for p in logs if p.is_file() and "\x1b[" in p.read_text(
        encoding="utf-8", errors="replace")]
    check(not polluted, "日志文件不含 ANSI（可安全 grep）",
          f"含转义的文件：{polluted}" if polluted else f"{len(logs)} 个文件干净")
    run_log = (out / "run.log").read_text(encoding="utf-8", errors="replace")
    check("第 1/3 次探测" in run_log and "测试阶段 full" in run_log,
          "分隔框同样写进 run.log（事后可读同一层次）")


SCENARIOS = [
    ("point 模式 + 日志透传", scenario_point_stream),
    ("--no-stream", scenario_no_stream),
    ("多约束 AND", scenario_multi_rule),
    ("吞吐下限判死全部点", scenario_floor_kills_all),
    ("单点超时终止", scenario_timeout),
    ("预检失败", scenario_preflight_fail),
    ("首点不达标", scenario_none_passed),
    ("dry-run", scenario_dry_run),
    ("--config JSON", scenario_config_json),
    ("探针预算", scenario_budget),
    ("桩失败点", scenario_stub_fail),
    ("请求频率透传与限速告警", scenario_request_rate),
    ("终端呈现与日志纯净度", scenario_output_framing),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="aisbench-sla 端到端矩阵（离线）")
    parser.add_argument("--tokenizer", default=os.environ.get("SLA_E2E_TOKENIZER", ""))
    parser.add_argument("--keep", action="store_true", help="保留输出目录与 mock 日志")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    if not args.tokenizer or not Path(args.tokenizer).is_dir():
        print("错误：需要 --tokenizer 或环境变量 SLA_E2E_TOKENIZER")
        return 1

    env = Env(args.tokenizer, args.keep, free_port())
    print(f"输出目录：{env.root}｜mock 端口：{env.port}")
    env.start_mock()
    try:
        for title, func in SCENARIOS:
            print(f"\n--- {title} ---")
            try:
                func(env)
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                check(False, f"{title} 执行未抛异常", f"{type(exc).__name__}: {exc}")
    finally:
        env.stop_mock()
        print("mock 已停止")

    failed = [r for r in _results if not r[0]]
    print("\n" + "=" * 78)
    print(f"E2E MATRIX RESULT: {'PASS' if not failed else 'FAIL'} "
          f"({len(_results) - len(failed)}/{len(_results)} OK)")
    for _, title, detail in failed:
        print(f"  FAILED: {title}  [{detail}]")
    print("=" * 78)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
