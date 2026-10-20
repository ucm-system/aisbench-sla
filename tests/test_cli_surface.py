#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI 表面回归：所有 --help 路径必须可用，且关键开关可见。

起因：argparse 会对每个选项的 help 文本做 `%` 格式化，帮助里出现单个 `%`
（如 "~100%、"）会让 `search --help` 直接抛 ValueError 崩溃 —— 这类问题
`--help` 一跑就现形，但没人跑就会带进发布版。此用例守住这条底线。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "sla_search.py"

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, title: str, detail: str = "") -> None:
    _results.append((bool(ok), title, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {title}" + (f"  [{detail}]" if detail else ""))


def run(args: list) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CLI), *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
                          env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    for sub in ("", "search", "point", "dataset"):
        args = ([sub] if sub else []) + ["--help"]
        proc = run(args)
        label = f"{sub or '(top)'} --help"
        check(proc.returncode == 0, f"{label} 可用",
              f"rc={proc.returncode} {proc.stderr.strip().splitlines()[-1] if proc.returncode else ''}")
        if sub == "search":
            for needle in ("--target", "--ttft-p90", "--tpot-p90", "--max-probes",
                           "--seed-per-point", "--no-seed-per-point", "--reset-cache",
                           "--data-num", "--work-path", "--dry-run"):
                check(needle in proc.stdout, f"search --help 含 {needle}")
            check("2~4 倍" in proc.stdout, "search --help 说明 data-num 与 max 的倍数关系")
        if sub == "point":
            check("--concurrency" in proc.stdout, "point --help 含 --concurrency")

    # 缺参时的报错必须可读且指名缺哪个参数（校验顺序：work-path → model-path）
    no_work = run(["search", "--ttft-p90", "1"])
    check(no_work.returncode == 1, "缺 --work-path 时退出码为 1", f"rc={no_work.returncode}")
    check("--work-path" in (no_work.stdout + no_work.stderr),
          "缺 --work-path 的报错指名该参数")

    no_tok = run(["search", "--ttft-p90", "1", "--work-path", str(ROOT),
                  "--input-len", "8", "--data-num", "1"])
    check(no_tok.returncode == 1, "缺 --model-path 时退出码为 1", f"rc={no_tok.returncode}")
    check("--model-path" in (no_tok.stdout + no_tok.stderr),
          "缺 --model-path 的报错指名该参数")

    failed = [r for r in _results if not r[0]]
    print("\n" + "=" * 70)
    print(f"CLI SURFACE RESULT: {'PASS' if not failed else 'FAIL'} "
          f"({len(_results) - len(failed)}/{len(_results)} OK)")
    print("=" * 70)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
