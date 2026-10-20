"""运行日志：把工具自身的日志也落到输出目录（对齐参考工具的日志处理）。

参考工具用 `2>&1 | tee -a aisbench.log | tee -a aisbench_all.log`，并且把**它自己的**
日志同时写进这两个文件，于是单个文件里就是「工具消息 + ais_bench 原始输出」的完整时序。
这里保留同样的可见性，并把职责分开：

    <output-dir>/run.log           工具自身的日志（本次运行覆盖）
    <output-dir>/aisbench_all.log  工具消息 + 各相位 ais_bench 原始输出（追加，带分隔块）
    <output-dir>/points/c<N>/*.log 单相位 ais_bench 原始输出（只有 ais_bench 内容）
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _has_handler(path: str) -> bool:
    target = os.path.abspath(path)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            if os.path.abspath(getattr(handler, "baseFilename", "")) == target:
                return True
    return False


def setup_run_logging(output_dir: str, banner: str = "") -> Dict[str, str]:
    """挂上 run.log / aisbench_all.log 两个文件处理器（重复调用不会重复挂）。"""
    os.makedirs(output_dir, exist_ok=True)
    run_log = os.path.join(output_dir, "run.log")
    all_log = os.path.join(output_dir, "aisbench_all.log")
    root = logging.getLogger()

    if not _has_handler(run_log):
        handler = logging.FileHandler(run_log, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
        root.setLevel(min(root.level or logging.INFO, logging.INFO))

    if not _has_handler(all_log):
        handler = logging.FileHandler(all_log, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
        with open(all_log, "a", encoding="utf-8") as handle:
            handle.write("\n" + "=" * 78 + "\n")
            handle.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} {banner or 'aisbench-sla'}\n")
            handle.write("=" * 78 + "\n")

    return {"run_log": run_log, "all_log": all_log}
