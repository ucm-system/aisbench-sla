"""ais_bench 调用层：模型配置注入 + 数据集挂载 + 子进程执行。

沿用参考工具的做法（已在真实环境验证）：把模型配置写成 ais_bench 的
`ais_bench/benchmark/configs/models/vllm_api/<name>.py`，把数据集软链到
`ais_bench/datasets/gsm8k/test.jsonl`，然后执行 `ais_bench --models <name> ...`。

与参考工具的区别：
* 不使用 shell 管道 tee，而是把 stdout/stderr 直接重定向到该点的日志文件
  （可解析、无死锁、Windows/Linux 行为一致）
* 支持单点超时与进程树终止
* 老版本 ais_bench 不认 `--num-warmups` 时自动去掉该参数重跑一次
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

MODEL_CFG_NAME = "vllm_api_chat_temp"
DATASET_CFG = "gsm8k_gen_0_shot_cot_str_perf"
SUMMARIZER = "default_perf"

# 占位符版模板（源自参考工具 default_api.py）
API_TEMPLATE = '''from ais_bench.benchmark.models import __API_TYPE__

models = [
    dict(
        attr="service",
        type=__API_TYPE__,
        abbr='__API_ABBR__',
        path="__MODEL_PATH__",
        model="__MODEL_NAME__",
        api_key="__API_KEY__",
        request_rate=__REQUEST_RATE__,
        retry=2,
        host_ip="__HOST_IP__",
        host_port=__HOST_PORT__,
        url="__URL__",
        max_out_len=__OUTPUT_LEN__,
        batch_size=__CONCURRENCY__,
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True__EXTRA_KWARGS__
        )
    )
]
'''

API_TYPES = {
    "stream": ("VLLMCustomAPIChatStream", "vllm-api-stream-chat"),
    "text": ("VLLMCustomAPIChat", "vllm-api-general-chat"),
}


def render_model_config(cfg, concurrency: int, output_len: int,
                        request_rate: float) -> str:
    api_type, api_abbr = API_TYPES.get(cfg.test_type, API_TYPES["stream"])
    extra = ""
    if cfg.enable_think:
        extra = ',\n            chat_template_kwargs={"enable_thinking": True}'
    # 目标寻址由 cfg.model_fields() 决定写 host_ip/host_port 还是 url，
    # 规避 ais_bench「host_ip 必须是 IP」的校验雷区（详见 sla_tool/target.py）
    target_fields = cfg.model_fields()
    text = API_TEMPLATE
    for placeholder, value in (
        ("__API_TYPE__", api_type),
        ("__API_ABBR__", api_abbr),
        ("__MODEL_PATH__", cfg.model_path or ""),
        ("__MODEL_NAME__", cfg.model),
        ("__API_KEY__", cfg.api_key or ""),
        ("__REQUEST_RATE__", repr(float(request_rate))),
        ("__HOST_IP__", str(target_fields["host_ip"])),
        ("__HOST_PORT__", str(target_fields["host_port"])),
        ("__URL__", str(target_fields["url"])),
        ("__OUTPUT_LEN__", str(int(output_len))),
        ("__CONCURRENCY__", str(int(concurrency))),
        ("__EXTRA_KWARGS__", extra),
    ):
        text = text.replace(placeholder, value)
    return text


def model_config_path(work_path: str) -> str:
    return os.path.join(work_path, "ais_bench", "benchmark", "configs", "models",
                        "vllm_api", f"{MODEL_CFG_NAME}.py")


def install_model_config(work_path: str, text: str) -> str:
    target = model_config_path(work_path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.islink(target):
        os.unlink(target)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(text)
    return target


def dataset_dir(work_path: str) -> str:
    return os.path.join(work_path, "ais_bench", "datasets", "gsm8k")


def _link_or_copy(source: str, target: str) -> None:
    if os.path.islink(target) or os.path.exists(target):
        try:
            os.unlink(target)
        except OSError:
            shutil.copy2(source, target)
            return
    try:
        os.symlink(source, target)
    except (OSError, NotImplementedError):
        shutil.copy2(source, target)


def mount_dataset(work_path: str, dataset_jsonl: str) -> str:
    """把数据集挂到 ais_bench 的 gsm8k/test.jsonl（train.jsonl 需存在）。"""
    target_dir = dataset_dir(work_path)
    os.makedirs(target_dir, exist_ok=True)
    train = os.path.join(target_dir, "train.jsonl")
    if not os.path.exists(train):
        open(train, "w", encoding="utf-8").close()
    target = os.path.join(target_dir, "test.jsonl")
    _link_or_copy(os.path.abspath(dataset_jsonl), target)
    return target


def build_command(cfg, run_work_dir: str, with_num_warmups: bool) -> str:
    parts = [
        cfg.aisbench_cmd,
        f"--models {MODEL_CFG_NAME}",
        f"--datasets {DATASET_CFG}",
        "--mode perf",
        f"--summarizer {SUMMARIZER}",
        f"--work-dir {run_work_dir}",
        "--debug",
    ]
    if with_num_warmups and cfg.num_warmups:
        parts.append("--num-warmups 0")
    if cfg.aisbench_extra_args:
        parts.append(cfg.aisbench_extra_args)
    return " ".join(parts)


def _terminate_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False)
        else:
            os.killpg(os.getpgid(proc.pid), 9)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


@dataclass
class CommandResult:
    command: str
    returncode: int
    duration_s: float
    log_path: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def run_command(command: str, log_path: str, cwd: str, timeout: float = 0.0,
                verbose: bool = False, stream: bool = True,
                all_log: str = "") -> CommandResult:
    """执行命令：输出**实时透传**到终端，同时写入 log_path（并追加到 all_log）。

    参考工具是把 ais_bench 的输出直接打到终端（不重定向），这里保留同样的可见性，
    但额外 tee 一份到该探测点的日志文件，便于事后排查。
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    logger.info("执行：%s", command)
    if not stream:
        logger.info("日志：%s（--no-stream，不实时打印）", log_path)
    started = time.time()
    popen_kwargs: Dict[str, object] = {}
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True
    all_handle = open(all_log, "a", encoding="utf-8", errors="replace") if all_log else None
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
            proc = subprocess.Popen(  # noqa: S602 - 用户可自定义 aisbench 命令
                command, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1, **popen_kwargs)

            def _tee() -> None:
                # 必须持续读管道：既是为了实时打印，也避免写满管道把子进程卡死
                for line in proc.stdout:  # type: ignore[union-attr]
                    log_file.write(line)
                    log_file.flush()
                    if all_handle is not None:
                        all_handle.write(line)
                        all_handle.flush()
                    if stream:
                        sys.stdout.write(line)
                        sys.stdout.flush()

            reader = threading.Thread(target=_tee, daemon=True, name="aisbench-tee")
            reader.start()
            timed_out = False
            try:
                proc.wait(timeout=timeout if timeout and timeout > 0 else None)
            except subprocess.TimeoutExpired:
                timed_out = True
                logger.warning("单点超时（%.0fs），终止进程树", timeout)
                _terminate_tree(proc)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            reader.join(timeout=10)
    finally:
        if all_handle is not None:
            all_handle.close()
    duration = time.time() - started
    return CommandResult(command=command, returncode=proc.returncode or 0,
                         duration_s=duration, log_path=log_path, timed_out=timed_out)


def copy_log_to_exp_folder(log_path: str, log_dir: str, tag: str) -> str:
    """把本次运行的日志回拷到 ais_bench 的 exp 目录（对齐参考工具的 save_log）。"""
    if not log_dir or not os.path.isdir(log_dir) or not os.path.isfile(log_path):
        return ""
    try:
        target = os.path.join(log_dir, f"aisbench_{tag}.log")
        shutil.copy2(log_path, target)
        return target
    except OSError as exc:
        logger.warning("回拷日志到 exp 目录失败（忽略）：%s", exc)
        return ""


def tail(path: str, lines: int = 20) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.readlines()
    except OSError:
        return ""
    return "".join(content[-lines:])
