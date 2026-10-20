"""运行配置：CLI 参数 / JSON 文件 → Config，含校验与归一化。"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


def parse_prefix_ratio(value: Any) -> float:
    """'90%' / '0.9' / 90 (int 视为百分数) → 0.9。"""
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1.0:
            v = v / 100.0
    else:
        text = str(value).strip()
        if text.endswith("%"):
            v = float(text[:-1]) / 100.0
        else:
            v = float(text)
            if v > 1.0:
                v = v / 100.0
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"repeat_rate 必须在 [0,1] 或 [0%,100%]：{value!r}")
    return v


# JSON 配置里允许的别名（CLI 惯用名 / 旧项目命名 → Config 字段名）
_FIELD_ALIASES = {
    "max": "max_concurrency",
    "max_concurrency_limit": "max_concurrency",
    "ttft_p90_ms": "ttft_p90",
    "tpot_p90_ms": "tpot_p90",
    "prefix_ratio": "repeat_rate",
    "dataset_mode": "dataset_type",
    "pods": "pods",
}


def _canonical_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    known = {f.name for f in fields(Config)}
    out: Dict[str, Any] = {}
    for key, value in data.items():
        target = _FIELD_ALIASES.get(key, key)
        if target not in known:
            raise ValueError(f"未知配置项：{key}")
        out[target] = value
    return out


@dataclass
class Config:
    """一次压测/SLA 搜索的完整配置。"""

    # ---- 目标服务 ----
    host: str = "127.0.0.1"
    port: int = 8000
    url: str = ""                      # 设置后忽略 host/port
    model: str = "model"               # 服务端模型名
    model_path: str = ""               # tokenizer 路径（生成数据集用）
    api_key: str = ""
    pods: List[str] = field(default_factory=list)   # metrics 端点，空则用 host:port
    npu_num: int = 1

    # ---- 负载形态 ----
    input_len: int = 2048
    output_len: int = 32
    data_num: int = 64
    request_rate: float = 0.0          # 0 = 不限制
    test_type: str = "stream"          # stream | text
    dataset_type: str = "prefix_cache"  # prefix_cache | normal
    prefix_num: int = 1
    repeat_rate: float = 0.9
    seed: int = 1
    dp: int = 1
    enable_think: bool = False
    length_mean: Optional[int] = None
    length_std: Optional[float] = None
    length_min: Optional[int] = None
    length_max: Optional[int] = None
    dataset: str = ""                  # 指定现成 jsonl（gsm8k 格式），跳过生成
    dataset_path: str = ""             # 生成数据集落盘目录
    gsm8k_source: str = ""             # 参考语料 jsonl（默认取工具自带 GSM8K.jsonl）

    # ---- 环境 / 执行 ----
    work_path: str = ""                # ais_bench benchmark 根目录
    output_dir: str = "./sla_out"
    aisbench_cmd: str = "ais_bench"    # 可整体覆盖
    aisbench_extra_args: str = ""      # 追加参数
    num_warmups: bool = True           # 是否附加 --num-warmups 0（老版本不支持时自动回退）
    point_timeout: float = 0.0         # 单点超时（秒），0 = 不限
    prefix_warmup: str = "per-point"   # none | once | per-point
    reset_cache: bool = False          # 每点前 POST /reset_prefix_cache
    dry_run: bool = False
    verbose: bool = False

    # ---- SLA 与搜索 ----
    ttft_p90: Optional[float] = None   # ms
    tpot_p90: Optional[float] = None   # ms
    start: int = 1
    max_concurrency: int = 256
    tolerance: int = 1                 # 二分收敛容差（并发数）
    max_probes: int = 20
    max_bisect: int = 5

    # ------------------------------------------------------------------ 工厂
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        cfg = cls(**_canonical_keys(data))
        cfg.normalize()
        return cfg

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def merge(self, data: Dict[str, Any]) -> "Config":
        """把 JSON/CLI 的显式值覆盖到当前配置（None 不覆盖）。"""
        for key, value in _canonical_keys(data).items():
            if value is None:
                continue
            setattr(self, key, value)
        self.normalize()
        return self

    def normalize(self) -> None:
        self.repeat_rate = parse_prefix_ratio(self.repeat_rate)
        self.port = int(self.port)
        self.start = max(1, int(self.start))
        self.max_concurrency = max(self.start, int(self.max_concurrency))
        self.tolerance = max(1, int(self.tolerance))
        self.max_probes = max(1, int(self.max_probes))
        self.max_bisect = max(0, int(self.max_bisect))
        if self.dp < 1:
            self.dp = 1
        if self.prefix_num < 1:
            self.prefix_num = 1
        if self.prefix_warmup not in ("none", "once", "per-point"):
            raise ValueError("prefix_warmup 只能是 none / once / per-point")
        if self.dataset_type not in ("prefix_cache", "normal"):
            raise ValueError("dataset_type 只能是 prefix_cache / normal")
        if self.test_type not in ("stream", "text"):
            raise ValueError("test_type 只能是 stream / text")
        if not self.output_dir:
            self.output_dir = "./sla_out"
        self.output_dir = os.path.abspath(self.output_dir)
        if self.work_path:
            # 绝对化：模型配置要写进 <work_path>/ais_bench/benchmark/configs/...，
            # 相对路径会在 cwd 变化（或容器/桩）时写错位置
            self.work_path = os.path.abspath(self.work_path)
        if self.model_path:
            self.model_path = os.path.abspath(self.model_path)
        if self.dataset:
            self.dataset = os.path.abspath(self.dataset)
        if not self.dataset_path:
            self.dataset_path = os.path.join(self.output_dir, "datasets")
        self.dataset_path = os.path.abspath(self.dataset_path)
        if not self.gsm8k_source:
            self.gsm8k_source = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "GSM8K.jsonl")

    # ------------------------------------------------------------------ 校验
    def validate_for_search(self) -> None:
        if self.ttft_p90 is None and self.tpot_p90 is None:
            raise ValueError(
                "SLA 搜索至少需要一个延迟阈值：--ttft-p90 和/或 --tpot-p90（单位 ms）")
        for name in ("ttft_p90", "tpot_p90"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} 必须 > 0（{value}）")

    def validate_for_run(self) -> None:
        if self.dry_run:
            return
        if not self.work_path:
            raise ValueError("必须提供 --work-path（ais_bench 的 benchmark 目录）")
        if not os.path.isdir(self.work_path):
            raise ValueError(f"--work-path 不存在：{self.work_path}")
        # ais_bench 的模型配置必须带 tokenizer 路径（即使数据集已由 --dataset 提供）
        if not self.model_path:
            raise ValueError("必须提供 --model-path（tokenizer 路径）："
                             "ais_bench 的模型配置需要它，否则报 UTILS-FILE-002")
        if not os.path.isdir(self.model_path):
            raise ValueError(f"--model-path 目录不存在：{self.model_path}")

    # ------------------------------------------------------------------ 便捷
    def metrics_pods(self) -> List[str]:
        if self.pods:
            return list(self.pods)
        return [f"{self.host}:{self.port}"]

    def base_url(self) -> str:
        if self.url:
            return self.url.rstrip("/")
        return f"http://{self.host}:{self.port}"

    def workload_signature(self) -> str:
        """数据集缓存键：形态一致才可复用同一份数据集。"""
        parts = [self.dataset_type, self.input_len, self.data_num, self.prefix_num,
                 f"{self.repeat_rate:.4f}", self.seed, self.dp, self.length_mean,
                 self.length_std, self.length_min, self.length_max]
        return "-".join(str(p) for p in parts)

    def describe(self) -> str:
        sla = []
        if self.ttft_p90 is not None:
            sla.append(f"TTFT P90<={self.ttft_p90:.0f}ms")
        if self.tpot_p90 is not None:
            sla.append(f"TPOT P90<={self.tpot_p90:.0f}ms")
        return (f"目标 {self.base_url()} model={self.model} | "
                f"输入{self.input_len}/输出{self.output_len}/条数{self.data_num} | "
                f"并发{self.start}→{self.max_concurrency} | SLA {' '.join(sla)}")
