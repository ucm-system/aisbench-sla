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
    target: str = ""                   # 统一入口：IP:PORT / host:port / http(s)://host[:port][/path]
    host: str = "127.0.0.1"
    port: int = 8000
    url: str = ""                      # 兼容旧用法（等价于 --target 传完整 URL）
    model: str = "model"               # 服务端模型名
    model_path: str = ""               # tokenizer 路径（生成数据集用）
    api_key: str = ""
    pods: List[str] = field(default_factory=list)   # metrics 端点，空则用 host:port
    npu_num: int = 1

    # ---- 负载形态 ----
    input_len: int = 2048
    output_len: int = 32
    data_num: int = 64
    request_rate: float = 3.0          # 请求频率（RPS，ais_bench 模型配置的 request_rate）；0 = 不限速
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
    reset_cache: bool = False          # 每点前 POST /reset_prefix_cache（vLLM 调试接口）
    seed_per_point: bool = True        # 默认：每点用 seed+i 生成独立数据集（保证命中率各点可比）
    dry_run: bool = False
    verbose: bool = False
    stream_logs: bool = True           # ais_bench 输出实时透传到终端（同时落盘）

    # ---- SLA 与搜索 ----
    # 规范键 → 阈值，如 {"ttft_p90": 2000, "tpot_avg": 30, "output_token_throughput_avg": 500}
    # 键由 sla_spec 解析（metric_agg），延迟为上限、吞吐为下限，多条 AND。
    # JSON 里也接受 {"ttft.p90": 2000} 或 ["ttft.p90:2000", ...] 两种写法。
    sla: Any = field(default_factory=dict)
    sla_direction: Dict[str, bool] = field(default_factory=dict)  # 显式方向覆盖（可选）
    ttft_p90: Optional[float] = None   # 兼容旧参数：等价 sla.ttft_p90（ms）
    tpot_p90: Optional[float] = None   # 兼容旧参数：等价 sla.tpot_p90（ms）
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
        # SLA：统一收敛到 sla 字典（旧字段并入，保持向后兼容）。
        # 支持三种输入：
        #   {"ttft_p90": 2000, "tpot_avg": 30}          # 键 = metric.agg / metric_agg
        #   {"ttft.p90:2000": null}                     # 键 = 完整约束串（值忽略）
        #   ["ttft.p90:2000", "tpot.avg:30"]            # 列表形式
        from . import sla_spec  # 局部导入避免循环
        raw = self.sla
        items = []
        if isinstance(raw, dict):
            items = list(raw.items())
        elif isinstance(raw, (list, tuple)):
            items = [(str(item), None) for item in raw]
        merged: Dict[str, float] = {}
        for key, limit in items:
            text = str(key).strip()
            if not text:
                continue
            if ":" in text or "=" in text:
                norm_key, value = sla_spec.parse_spec(text)
            else:
                norm_key = sla_spec.canonical_key(text)
                if limit is None:
                    raise ValueError(
                        f"SLA 配置缺少阈值：{key!r}（字典形式的值需给阈值，"
                        "或改用 'ttft.p90:2000' 这种完整串）")
                value = float(limit)
            merged[norm_key] = value
        for legacy in ("ttft_p90", "tpot_p90"):
            value = getattr(self, legacy)
            if value is not None:
                merged.setdefault(legacy, float(value))
        self.sla = merged
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
        # 目标地址归一化：统一 --target / --url / --host:--port，自动决定写
        # ais_bench 的 host_ip/host_port 还是 url（见 sla_tool/target.py 说明）
        from .target import resolve  # 局部导入避免循环
        parsed, fields = resolve(self.target, host=self.host, port=self.port,
                                 url=self.url)
        self._target_obj = parsed
        self.host = parsed.host
        self.port = int(parsed.port or (443 if parsed.effective_scheme == "https" else 80))
        self.url = str(fields["url"]) if parsed.use_url_field else ""
        if not self.dataset_path:
            self.dataset_path = os.path.join(self.output_dir, "datasets")
        self.dataset_path = os.path.abspath(self.dataset_path)
        if not self.gsm8k_source:
            self.gsm8k_source = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "GSM8K.jsonl")

    # ------------------------------------------------------------------ 校验
    def sla_rules(self):
        """→ [SlaRule]，按配置顺序。"""
        from . import sla_spec
        return sla_spec.build_rules(self.sla, self.sla_direction)

    def validate_for_search(self) -> None:
        if not self.sla:
            raise ValueError(
                "SLA 搜索至少需要一条约束，例如 --ttft-p90 2000 --tpot-avg 30，"
                "或 ClawPerf 风格 --sla ttft.p90:2000 --sla tpot.avg:30"
                "（可用统计量：avg/min/max/median/p50/p75/p90/p99）")
        for key, limit in self.sla.items():
            if limit is None or float(limit) <= 0:
                raise ValueError(f"SLA 阈值必须 > 0：{key}={limit}")
        # 触发一次解析，尽早暴露未知指标/统计量
        self.sla_rules()

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
    def model_fields(self) -> Dict[str, object]:
        """写入 ais_bench 模型配置的 host_ip / host_port / url 三件套。"""
        parsed = getattr(self, "_target_obj", None)
        if parsed is not None:
            return parsed.model_fields()
        if self.url:
            url = self.url if self.url.startswith(("http://", "https://")) else f"http://{self.url}"
            return {"host_ip": "localhost", "host_port": self.port or 80, "url": url}
        return {"host_ip": self.host, "host_port": self.port or 8000, "url": ""}

    def target_mode(self) -> str:
        parsed = getattr(self, "_target_obj", None)
        return "url" if parsed is not None and parsed.use_url_field else "host_ip"

    def metrics_pods(self) -> List[str]:
        if self.pods:
            return list(self.pods)
        parsed = getattr(self, "_target_obj", None)
        if parsed is not None:
            return [parsed.metrics_endpoint()]
        return [f"{self.host}:{self.port}"]

    def base_url(self) -> str:
        parsed = getattr(self, "_target_obj", None)
        if parsed is not None:
            return parsed.base_url().rstrip("/")
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
        sla = [rule.label() for rule in self.sla_rules()]
        return (f"目标 {self.base_url()} model={self.model} | "
                f"输入{self.input_len}/输出{self.output_len}/条数{self.data_num} | "
                f"并发{self.start}→{self.max_concurrency} | SLA {' '.join(sla)}")
