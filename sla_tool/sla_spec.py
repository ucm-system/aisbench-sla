"""SLA 约束：解析与判定。

对齐 ClawPerf 的 `--slo <metric>.<agg>:<value>` 语法（多条约束 AND），
并覆盖 ais_bench 汇总表**实际提供的全部统计量**。实测 ais_bench
v3.1.20260330 的 Performance Parameters 表列：

    │ Performance Parameters │ Stage │ Average │ Min │ Max │ Median │ P75 │ P90 │ P99 │ N │

支持写法（分隔符 `.` `_` `-` 均可；阈值分隔用 `:` 或 `=`；可显式写 `<=` / `>=`）：

    --sla ttft.p90:2000        # TTFT P90 <= 2000ms
    --sla ttft.max:5000        # TTFT Max <= 5000ms
    --sla ttft.avg:1200        # TTFT Average <= 1200ms
    --sla tpot.avg:30          # TPOT Average <= 30ms
    --sla e2e.p99:20000        # E2EL P99 <= 20000ms
    --sla output_throughput.avg:>=500   # 输出吞吐 Average >= 500 tok/s（下限语义）

对应 CLI 也提供离散开关：`--ttft-avg/--ttft-max/--tpot-p90/--e2el-p99/...`。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# 指标别名 → 内部指标名（与 parse.py 的键前缀一致）
METRIC_ALIASES: Dict[str, str] = {
    "ttft": "ttft",
    "tpot": "tpot",
    "e2e": "e2el",
    "e2el": "e2el",
    "output_throughput": "output_token_throughput",
    "outputtokenthroughput": "output_token_throughput",
    "output_token_throughput": "output_token_throughput",
}
METRIC_LABELS = {"ttft": "TTFT", "tpot": "TPOT", "e2el": "E2EL",
                 "output_token_throughput": "输出吞吐"}
# 统计量别名 → 规范名（median 与 p50 是同一列，两者都接受）
AGG_ALIASES: Dict[str, str] = {
    "avg": "avg", "average": "avg", "mean": "avg",
    "min": "min", "max": "max",
    "median": "median", "p50": "p50",
    "p75": "p75", "p90": "p90", "p99": "p99",
}
AGG_ORDER = ("avg", "min", "max", "median", "p50", "p75", "p90", "p99")
UNITS = {"output_token_throughput": "tok/s"}
DEFAULT_AGG = "avg"

_SPEC_RE = re.compile(
    r"^\s*(?P<metric>[A-Za-z_]+)[.\-_](?P<agg>[A-Za-z0-9]+)\s*[:=]\s*"
    r"(?P<op><=|>=|<|>)?\s*(?P<value>[0-9]*\.?[0-9]+)\s*$")


@dataclass
class SlaRule:
    """单条 SLA 约束。latency 指标是上限（<=），吞吐是下限（>=）。"""

    key: str            # 规范键，如 ttft_p90 / output_token_throughput_avg
    metric: str
    agg: str
    limit: float
    lower_bound: bool

    @property
    def metrics_key(self) -> str:
        """在 parse.py 产出的指标字典里对应的键。"""
        return f"{self.metric}_{self.agg}" + ("" if self.metric == "output_token_throughput" else "_ms")

    def label(self) -> str:
        op = ">=" if self.lower_bound else "<="
        unit = UNITS.get(self.metric, "ms")
        agg = "AVG" if self.agg == "avg" else self.agg.upper()
        return f"{METRIC_LABELS.get(self.metric, self.metric)} {agg} {op} {self.limit:g}{unit}"

    def check(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """→ (是否满足, 说明)。指标缺失（-1/None）一律判不满足，避免静默放行。"""
        raw = metrics.get(self.metrics_key, -1)
        if raw is None or raw < 0:
            return False, f"{METRIC_LABELS.get(self.metric, self.metric)} {self.agg.upper()} 无数据"
        value = float(raw)
        op = ">=" if self.lower_bound else "<="
        ok = value >= self.limit if self.lower_bound else value <= self.limit
        if ok:
            return True, ""
        return False, f"{self.label()}（实测 {value:.4g}）"


def normalize_metric(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    if key in METRIC_ALIASES:
        return METRIC_ALIASES[key]
    raise ValueError(f"未知 SLA 指标：{name}（可用：ttft / tpot / e2e / output_throughput）")


def normalize_agg(name: str) -> str:
    key = name.strip().lower()
    if key in AGG_ALIASES:
        return AGG_ALIASES[key]
    raise ValueError(f"未知统计量：{name}（可用：{' / '.join(AGG_ORDER)}）")


def make_key(metric: str, agg: str) -> str:
    return f"{metric}_{agg}"


def split_key(key: str) -> Tuple[str, str]:
    metric, _, agg = key.rpartition("_")
    if not metric or not agg:
        raise ValueError(f"非法 SLA 键：{key}")
    return metric, agg


def parse_spec(text: str) -> Tuple[str, float]:
    """`ttft.p90:2000` / `tpot_avg=30` / `output_throughput.avg:>=500` → (key, limit)。"""
    match = _SPEC_RE.match(str(text))
    if not match:
        raise ValueError(
            f"无法解析 SLA 约束：{text!r}（示例：ttft.p90:2000 / tpot.avg:30 / "
            "e2e.max:20000 / output_throughput.avg:>=500）")
    metric = normalize_metric(match.group("metric"))
    agg = normalize_agg(match.group("agg"))
    value = float(match.group("value"))
    op = match.group("op") or ""
    is_throughput = metric == "output_token_throughput"
    if op in ("<", "<="):
        lower_bound = False
    elif op in (">", ">="):
        lower_bound = True
    else:
        lower_bound = is_throughput  # 吞吐默认下限，延迟默认上限
    if lower_bound and not is_throughput and op not in (">", ">="):
        pass  # 允许对延迟设下限（比如要求 TPOT 不低于某值，语义合法）
    if value <= 0:
        raise ValueError(f"SLA 阈值必须 > 0：{text!r}")
    return make_key(metric, agg), value


def build_rule(key: str, limit: float) -> SlaRule:
    metric, agg = split_key(key)
    normalize_metric(metric)
    normalize_agg(agg)
    lower_bound = metric == "output_token_throughput"
    return SlaRule(key=key, metric=metric, agg=agg, limit=float(limit),
                   lower_bound=lower_bound)


def build_rules(sla: Dict[str, float],
                direction_overrides: Optional[Dict[str, bool]] = None) -> List[SlaRule]:
    overrides = direction_overrides or {}
    rules = [build_rule(key, limit) for key, limit in sla.items()]
    for rule in rules:
        if rule.key in overrides:
            rule.lower_bound = overrides[rule.key]
    return rules


def describe_values(rules: List[SlaRule], metrics: Dict[str, float]) -> str:
    """把一组规则对应的实测值拼成一行说明（用于结论/报告）。"""
    parts = []
    for rule in rules:
        raw = metrics.get(rule.metrics_key, -1)
        unit = UNITS.get(rule.metric, "ms")
        agg = "AVG" if rule.agg == "avg" else rule.agg.upper()
        name = f"{METRIC_LABELS.get(rule.metric, rule.metric)} {agg}"
        if raw is None or raw < 0:
            parts.append(f"{name} 无数据")
        else:
            parts.append(f"{name} {float(raw):.4g}{unit}")
    return "，".join(parts)


def canonical_key(text: str) -> str:
    """`ttft.p90` / `ttft-p90` / `ttft_p90` / `e2e.max` → 规范键（不含阈值）。"""
    body = str(text).strip().lower().replace("-", "_").replace(".", "_")
    metric, _, agg = body.rpartition("_")
    if not metric or not agg:
        raise ValueError(f"非法 SLA 键：{text!r}（示例：ttft.p90 / tpot_avg / e2e.max）")
    return make_key(normalize_metric(metric), normalize_agg(agg))


def discrete_flag_names() -> List[str]:
    """CLI 离散开关名：--ttft-avg …（延迟三指标 × 全部统计量）。"""
    names = []
    for metric in ("ttft", "tpot", "e2el"):
        for agg in AGG_ORDER:
            names.append(f"--{metric}-{agg}")
    return names


def flag_to_key(flag: str) -> str:
    """`--ttft-p90` → `ttft_p90`（CLI 开关转规范键）。"""
    body = flag.lstrip("-").replace("-", "_")
    metric, _, agg = body.rpartition("_")
    if metric == "e2e":
        metric = "e2el"
    if not metric or not agg:
        raise ValueError(f"非法开关：{flag}")
    return make_key(normalize_metric(metric), normalize_agg(agg))
