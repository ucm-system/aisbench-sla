"""vLLM /metrics 前缀缓存指标抓取与快照差分。

计数口径（与参考工具一致）：
    hit rate = (测试后累计 hits - 测试前累计 hits) / (测试后累计 queries - 测试前累计 queries)

同时区分两类缓存：
    vllm:prefix_cache_*            —— HBM（显存内）前缀缓存
    external_prefix_cache_*        —— 外部缓存（如 UCM），不存在时报告为空
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_ENGINE_RE = re.compile(r'engine="(\d+)"')

FAMILIES = {
    "hbm_q": "vllm:prefix_cache_queries_total",
    "hbm_h": "vllm:prefix_cache_hits_total",
    "ext_q": "external_prefix_cache_queries_total",
    "ext_h": "external_prefix_cache_hits_total",
}


def _requests():
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("缺少依赖 requests：pip install requests") from exc
    return requests


def fetch_metrics_text(host: str, port: str, timeout: float = 10.0) -> str:
    """抓取 /metrics（显式绕过系统代理，内网地址常被代理劫持）。"""
    url = f"http://{host}:{port}/metrics"
    response = _requests().get(url, proxies={"http": None, "https": None}, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_metrics(text: str) -> Dict[str, Dict[str, float]]:
    """→ {engine_id: {hbm_q, hbm_h, ext_q, ext_h, running, waiting}}

    兼容两种导出风格：带 `engine="N"` 标签的（多引擎/多 DP）与不带该标签的
    （部分 vLLM 变体与 mock 服务只给 model_name）——缺标签时按 engine="0" 归并。
    """
    result: Dict[str, Dict[str, float]] = {}
    for line in text.splitlines():
        if line.startswith("#") or "{" not in line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = float(parts[-1])
        except ValueError:
            continue
        if value.is_integer():
            value = int(value)
        engine_match = _ENGINE_RE.search(line)
        matched = False
        for key, metric in FAMILIES.items():
            if line.startswith(metric) or f" {metric}" in line:
                bucket = result.setdefault(engine_match.group(1) if engine_match else "0", {})
                bucket[key] = bucket.get(key, 0) + value
                matched = True
                break
        if matched:
            continue
        if "num_requests_running" in line or "num_requests_waiting" in line:
            bucket = result.setdefault(engine_match.group(1) if engine_match else "0", {})
            key = "running" if "num_requests_running" in line else "waiting"
            bucket[key] = max(bucket.get(key, 0), value)
    return result


def snapshot_pods(pods: List[str], timeout: float = 10.0
                  ) -> Dict[str, Dict[str, Dict[str, float]]]:
    """→ {pod: {engine: counters}}；抓取失败的 pod 记为 {}。"""
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pod in pods:
        host, _, port = pod.rpartition(":")
        if not host:
            host, port = pod, "80"
        try:
            result[pod] = parse_metrics(fetch_metrics_text(host, port, timeout))
        except Exception as exc:  # noqa: BLE001 - 单个 pod 失败不影响其他
            logger.warning("抓取 %s 指标失败：%s", pod, exc)
            result[pod] = {}
    return result


def _rate(hits: float, queries: float) -> Optional[float]:
    if not queries:
        return None
    return hits / queries


@dataclass
class HitRate:
    """一次压测阶段的命中率结果（快照差分）。"""

    per_pod: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    hbm_hits: float = 0.0
    hbm_queries: float = 0.0
    ext_hits: float = 0.0
    ext_queries: float = 0.0
    skipped: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def hbm_hit_rate(self) -> Optional[float]:
        return _rate(self.hbm_hits, self.hbm_queries)

    @property
    def ext_hit_rate(self) -> Optional[float]:
        return _rate(self.ext_hits, self.ext_queries)

    def as_dict(self) -> dict:
        return {
            "hbm_hits": self.hbm_hits, "hbm_queries": self.hbm_queries,
            "hbm_hit_rate": self.hbm_hit_rate,
            "ext_hits": self.ext_hits, "ext_queries": self.ext_queries,
            "ext_hit_rate": self.ext_hit_rate,
            "per_pod": self.per_pod, "skipped_pods": self.skipped, "notes": self.notes,
        }


def diff_snapshots(before: Dict[str, Dict[str, Dict[str, float]]],
                   after: Dict[str, Dict[str, Dict[str, float]]]) -> HitRate:
    """计算两帧快照之间四类计数的增量与命中率（按 pod×engine 汇总）。"""
    result = HitRate()
    for pod, engines_after in sorted(after.items()):
        engines_before = before.get(pod, {})
        if not engines_after or not engines_before:
            result.skipped.append(pod)
            continue
        pod_rows: Dict[str, Dict[str, float]] = {}
        pod_totals = {"hbm_q": 0.0, "hbm_h": 0.0, "ext_q": 0.0, "ext_h": 0.0}
        missing_ext = True
        for engine in sorted(set(engines_after) | set(engines_before)):
            cur = engines_after.get(engine)
            prev = engines_before.get(engine)
            if cur is None or prev is None:
                result.notes.append(f"{pod}/engine{engine} 快照不完整，已跳过")
                continue
            row = {
                "hbm_q": max(0.0, cur.get("hbm_q", 0) - prev.get("hbm_q", 0)),
                "hbm_h": max(0.0, cur.get("hbm_h", 0) - prev.get("hbm_h", 0)),
                "ext_q": max(0.0, cur.get("ext_q", 0) - prev.get("ext_q", 0)),
                "ext_h": max(0.0, cur.get("ext_h", 0) - prev.get("ext_h", 0)),
            }
            if cur.get("ext_q") or prev.get("ext_q"):
                missing_ext = False
            row["hbm_hit_rate"] = _rate(row["hbm_h"], row["hbm_q"])
            row["ext_hit_rate"] = _rate(row["ext_h"], row["ext_q"])
            pod_rows[f"engine{engine}"] = row
            for key in pod_totals:
                pod_totals[key] += row[key]
        if not pod_rows:
            result.skipped.append(pod)
            continue
        pod_rows["ALL"] = {
            **pod_totals,
            "hbm_hit_rate": _rate(pod_totals["hbm_h"], pod_totals["hbm_q"]),
            "ext_hit_rate": _rate(pod_totals["ext_h"], pod_totals["ext_q"]),
        }
        if missing_ext:
            result.notes.append(f"{pod} 无 external_prefix_cache 指标（非 UCM 场景属正常）")
        result.per_pod[pod] = pod_rows
        result.hbm_hits += pod_totals["hbm_h"]
        result.hbm_queries += pod_totals["hbm_q"]
        result.ext_hits += pod_totals["ext_h"]
        result.ext_queries += pod_totals["ext_q"]
    hbm_rate = result.hbm_hit_rate
    if hbm_rate is not None and hbm_rate > 1.0:
        # 真实 vLLM 不会 >100%；出现说明服务端计数器口径异常（如 mock 自身统计方式），
        # 这里如实标注而不做截断，避免掩盖数据问题
        result.notes.append(
            f"HBM 命中率 {(hbm_rate * 100):.2f}% > 100%：服务端计数器口径可能异常，请核对")
    return result


class GaugeSampler:
    """压测期间后台采样在途请求数。

    注意：这是**采样值**，请求窗口很短时（小模型/短输出）可能整个窗口都落在
    两次采样之间而采到 0 —— 「并发是否真压上去」的权威依据是 ais_bench 汇总里的
    实测并发（measured_concurrency），本采样只作辅助证据，默认 0.5s 一次。
    """

    def __init__(self, pods: List[str], interval: float = 0.5) -> None:
        self.pods = pods
        self.interval = max(0.2, interval)
        self.max_running = 0.0
        self.max_waiting = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                for pod in self.pods:
                    host, _, port = pod.rpartition(":")
                    text = fetch_metrics_text(host or pod, port or "80", timeout=5.0)
                    for counters in parse_metrics(text).values():
                        self.max_running = max(self.max_running, counters.get("running", 0))
                        self.max_waiting = max(self.max_waiting, counters.get("waiting", 0))
                self.samples += 1
            except Exception:  # noqa: BLE001 - 采样失败静默重试
                pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="gauge-sampler")
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 2)
        return {"max_running": self.max_running, "max_waiting": self.max_waiting,
                "samples": self.samples}


def reset_prefix_cache(base_url: str, api_key: str = "", timeout: float = 10.0) -> bool:
    """POST /reset_prefix_cache（服务不支持时返回 False，不视为错误）。"""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = _requests().post(f"{base_url.rstrip('/')}/reset_prefix_cache",
                                    headers=headers,
                                    proxies={"http": None, "https": None},
                                    timeout=timeout)
        return response.status_code < 400
    except Exception as exc:  # noqa: BLE001
        logger.warning("reset_prefix_cache 失败（忽略）：%s", exc)
        return False


def probe_service(base_url: str, api_key: str = "", timeout: float = 10.0
                  ) -> Tuple[bool, str]:
    """/v1/models 可达性探测，返回 (是否可达, 摘要)。"""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = _requests().get(f"{base_url.rstrip('/')}/v1/models", headers=headers,
                                   proxies={"http": None, "https": None}, timeout=timeout)
        if response.status_code >= 400:
            return False, f"HTTP {response.status_code}"
        try:
            models = [m.get("id") for m in response.json().get("data", [])]
        except Exception:  # noqa: BLE001
            models = []
        return True, "模型：" + (", ".join(str(m) for m in models[:5]) or "未知")
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def wait_for_run_metrics(pods: List[str], timeout: float = 10.0) -> None:
    """等待首个 pod 的指标可用（避免拿不到差分基线）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        frame = snapshot_pods(pods[:1], timeout=5.0)
        if any(frame.values()):
            return
        time.sleep(0.5)
