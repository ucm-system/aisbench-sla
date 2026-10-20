"""SLA 自动搜索：预检 -> 指数阶梯（×2）-> 二分收敛。

判定口径（按需求：只看延迟）：
    通过 <=> returncode == 0 且（若设定）TTFT P90 <= 阈值 且 TPOT P90 <= 阈值
命中率不参与判定，仅作为前缀缓存场景的证据随点输出（每个点都会算）。

策略说明（沿用本项目既有实现，已在真实服务验证）：
1. 预检：若 --start > 1，先在并发=1 跑一次。延迟最低的点都不达标，说明阈值与
   服务能力矛盾，直接结束（退出码 2），不浪费后续探针。
2. 阶梯：从 --start 起每次 ×2，直到某点不达标或到达 --max。
3. 二分：在「最后一个通过 (lo)」与「第一个不达标 (hi)」之间收敛，直到区间
   ≤ max(--tolerance, lo//8) 或达到 --max-bisect 次、或探针预算耗尽。
"""
from __future__ import annotations

import ipaddress
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import aisbench, dataset, metrics
from .parse import parse_aisbench_log, looks_like_bad_num_warmups

logger = logging.getLogger(__name__)


@dataclass
class PointResult:
    concurrency: int
    ok: bool = False
    reason: str = ""
    kind: str = "probe"                     # probe | preflight
    returncode: Optional[int] = None
    elapsed_s: float = 0.0
    log_path: str = ""
    run_dir: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    hit_rate: Dict[str, object] = field(default_factory=dict)
    gauges: Dict[str, float] = field(default_factory=dict)
    warmup_hit_rate: Optional[float] = None
    output_len: int = 0

    # ------------------------------------------------------------- SLA 判定
    def failing_keys(self, cfg) -> List[str]:
        bad: List[str] = []
        if self.returncode not in (0, None):
            bad.append(f"进程退出码 {self.returncode}")
        if cfg.ttft_p90 is not None:
            value = self.metrics.get("ttft_p90_ms", -1)
            if value is None or value < 0:
                bad.append("TTFT P90 无数据")
            elif value > cfg.ttft_p90:
                bad.append(f"TTFT P90 {value:.0f} > {cfg.ttft_p90:.0f}")
        if cfg.tpot_p90 is not None:
            value = self.metrics.get("tpot_p90_ms", -1)
            if value is None or value < 0:
                bad.append("TPOT P90 无数据")
            elif value > cfg.tpot_p90:
                bad.append(f"TPOT P90 {value:.0f} > {cfg.tpot_p90:.0f}")
        return bad

    def as_row(self) -> Dict[str, object]:
        hit = self.hit_rate or {}
        notes = list(hit.get("notes") or [])
        skipped = list(hit.get("skipped_pods") or [])
        if skipped:
            notes.append("指标未取到（快照不完整）：" + ",".join(skipped))
        return {
            "kind": self.kind,
            "concurrency": self.concurrency,
            "output_len": self.output_len,
            "verdict": "PASS" if self.ok else "FAIL",
            "reason": self.reason,
            "ttft_avg_ms": self.metrics.get("ttft_avg_ms", -1),
            "ttft_p90_ms": self.metrics.get("ttft_p90_ms", -1),
            "tpot_avg_ms": self.metrics.get("tpot_avg_ms", -1),
            "tpot_p90_ms": self.metrics.get("tpot_p90_ms", -1),
            "e2el_p90_ms": self.metrics.get("e2el_p90_ms", -1),
            "output_token_throughput": self.metrics.get("output_token_throughput", -1),
            "request_throughput_qps": self.metrics.get("request_throughput_qps", -1),
            "total_requests": self.metrics.get("total_requests", -1),
            "measured_concurrency": self.metrics.get("measured_concurrency", -1),
            "hbm_hit_rate": hit.get("hbm_hit_rate"),
            "ext_hit_rate": hit.get("ext_hit_rate"),
            "hit_rate_note": "；".join(notes),
            "warmup_hit_rate": self.warmup_hit_rate,
            "max_running": self.gauges.get("max_running", 0),
            "elapsed_s": round(self.elapsed_s, 1),
            "log_path": self.log_path,
        }


@dataclass
class SearchResult:
    points: List[PointResult] = field(default_factory=list)
    max_ok: Optional[int] = None
    status: str = "ok"          # ok | preflight_failed | none_passed | max_unbounded
                                # | dry_run | error
    note: str = ""
    dataset_info: Dict[str, str] = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""

    @property
    def exit_code(self) -> int:
        if self.status == "error":
            return 1
        if self.status == "preflight_failed":
            return 2
        if self.status in ("none_passed",):
            return 3
        return 0


class SlaSearch:
    def __init__(self, cfg, on_point: Optional[Callable[[PointResult], None]] = None,
                 on_event: Optional[Callable[[str], None]] = None) -> None:
        self.cfg = cfg
        self.on_point = on_point
        self.on_event = on_event or (lambda message: logger.info("%s", message))
        self.full_probes = 0
        self._warmup_done = False
        self.prefix_path = ""
        self.dataset_path = ""

    # ---------------------------------------------------------------- 执行
    def run(self) -> SearchResult:
        result = SearchResult(started_at=time.time())
        cfg = self.cfg
        try:
            cfg.validate_for_search()
            cfg.validate_for_run()
        except ValueError as exc:
            result.status = "error"
            result.error = str(exc)
            return result
        try:
            self._prepare()
            result.dataset_info = {"prefix_path": self.prefix_path,
                                   "dataset_path": self.dataset_path}
            if cfg.dry_run:
                result.status = "dry_run"
                result.note = "dry-run：未执行任何压测"
                return result
            self._preflight(result)
            if result.status in ("preflight_failed", "error"):
                return result
            self._ladder(result)
            self._bisect(result)
            self._finalize(result)
        except Exception as exc:  # noqa: BLE001 - CLI 需给出可读错误
            logger.exception("SLA 搜索失败")
            result.status = "error"
            result.error = str(exc)
        finally:
            result.finished_at = time.time()
            if result.status in ("ok", "none_passed", "max_unbounded"):
                self._finalize_note(result)
        return result

    # ---------------------------------------------------------------- 准备
    def _prepare(self) -> None:
        cfg = self.cfg
        os.makedirs(cfg.output_dir, exist_ok=True)
        self._check_target_host()
        reachable, message = metrics.probe_service(cfg.base_url(), cfg.api_key)
        if reachable:
            self.on_event(f"目标服务可达：{cfg.base_url()}（{message}）")
        else:
            self.on_event(f"目标服务探测失败（{cfg.base_url()}）：{str(message)[:160]}　"
                          + ("dry-run 继续展示计划" if cfg.dry_run else "仍继续尝试压测"))

        if cfg.dry_run:
            self.prefix_path, self.dataset_path = (cfg.dataset, cfg.dataset)
            return
        self.prefix_path, self.dataset_path = dataset.ensure_dataset(cfg)
        metrics.wait_for_run_metrics(cfg.metrics_pods(), timeout=15.0)

    def _check_target_host(self) -> None:
        """报告最终寻址方式，并提醒 ais_bench 的 host_ip 校验限制。

        归一化后主机名一律走 url 字段（host_ip 只写 "localhost"），因此不会
        再触发 [UTILS-CFG-002]；这里仅把结果说清楚，便于对照日志排查。
        """
        cfg = self.cfg
        parsed = getattr(cfg, "_target_obj", None)
        if parsed is None:
            return
        if parsed.use_url_field:
            self.on_event(f"寻址方式：ais_bench url 字段 = {cfg.url}"
                          f"（host_ip 写 localhost 以通过校验；原始输入 {parsed.raw!r}）")
        else:
            self.on_event(f"寻址方式：ais_bench host_ip={cfg.host} host_port={cfg.port}"
                          f"（原始输入 {parsed.raw!r}）")

    # ---------------------------------------------------------------- 单点
    def _run_phase(self, run_dir: str, tag: str, concurrency: int, output_len: int,
                   request_rate: float, dataset_file: str) -> tuple:
        """执行一个阶段（预热或全量），返回 (CommandResult, 指标, 命中率, 采样)。"""
        cfg = self.cfg
        aisbench.install_model_config(
            cfg.work_path,
            aisbench.render_model_config(cfg, concurrency, output_len, request_rate))
        aisbench.mount_dataset(cfg.work_path, dataset_file)
        command = aisbench.build_command(cfg, run_dir, with_num_warmups=True)
        log_path = os.path.join(run_dir, f"{tag}.log")
        before = metrics.snapshot_pods(cfg.metrics_pods())
        sampler = metrics.GaugeSampler(cfg.metrics_pods())
        sampler.start()
        try:
            command_result = aisbench.run_command(command, log_path, cwd=run_dir,
                                                  timeout=cfg.point_timeout,
                                                  verbose=cfg.verbose)
            if (not command_result.ok and cfg.num_warmups
                    and looks_like_bad_num_warmups(log_path)):
                self.on_event("检测到 ais_bench 不识别 --num-warmups，去掉该参数重试"
                              "（首次日志保留为 .retry0.log）")
                retry_log = os.path.join(run_dir, f"{tag}.retry0.log")
                os.replace(log_path, retry_log)
                command = aisbench.build_command(cfg, run_dir, with_num_warmups=False)
                command_result = aisbench.run_command(command, log_path, cwd=run_dir,
                                                      timeout=cfg.point_timeout,
                                                      verbose=cfg.verbose)
        finally:
            gauges = sampler.stop()
        after = metrics.snapshot_pods(cfg.metrics_pods())
        hit = metrics.diff_snapshots(before, after)
        parsed, _ = parse_aisbench_log(log_path, npu_num=cfg.npu_num)
        return command_result, parsed, hit, gauges

    def _probe(self, concurrency: int, kind: str = "probe") -> PointResult:
        cfg = self.cfg
        started = time.time()
        run_dir = os.path.join(cfg.output_dir, "points", f"c{concurrency}")
        os.makedirs(run_dir, exist_ok=True)
        point = PointResult(concurrency=concurrency, kind=kind, run_dir=run_dir,
                           output_len=cfg.output_len)

        if cfg.reset_cache:
            metrics.reset_prefix_cache(cfg.base_url(), cfg.api_key)

        # ---- 前缀预热阶段 ----
        do_warmup = (cfg.dataset_type == "prefix_cache" and cfg.prefix_warmup != "none"
                     and bool(self.prefix_path))
        if do_warmup and cfg.prefix_warmup == "once" and self._warmup_done:
            do_warmup = False
        if do_warmup:
            warm_c = cfg.dp if cfg.dp > 1 else cfg.prefix_num
            self.on_event(f"[c={concurrency}] 前缀预热阶段（并发={warm_c}, output_len=1）")
            warm_result, _, warm_hit, _ = self._run_phase(
                run_dir, "warmup", warm_c, 1, 0.0, self.prefix_path)
            point.warmup_hit_rate = warm_hit.hbm_hit_rate
            self._warmup_done = True
            if not warm_result.ok:
                logger.warning("前缀预热阶段失败（returncode=%s）：%s",
                               warm_result.returncode, aisbench.tail(warm_result.log_path, 10))

        # ---- 全量阶段 ----
        self.on_event(f"[c={concurrency}] 全量阶段（output_len={cfg.output_len}, "
                      f"request_rate={cfg.request_rate}）")
        command_result, parsed, hit, gauges = self._run_phase(
            run_dir, "full", concurrency, cfg.output_len, cfg.request_rate,
            self.dataset_path)
        point.returncode = command_result.returncode
        point.log_path = os.path.join(run_dir, "full.log")
        point.elapsed_s = time.time() - started
        point.metrics = parsed
        point.hit_rate = hit.as_dict()
        point.gauges = gauges

        if command_result.timed_out:
            point.ok = False
            point.reason = f"单点超时（>{cfg.point_timeout:.0f}s）"
        else:
            bad = point.failing_keys(cfg)
            point.ok = not bad
            point.reason = "SLA 满足" if point.ok else "；".join(bad)
        if not point.ok and command_result.returncode not in (0, None):
            point.reason = (point.reason + "；日志尾部：" +
                            aisbench.tail(point.log_path, 3).strip().replace("\n", " "))[:300]
        return point

    def _record(self, result: SearchResult, point: PointResult) -> None:
        result.points.append(point)
        if point.kind != "warmup":
            self.full_probes += 1
        if self.on_point:
            self.on_point(point)

    def _budget_left(self) -> bool:
        return self.full_probes < self.cfg.max_probes

    # ---------------------------------------------------------------- 三阶段
    def _preflight(self, result: SearchResult) -> None:
        cfg = self.cfg
        if cfg.start <= 1:
            return
        self.on_event("预检：并发=1 探测阈值是否可达（延迟最低点）")
        point = self._probe(1, kind="preflight")
        self._record(result, point)
        if not point.ok:
            result.status = "preflight_failed"
            result.max_ok = 0
            result.note = (f"预检失败：并发=1 仍不满足 SLA（{point.reason}）。"
                           "阈值与服务实际能力矛盾，请放宽 --ttft-p90/--tpot-p90 后重试")
            self.on_event(result.note)
        else:
            # 预检本身就是"已知可行"的并发点：记入 max_ok，避免预算不足时
            # 误报"没有任何并发点满足 SLA"（预检行明明是 PASS）
            result.max_ok = 1

    def _ladder(self, result: SearchResult) -> None:
        cfg = self.cfg
        current = cfg.start
        while current <= cfg.max_concurrency:
            if not self._budget_left():
                self.on_event(f"探针预算用尽（--max-probes {cfg.max_probes}）")
                break
            point = self._probe(current)
            self._record(result, point)
            if point.ok:
                result.max_ok = current
                nxt = current * 2
                if nxt > cfg.max_concurrency:
                    result.status = "max_unbounded"
                    result.note = (f"并发达上限 {cfg.max_concurrency} 仍满足 SLA，"
                                   "可提高 --max 继续探测")
                    return
                current = nxt
            else:
                break
        else:
            if result.max_ok is None:
                result.status = "none_passed"
                result.note = f"起始并发 {cfg.start} 即不满足 SLA"

    def _bisect(self, result: SearchResult) -> None:
        cfg = self.cfg
        if result.status in ("preflight_failed", "max_unbounded", "none_passed"):
            return
        lows = [p for p in result.points if p.ok]
        highs = [p for p in result.points if not p.ok and p.kind != "preflight"]
        if not lows or not highs:
            return
        lo = max(p.concurrency for p in lows)
        hi = min(p.concurrency for p in highs if p.concurrency > lo)
        if hi is None or hi - lo <= 1:
            return
        iterations = 0
        while hi - lo > max(cfg.tolerance, lo // 8) and iterations < cfg.max_bisect:
            if not self._budget_left():
                self.on_event("探针预算用尽，二分提前结束")
                break
            mid = (lo + hi) // 2
            if mid in (lo, hi):
                break
            point = self._probe(mid)
            self._record(result, point)
            if point.ok:
                lo = mid
                result.max_ok = max(result.max_ok or 0, mid)
            else:
                hi = mid
            iterations += 1

    def _finalize(self, result: SearchResult) -> None:
        cfg = self.cfg
        if result.status in ("preflight_failed", "error", "dry_run"):
            return
        if result.max_ok is None:
            result.status = "none_passed"
        elif result.max_ok >= cfg.max_concurrency:
            result.status = "max_unbounded"
        else:
            result.status = "ok"

    def _finalize_note(self, result: SearchResult) -> None:
        if result.note:
            return
        passing = [p for p in result.points if p.ok and p.kind != "preflight"]
        failing = [p for p in result.points if not p.ok and p.kind != "preflight"]
        if result.max_ok:
            best = max(passing, key=lambda p: p.concurrency) if passing else None
            detail = ""
            if best:
                detail = (f"（TTFT P90 {best.metrics.get('ttft_p90_ms', -1):.0f}ms，"
                          f"TPOT P90 {best.metrics.get('tpot_p90_ms', -1):.0f}ms，"
                          f"HBM 命中率 {self._pct(best.hit_rate.get('hbm_hit_rate'))}）")
            result.note = f"SLA 内最大并发 = {result.max_ok}{detail}"
            if failing:
                first_bad = min(failing, key=lambda p: p.concurrency)
                result.note += f"；首个不达标点 c={first_bad.concurrency}（{first_bad.reason}）"
        else:
            result.note = "没有任何并发点满足 SLA"

    @staticmethod
    def _pct(value) -> str:
        return "无数据" if value is None else f"{float(value) * 100:.2f}%"


def dry_run_points(cfg) -> List[int]:
    """dry-run 时展示的阶梯探测点。"""
    points, current = [], cfg.start
    while current <= cfg.max_concurrency and len(points) < 12:
        points.append(current)
        current *= 2
    return points
