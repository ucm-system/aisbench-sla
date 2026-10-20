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

from . import aisbench, console, dataset, logs, metrics
from . import __version__
from .parse import METRIC_DEFAULTS, parse_aisbench_log, looks_like_bad_num_warmups

logger = logging.getLogger(__name__)

# 汇总/CSV 里导出的全部指标键（= ais_bench 汇总表提供的统计量）
ALL_METRIC_KEYS = tuple(METRIC_DEFAULTS.keys())


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
    seed: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    hit_rate: Dict[str, object] = field(default_factory=dict)
    gauges: Dict[str, float] = field(default_factory=dict)
    warmup_hit_rate: Optional[float] = None
    output_len: int = 0
    request_rate: float = 0.0

    # ------------------------------------------------------------- SLA 判定
    def failing_keys(self, cfg) -> List[str]:
        """按配置的 SLA 规则逐条判定，返回未满足项的可读说明。"""
        bad: List[str] = []
        if self.returncode not in (0, None):
            bad.append(f"进程退出码 {self.returncode}")
        for rule in cfg.sla_rules():
            ok, detail = rule.check(self.metrics)
            if not ok:
                bad.append(detail)
        return bad

    def as_row(self) -> Dict[str, object]:
        hit = self.hit_rate or {}
        notes = list(hit.get("notes") or [])
        skipped = list(hit.get("skipped_pods") or [])
        if skipped:
            notes.append("指标未取到（快照不完整）：" + ",".join(skipped))
        row: Dict[str, object] = {
            "kind": self.kind,
            "concurrency": self.concurrency,
            "seed": self.seed,
            "output_len": self.output_len,
            "verdict": "PASS" if self.ok else "FAIL",
            "reason": self.reason,
            "output_token_throughput": self.metrics.get("output_token_throughput", -1),
            "request_throughput_qps": self.metrics.get("request_throughput_qps", -1),
            "request_rate": self.request_rate,
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
        # ais_bench 汇总表里的全部统计量（avg/min/max/median/p50/p75/p90/p99 …）
        for key in ALL_METRIC_KEYS:
            row.setdefault(key, self.metrics.get(key, -1))
        return row


@dataclass
class SearchResult:
    points: List[PointResult] = field(default_factory=list)
    max_ok: Optional[int] = None
    status: str = "ok"          # ok | preflight_failed | none_passed | max_unbounded
                                # | dry_run | error
    note: str = ""
    warnings: List[str] = field(default_factory=list)
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
        self._probe_seq = 0
        self._round = 0
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
            self._collect_warnings(result)
            if result.status in ("ok", "none_passed", "max_unbounded"):
                self._finalize_note(result)
        return result

    def _collect_warnings(self, result: SearchResult) -> None:
        """命中率可比性告警。"""
        self._warn_hit_rate_saturation(result)
        self._warn_rate_capped(result)

    def _warn_rate_capped(self, result: SearchResult) -> None:
        """请求频率成为瓶颈时告警：rate × 时延 < 目标并发 → 并发阶梯压不满。

        例：--request-rate 3 配 500ms 时延，在途并发最多约 1.5，此时「最大并发」
        测的是频率上限而不是服务能力，实测并发列会明显低于目标并发。
        """
        cfg = self.cfg
        if cfg.request_rate <= 0:
            return
        capped = []
        for point in result.points:
            if point.kind != "probe":
                continue
            measured = point.metrics.get("measured_concurrency", -1)
            if measured is None or measured < 0:
                continue
            if measured < 0.6 * point.concurrency:
                capped.append((point.concurrency, measured))
        if len(capped) < 2:
            return
        sample = "，".join(f"c={c} 实测 {m:.1f}" for c, m in capped[:3])
        result.warnings.append(
            f"实测并发明显低于目标并发（{sample} …）：请求频率 --request-rate "
            f"{cfg.request_rate:g} 成了压力瓶颈（频率 × 时延 < 目标并发），并发阶梯压不满，"
            "此时的结论反映的是该频率下的表现，而不是并发上限。要测并发上限请调大 "
            "--request-rate，或设为 0（不限速）。")

    def _warn_hit_rate_saturation(self, result: SearchResult) -> None:
        """识别「命中率逐点饱和」：各点缓存状态不同会让命中率失去可比性。

        典型成因：同一份数据集在探测点之间反复播放，缓存(即使被预热过)会逐步累积
        到整行命中 -> 命中率趋近 100%，首点与后续点不可比。
        """
        rates = [(p.concurrency, (p.hit_rate or {}).get("hbm_hit_rate"))
                 for p in result.points if p.kind != "warmup"]
        rates = sorted((c, float(r)) for c, r in rates if r is not None)
        if len(rates) < 3:
            return
        first_rate, last_rate = rates[0][1], rates[-1][1]
        saturated = sum(1 for _, r in rates if r >= 0.99)
        if saturated >= 2 and last_rate - first_rate > 0.05:
            if self.cfg.seed_per_point:
                fix = ("已开每点换 seed 仍出现饱和，说明各点前缀仍被复用"
                       "（请检查数据集是否被沿用）")
            elif self.cfg.reset_cache:
                fix = ("去掉 --no-seed-per-point，恢复默认的每点换 seed；"
                       "注意 /reset_prefix_cache 是调试接口且只清显存侧缓存，"
                       "外部缓存命中不受影响")
            else:
                fix = ("去掉 --no-seed-per-point，恢复默认的每点换 seed"
                       "（不依赖服务支持，HBM 与外部缓存都从「该前缀从未出现过」起测）")
            result.warnings.append(
                f"命中率逐点上升并趋近 100%（首点 {first_rate:.1%} → 末点 {last_rate:.1%}，"
                f"{saturated}/{len(rates)} 个点 ≥99%）：缓存未在探测点之间复位，"
                f"各点的缓存状态不同、命中率不可直接比较。建议：{fix}。")

    # ---------------------------------------------------------------- 准备
    def _prepare(self) -> None:
        cfg = self.cfg
        os.makedirs(cfg.output_dir, exist_ok=True)
        if not cfg.dry_run:
            # 尽早挂上文件日志，让 run.log / aisbench_all.log 覆盖从第一行起的全部消息
            paths = logs.setup_run_logging(cfg.output_dir, cfg.describe())
            logger.debug("运行日志：%s / %s", paths["run_log"], paths["all_log"])
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
        self._check_hit_rate_regime()
        metrics.wait_for_run_metrics(cfg.metrics_pods(), timeout=15.0)

    def _check_hit_rate_regime(self) -> None:
        """开跑前提示命中率的可比性风险（默认每点换 seed，故仅在被显式关闭时提示）。"""
        cfg = self.cfg
        if cfg.dataset_type != "prefix_cache" or cfg.seed_per_point:
            return
        if cfg.reset_cache:
            self.on_event(
                "提示：已关闭每点独立数据集（--no-seed-per-point）并启用 --reset-cache。"
                "注意 vLLM 的 /reset_prefix_cache 是调试接口（需 VLLM_SERVER_DEV_MODE=1）且"
                "只清显存侧缓存——测外部缓存(UCM)命中时它无效，此类场景请去掉 "
                "--no-seed-per-point，改用默认的每点换 seed。")
            return
        self.on_event(
            "提示：已关闭每点独立数据集（--no-seed-per-point），各点复用同一份数据集且未复位"
            "缓存 —— 第 1 点是冷缓存，后续点命中率会逐点饱和到 ~100%，各点不可比。"
            "去掉该参数即可恢复默认的「每点换 seed」（通用、不依赖服务）。")

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
        all_log = os.path.join(cfg.output_dir, "aisbench_all.log")
        before = metrics.snapshot_pods(cfg.metrics_pods())
        sampler = metrics.GaugeSampler(cfg.metrics_pods())
        sampler.start()
        try:
            command_result = aisbench.run_command(
                command, log_path, cwd=run_dir, timeout=cfg.point_timeout,
                verbose=cfg.verbose, stream=cfg.stream_logs, all_log=all_log)
            if (not command_result.ok and cfg.num_warmups
                    and looks_like_bad_num_warmups(log_path)):
                self.on_event("检测到 ais_bench 不识别 --num-warmups，去掉该参数重试"
                              "（首次日志保留为 .retry0.log）")
                retry_log = os.path.join(run_dir, f"{tag}.retry0.log")
                os.replace(log_path, retry_log)
                command = aisbench.build_command(cfg, run_dir, with_num_warmups=False)
                command_result = aisbench.run_command(
                    command, log_path, cwd=run_dir, timeout=cfg.point_timeout,
                    verbose=cfg.verbose, stream=cfg.stream_logs, all_log=all_log)
        finally:
            gauges = sampler.stop()
        after = metrics.snapshot_pods(cfg.metrics_pods())
        hit = metrics.diff_snapshots(before, after)
        parsed, log_dir = parse_aisbench_log(log_path, npu_num=cfg.npu_num)
        aisbench.copy_log_to_exp_folder(log_path, log_dir, tag)
        return command_result, parsed, hit, gauges

    def _point_dataset(self, concurrency: int) -> tuple:
        """该探测点使用的（前缀文件, 数据集, seed）。

        `--seed-per-point` 下每点换 seed 生成独立数据集：各点前缀对缓存而言都是新的，
        即便服务不支持 /reset_prefix_cache，命中率也不会因复用同一份数据集而逐点饱和。
        代价是每点都要重新生成数据集（64K 长上下文尤其明显）。
        """
        cfg = self.cfg
        seed = cfg.seed + self._probe_seq
        if not cfg.seed_per_point:
            return self.prefix_path, self.dataset_path, cfg.seed
        save_path = os.path.join(cfg.output_dir, "datasets", f"seed{seed}")
        prefix_path, dataset_path = dataset.ensure_seed_dataset(cfg, seed, save_path)
        self.on_event(f"[c={concurrency}] 每点独立数据集：seed={seed}"
                      "（各点前缀对缓存都是新的，命中率可比）")
        return prefix_path, dataset_path, seed

    def _probe(self, concurrency: int, kind: str = "probe") -> PointResult:
        cfg = self.cfg
        started = time.time()
        run_dir = os.path.join(cfg.output_dir, "points", f"c{concurrency}")
        os.makedirs(run_dir, exist_ok=True)
        self._round += 1
        total = cfg.max_probes
        # 先备好本点数据集（可能耗时），再打轮次框，避免准备日志夹在框线中间
        point_prefix, point_dataset, point_seed = self._point_dataset(concurrency)
        self._probe_seq += 1
        console.banner(
            f"第 {self._round}/{total} 次探测 ┃ 并发 c={concurrency}"
            + (f" ┃ 数据集 seed={point_seed}"
               if cfg.seed_per_point else " ┃ 复用数据集"),
            right=f"aisbench-sla {__version__} ┃ {time.strftime('%H:%M:%S')}")
        console.emit(f"  SLA 口径：{'；'.join(rule.label() for rule in cfg.sla_rules())}",
                     (console.DIM,))
        console.emit(f"  请求频率：{cfg.request_rate:g} RPS"
                     + ("（不限速）" if cfg.request_rate <= 0 else "")
                     + f" ┃ 负载：输入 {cfg.input_len} / 输出 {cfg.output_len}"
                     f" / {cfg.data_num} 条 ┃ 目标 {cfg.base_url()}", (console.DIM,))
        point = PointResult(concurrency=concurrency, kind=kind, run_dir=run_dir,
                            output_len=cfg.output_len, seed=point_seed,
                            request_rate=cfg.request_rate)

        if cfg.reset_cache:
            ok, detail = metrics.reset_prefix_cache(cfg.base_url(), cfg.api_key)
            self.on_event(f"[c={concurrency}] 缓存复位{'成功' if ok else '未生效'}：{detail}")

        # ---- 前缀预热阶段 ----
        do_warmup = (cfg.dataset_type == "prefix_cache" and cfg.prefix_warmup != "none"
                     and bool(point_prefix))
        if do_warmup and cfg.prefix_warmup == "once" and self._warmup_done:
            do_warmup = False
        if do_warmup:
            warm_c = cfg.dp if cfg.dp > 1 else cfg.prefix_num
            console.rule(f"预热阶段 warmup ┃ 并发={warm_c} ┃ output_len=1 ┃ 只写缓存，不计入 SLA",
                         codes=console.phase_label("warmup"))
            warm_result, _, warm_hit, _ = self._run_phase(
                run_dir, "warmup", warm_c, 1, 0.0, point_prefix)
            point.warmup_hit_rate = warm_hit.hbm_hit_rate
            self._warmup_done = True
            console.rule(f"预热结束 ┃ HBM 命中率 {self._pct(point.warmup_hit_rate)}",
                         codes=console.phase_label("warmup"))
            if not warm_result.ok:
                logger.warning("前缀预热阶段失败（returncode=%s）：%s",
                               warm_result.returncode, aisbench.tail(warm_result.log_path, 10))

        # ---- 全量阶段 ----
        console.rule(f"测试阶段 full ┃ 并发={concurrency} ┃ output_len={cfg.output_len}"
                     f" ┃ request_rate={cfg.request_rate:g}"
                     f" ┃ 计入 SLA 判定", codes=console.phase_label("full"))
        command_result, parsed, hit, gauges = self._run_phase(
            run_dir, "full", concurrency, cfg.output_len, cfg.request_rate,
            point_dataset)
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
                           "阈值与服务实际能力矛盾，请放宽 SLA 约束（--sla / --ttft-* 等）后重试")
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
        cfg = self.cfg
        if result.note:
            return
        passing = [p for p in result.points if p.ok and p.kind != "preflight"]
        failing = [p for p in result.points if not p.ok and p.kind != "preflight"]
        if result.max_ok:
            best = max(passing, key=lambda p: p.concurrency) if passing else None
            detail = ""
            if best:
                from . import sla_spec
                detail = ("（" + sla_spec.describe_values(cfg.sla_rules(), best.metrics) +
                          f"，HBM 命中率 {self._pct(best.hit_rate.get('hbm_hit_rate'))}）")
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
