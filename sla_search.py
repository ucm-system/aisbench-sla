#!/usr/bin/env python3
"""aisbench-sla —— 前缀缓存 SLA 自动调优 CLI。

示例：
  # 自动搜索满足延迟 SLA 的最大并发（前缀缓存场景）
  python3 sla_search.py search \
      --host 192.168.1.10 --port 8000 --model ds --model-path /home/weights/ds \
      --work-path /home/benchmark --dataset-path /home/dataset \
      --input-len 2048 --output-len 32 --data-num 160 \
      --prefix-num 1 --repeat-rate 90% --seed 1 --dp 1 \
      --ttft-p90 2000 --tpot-p90 30 \
      --start 8 --max 256 --output-dir ./sla_out

  # 单并发点压测（等价于参考工具的 aisbench_test.py）
  python3 sla_search.py point --host ... --concurrency 64 ...

  # 只生成数据集
  python3 sla_search.py dataset --model-path /home/weights/ds --input-len 2048 ...

退出码：0 找到可用并发 / 2 预检失败（阈值与能力矛盾）/ 3 无可用并发 / 1 参数或执行错误
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import fields
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sla_tool import __version__, dataset, report  # noqa: E402
from sla_tool.config import Config  # noqa: E402
from sla_tool.search import SlaSearch  # noqa: E402

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


# --------------------------------------------------------------------- 参数
def add_target_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("目标服务")
    group.add_argument("--target",
                       help="统一目标地址：IP:PORT / host:port / http(s)://host[:port][/path]"
                            "（自动转成 ais_bench 需要的 host_ip/host_port 或 url；"
                            "优先级高于 --host/--port/--url）")
    group.add_argument("--host", help="服务 IP 或主机名（也可直接写 IP:PORT / http://...）")
    group.add_argument("--port", type=int, help="服务端口")
    group.add_argument("--url", help="完整 baseUrl（等价 --target）")
    group.add_argument("--model", help="服务端模型名")
    group.add_argument("--model-path", help="tokenizer 路径（生成数据集用）")
    group.add_argument("--api-key", help="鉴权 key")
    group.add_argument("--pods", help="metrics 端点，逗号分隔（PD 分离填各 P 节点）")
    group.add_argument("--npu-num", type=int, help="卡数，用于换算单卡吞吐")


def add_workload_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("负载形态")
    group.add_argument("--input-len", type=int, help="输入 token 长度")
    group.add_argument("--output-len", type=int, help="输出 token 长度")
    group.add_argument("--data-num", type=int,
                       help="数据集条数；建议取 --max 的 2~4 倍、下限 >= --max，"
                            "否则最高并发点压不满（measured_concurrency 低于目标、吞吐被低估）")
    group.add_argument("--request-rate", type=float, help="请求频率，0=不限制")
    group.add_argument("--test-type", choices=["stream", "text"], help="流式/非流式")
    group.add_argument("--dataset-type", choices=["prefix_cache", "normal"],
                       help="前缀数据集或普通数据集")
    group.add_argument("--prefix-num", type=int, help="前缀种类数")
    group.add_argument("--repeat-rate", help="前缀重复率，支持 90%% 或 0.9")
    group.add_argument("--seed", type=int, help="随机种子")
    group.add_argument("--dp", type=int, help="dp 域数量")
    group.add_argument("--enable-think", action="store_true", default=None,
                       help="开启思考模式（DeepSeek V3.1 等）")
    group.add_argument("--length-mean", type=int, help="变长输入：高斯均值")
    group.add_argument("--length-std", type=float, help="变长输入：高斯标准差")
    group.add_argument("--length-min", type=int, help="变长输入：最小值")
    group.add_argument("--length-max", type=int, help="变长输入：最大值")
    group.add_argument("--dataset", help="直接指定 gsm8k 格式 jsonl（跳过生成）")
    group.add_argument("--dataset-path", help="生成数据集落盘目录")
    group.add_argument("--gsm8k-source", help="参考语料 jsonl（默认工具自带）")


def add_env_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("执行环境")
    group.add_argument("--work-path", help="ais_bench benchmark 根目录")
    group.add_argument("--output-dir", help="结果与日志输出目录")
    group.add_argument("--aisbench-cmd", help="ais_bench 命令（可整体替换，便于测试）")
    group.add_argument("--aisbench-extra-args", help="追加到 ais_bench 的参数")
    group.add_argument("--no-num-warmups", action="store_true", default=None,
                       help="不附加 --num-warmups 0（老版本 ais_bench）")
    group.add_argument("--point-timeout", type=float, help="单点超时秒数，0=不限")
    group.add_argument("--prefix-warmup", choices=["none", "once", "per-point"],
                       help="前缀预热时机（默认 per-point）")
    group.add_argument("--reset-cache", action="store_true", default=None,
                       help="每点前 POST /reset_prefix_cache —— 注意：vLLM 该端点属调试接口，"
                            "需 VLLM_SERVER_DEV_MODE=1 才存在；且只清显存侧缓存，"
                            "测外部缓存(UCM)命中时无效。各点可比性请优先用 --seed-per-point")
    group.add_argument("--seed-per-point", action="store_true", default=None,
                       help="【默认开启】每个探测点用 seed+i 生成独立数据集：各点前缀都是新的，"
                            "HBM 与外部缓存都从干净状态起测，命中率各点可比")
    group.add_argument("--no-seed-per-point", action="store_true", default=None,
                       help="关闭默认行为：各探测点复用同一份数据集（省去逐点重生成，"
                            "适合长上下文；代价是后续点命中率会逐点饱和到 ~100%、各点不可比）")
    group.add_argument("--dry-run", action="store_true", default=None,
                       help="只打印计划，不真正压测")


def add_sla_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("SLA 与搜索")
    group.add_argument("--ttft-p90", type=float, help="TTFT P90 上限（ms）")
    group.add_argument("--tpot-p90", type=float, help="TPOT P90 上限（ms）")
    group.add_argument("--start", type=int, help="起始并发（默认 1）")
    group.add_argument("--max", "--max-concurrency", dest="max_concurrency",
                       type=int, help="并发上限")
    group.add_argument("--tolerance", type=int, help="二分收敛容差（并发数）")
    group.add_argument("--max-probes", type=int, help="最大探测点数（预算）")
    group.add_argument("--max-bisect", type=int, help="二分最大迭代次数")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sla_search.py",
        description="AISBench 前缀缓存 SLA 自动调优（搜索满足延迟阈值的最小/最大并发）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--config", help="JSON 配置文件（CLI 参数优先级更高）")
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    parser.add_argument("--version", action="version", version=f"aisbench-sla {__version__}")
    # 子命令也要能接收 --verbose / --config（argparse 不继承父解析器选项）
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--verbose", action="store_true", help="输出调试日志")
    common.add_argument("--config", help="JSON 配置文件（CLI 参数优先级更高）")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="自动搜索 SLA 内最大并发",
                            parents=[common],
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    add_target_opts(search)
    add_workload_opts(search)
    add_env_opts(search)
    add_sla_opts(search)

    point = sub.add_parser("point", help="单个并发点压测",
                           parents=[common],
                           formatter_class=argparse.RawDescriptionHelpFormatter)
    add_target_opts(point)
    add_workload_opts(point)
    add_env_opts(point)
    point.add_argument("--concurrency", type=int, required=True, help="并发数")
    add_sla_opts(point)

    make = sub.add_parser("dataset", help="只生成前缀数据集", parents=[common])
    add_target_opts(make)
    add_workload_opts(make)
    make.add_argument("--output-dir", help="数据集默认落盘目录")
    return parser


def namespace_to_config(args: argparse.Namespace) -> Config:
    values: Dict[str, Any] = {}
    known = {f.name for f in fields(Config)}
    for key, value in vars(args).items():
        if key in ("command", "config", "verbose", "concurrency"):
            continue
        if value is None:
            continue
        target = key
        if key == "no_num_warmups":
            target, value = "num_warmups", False
        if key == "no_seed_per_point":
            target, value = "seed_per_point", False
        if key == "pods":
            value = [item.strip() for item in str(value).split(",") if item.strip()]
        if target in known:
            values[target] = value
    cfg = Config()
    if args.config:
        cfg = Config.from_json(args.config)
    cfg.merge(values)
    cfg.verbose = bool(getattr(args, "verbose", False))
    return cfg


def setup_logging(verbose: bool) -> None:
    try:  # Windows 控制台默认 GBK，保证中文/符号不炸
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format=LOG_FORMAT)


# --------------------------------------------------------------------- 子命令
def cmd_search(cfg: Config) -> int:
    search = SlaSearch(cfg, on_point=lambda point: print(report.point_line(point)))
    if cfg.dry_run:
        report.print_dry_run(cfg)
    result = search.run()
    if result.status == "dry_run":
        return result.exit_code
    if result.points:
        report.print_table(result.points)
    report.print_summary(result, cfg)
    csv_path = report.write_csv(cfg.output_dir, result.points)
    json_path = report.write_summary(cfg.output_dir, cfg, result)
    print(f"  明细 CSV：{csv_path}")
    print(f"  摘要 JSON：{json_path}")
    return result.exit_code


def cmd_point(cfg: Config, args: argparse.Namespace) -> int:
    cfg.request_rate = cfg.request_rate if cfg.request_rate is not None else 0.0
    try:
        cfg.validate_for_run()
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    search = SlaSearch(cfg)
    search._prepare()  # noqa: SLF001 - 单点模式复用准备流程
    point = search._probe(int(args.concurrency))  # noqa: SLF001
    print(report.point_line(point))
    report.print_table([point])
    if point.hit_rate:
        for pod, engines in (point.hit_rate.get("per_pod") or {}).items():
            for engine, row in engines.items():
                print(f"  {pod}/{engine}: HBM {report._pct(row.get('hbm_hit_rate'))} "  # noqa: SLF001
                      f"({row.get('hbm_h')}/{row.get('hbm_q')})  "
                      f"ext {report._pct(row.get('ext_hit_rate'))} "
                      f"({row.get('ext_h')}/{row.get('ext_q')})")
    return 0 if point.ok else 3


def cmd_dataset(cfg: Config) -> int:
    cfg.output_dir = cfg.output_dir or "./sla_datasets"
    cfg.normalize()
    prefix_path, dataset_path = dataset.ensure_dataset(cfg)
    print(f"前缀文件：{prefix_path or '(无)'}")
    print(f"数据集：{dataset_path}")
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(getattr(args, "verbose", False))
    try:
        cfg = namespace_to_config(args)
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 1
    try:
        if args.command == "search":
            return cmd_search(cfg)
        if args.command == "point":
            return cmd_point(cfg, args)
        if args.command == "dataset":
            return cmd_dataset(cfg)
    except (ValueError, RuntimeError, OSError) as exc:
        logging.error("%s", exc)
        return 1
    parser.error("未知子命令")
    return 1


if __name__ == "__main__":
    sys.exit(main())
