# aisbench-sla

AISBench 前缀缓存 **SLA 自动调优 CLI**：预检 → 指数阶梯 → 二分收敛，搜出满足延迟约束的**最大并发**与**最高吞吐**。

## 快速开始

```bash
pip install requests          # 生成数据集还需 transformers（ais_bench 容器已自带）
```

```bash
python3 sla_search.py search \
    --target 10.0.0.5:8000 \
    --model my-model --model-path /weights/my-model \
    --work-path /benchmark --dataset-path /data/datasets \
    --input-len 2048 --output-len 32 --data-num 160 \
    --dataset-type prefix_cache --prefix-num 1 --repeat-rate 90% --dp 1 \
    --sla ttft.avg:1200 --sla tpot.avg:30 \
    --start 8 --max 256 --output-dir ./sla_out
```

其余两个子命令：

```bash
python3 sla_search.py point   --target ... --concurrency 64                     # 单并发点压测
python3 sla_search.py dataset --model-path /weights/my-model --input-len 2048   # 只生成数据集
```

* `--work-path` 是 ais_bench 源码根（官方镜像里为 `/benchmark`），`--dataset-path` 是数据集落盘目录。
* `--target` 收 `IP:PORT` / `host:port` / `http(s)://host[:port][/path]`，自动转成 ais_bench 需要的字段。
* 容器内访问宿主机服务：`--target host.docker.internal:8000`。
* `--model-path` 必填（ais_bench 模型配置需要 tokenizer，即使已用 `--dataset`）。

## 示例：64K 输入 / 1K 输出 / 90% 命中，找最大并发与最高吞吐

```bash
python3 sla_search.py search \
    --target 10.0.0.5:8000 \
    --model my-model --model-path /weights/my-model \
    --work-path /benchmark --dataset-path /data/datasets \
    --input-len 65536 --output-len 1024 --data-num 160 \
    --dataset-type prefix_cache --prefix-num 1 --repeat-rate 90% --dp 1 \
    --sla ttft.avg:30000 --sla tpot.avg:40 \
    --start 1 --max 64 --max-probes 12 --point-timeout 1800 \
    --output-dir ./sla_out_64k
```

`--data-num` 取 `--max` 的 **2~4 倍**、下限 ≥ `--max`：请求数不足时最高几个并发点压不满（看 CSV 的
`measured_concurrency`）、吞吐被低估。长上下文很贵，可收窄：`--max 16 --data-num 48`。

实测输出（真实 ais_bench + 官方镜像，`--sla ttft.avg:900`；节选）：

```
  PASS c=1    TTFT AVG 479  TPOT P90 0  out    133.1 tok/s  HBM 89.39%  ext   -    SLA 满足
  PASS c=8    TTFT AVG 482  TPOT P90 0  out   1011.8 tok/s  HBM 89.07%  ext   -    SLA 满足
  FAIL c=16   TTFT AVG 1006 TPOT P90 0  out    413.9 tok/s  HBM 88.96%  ext   -    TTFT AVG <= 900ms（实测 1006）
  PASS c=12   TTFT AVG 649  TPOT P90 0  out    693.9 tok/s  HBM 89.68%  ext   -    SLA 满足
  PASS c=14   TTFT AVG 877  TPOT P90 0  out    516.5 tok/s  HBM 89.90%  ext   -    SLA 满足
  FAIL c=15   TTFT AVG 945  TPOT P90 0  out    465.8 tok/s  HBM 89.42%  ext   -    TTFT AVG <= 900ms（实测 945.1）

结论：SLA 内最大并发 = 14（TTFT AVG 877.4ms，TPOT P90 0.2ms，HBM 命中率 89.90%）；首个不达标点 c=15
  判定口径：TTFT AVG <= 900ms；TPOT P90 <= 40ms
  SLA 内最大并发：14
  最高输出吞吐：1011.8 tok/s @ 并发=8（达标）
  状态：ok（退出码 0）
```

## SLA 搜索策略

```
预检（并发=1）    不达标即判定「阈值与能力矛盾」，退出码 2
阶梯（start ×2）  直到某点不达标或到 --max
二分（lo, hi）    收敛到 lo//8 或 --tolerance，最多 --max-bisect 次
预算              --max-probes 限制总探测点数
```

约束可多条、AND 生效。指标 `ttft` / `tpot` / `e2e` / `output_throughput` × 统计量
`avg` / `min` / `max` / `median` / `p50` / `p75` / `p90` / `p99`（吞吐为下限语义）：

```bash
--sla ttft.avg:1200 --sla ttft.p99:3000 --sla tpot.avg:30 --sla e2e.max:20000
--sla output_throughput.avg:>=500

--ttft-avg 1200 --ttft-p99 3000 --tpot-avg 30 --e2el-max 20000    # 等价离散开关
```

**退出码**：`0` 找到可用并发 ｜ `2` 预检失败 ｜ `3` 无可用并发 ｜ `1` 参数/执行错误。

命中率按 vLLM `/metrics` 快照差分（`Δhits/Δqueries`），不参与判定、只作证据；默认**每个探测点换 seed**
（各点前缀都是新的，命中率各点可比），`--no-seed-per-point` 复用同一数据集（省时间，命中率逐点饱和）。
`--reset-cache` 需服务开调试模式（vLLM 为 `VLLM_SERVER_DEV_MODE=1`），且只清显存侧缓存，测外部缓存时无效。

## 输出

```
sla_out/
├── sla_points.csv      每点一行：并发/判定/各项统计量/吞吐/命中率/实测并发/峰值在途/日志路径
├── sla_summary.json    结论、SLA 口径、全部探测点
├── aisbench_all.log    本轮所有相位拼接的 ais_bench 原始输出
├── datasets/           数据集（默认 seed<N>/ 每点一份，可复用）
└── points/c<N>/        warmup.log、full.log（原始 ais_bench 日志）
```

ais_bench 的输出**实时打印到终端**（同时写入 `points/c<N>/*.log`，并回拷一份到它的 exp 目录）；
不想刷屏用 `--no-stream`。

## 参数速查

| 参数 | 说明 |
| --- | --- |
| `--target` | 目标服务：`IP:PORT` / `host:port` / `http(s)://…`（优先于 `--host/--port/--url`） |
| `--model` / `--model-path` / `--api-key` | 服务端模型名 / tokenizer 路径（必填）/ 鉴权 |
| `--pods` | 采集 `/metrics` 的端点，逗号分隔（PD 分离填各 P 节点，默认同 target） |
| `--input-len` / `--output-len` / `--data-num` | 负载形态；`--data-num` 建议 `--max` 的 2~4 倍 |
| `--dataset-type` | `prefix_cache`（默认）/ `normal` |
| `--prefix-num` / `--repeat-rate` / `--seed` / `--dp` | 前缀数据集参数（`--repeat-rate` 支持 `90%` 或 `0.9`） |
| `--length-mean/std/min/max` | 变长输入（高斯或均匀分布） |
| `--dataset` | 直接指定现成 gsm8k jsonl，跳过生成 |
| `--sla` / `--ttft-avg`…`--e2el-p99` / `--output-throughput-min` | SLA 约束（至少一条） |
| `--start` / `--max` / `--tolerance` / `--max-probes` / `--max-bisect` | 搜索控制 |
| `--prefix-warmup` | `per-point`（默认）/ `once` / `none` |
| `--seed-per-point` / `--no-seed-per-point` | 每点换 seed（默认）/ 复用同一数据集 |
| `--reset-cache` | 每点前 `POST /reset_prefix_cache`（调试接口、仅显存侧） |
| `--work-path` | ais_bench 源码根（官方镜像 `/benchmark`） |
| `--dataset-path` / `--output-dir` | 数据集目录 / 结果目录 |
| `--aisbench-cmd` / `--aisbench-extra-args` | 替换 / 追加 ais_bench 命令 |
| `--point-timeout` | 单点超时秒数（超时终止进程树） |
| `--no-stream` | 不把 ais_bench 输出实时打到终端（仍写入日志文件） |
| `--no-num-warmups` | 不附加 `--num-warmups 0`（老版本 ais_bench） |
| `--dry-run` | 只打印探测计划 |
| `--config` | JSON 配置文件（CLI 参数优先） |

## 测试

```bash
py -3.11 tests/test_cli_surface.py      # CLI 表面（--help、报错可读）
py -3.11 tests/test_offline.py          # 离线：确定性桩 + 自建 mock
py -3.11 tests/e2e_matrix.py            # 端到端矩阵（11 个场景：超时/退出码/多约束/配置…）
py -3.11 tests/e2e_real.py --no-docker  # 离线全链路
py -3.11 tests/e2e_real.py              # 真实 ais_bench 镜像 + 自带 mock
```

## License

MIT（见 [LICENSE](LICENSE)）；第三方来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
