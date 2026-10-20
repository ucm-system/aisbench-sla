# aisbench-sla

**AISBench 前缀缓存 SLA 自动调优命令行工具** —— 一条命令搜出「仍然满足延迟 SLA 的最大并发」。

本工具基于 [rayn-zzz/aisbench_auto_tools_prefix](https://github.com/rayn-zzz/aisbench_auto_tools_prefix)（MIT）改造：
沿用它的前缀缓存数据集生成、ais_bench 调用方式与 vLLM 前缀缓存命中率口径，
补上它缺的那一环——**并发点自动搜索**（原工具需要手写 `for i in ${bs[@]}` 循环逐个并发跑）。

- 无图形界面，纯命令行，可直接跑在 ais_bench 容器里（Linux / Python 3.8+）
- 运行时依赖只有 `requests`；生成数据集时才需要 `transformers`（ais_bench 容器通常自带）
- SLA 判定口径：**TTFT P90 与 TPOT P90**（前缀缓存命中率作为证据随点输出，不参与判定）

## 快速开始

```bash
# 0) 依赖（容器内一般已具备 ais_bench 与 transformers）
pip install requests
export PYTHONPATH=...   # 能 import ais_bench 即可

# 1) 自动搜索：前缀缓存场景下满足 SLA 的最大并发
python3 sla_search.py search \
    --target 192.168.1.10:8000 \
    --model my-model --model-path /home/weights/my-model \
    --work-path /home/benchmark \
    --dataset-path /home/dataset \
    --input-len 2048 --output-len 32 --data-num 160 \
    --dataset-type prefix_cache --prefix-num 1 --repeat-rate 90% --seed 1 --dp 1 \
    --ttft-p90 2000 --tpot-p90 30 \
    --start 8 --max 256 --output-dir ./sla_out

# 2) 单并发点压测（等价于参考工具的 aisbench_test.py）
python3 sla_search.py point \
    --target host.docker.internal:8000 \
    --model my-model --model-path /home/weights/my-model \
    --work-path /home/benchmark --dataset-path /home/dataset \
    --input-len 2048 --output-len 32 --data-num 160 --concurrency 64 \
    --dataset-type prefix_cache --repeat-rate 90% --prefix-num 1

# 3) 只生成数据集（随后可复用：--dataset <jsonl>）
python3 sla_search.py dataset \
    --model-path /home/weights/my-model --input-len 2048 --data-num 160 \
    --dataset-type prefix_cache --repeat-rate 90% --prefix-num 1 --dp 2 \
    --dataset-path /home/dataset
```

`--help` / `python3 sla_search.py search --help` 查看全部参数。

## 示例：64K 输入 / 1K 输出 / 90% 命中 —— 找最大并发与最高吞吐

场景：长上下文（**输入 65536 token**）+ **输出 1024 token**，前缀重复率 **90%**，
SLA 为 **TTFT P90 < 30s** 且 **TPOT P90 < 40ms**；目标是给出满足 SLA 的
**最大并发**与**最高输出吞吐**。

```bash
python3 sla_search.py search \
    --target 192.168.1.10:8000 \
    --model my-model --model-path /home/weights/my-model \
    --work-path /home/benchmark --dataset-path /home/dataset \
    --input-len 65536 --output-len 1024 --data-num 24 \
    --dataset-type prefix_cache --prefix-num 1 --repeat-rate 90% --seed 1 --dp 1 \
    --ttft-p90 30000 --tpot-p90 40 \
    --start 1 --max 64 --max-probes 12 --point-timeout 1800 \
    --output-dir ./sla_out_64k
```

长上下文场景的四个注意点：

1. **`--data-num` 要不小于计划探测的最大并发**（示例 24 条，配 `--max 64` 建议 2~4 倍于上限），
   否则单点请求数不够、`measured_concurrency` 压不满，吞吐会被低估；
2. **单点很慢**：每点 = 前缀预热 + 全量两阶段，64K×1K 的输出按秒计；用 `--point-timeout 1800`
   兜底、`--max-probes` 控预算，赶时间可加 `--prefix-warmup once`（只在首次预热前缀）；
3. **`--max` 要给足**：上限内一直达标时结论会提示「可提高 --max 继续探测」（退出码 0），
   这时把 `--max` 加倍重跑即可，数据集与命中率口径不变；
4. 变长输入（如 8k~128k 不定长、均值 32k）追加 `--length-mean 32768 --length-std 49152
   --length-min 8192 --length-max 131072`，工具按同分布生成并逐条截断。

**实测读数示例**（本轮用官方 ais_bench 镜像 + 本地 mock 服务跑通 65536/1024/90%，
mock 延迟恒定故各点全达标；真实 NPU 上会出现拐点，读数方式相同）：

```
  PASS c=1    TTFT P90      500  TPOT P90     0.1  out   1893.9 tok/s  HBM 89.51%  SLA 满足
  PASS c=2    TTFT P90      495  TPOT P90     0.1  out   3708.9 tok/s  HBM 100.00%  SLA 满足
  PASS c=4    TTFT P90      517  TPOT P90     0.4  out   3678.5 tok/s  HBM 100.00%  SLA 满足
  PASS c=8    TTFT P90      499  TPOT P90     0.3  out   7971.2 tok/s  HBM 100.00%  SLA 满足

结论：并发达上限 8 仍满足 SLA，可提高 --max 继续探测
  SLA 内最大并发：8
  最高输出吞吐：7971.2 tok/s @ 并发=8（达标：TTFT P90 499ms，TPOT P90 0.3ms）
  状态：max_unbounded（退出码 0）
```

结果怎么读：

- **最大并发** → 结论里的「SLA 内最大并发」，即 `sla_summary.json` 的 `max_ok_concurrency`
- **最高吞吐** → 「最高输出吞吐」，即 `throughput.peak`（含所在并发与是否达标）；
  若它本身达标，还会给出 `throughput.best_within_sla`
- **逐点曲线** → `sla_points.csv`：并发、判定、TTFT/TPOT 的 avg 与 P90、输出吞吐、QPS、
  HBM/ext 命中率、预热命中率、实测并发、峰值在途、耗时、日志路径

### 命中率口径与「各点缓存状态可比」（重要）

命中率一律按**快照差分**算（测试后累计 − 测试前累计），反映的是「该探测点窗口内缓存的表现」；
而窗口开始时缓存里已经有什么，取决于**上一个点留下了什么**。默认行为下会出现一个陷阱：

- 第 1 个点是冷缓存起步，命中率 ≈ 前缀重复率；
- 同一份数据集在后续点被反复播放，缓存逐步累积到**整行命中** → 命中率趋近 100%，
  各点不再可比（工具会打印「注意：命中率逐点上升并趋近 100%…」并在 JSON 的 `warnings`
  里给出该提示，提示文案会指出该用哪个修法）。

两种修法（按需选一）：

| 修法 | 做法 | 适用场景 |
| --- | --- | --- |
| `--reset-cache` | 每点前调用 `POST /reset_prefix_cache`（vLLM 原生支持；自带的 mock 也实现了） | 同一数据集、要求各点从**同一缓存状态**起测（推荐） |
| `--seed-per-point` | 每点用 `seed+i` 重新生成数据集：各点前缀对缓存都是新的 | 服务不支持缓存复位；代价是每点都要重生成数据集（64K 长上下文尤其明显） |

实测（官方镜像 + 自带 mock，阈值 TTFT P90 1200ms，同一负载形态）：

| 运行方式 | 各点命中率 | 结果 |
| --- | --- | --- |
| 默认 | 首点 90.8% → 末点 100.0%（7/8 个点 ≥99%） | 触发「不可比」告警 |
| `--reset-cache` | 全程 ≤ 90.36%，各点稳定 | 无告警，最大并发 12 |
| `--seed-per-point` | 全程 ≤ 90.44%，各点稳定 | 无告警，最大并发 12 |

> 顺带一提：命中率偶尔略高于 100%（如 100.01%）不是我们的算术问题——服务端 `queries` 与
> `hits` 可能来自两套 token 估算（mock 的 chars/4 估算 vs trie 的 token 计数）。工具会照实
> 显示并在 `hit_rate_note` 列标注「服务端计数器口径可能异常」。

## 它做了什么

每个探测点都是一次真实的「前缀预热 + 全量压测」两阶段运行（与参考工具一致）：

1. **前缀预热阶段**（默认每个探测点都做，可用 `--prefix-warmup` 调整）：并发 = dp（dp>1）
   或 prefix_num，`output_len=1`、`request_rate=0`，把公共前缀先灌进 KV cache；
2. **全量阶段**：目标并发 + 目标输出长度跑完整数据集；
3. 两个阶段各自做 **vLLM `/metrics` 快照差分**，得到该点的显存内（HBM）与外部缓存命中率；
4. 压测期间后台采样 `num_requests_running`，确认并发真的压上去了（峰值在途请求数）；
5. 解析 ais_bench 汇总日志拿到 TTFT/TPOT 的 avg 与 P90、吞吐、QPS、实测并发；
6. 用 TTFT P90 / TPOT P90 与阈值比较，判定该点通过与否。

命中率口径（与参考工具一致）：

```
hit rate = (测试后累计 hits - 测试前累计 hits) / (测试后累计 queries - 测试前累计 queries)
```

## SLA 搜索策略

```
预检（并发=1）      → 延迟最低的点都不达标 ⇒ 直接判定「阈值与能力矛盾」，退出码 2
阶梯（start ×2…）   → 直到某点不达标或撞到 --max
二分（lo,hi 收敛）  → 直到区间 ≤ max(--tolerance, lo//8)、或 --max-bisect 次、或探针预算耗尽
```

- `--start`：起始并发（默认 1；设 >1 时会先跑一次并发=1 的预检）
- `--max`：并发上限。全程都通过则为「上限内未触及瓶颈」，退出码 0 并提示提高 `--max`
- `--max-probes`：探测点总预算（默认 20），防止无人值守跑太久
- `--tolerance` / `--max-bisect`：二分收敛条件

**退出码**：`0` 找到可用并发 ｜ `2` 预检失败（阈值与服务能力矛盾）｜ `3` 起始并发即不达标
｜ `1` 参数或执行错误。便于接 CI / 批处理脚本。

## 输出

```
sla_out/
├── sla_points.csv          # 每个探测点一行：并发、判定、TTFT/TPOT P90、吞吐、命中率、峰值在途…
├── sla_summary.json        # 结论 + 配置快照 + 全部探测点（含 max_ok_concurrency）
├── datasets/               # 生成的数据集与前缀文件（可跨 run 复用）
└── points/c<并发>/         # 每个探测点的原始证据
    ├── warmup.log          # 前缀预热阶段 ais_bench 日志
    └── full.log            # 全量阶段 ais_bench 日志
```

stdout 会实时打印每个点的判定行，结束后打印明细表和结论，例如：

```
结论：SLA 内最大并发 = 96（TTFT P90 1789ms，TPOT P90 26.4ms，HBM 命中率 92.31%）；首个不达标点 c=128（TTFT P90 2340 > 2000）
```

## 参数速查

| 参数 | 说明 |
| --- | --- |
| `--target` | **推荐**：`IP:PORT` / `host:port` / `http(s)://host[:port][/path]`，自动转成 ais_bench 的 `host_ip/host_port` 或 `url`（见上一节） |
| `--host/--port/--url` | 兼容旧用法（`--url` 等价于 `--target` 传完整 URL） |
| `--model/--model-path/--api-key` | 服务端模型名 / tokenizer 路径 / 鉴权 |
| `--pods` | 采集 metrics 的端点，逗号分隔；PD 分离填各 P 节点（默认 `host:port`） |
| `--input-len/--output-len/--data-num` | 工作负载形态 |
| `--prefix-num/--repeat-rate/--seed/--dp` | 前缀数据集参数（`--repeat-rate` 支持 `90%` 或 `0.9`） |
| `--length-mean/std/min/max` | 变长输入（高斯或均匀分布），如 8k~128k 不定长 |
| `--dataset-type` | `prefix_cache`（默认）或 `normal` |
| `--ttft-p90/--tpot-p90` | SLA 阈值（ms），至少给一个 |
| `--start/--max/--tolerance/--max-probes/--max-bisect` | 搜索控制 |
| `--prefix-warmup` | `per-point`（默认）/ `once` / `none`；`once` 更快，`none` 不预热（命中率会显著偏低） |
| `--reset-cache` | 每个探测点前调用 `POST /reset_prefix_cache`（保证各点缓存状态可比，推荐） |
| `--seed-per-point` | 每个探测点用 `seed+i` 生成独立数据集（服务不支持复位时的替代方案） |
| `--work-path` | ais_bench 的 benchmark 目录（必填） |
| `--dataset` | 直接指定现成 gsm8k jsonl，跳过生成（此时无需 transformers） |
| `--aisbench-cmd` | 覆盖 ais_bench 命令（测试/适配用） |
| `--point-timeout` | 单点超时秒数，超时终止整个进程树（默认不限） |
| `--no-num-warmups` | 不附加 `--num-warmups 0`（老版本 ais_bench 不支持该参数） |
| `--dry-run` | 只打印探测计划，不真正压测 |
| `--config` | JSON 配置文件（CLI 参数优先） |

## 与参考工具的关系

| 能力 | 参考工具 | 本工具 |
| --- | --- | --- |
| 前缀数据集生成 | ✅ | ✅ 沿用（去掉 torch 依赖，tokenizer 只加载一次） |
| 单并发点压测 + 命中率 | ✅（`--prefix_test`） | ✅ `point` 子命令 |
| 多并发点扫描 | ❌ 手写 shell 循环 | ✅ **自动搜索（预检 + 阶梯 + 二分）** |
| 在途并发峰值 / 阶段级命中率 | 部分 | ✅ 每个探测点都有 |
| 结构化输出 | CSV + 日志 | ✅ CSV + JSON 摘要 + 每点日志 |
| 无人值守 | 需自行拼脚本 | ✅ 探针预算 + 单点超时 + 语义化退出码 |

## 目标地址：统一塞一个 `--target`（推荐）

ais_bench 的 API 模型有**两套寻址字段**，而且不是干净的二选一（源码实读 + 实测）：

| | `host_ip` + `host_port` | `url` |
| --- | --- | --- |
| 运行时 | 拼 `{http\|https}://{host_ip}:{host_port}/`（IPv6 自动加方括号） | 直接用 `url`（无 scheme 补 `http://`），**host_ip/host_port 被忽略** |
| 配置校验 | `host_ip` 必须是合法 IPv4/IPv6，唯一放行的非 IP 值是 `"localhost"` | 校验表里**没有 url** |
| 主机名 | 运行时代码其实支持（注释写着 "it's a hostname, use it as is"），却**被前置校验拦死** | 天然支持 |
| `host_port` | 必须 int ∈ (0,65536)，**与 url 无关地强制校验** | 同左 |
| 请求 URL | `urljoin(base_url, "v1/chat/completions")` | 同左 → **url 带路径必须以 `/` 结尾**，否则路径被替换 |

后果就是：给域名/容器别名（如 `host.docker.internal`）时，必须知道"要改用 `url` 字段、
还得把 `host_ip` 写成 `localhost`"才能过校验——这不该让使用者背。

本工具用**一个 `--target`** 抹平（`--host/--port/--url` 仍兼容）：

```bash
--target 192.168.65.254:8000          # 裸 IP:PORT      -> host_ip/host_port
--target host.docker.internal:8099    # 主机名          -> url=http://host.docker.internal:8099
--target https://gw.example.com/v1/   # https/带路径     -> url=.../v1/（自动补末尾 /）
--target [::1]:8000                   # IPv6            -> host_ip=::1
```

规则：**裸 IP[:端口] 走 `host_ip`/`host_port`；主机名、带 scheme、带路径一律走 `url`，
并同时写 `host_ip="localhost"` 规避校验雷区**；`--pods` 也从同一 target 推导。
实测：`--target host.docker.internal:8099` 直连成功（此前同一输入必报 `[UTILS-CFG-002]`）。

## 在官方 ais_bench 镜像里运行（已实测）

推荐直接在 AISBench 官方镜像里跑本工具（`ais_bench` 与依赖都在镜像内），mock/真实服务留在宿主机：

```bash
docker run --rm \
  -v $PWD:/tool -v /path/to/work:/rtest -v /path/to/tokenizer:/tok:ro \
  ghcr.io/aisbench/aisbench_benchmark:v3.1-20260330-master_x86_64_py_310 \
  python3 /tool/sla_search.py search \
    --host <宿主机IP> --port 8000 --model <模型名> --model-path /tok \
    --work-path /benchmark --dataset-path /rtest/ds --output-dir /rtest/out \
    --input-len 2048 --output-len 32 --data-num 160 \
    --dataset-type prefix_cache --prefix-num 1 --repeat-rate 90% --dp 1 \
    --ttft-p90 2000 --tpot-p90 30 --start 8 --max 256
```

实测环境：镜像 `v3.1-20260330-master_x86_64_py_310`（Python 3.10.18 / ais_bench 3.1.20260330 /
transformers 4.57.6 / requests 2.33.1），`--work-path` 取 **`/benchmark`**（镜像内 ais_bench 源码根，
CLI 为 `/usr/local/bin/ais_bench`），数据集最终落在 `/benchmark/ais_bench/datasets/gsm8k/test.jsonl`。

容器里访问宿主机服务时，**直接用 `--target host.docker.internal:<端口>` 即可**
（工具会自动走 url 字段，无需手工解析 IP）。

另一个必知项：**`--model-path` 必填**——即使已用 `--dataset` 指定现成数据集，ais_bench 的模型配置
仍需要 tokenizer 路径，否则报 `[UTILS-FILE-002] tokenizer path not found`。工具已把该参数设为必填
并给出明确报错。

## 常见问题

1. **`ais_bench: error: unrecognized arguments: --num-warmups`**
   旧版 ais_bench 不支持该参数。加 `--no-num-warmups`，或直接不管——工具会自动去掉该参数重跑一次。
2. **`生成数据集失败，请清空 picked ids`**
   删除 `<dataset-path>/picked_ids.txt` 后重试（参考语料已用尽时会这样，属于预期保护）。
3. **命中率打不出来 / 全是 0**
   检查 `--pods`（PD 分离要填各 P 节点）与系统代理：内网地址经代理会 502/超时，工具已强制绕过代理，
   但**服务端**指标端点本身要可达，可先 `curl -s http://<ip>:<port>/metrics | grep prefix_cache` 自检。
4. **超时**：`--point-timeout 1800` 给单点加上限；高并发大输入的单点跑很久是正常的。
5. **只想要性能数据，不要 SLA 搜索**：用 `point` 子命令，或把 `--ttft-p90/--tpot-p90` 都省掉并只跑一个点。

## 已知限制

- SLA 判定只看延迟阈值（按需求）；吞吐/成功率不作为门限，只在报告中展示。
- 每次探测都是独立进程的真实压测，**总耗时 = 探测点数 × 单点时长**；用 `--prefix-warmup once`
  与合理的 `--max-probes` 控制时长。
- 数据集按「工作负载形态 + seed」生成一次后被各探测点复用（保证命中率口径一致）；
  若用 `--data-num` 随并发变化的需求，请按并发分别调用工具的 `point` 子命令。
- 需要目标服务暴露 vLLM 风格 `/metrics` 的前缀缓存计数器才能给出命中率；
  缺失时命中率显示为「-」，不影响 SLA 判定。

## 测试

```bash
# 1) 纯离线（stdlib，不需要 NPU / Docker / 真实服务）：确定性桩 + 自建 mock
py -3.11 tests/test_offline.py        # 17 个用例：指标契约、解析、SLA 边界搜索、失败分支

# 2) 完整链路（真 ais_bench + 本仓库自带的 mock）
export SLA_E2E_TOKENIZER=/path/to/Qwen3-0.6B     # 或运行时传 --tokenizer
py -3.11 tests/e2e_real.py --no-docker   # 不依赖 Docker：确定性桩 + vendored mock
py -3.11 tests/e2e_real.py               # 默认：官方 ais_bench 镜像 + vendored mock
```

`tests/e2e_real.py` 自己拉起 mock、跑一轮完整 SLA 搜索、再对结果做**性质断言**
（不写死具体并发）：通过点/失败点都存在、**判定随并发单调**、最大并发等于通过点最大值、
**二分收敛在容差内**、拐点两侧 TTFT 分居阈值两边、每个点都算出命中率、最高吞吐有值、
CSV/JSON 落盘；`--expect-max-ok N` 可改为严格断言。

**自带的 mock**（`tests/mocks/clawperf_mock_server.py`）是 ClawPerf 的 mock LLM 服务
vendored 副本（Apache-2.0，全文见 `tests/mocks/LICENSE.ClawPerf`）：真 trie 前缀缓存、
vLLM 风格 `/metrics`（`vllm:prefix_cache_*` / `external_prefix_cache_*`）、SSE 流式、
在途请求 gauge。相对上游加了三处**测试用**扩展（文件头有说明）：

- `--free-concurrency` / `--queue-ms`：**负载相关 TTFT 模型**——在途请求超过 free 后
  每个 +queue-ms。上游 TTFT 恒定，真实压测跑不出延迟拐点，也就无法验证阶梯失败与二分；
- `POST /__sim__/traffic`：直接累加缓存计数器，供确定性桩模拟流量；
- `--ready-file` / `POST /__sim__/reset`：测试脚本等待就绪与复位。

实测（官方镜像 + 该 mock，free=8 / queue-ms=250 / ttft=500，阈值 TTFT P90 1200ms）：
阶梯 c=1/2/4/8 通过 → c=16 失败 → 二分 12（通过，1186ms）/ 14（失败）/ 13（失败）
→ **最大并发 12**，边界缺口 1 在容差内；三次运行结论一致。

## License

MIT（见 [LICENSE](LICENSE)）。移植自 [aisbench_auto_tools_prefix](https://github.com/rayn-zzz/aisbench_auto_tools_prefix)（MIT）；
致谢与移植范围见 LICENSE 与各文件头部注释。
