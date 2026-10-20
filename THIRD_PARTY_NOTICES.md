# 第三方代码与数据声明（Third-Party Notices）

本项目的自有代码以 [MIT License](LICENSE) 发布。以下内容来自第三方，按其原许可使用并致谢。

## 1. aisbench_auto_tools_prefix

- 来源：<https://github.com/rayn-zzz/aisbench_auto_tools_prefix>
- 许可：MIT License（原许可全文：`third_party/LICENSE.aisbench_auto_tools_prefix`）
- 收录范围（移植/改写）：
  - 前缀缓存数据集生成 —— `generate_dataset.py` / `data_picker.py` → `sla_tool/dataset.py`
  - ais_bench 模型配置模板 —— `default_api.py` → `sla_tool/aisbench.py`
  - vLLM 前缀缓存命中率抓取 —— `cal_prefix_hit_rate.py` → `sla_tool/metrics.py`
  - ais_bench 汇总日志解析规则 —— `save_file.py` → `sla_tool/parse.py`

## 2. AISBench_PrefixTest_Tools

- 来源：本项目作者的另一仓库（AISBench 前缀复用测试器桌面工具）
- 许可：Apache License 2.0
- 收录范围（移植/改写）：
  - ais_bench 输出解析（传统行格式 + `│` 表格格式）→ `sla_tool/parse.py`
  - SLA 搜索策略（预检 → 指数阶梯 → 二分收敛）→ `sla_tool/search.py`

## 3. ClawPerf mock LLM 服务

- 来源：ClawPerf 的 `src/clawperf/mock_server.py`
- 许可：Apache License 2.0（原许可全文：`tests/mocks/LICENSE.ClawPerf`）
- 收录范围：`tests/mocks/clawperf_mock_server.py`（vendored 副本，仅用于测试）
- 本副本相对上游的改动（见该文件头注释）：新增负载相关 TTFT 模型
  (`--free-concurrency` / `--queue-ms`）、`POST /__sim__/traffic`、`POST /__sim__/reset`、
  `--ready-file`，并补充 `POST /reset_prefix_cache`（上游为真实 vLLM 的调试接口）。

## 4. GSM8K 数据集

- 来源：公开的 GSM8K 数据集（OpenAI）
- 许可：MIT License
- 收录范围：`sla_tool/GSM8K.jsonl` —— 作为生成压测数据集的参考语料。
