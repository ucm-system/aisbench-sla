"""aisbench-sla: 前缀缓存 SLA 自动调优 CLI 工具。

在参考工具 aisbench_auto_tools_prefix（MIT，rayn-zzz）的单并发点压测能力之上，
增加「自动搜索满足 SLA 的最大并发」：预检 → 指数阶梯 → 二分收敛。

模块划分：
  config   运行配置（CLI / JSON）
  dataset  前缀缓存数据集生成（移植自参考工具）
  metrics  vLLM /metrics 前缀缓存计数抓取与快照差分
  parse    ais_bench 汇总日志解析
  aisbench ais_bench 调用（模型配置注入、数据集挂载、子进程执行）
  search   SLA 搜索状态机
  report   控制台表格 / CSV / JSON 输出
"""

__version__ = "0.5.0"
