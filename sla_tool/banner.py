"""启动大字横幅（内嵌常量，运行时不联网、不依赖第三方库）。

来源：patorjk TAAG（https://patorjk.com/software/taag/），figlet 字体 `Standard`，
文本 `AISBench SLA` —— 与 AISBench 自身横幅同一款字体。

重新生成（任意 figlet 字体，如 Standard / Big / ANSI Shadow / Doom / Slant）：

    py -3.11 tools/make_banner.py "AISBench SLA" Standard

脚本会打印可直接替换下面的常量；字体文件默认从
`https://patorjk.com/software/taag/fonts/<font>.flf` 拉取。

注意：TAAG 上以 “Atom …” 命名的字体属于 TheDraw 字体（`.tdf`），站点不提供
`.flf`，因此无法用上面的方式渲染；需要时得单独解析 TDF 容器。
"""
from __future__ import annotations

BANNER_TEXT = "AISBench SLA"
BANNER_FONT = "Standard"

# figlet Standard / "AISBench SLA"（5 行正文，尾部空行已去）
ART = (
    r"     _      ___   ____    ____                          _         ____    _          _",
    r"    / \    |_ _| / ___|  | __ )    ___   _ __     ___  | |__     / ___|  | |        / \ ",
    r"   / _ \    | |  \___ \  |  _ \   / _ \ | '_ \   / __| | '_ \    \___ \  | |       / _ \ ",
    r"  / ___ \   | |   ___) | | |_) | |  __/ | | | | | (__  | | | |    ___) | | |___   / ___ \ ",
    r" /_/   \_\ |___| |____/  |____/   \___| |_| |_|  \___| |_| |_|   |____/  |_____| /_/   \_\ ",
)

# 副标题（工具定位，一行）
TAGLINE = "AISBench 前缀缓存 SLA 自动调优 CLI ┃ 预检 → 阶梯 → 二分"
