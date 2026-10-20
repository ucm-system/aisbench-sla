#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 figlet 字体渲染大字，用于生成 sla_tool/banner.py 里的内嵌横幅。

字体取自 patorjk TAAG（https://patorjk.com/software/taag/），与 AISBench 自身横幅同源：

    py -3.11 tools/make_banner.py "AISBench SLA" Standard
    py -3.11 tools/make_banner.py "AISBench SLA" "ANSI Shadow"

不带第二个参数时使用 Standard。字体文件会缓存到 tools/.fonts/，可离线复用。
仅依赖标准库；需要联网时走 HTTP(S)_PROXY / --proxy。
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

FONT_URL = "https://patorjk.com/software/taag/fonts/{name}.flf"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fonts")


def load_font(name: str, proxy: str = "") -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    local = os.path.join(CACHE_DIR, f"{name.replace(' ', '_')}.flf")
    if os.path.isfile(local):
        with open(local, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    url = FONT_URL.format(name=name.replace(" ", "%20"))
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)
    with opener.open(url, timeout=60) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    with open(local, "w", encoding="utf-8", errors="replace") as handle:
        handle.write(text)
    return text


def parse_flf(text: str):
    """解析 figlet .flf：返回 (hardblank, height, layout, {字符: 行列表})。"""
    lines = text.split("\n")
    header = lines[0].split()
    if not header or not header[0].startswith("flf2a"):
        raise ValueError("不是 figlet .flf 字体文件")
    hardblank = header[0][-1]
    height = int(header[1])
    layout = int(header[4]) if len(header) > 4 else 1
    comment_lines = int(header[5]) if len(header) > 5 else 0
    glyphs = {}
    current = []
    code = 32
    for line in lines[1 + comment_lines:]:
        current.append(line.rstrip("\n"))
        if len(current) == height:
            endmark = current[-1][-1] if current[-1] else "@"
            glyphs[chr(code)] = [row.rstrip(endmark) for row in current]
            code += 1
            current = []
    return hardblank, height, layout, glyphs


def render(text: str, glyphs: dict, hardblank: str, height: int) -> list:
    rows = [""] * height
    for char in text:
        glyph = glyphs.get(char) or glyphs.get("?")
        if glyph is None:
            continue
        for index in range(height):
            rows[index] += glyph[index]
    return [row.replace(hardblank, " ").rstrip() for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description="figlet 大字渲染（生成内嵌横幅用）")
    parser.add_argument("text", nargs="?", default="AISBench SLA")
    parser.add_argument("font", nargs="?", default="Standard")
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY", ""))
    parser.add_argument("--python-literal", action="store_true",
                        help="输出可直接粘进 banner.py 的 ART 常量")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    hardblank, height, layout, glyphs = parse_flf(load_font(args.font, args.proxy))
    rows = [row for row in render(args.text, glyphs, hardblank, height) if row.strip()]
    print(f"# font={args.font} height={height} layout={layout} glyphs={len(glyphs)}",
          file=sys.stderr)
    if args.python_literal:
        print("ART = (")
        for row in rows:
            print(f'    r"{row}",')
        print(")")
    else:
        for row in rows:
            print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
