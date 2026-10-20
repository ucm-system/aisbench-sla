"""终端呈现：分隔框、分节线、颜色。

目标：一轮探测里「预热阶段 / 全量阶段」以及不同并发点之间一眼可分。
规则：

* 颜色只在终端（TTY）启用；被重定向/管道接走时自动关闭，保证日志可 grep。
  `--color always` / `never` 可强制。遵循 NO_COLOR / FORCE_COLOR 约定。
* ais_bench 自带颜色（它的 ColoredLevelFormatter 无条件上色），透传到终端；
  但**写入日志文件前会剥掉 ANSI**，让落盘日志保持纯文本。
* 分隔框与分节线始终输出（`*` / `─` 框线），即使没有颜色也能看清层次；
  同样的纯文本也会写进 run.log 与 aisbench_all.log，便于事后按轮次阅读。
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import List

# --------------------------------------------------------------------------- #
# 颜色
# --------------------------------------------------------------------------- #
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
UNDERLINE = "\033[4m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"

_enabled: bool | None = None
_record_handlers: List[object] = []
WIDTH = 100

LEVEL_COLORS = {
    logging.DEBUG: DIM + BLUE,
    logging.INFO: CYAN,           # 与 ais_bench 的绿色 INFO 区分开
    logging.WARNING: YELLOW,
    logging.ERROR: BOLD + RED,
    logging.CRITICAL: BG_RED + WHITE,
}


def _enable_windows_vt() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        for handle_id in (-11, -12):        # stdout, stderr
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VT
    except Exception:  # noqa: BLE001
        pass


def init_color(mode: str = "auto") -> bool:
    """决定是否启用颜色。mode: auto | always | never。"""
    global _enabled
    if os.environ.get("NO_COLOR") and mode != "always":
        _enabled = False
    elif os.environ.get("FORCE_COLOR") and mode != "never":
        _enabled = True
    elif mode == "always":
        _enabled = True
    elif mode == "never":
        _enabled = False
    else:
        try:
            _enabled = bool(sys.stdout.isatty())
        except Exception:  # noqa: BLE001
            _enabled = False
    if _enabled:
        _enable_windows_vt()
    return _enabled


def color_enabled() -> bool:
    if _enabled is None:
        init_color("auto")
    return bool(_enabled)


def c(text: str, *codes: str) -> str:
    """着色（未启用时原样返回）。"""
    if not text or not color_enabled():
        return text
    return "".join(codes) + text + RESET


def strip_ansi(text: str) -> str:
    if not text or "\033" not in text:
        return text
    out = []
    i = 0
    while i < len(text):
        if text[i] == "\033" and i + 1 < len(text) and text[i + 1] == "[":
            j = i + 2
            while j < len(text) and not ("@" <= text[j] <= "~"):
                j += 1
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


# --------------------------------------------------------------------------- #
# 输出（终端 + 记录文件）
# --------------------------------------------------------------------------- #
def register_record_handler(handler) -> None:
    """把分隔框/分节线写进同一个文件处理器。

    必须复用 logging 的 FileHandler（而不是自己再开一个句柄）：两个文件对象各自
    维护写位置，后写的那个会把先写的内容覆盖掉。
    """
    if handler is not None and handler not in _record_handlers:
        _record_handlers.append(handler)


def emit(text: str = "", color_codes: tuple = (), record: bool = True) -> None:
    """写一行：终端按需着色，记录文件写纯文本。"""
    plain = strip_ansi(text)
    if record and _record_handlers:
        for handler in list(_record_handlers):
            try:
                handler.flush()
                handler.stream.write(plain + "\n")
                handler.flush()
            except Exception:  # noqa: BLE001
                pass
    if color_codes and color_enabled():
        print("".join(color_codes) + plain + RESET)
    else:
        print(plain)


def dwidth(text: str) -> int:
    """显示宽度：中文/全角按 2 列计（框线对齐用）。"""
    import unicodedata
    total = 0
    for char in strip_ansi(text):
        total += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return total


def pad(text: str, width: int, align: str = "<") -> str:
    """按显示宽度补空格。"""
    filler = max(0, width - dwidth(text))
    if align == ">":
        return " " * filler + text
    if align == "^":
        left = filler // 2
        return " " * left + text + " " * (filler - left)
    return text + " " * filler


def banner(title: str, right: str = "", char: str = "*", width: int = WIDTH,
           codes: tuple = (BOLD, CYAN)) -> None:
    """醒目的轮次分隔框。"""
    line = char * width
    left = f"{char} {title}"
    right_text = f"{right} {char}" if right else char
    middle = pad(left, max(dwidth(left), width - dwidth(right_text))) + right_text
    emit(line, codes)
    emit(middle, codes)
    emit(line, codes)


def rule(label: str = "", char: str = "─", width: int = WIDTH,
         codes: tuple = (DIM,)) -> None:
    """分节线（阶段之间）。"""
    if not label:
        emit(char * width, codes)
        return
    text = f" {label} "
    emit(char * 2 + pad(text, width - 2, "<"), codes)


def verdict(ok: bool, text: str) -> str:
    return c(text, BOLD, GREEN if ok else RED)


def phase_label(stage: str) -> tuple:
    """阶段配色：预热=品红、全量=绿。"""
    return (BOLD, MAGENTA) if stage == "warmup" else (BOLD, GREEN)


class ConsoleFormatter(logging.Formatter):
    """终端用：levelname 着色（文件处理器用普通 Formatter，保持纯文本）。"""

    def __init__(self, fmt: str, datefmt: str = "%H:%M:%S") -> None:
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        original = record.levelname
        record.levelname = c(original, *LEVEL_COLORS.get(record.levelno, ()))
        try:
            return super().format(record)
        finally:
            record.levelname = original
