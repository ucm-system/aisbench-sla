"""目标服务寻址归一化。

背景（对 ais_bench v3.1.20260330 实读源码 + 实测确认）：
ais_bench 的 API 模型有两套寻址字段，且**不是干净的二选一**：

* `host_ip` + `host_port`：运行时拼 `{http|https}://{host_ip}:{host_port}/`
  （IPv6 自动加方括号）；但 `utils/config/build.py` 会校验 `host_ip` 必须是合法
  IPv4/IPv6，唯一放行的非 IP 值是 `"localhost"`；`host_port` 也必须是无条件合法的
  整数端口。运行时代码其实容忍 hostname（注释：it's a hostname, use it as is），
  却被前置校验拦死 —— 这就是 `host.docker.internal` 报 [UTILS-CFG-002] 的根因。
* `url`：完整 baseUrl，设置后 host_ip/host_port 被忽略（运行时打日志提示），
  但校验器里没有 url 这一项、而 host_ip/host_port 的校验依旧独立生效。

于是本模块把用户输入统一成一个 `--target`，自动决定写哪套字段：
* 裸 IP[:端口]        -> host_ip/host_port（最贴合原生用法）
* 主机名 / 带 scheme / 带路径 -> url，并同时写 host_ip="localhost" 规避校验雷区
* url 带路径时补末尾 '/'，否则 urljoin 会把路径替换掉
"""
from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

_SCHEME_RE = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*)://(?P<rest>.*)$")
_DEFAULT_PORTS = {"http": 80, "https": 443}


@dataclass
class Target:
    """归一化后的目标服务地址。"""

    raw: str
    scheme: str = ""            # 用户显式给的 scheme（"" 表示没给）
    host: str = ""
    port: Optional[int] = None
    path: str = ""              # 形如 "api/"（已去首斜杠）
    is_ip: bool = False
    ip_version: int = 0

    # ------------------------------------------------------------- 判定
    @property
    def use_url_field(self) -> bool:
        """是否应写入 ais_bench 的 url 字段。"""
        return bool(self.scheme or self.path or not self.is_ip)

    @property
    def effective_scheme(self) -> str:
        return self.scheme or "http"

    def base_url(self) -> str:
        host = f"[{self.host}]" if self.ip_version == 6 else self.host
        url = f"{self.effective_scheme}://{host}"
        if self.port:
            url += f":{self.port}"
        if self.path:
            # urljoin 会用它替换 base 的路径，必须以 '/' 结尾才保留
            url += "/" + self.path.rstrip("/") + "/"
        return url

    def metrics_endpoint(self) -> str:
        """metrics 采集端点 host:port（默认端口按 scheme 推断）。"""
        port = self.port or _DEFAULT_PORTS.get(self.effective_scheme, 80)
        host = f"[{self.host}]" if self.ip_version == 6 else self.host
        return f"{host}:{port}"

    def model_fields(self) -> Dict[str, object]:
        """→ 写入 ais_bench 模型配置的三件套。"""
        if self.use_url_field:
            # host_ip 只写 "localhost"：校验唯一放行的非 IP 值，运行时不使用
            return {"host_ip": "localhost",
                    "host_port": self.port or _DEFAULT_PORTS.get(self.effective_scheme, 80),
                    "url": self.base_url()}
        return {"host_ip": self.host,
                "host_port": self.port or _DEFAULT_PORTS.get(self.effective_scheme, 80),
                "url": ""}

    def describe(self) -> str:
        mode = "url" if self.use_url_field else "host_ip:host_port"
        return f"{self.base_url()}（{mode} 模式）"


def parse_target(text: str) -> Target:
    """解析 `1.2.3.4:8000` / `host:8000` / `http(s)://host[:port][/path]` / `[::1]:8000`。"""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("目标地址为空")
    target = Target(raw=raw)
    rest = raw
    match = _SCHEME_RE.match(raw)
    if match:
        target.scheme = match.group("scheme").lower()
        rest = match.group("rest")
    # 去掉 userinfo（token@host 形式）
    if "@" in rest.split("/")[0]:
        head, _, tail = rest.partition("@")
        rest = tail
    rest, _, path = rest.partition("/")
    target.path = path.strip("/")

    # 裸 IPv6（无方括号）优先整体识别，避免被端口分割误伤
    bare_host = rest
    if not bare_host.startswith("["):
        try:
            ip = ipaddress.ip_address(bare_host)
            target.host = str(ip)
            target.is_ip = True
            target.ip_version = ip.version
            return target
        except ValueError:
            pass

    if rest.startswith("["):                      # [::1]:8000
        host, _, tail = rest[1:].partition("]")
        target.host = host
        port_text = tail.lstrip(":")
        target.port = int(port_text) if port_text else None
    else:
        host, sep, port_text = rest.rpartition(":")
        if sep and port_text.isdigit():
            target.host, target.port = host, int(port_text)
        else:
            target.host = rest
    if not target.host:
        raise ValueError(f"无法从 {text!r} 解析主机名")
    try:
        ip = ipaddress.ip_address(target.host)
        target.is_ip = True
        target.ip_version = ip.version
        target.host = str(ip)
    except ValueError:
        target.is_ip = False
    if target.port is not None and not (0 < target.port < 65536):
        raise ValueError(f"端口越界：{target.port}（需在 0-65535）")
    return target


def resolve(target_text: str, *, host: str = "", port: Optional[int] = None,
            url: str = "") -> Tuple[Target, Dict[str, object]]:
    """把 target / host / url 三种输入收敛为一个 Target 与模型字段。

    优先级：显式 --url > --target > --host/--port。
    """
    if url:
        combined = url if _SCHEME_RE.match(url) else f"http://{url}"
        parsed = parse_target(combined)
        return parsed, parsed.model_fields()
    if target_text:
        parsed = parse_target(target_text)
        return parsed, parsed.model_fields()
    text = str(host or "")
    if port:
        text = f"{text}:{port}" if not text.endswith(f":{port}") else text
    parsed = parse_target(text)
    return parsed, parsed.model_fields()
