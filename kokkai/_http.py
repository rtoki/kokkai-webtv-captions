"""共通 HTTP クライアント。

- ``POLITE_UA``: 全モジュール共通の User-Agent。
- ``polite_get(url, ...)``: robots.txt + per-host レート制限 + UA を付与した GET。
- ``polite_get_bytes(url, ...)``: bytes を返すバリアント (HLS セグメント等)。
- ``RobotsDisallowed``: robots.txt が当該 URL を拒否しているときの例外。

設計方針:

- robots.txt はホスト単位で 1 回だけ取得しキャッシュする。
  404 / その他取得失敗は「制限なし」として扱う (RFC 9309 準拠)。
- ``_DEFAULT_INTERVAL`` に挙げたオリジンサーバには **最低 0.5 秒** の
  per-host 間隔を空ける。CDN (public.mediasp.jp など) は 0 で並列化を維持。
- robots.txt の ``Crawl-delay`` が POLITE_UA 向けに指定されていれば、
  ``_DEFAULT_INTERVAL`` より優先する。
- 並列スレッドから呼ばれても per-host の last-access 更新が壊れないよう
  ``threading.Lock`` で保護する。
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from typing import Optional


POLITE_UA = "kokkai-webtv-captions/0.2.0"

# 「オリジン (国会側) サーバ」と判定したホストの最小アクセス間隔 (秒)。
# robots.txt に Crawl-delay があればそちらを優先。CDN は 0。
_DEFAULT_INTERVAL: dict[str, float] = {
    "www.webtv.sangiin.go.jp": 0.5,
    "www.sangiin.go.jp": 0.5,
    "www.shugiintv.go.jp": 0.5,
    "rss.shugiintv.go.jp": 0.5,
    "www.shugiin.go.jp": 0.5,
}

_robots_cache: dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}
_last_access: dict[str, float] = {}
_robots_lock = threading.Lock()
_rate_lock = threading.Lock()


class RobotsDisallowed(Exception):
    """robots.txt が User-Agent に対し当該 URL を拒否している。"""


def _fetch_robots(host: str, scheme: str) -> Optional[urllib.robotparser.RobotFileParser]:
    """robots.txt をホスト単位で取得。

    Returns:
        RobotFileParser: robots.txt が存在し有効にパースできたとき。
        None: robots.txt が存在しない (404) か、取得に失敗したとき (制限なし扱い)。
    """
    robots_url = f"{scheme}://{host}/robots.txt"
    req = urllib.request.Request(robots_url, headers={"User-Agent": POLITE_UA})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            body = r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError:
        return None  # 404 等 → 制限なし
    except Exception:
        return None  # ネットワーク失敗 → 制限なし (寛容寄り)
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    rp.parse(body.splitlines())
    return rp


def _get_robots(host: str, scheme: str) -> Optional[urllib.robotparser.RobotFileParser]:
    with _robots_lock:
        if host in _robots_cache:
            return _robots_cache[host]
    rp = _fetch_robots(host, scheme)
    with _robots_lock:
        # 競合した場合は先勝ち (両者とも同じ結論になるはず)
        _robots_cache.setdefault(host, rp)
        return _robots_cache[host]


def _resolve_min_interval(
    host: str, rp: Optional[urllib.robotparser.RobotFileParser]
) -> float:
    if rp is not None:
        try:
            delay = rp.crawl_delay(POLITE_UA)
            if delay:
                return float(delay)
        except Exception:
            pass
    return _DEFAULT_INTERVAL.get(host, 0.0)


def _wait_for_host(host: str, interval: float) -> None:
    if interval <= 0:
        return
    with _rate_lock:
        now = time.monotonic()
        last = _last_access.get(host, 0.0)
        wait = interval - (now - last)
        if wait > 0:
            time.sleep(wait)
        _last_access[host] = time.monotonic()


def _check_and_wait(url: str) -> None:
    """robots.txt チェック + per-host レート制限。"""
    parsed = urllib.parse.urlsplit(url)
    host = parsed.netloc
    if not host:
        return
    rp = _get_robots(host, parsed.scheme or "https")
    if rp is not None and not rp.can_fetch(POLITE_UA, url):
        raise RobotsDisallowed(
            f"robots.txt disallows {url} for User-Agent {POLITE_UA!r}"
        )
    interval = _resolve_min_interval(host, rp)
    _wait_for_host(host, interval)


def polite_get_bytes(
    url: str,
    *,
    referer: Optional[str] = None,
    timeout: int = 30,
) -> bytes:
    """robots.txt + レート制限 + UA を付与した GET (bytes)。"""
    _check_and_wait(url)
    headers = {"User-Agent": POLITE_UA}
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def polite_get(
    url: str,
    *,
    referer: Optional[str] = None,
    timeout: int = 30,
    encoding: str = "utf-8",
) -> str:
    """robots.txt + レート制限 + UA を付与した GET (str)。

    Args:
        encoding: ``"utf-8"`` / ``"euc_jp"`` / ``"shift_jis"`` 等。
            ``errors="replace"`` で安全側にデコードする。
    """
    raw = polite_get_bytes(url, referer=referer, timeout=timeout)
    return raw.decode(encoding, errors="replace")
