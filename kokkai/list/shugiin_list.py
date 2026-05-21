"""衆議院の中継一覧を日別に取得 (shugiintv.go.jp の `?u_day=YYYYMMDD` を解析)。"""

from __future__ import annotations

import re
from datetime import date, timedelta

from kokkai._http import polite_get


REFERER = "https://www.shugiintv.go.jp/"


def _http_get_eucjp(url: str) -> str:
    return polite_get(url, referer=REFERER, timeout=30, encoding="euc_jp")


_LIST_URL = "https://www.shugiintv.go.jp/jp/index.php?ex=VL&u_day={}"


def fetch_for_date(target_date: date) -> list[dict]:
    """指定日の衆議院中継一覧を返す。

    各エントリ: ``{house: "shugiin", id: <deli_id>, date: ISO, title, duration, page_url}``
    """
    ymd = target_date.strftime("%Y%m%d")
    url = _LIST_URL.format(ymd)
    try:
        html = _http_get_eucjp(url)
    except Exception:
        return []

    # 「該当する検索結果はありません」の場合は空リスト
    if "該当する検索結果はありません" in html:
        return []

    # 委員会名は <A href="...&deli_id=N..."> 内の text、長さは別途
    items: list[dict] = []
    # 1 行ごとに deli_id + 委員会名 + 時間 をまとめて取れるよう、行ブロックで切る
    # row blocks: <TR ...> ... <A ...&deli_id=N>...</A> ... <TD>...分</TD> ... </TR>
    row_re = re.compile(
        r'<A[^>]*deli_id=(\d+)[^>]*>([^<]+?)</A>'
        r'.*?<TD[^>]*class="s14_24"[^>]*>([^<]+)</TD>',
        re.DOTALL | re.IGNORECASE,
    )
    for m in row_re.finditer(html):
        deli_id, title, duration = m.group(1), m.group(2).strip(), m.group(3).strip()
        if not any(k in title for k in ("委員会", "本会議", "審査会", "調査会", "公聴会")):
            continue
        items.append({
            "house": "shugiin",
            "id": deli_id,
            "date": target_date.strftime("%Y-%m-%d"),
            "title": title,
            "duration": duration,
            "page_url": (
                f"https://www.shugiintv.go.jp/jp/index.php?ex=VL"
                f"&deli_id={deli_id}&media_type="
            ),
        })

    # 重複除去 (同一 deli_id で複数行マッチした場合)
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        if it["id"] in seen:
            continue
        seen.add(it["id"])
        out.append(it)
    return out


def fetch_for_range(start: date, end: date, max_days: int = 60) -> list[dict]:
    """期間内の衆議院中継一覧 (日ごとに HTTP 取得を ``max_days`` 日まで繰り返す)。"""
    if start > end:
        start, end = end, start
    span = (end - start).days
    if span > max_days:
        start = end - timedelta(days=max_days)

    out: list[dict] = []
    d = start
    while d <= end:
        out.extend(fetch_for_date(d))
        d += timedelta(days=1)
    return out
