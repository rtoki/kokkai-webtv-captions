"""参議院の中継一覧を取得 (webtv.sangiin.go.jp の AJAX endpoints を利用)。

エンドポイント仕様 (2026-05 現在):

- ``calendar.php?calendarmove=1&dt_calendarpoint=YYYY-MM-DD``
    指定月のカレンダー HTML を返す。各 ``<td class=" clickable" title="...">`` の
    title に当日開催された委員会名が改行区切りで入っている。**WAF なし。
    2022 年〜現在まで取得可能。** 過去日の発見はこの経路を使う。
- ``detail.php?sid=N``
    個別 sid の詳細 (開会日, 会議名, HLS URL 等)。WAF なし。
    sid → date の逆引きは ``sangiin_sid.py`` で実装。
- ``result_selecter.php``
    業務時間中は今日のライブ中継 (sid 付き)、終了後は翌日予定 (sid なし)。
- ``keyword_search.php``
    本来の検索 API だが F5 ASM (WAF) に弾かれる。curl-impersonate / Playwright
    が無いと突破不可。本パッケージでは使わない。
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from kokkai._http import polite_get

from . import sangiin_sid


REFERER = "https://www.webtv.sangiin.go.jp/webtv/index.php"


def _http_get(url: str) -> str:
    return polite_get(url, referer=REFERER, timeout=20)


# result_selecter.php 用 (今日のライブ): sid + 名称を抽出
_SID_LABEL_RE = re.compile(
    r'sid=(\d+)[^>]*>([^<]+(?:委員会|本会議|審査会|調査会|公聴会|連合審査会))',
    re.IGNORECASE,
)

# calendar.php 用: clickable な日付セルを抽出
#   <td class=' clickable' title="内閣委員会\n総務委員会">
#     <a href='#' name="2026-05-14" id="2026-05-14">14</a></td>
_CALENDAR_CELL_RE = re.compile(
    r'<td[^>]*\bclickable\b[^>]*\btitle="([^"]*)"[^>]*>\s*'
    r'<a[^>]*\bname="(\d{4}-\d{2}-\d{2})"',
    re.DOTALL,
)


def fetch_today() -> list[dict]:
    """webtv トップの「今日の中継」リスト (result_selecter.php)。

    sid 付きで返るので、業務時間中のライブ取得には最速。
    終了後は翌日予定にフリップするため、何も拾えないこともある。
    """
    url = "https://www.webtv.sangiin.go.jp/webtv/result_selecter.php"
    try:
        html = _http_get(url)
    except Exception:
        return []
    out: list[dict] = []
    today = date.today().strftime("%Y-%m-%d")
    for m in _SID_LABEL_RE.finditer(html):
        sid, title = m.group(1), m.group(2).strip()
        out.append({
            "house": "sangiin",
            "id": sid,
            "date": today,
            "title": title,
            "duration": "",
            "page_url": f"https://www.webtv.sangiin.go.jp/webtv/detail.php?sid={sid}",
        })
    return out


def _fetch_calendar_month(year: int, month: int) -> dict[str, list[str]]:
    """指定月の calendar.php を取得し、{YYYY-MM-DD: [委員会名, ...]} を返す。"""
    anchor = f"{year:04d}-{month:02d}-01"
    url = (
        "https://www.webtv.sangiin.go.jp/webtv/calendar.php"
        f"?calendarmove=1&dt_calendarpoint={anchor}"
    )
    try:
        html = _http_get(url)
    except Exception:
        return {}
    out: dict[str, list[str]] = {}
    for m in _CALENDAR_CELL_RE.finditer(html):
        title_raw, day_iso = m.group(1), m.group(2)
        # title は改行区切りで委員会名が並ぶ
        names = [n.strip() for n in title_raw.split("\n") if n.strip()]
        if names:
            out[day_iso] = names
    return out


def _months_between(start: date, end: date) -> list[tuple[int, int]]:
    """[start..end] が跨る (year, month) を昇順で。"""
    out: list[tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def fetch_for_date(target_date: date, resolve_sids: bool = True) -> list[dict]:
    """指定日の参議院中継一覧を返す。

    Args:
        target_date: 対象日。
        resolve_sids: True なら detail.php 経由で sid を逆引きする
            (HTTP リクエスト数が増える代わりに ``kkcap fetch`` に直接渡せる)。

    Returns:
        中継一覧 (取得不能なら空)。
    """
    cal = _fetch_calendar_month(target_date.year, target_date.month)
    iso = target_date.isoformat()
    names = cal.get(iso, [])

    # 今日のライブ (sid 付き) で補完できるか試す
    live_by_name: dict[str, str] = {}
    if target_date == date.today():
        for it in fetch_today():
            live_by_name.setdefault(it["title"], it["id"])
            if it["title"] not in names:
                names = names + [it["title"]]

    if not names:
        return []

    sid_map: dict[str, str | None] = {}
    if resolve_sids:
        sid_map = sangiin_sid.resolve_sids_for_date(target_date, names)
    # live で取れたものは優先 (キャッシュより新しい)
    for nm, sid in live_by_name.items():
        sid_map[nm] = sid

    out: list[dict] = []
    for name in names:
        sid = sid_map.get(name)
        page_url = (
            f"https://www.webtv.sangiin.go.jp/webtv/detail.php?sid={sid}"
            if sid else "https://www.webtv.sangiin.go.jp/webtv/index.php"
        )
        out.append({
            "house": "sangiin",
            "id": sid or "",
            "date": iso,
            "title": name,
            "duration": "",
            "page_url": page_url,
        })
    return out


def fetch_for_range(
    start: date, end: date, max_days: int = 60, resolve_sids: bool = True,
) -> list[dict]:
    """期間内の参議院中継 (calendar.php を月単位でまとめ取り)。"""
    if start > end:
        start, end = end, start
    span = (end - start).days
    if span > max_days:
        start = end - timedelta(days=max_days)

    # 月単位で calendar をまとめ取り
    month_cache: dict[tuple[int, int], dict[str, list[str]]] = {}
    for ym in _months_between(start, end):
        month_cache[ym] = _fetch_calendar_month(*ym)

    out: list[dict] = []
    d = start
    today = date.today()
    while d <= end:
        iso = d.isoformat()
        names = list(month_cache.get((d.year, d.month), {}).get(iso, []))

        live_by_name: dict[str, str] = {}
        if d == today:
            for it in fetch_today():
                live_by_name.setdefault(it["title"], it["id"])
                if it["title"] not in names:
                    names.append(it["title"])

        if names:
            sid_map: dict[str, str | None] = {}
            if resolve_sids:
                sid_map = sangiin_sid.resolve_sids_for_date(d, names)
            for nm, sid in live_by_name.items():
                sid_map[nm] = sid
            for name in names:
                sid = sid_map.get(name)
                page_url = (
                    f"https://www.webtv.sangiin.go.jp/webtv/detail.php?sid={sid}"
                    if sid else "https://www.webtv.sangiin.go.jp/webtv/index.php"
                )
                out.append({
                    "house": "sangiin",
                    "id": sid or "",
                    "date": iso,
                    "title": name,
                    "duration": "",
                    "page_url": page_url,
                })
        d += timedelta(days=1)
    return out
