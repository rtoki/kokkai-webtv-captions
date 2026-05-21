"""
衆議院インターネット審議中継 (shugiintv.go.jp) からメタ情報を抽出するモジュール。

主な公開関数:
- ``resolve_from_shugiin_detail(target)`` — deli_id または URL から (m3u8_url, meta) を返す
- ``parse_meta_html(html, deli_id)`` — フェッチ済み HTML 文字列を直接パース (テスト用)

衆議院ページは EUC-JP エンコード、HLS は HTTP → HTTPS 301 で飛ぶ点に注意。
"""

from __future__ import annotations

import re

from kokkai._http import polite_get


REFERER = "https://www.shugiintv.go.jp/"
DETAIL_URL_TMPL = (
    "https://www.shugiintv.go.jp/jp/index.php?ex=VL&deli_id={}&media_type="
)


def http_get_html(url: str) -> str:
    """衆議院ページは EUC-JP なので UTF-8 デコードして返す。"""
    return polite_get(url, referer=REFERER, timeout=30, encoding="euc_jp")


def _parse_deli_id(target: str) -> str:
    """deli_id 単体 (例: "56246") または URL 形式から deli_id を取り出す。"""
    s = target.strip()
    if s.isdigit():
        return s
    m = re.search(r"deli_id=(\d+)", s)
    if m:
        return m.group(1)
    raise ValueError(
        f"deli_id を抽出できません: {target!r}  "
        f"(数字だけか、index.php?...deli_id=NNNN を含む URL を指定してください)"
    )


def _extract_hls_url(html: str) -> str | None:
    """vtag_src_base_vod hidden input から HLS m3u8 URL を抜き出し、https に正規化。"""
    m = re.search(
        r'id="vtag_src_base_vod"\s+value="(http[^"]+\.m3u8)"', html
    )
    if not m:
        return None
    url = m.group(1)
    return url.replace("http://", "https://", 1)


def _extract_date(html: str) -> str:
    """例: "2026年5月15日" → "2026-05-15" """
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", html)
    if not m:
        return ""
    y, mo, d = m.groups()
    return f"{y}-{int(mo):02d}-{int(d):02d}"


# 会議名は "○○委員会" "○○本会議" "○○審査会" "○○調査会" のいずれか。
# 直後に "(時間表記)" が続くパターンで識別。
_MEETING_RE = re.compile(
    r"<td[^>]*>([^<(（]*?(?:委員会|本会議|審査会|調査会|公聴会))\s*[\(（]\d"
)


def _extract_meeting_name(html: str) -> str:
    m = _MEETING_RE.search(html)
    return m.group(1).strip() if m else ""


_SPEAKER_RE = re.compile(
    r'time=([\d.]+)"[^>]*>([^<]+)</a>'
)
_NAME_GROUP_RE = re.compile(r"(.+?)\s*[\(（](.+)[\)）]\s*$")


def _extract_speakers(html: str) -> list[dict]:
    """
    発言者一覧を ``[{start: float, name: str, group: str}, ...]`` で返す。

    HTML には「はじめから再生」ボタンも同じ ``time=...`` 形式で含まれるため、
    リスト内で重複する場合は最初の発言者リスト出現を優先するよう dedup する。
    """
    seen: set[tuple[float, str]] = set()
    out: list[dict] = []
    for m in _SPEAKER_RE.finditer(html):
        start = float(m.group(1))
        label = re.sub(r"\s+", " ", m.group(2)).strip()
        if not label or label in ("はじめから再生",):
            continue
        nm = _NAME_GROUP_RE.match(label)
        if nm:
            name, group = nm.group(1).strip(), nm.group(2).strip()
        else:
            name, group = label, ""
        key = (round(start, 1), name)
        if key in seen:
            continue
        seen.add(key)
        out.append({"start": start, "name": name, "group": group})
    out.sort(key=lambda s: s["start"])
    return out


# 議題セルは width="595" で他のセル (発言者リスト=380, 時間=100) と
# 区別される。spacer.gif の <IMG> タグが前後に挟まるので DOTALL でテキストだけ抜く。
_AGENDA_RE = re.compile(
    r'<TD\s+[^>]*width="595"[^>]*bgcolor="#CCCCFF"[^>]*>(.*?)</TD>',
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _extract_agenda(html: str) -> list[str]:
    """
    案件 (法案名・議題) を抽出する。

    視聴ページの `<TD width="595" bgcolor="#CCCCFF">` セル群に各議題が
    入っている (spacer.gif 等の画像タグが前後に混じる)。Phase 3 の ASR
    initial_prompt で固有名詞バイアスとして使うため、議題名のテキスト
    部分だけを取り出す。
    """
    out: list[str] = []
    for m in _AGENDA_RE.finditer(html):
        text = _TAG_RE.sub("", m.group(1))
        text = re.sub(r"[\s　]+", " ", text).strip()
        if text:
            out.append(text)
    return out


def parse_meta_html(html: str, deli_id: str) -> tuple[str | None, dict]:
    """フェッチ済み HTML から (m3u8_url, meta) を組み立てる純関数 (テスト用)。"""
    m3u8 = _extract_hls_url(html)
    meta = {
        "deli_id": deli_id,
        "date": _extract_date(html),
        "title": _extract_meeting_name(html),
        "speakers": _extract_speakers(html),
        "agenda": _extract_agenda(html),
        "page_url": DETAIL_URL_TMPL.format(deli_id),
    }
    return m3u8, meta


def resolve_from_shugiin_detail(target: str) -> tuple[str, dict]:
    """
    deli_id (例: "56246") または URL を受け取り、(m3u8_url, meta) を返す。

    meta:
        - ``deli_id``: 文字列
        - ``date``: ISO 形式 (YYYY-MM-DD)
        - ``title``: 会議名 (例: "内閣委員会")
        - ``speakers``: ``[{start, name, group}, ...]`` (開始時刻昇順)
        - ``page_url``: 元の視聴ページ URL
    """
    deli_id = _parse_deli_id(target)
    url = DETAIL_URL_TMPL.format(deli_id)
    html = http_get_html(url)
    m3u8, meta = parse_meta_html(html, deli_id)
    if not m3u8:
        raise ValueError(
            f"deli_id={deli_id}: HLS m3u8 URL が見つかりません。"
            f" ページ構造が変わった可能性があります。"
        )
    if not meta["speakers"]:
        # 発言者リストが無い動画 (会議冒頭のみの短尺等) もあり得るので警告に留める
        import sys
        print(
            f"[!] deli_id={deli_id}: 発言者一覧が抽出できませんでした",
            file=sys.stderr,
        )
    return m3u8, meta
