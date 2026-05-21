"""
参議院インターネット審議中継から WebVTT 字幕を取得・解析するモジュール.

主な公開関数:
- ``resolve_from_sangiin_detail(url)`` — detail.php URL から (m3u8_url, meta) を返す
- ``find_subtitle_playlist(master_url, master_body)`` — m3u8 マスターから字幕プレイリスト URL を抽出
- ``fetch_vtt_segments(sub_playlist_url)`` — 全 WebVTT セグメントを並列ダウンロード&連結
- ``vtt_to_text(vtt)`` — WebVTT から純テキスト (タイムコード除去)
- ``parse_vtt_cues(vtt)`` — WebVTT を {start, end, text} のリストへ
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import re
import sys
import urllib.parse

from kokkai._http import polite_get


REFERER = "https://www.webtv.sangiin.go.jp/"


def http_get(url: str) -> str:
    return polite_get(url, referer=REFERER, timeout=30)


SANGIIN_DETAIL_RE = re.compile(
    r"webtv\.sangiin\.go\.jp/webtv/detail\.php\?sid=(\d+)"
)


def resolve_from_sangiin_detail(url: str) -> tuple[str, dict]:
    """
    detail.php?sid=XXXX を受けて (m3u8_url, meta) を返す。

    現行ページ構造:
        detail.php → <script src="public.mediasp.jp/v1/player?hash=..."> → m3u8 URL

    VOD と LIVE の 2 ケースに対応:
    - VOD: player JS に ``url:"...m3u8"`` が直接埋まっている
    - LIVE: ``url:""`` で、``channel_info:{url:"...channel-info.json"}`` から
      JSON を fetch して ``manifest`` フィールドで HLS URL を組み立てる
    """
    html = http_get(url)

    m_player = re.search(
        r'<script\s+src="(https?://public\.mediasp\.jp/v1/player\?[^"]+)"', html
    )
    if not m_player:
        raise SystemExit(
            "mediasp player URL が見つかりません。ページ構造が変わった可能性があります。"
        )
    player_url = m_player.group(1)

    js = http_get(player_url)
    m3u8, is_live = _extract_m3u8_url(js)

    m_caption = re.search(r'captionUrl:"([^"]*)"', js)
    caption_url = m_caption.group(1) if m_caption else ""

    date = _extract_dd(html, "開会日") or ""
    title = _extract_dd(html, "会議名") or ""
    speakers = _extract_speakers(html)
    meta = {
        "date": date, "title": title,
        "caption_url": caption_url, "player_url": player_url,
        "speakers": speakers,
        "is_live": is_live,
    }
    return m3u8, meta


def _extract_m3u8_url(player_js: str) -> tuple[str, bool]:
    """
    player JS から HLS マスター m3u8 URL を取り出す。

    Returns:
        (m3u8_url, is_live)
        VOD (録画済) なら ``url:"..."`` を直接利用、LIVE なら ``channel_info.url``
        の JSON を fetch して ``manifest`` フィールドから HLS マスターを組み立てる。
    """
    m_vod = re.search(r'url:"([^"]+\.m3u8[^"]*)"', player_js)
    if m_vod:
        return m_vod.group(1), False

    # LIVE: video_info の url が空、channel_info.url から channel-info.json を引く
    m_ch = re.search(r'channel_info:\{[^}]*url:"([^"]+)"', player_js)
    if not m_ch:
        raise SystemExit(
            "player JS から m3u8 URL も channel_info URL も取れません。"
        )
    ch_url = m_ch.group(1)
    ch_json = http_get(ch_url)
    try:
        ch = json.loads(ch_json)
    except json.JSONDecodeError as e:
        raise SystemExit(f"channel-info.json の JSON パース失敗: {e}") from e
    manifest = ch.get("manifest", "index.m3u8")
    m3u8 = urllib.parse.urljoin(ch_url.rsplit("/", 1)[0] + "/", manifest)
    return m3u8, True


def _extract_dd(html: str, label: str) -> str | None:
    m = re.search(
        rf"<dt[^>]*>\s*{re.escape(label)}\s*</dt>\s*<dd[^>]*>\s*(.+?)\s*</dd>",
        html,
    )
    return m.group(1).strip() if m else None


def _extract_speakers(html: str) -> list[dict]:
    """発言者一覧 (名前 + 会派/役職 + 開始秒数) を抽出。"""
    m_block = re.search(
        r"<h3>\s*発言者一覧\s*</h3>\s*<ul>(.*?)</ul>", html, re.DOTALL
    )
    if not m_block:
        return []
    out = []
    for m in re.finditer(
        r"<a\s+href=['\"]#([\d.]+)['\"][^>]*>(.+?)</a>",
        m_block.group(1), re.DOTALL,
    ):
        sec = float(m.group(1))
        label = re.sub(r"\s+", " ", m.group(2)).strip()
        name_m = re.match(r"(.+?)\s*[\(（](.+)[\)）]\s*$", label)
        if name_m:
            name, group = name_m.group(1).strip(), name_m.group(2).strip()
        else:
            name, group = label, ""
        out.append({"start": sec, "name": name, "group": group})
    out.sort(key=lambda s: s["start"])
    return out


def find_subtitle_playlist(master_url: str, master_body: str) -> str | None:
    """master playlist から WebVTT サブタイトルプレイリスト URL を抽出。"""
    for line in master_body.splitlines():
        if line.startswith("#EXT-X-MEDIA") and "TYPE=SUBTITLES" in line:
            m = re.search(r'URI="([^"]+)"', line)
            if m:
                return urllib.parse.urljoin(master_url, m.group(1))
    return None


def fetch_vtt_segments(sub_playlist_url: str, workers: int = 32) -> str:
    """字幕プレイリストの全 .vtt セグメントを並列取得し、順序を保って連結。"""
    body = http_get(sub_playlist_url)
    segments = [
        urllib.parse.urljoin(sub_playlist_url, line.strip())
        for line in body.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    print(
        f"[*] WebVTT セグメント数: {len(segments)} (並列 {workers})",
        file=sys.stderr,
    )
    results: list[str | None] = [None] * len(segments)

    def get(i: int) -> tuple[int, str | None]:
        try:
            return i, http_get(segments[i])
        except Exception as e:
            print(f"  ! セグメント {i+1} の取得失敗: {e}", file=sys.stderr)
            return i, None

    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(get, i) for i in range(len(segments))]
        for fut in cf.as_completed(futures):
            i, text = fut.result()
            results[i] = text
            done += 1
            if done % 100 == 0 or done == len(segments):
                print(f"    {done}/{len(segments)}", file=sys.stderr)
    return "\n".join(r for r in results if r)


# ---------- VTT 解析 ----------

_TS_RE = re.compile(r"(\d+):(\d{2}):(\d{2})\.(\d{3})")


def _ts_to_sec(ts: str) -> float:
    m = _TS_RE.match(ts)
    if not m:
        return 0.0
    h, m_, s, ms = (int(x) for x in m.groups())
    return h * 3600 + m_ * 60 + s + ms / 1000


def parse_vtt_cues(vtt: str) -> list[dict]:
    """
    WebVTT を ``{start, end, text}`` のリストに変換。

    HLS ライブ字幕はローリングバッファで同じフレーズが連続 cue に繰り返されるため、
    ``直前 cue と text 完全一致`` ``末尾と先頭の長い重複`` を除去する。
    """
    raw: list[dict] = []
    seen = set()
    for block in re.split(r"\n\n+", vtt):
        lines = [l for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        if lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION", "X-TIMESTAMP-MAP")):
            continue
        if lines and "-->" not in lines[0]:
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        m_ts = re.match(r"\s*(\S+)\s*-->\s*(\S+)", lines[0])
        if not m_ts:
            continue
        start = _ts_to_sec(m_ts.group(1))
        end = _ts_to_sec(m_ts.group(2))
        text = " ".join(lines[1:]).strip()
        if not text:
            continue
        key = (round(start, 1), text)
        if key in seen:
            continue
        seen.add(key)
        raw.append({"start": start, "end": end, "text": text})
    raw.sort(key=lambda c: c["start"])

    cues: list[dict] = []
    for c in raw:
        if cues and cues[-1]["text"] == c["text"]:
            cues[-1]["end"] = c["end"]
            continue
        if cues:
            prev = cues[-1]["text"]
            cur = c["text"]
            max_overlap = min(len(prev), len(cur), 200)
            cut = 0
            for n in range(max_overlap, 5, -1):
                if prev.endswith(cur[:n]):
                    cut = n
                    break
            if cut:
                cur = cur[cut:].lstrip(" 、。,.")
            if cur:
                cues.append({**c, "text": cur})
        else:
            cues.append(c)
    return cues


def vtt_to_text(vtt: str) -> str:
    """WebVTT から純テキストを抽出。タイムコード・ヘッダを落とす。"""
    return "\n".join(c["text"] for c in parse_vtt_cues(vtt))
