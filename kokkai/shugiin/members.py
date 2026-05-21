"""
衆議院議員名簿のスクレイプ + ローカルキャッシュ。

公式ページ ``itdb_annai.nsf/html/statics/syu/<NNN>kaiha.htm`` (Shift_JIS) を
会派 (kaiha) ごとに辿り、各議員の氏名・ふりがな・選挙区を取得して JSON にキャッシュする。

公開関数:
- ``load_members(refresh: bool = False, ttl_days: int = 7)``
   キャッシュを読み (古ければ更新)、 ``list[dict]`` を返す。
   各 dict: ``{"name": "逢沢一郎", "name_with_spaces": "逢沢　　一郎",
              "faction": "自由民主党・無所属の会", "furigana": "あいさわ いちろう",
              "district": "岡山1"}``
- ``parse_kaiha_html(html, faction_name)`` (テスト用、純関数)

キャッシュ場所: ``$XDG_CACHE_HOME/kokkai-webtv-captions/shugiin_members.json``
(既定 ``~/.cache/kokkai-webtv-captions/``)。
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from kokkai._http import polite_get


REFERER = "https://www.shugiin.go.jp/"
KAIHA_URL_TMPL = (
    "https://www.shugiin.go.jp/internet/itdb_annai.nsf/html/statics/syu/{}kaiha.htm"
)
# 任意の 1 会派ページの nav から全会派 URL を発見するためのスタート点
SEED_KAIHA_CODE = "011"


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "kokkai-webtv-captions"


def _cache_path() -> Path:
    return _cache_dir() / "shugiin_members.json"


def _http_get_sjis(url: str) -> str:
    # shift_jis ≒ cp932 (-ish)。errors=replace で安全に。
    return polite_get(url, referer=REFERER, timeout=30, encoding="shift_jis")


# 会派ページ内の <A HREF="NNNkaiha.htm">会派名</A> から (code, name) を取り出す
_NAV_KAIHA_RE = re.compile(r'<A\s+HREF="(\d+)kaiha\.htm">([^<]+)</A>')


def _discover_kaiha_list(seed_html: str) -> list[tuple[str, str]]:
    """seed ページの nav 部分から (会派コード, 会派名) のリストを返す。

    重複は除去、出現順を維持。
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in _NAV_KAIHA_RE.finditer(seed_html):
        code, name = m.group(1), m.group(2).strip()
        if code in seen:
            continue
        seen.add(code)
        out.append((code, name))
    return out


# 各議員エントリ: shu2td5 = 氏名, shu2td6 = ふりがな, shu2td7 = 選挙区
# 4 セル目 (shu2td8) は当選回数なのでスキップ。
_NAME_CELL_RE = re.compile(
    r'<TD\s+class="shu2td5">\s*<TT[^>]*>\s*([^<]+?)\s*</TT>', re.IGNORECASE
)
_FURI_CELL_RE = re.compile(
    r'<TD\s+class="shu2td6">\s*<TT[^>]*>\s*([^<]+?)\s*</TT>', re.IGNORECASE
)
_DISTRICT_CELL_RE = re.compile(
    r'<TD\s+class="shu2td7">\s*<TT[^>]*>\s*([^<]+?)\s*</TT>', re.IGNORECASE
)


def _normalize_name(raw: str) -> tuple[str, str]:
    """
    "逢沢　　一郎君\n" → (clean="逢沢一郎", with_spaces="逢沢　　一郎")

    末尾の「君」と空白文字を除去し、全角スペースを残した版と除いた版を返す。
    ASR hint としては両方の表記を持っておくと取りこぼしが減る。
    """
    s = raw.strip().rstrip("君").rstrip()
    # 末尾の全角スペースも落とす
    s = s.rstrip("　").rstrip()
    clean = re.sub(r"[\s　]+", "", s)
    return clean, s


def parse_kaiha_html(html: str, faction_name: str) -> list[dict]:
    """1 会派ページの HTML から議員リストを抽出する純関数 (テスト用)。"""
    names = _NAME_CELL_RE.findall(html)
    furis = _FURI_CELL_RE.findall(html)
    districts = _DISTRICT_CELL_RE.findall(html)
    # ヘッダ行 (氏名 / ふりがな / 選挙区) を skip — 「氏名」「ふりがな」「選挙区」が
    # 各セルの 1 件目として混じることがあるので落とす。
    if names and names[0].strip() == "氏名":
        names = names[1:]
    if furis and furis[0].strip() == "ふりがな":
        furis = furis[1:]
    if districts and districts[0].strip() == "選挙区":
        districts = districts[1:]
    n = min(len(names), len(furis), len(districts))
    out: list[dict] = []
    for i in range(n):
        clean, with_spaces = _normalize_name(names[i])
        if not clean:
            continue
        out.append({
            "name": clean,
            "name_with_spaces": with_spaces,
            "faction": faction_name,
            "furigana": re.sub(r"\s+", " ", furis[i]).strip(),
            "district": districts[i].strip(),
        })
    return out


def fetch_all_members() -> list[dict]:
    """全会派ページを fetch & parse して議員リストを返す (ネット必須)。"""
    seed = _http_get_sjis(KAIHA_URL_TMPL.format(SEED_KAIHA_CODE))
    kaiha_list = _discover_kaiha_list(seed)
    if not kaiha_list:
        raise SystemExit(
            "会派ナビゲーションの抽出に失敗しました。"
            " ページ構造が変わった可能性があります。"
        )

    all_members: list[dict] = []
    for code, faction in kaiha_list:
        url = KAIHA_URL_TMPL.format(code)
        print(f"[members] {code} {faction}", file=sys.stderr)
        html = _http_get_sjis(url)
        members = parse_kaiha_html(html, faction)
        all_members.extend(members)
    return all_members


def _cache_is_fresh(ttl_days: int) -> bool:
    p = _cache_path()
    if not p.exists():
        return False
    age_days = (time.time() - p.stat().st_mtime) / 86400
    return age_days < ttl_days


def load_members(refresh: bool = False, ttl_days: int = 7) -> list[dict]:
    """
    キャッシュから議員リストを取得 (期限切れ or refresh=True ならネットから再取得)。

    キャッシュフォーマット (JSON):
        {"fetched_at": <ISO timestamp>, "members": [...]}
    """
    cache = _cache_path()
    if not refresh and _cache_is_fresh(ttl_days):
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            return data["members"]
        except (KeyError, json.JSONDecodeError):
            pass  # 壊れていたら再取得

    members = fetch_all_members()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "members": members,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[members] 取得完了: {len(members)} 名 → {cache}", file=sys.stderr
    )
    return members
