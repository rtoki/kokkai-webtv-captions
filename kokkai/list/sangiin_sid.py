"""参議院 webtv の sid を、日付 + 会議名から逆引きする。

参議院は ``keyword_search.php`` (検索 API) が F5 ASM (WAF) で弾かれるため、
過去日の sid を直接リストできない。代わりに ``detail.php?sid=N`` は GET で
普通に開けるので、二分探索 + 隣接走査で「指定日に開催された全 sid」を
求める。

sid は単調増加する（古い会議ほど小さい）。同じ日に複数の会議があると
sid は連番で並ぶ傾向にある。

結果はディスクにキャッシュ (``~/.cache/kokkai/sangiin_sid_cache.json``)
して、2 回目以降はネットワークなしで返す。
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from pathlib import Path

from kokkai._http import polite_get


REFERER = "https://www.webtv.sangiin.go.jp/webtv/index.php"
DETAIL_URL_TMPL = "https://www.webtv.sangiin.go.jp/webtv/detail.php?sid={}"

# 空 sid の detail.php は ~87 bytes の短いリダイレクト/エラー画面を返す
EMPTY_RESPONSE_THRESHOLD = 500

# 二分探索の上限。現在 (2026-05) で ~9050。10 年後でも 30000 に到達しないはず。
SID_UPPER_BOUND = 30000

# 探索 / 走査の安全上限
MAX_BISECT_STEPS = 25
MAX_NEIGHBOR_WALK = 50

_OPEN_DATE_RE = re.compile(r"<dt>開会日</dt>\s*<dd>(\d{4})年(\d{1,2})月(\d{1,2})日</dd>")
_MEETING_NAME_RE = re.compile(r"<dt>会議名</dt>\s*<dd>([^<]+)</dd>")


class SidLookupUnavailable(Exception):
    """一時的な通信失敗などで sid 逆引きを続行できない。"""


def _cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    p = Path(base) / "kokkai" / "sangiin_sid_cache.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_cache() -> dict:
    p = _cache_path()
    if not p.exists():
        return {"detail": {}, "by_date": {}, "sid_max": None, "sid_max_at": None}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"detail": {}, "by_date": {}, "sid_max": None, "sid_max_at": None}
    data.setdefault("detail", {})
    data.setdefault("by_date", {})
    data.setdefault("sid_max", None)
    data.setdefault("sid_max_at", None)
    return data


def _save_cache(cache: dict) -> None:
    try:
        _cache_path().write_text(
            json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        pass


def _http_get(url: str) -> str:
    return polite_get(url, referer=REFERER, timeout=20)


def _fetch_detail(sid: int, cache: dict) -> dict | None:
    """detail.php?sid=N から (date, name) を取得。空 sid なら None。

    結果は cache["detail"][str(sid)] に保存する (None 含む)。
    """
    key = str(sid)
    if key in cache["detail"]:
        v = cache["detail"][key]
        return v  # None or {"date": "...", "name": "..."}
    try:
        html = _http_get(DETAIL_URL_TMPL.format(sid))
    except Exception as e:
        raise SidLookupUnavailable(f"detail.php?sid={sid} の取得に失敗: {e}") from e
    if len(html) < EMPTY_RESPONSE_THRESHOLD:
        cache["detail"][key] = None
        return None
    m = _OPEN_DATE_RE.search(html)
    if not m:
        cache["detail"][key] = None
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    name_m = _MEETING_NAME_RE.search(html)
    name = name_m.group(1).strip() if name_m else ""
    rec = {"date": f"{y:04d}-{mo:02d}-{d:02d}", "name": name}
    cache["detail"][key] = rec
    return rec


def _probe_sid_max(cache: dict, force: bool = False) -> int:
    """現存する最大の sid を二分探索で求める。

    1 日 1 回キャッシュを更新 (24h)。
    """
    sid_max = cache.get("sid_max")
    sid_max_at = cache.get("sid_max_at")
    if not force and sid_max and sid_max_at:
        try:
            ts = datetime.fromisoformat(sid_max_at)
            if int(sid_max) > 1 and (datetime.utcnow() - ts).total_seconds() < 24 * 3600:
                return int(sid_max)
        except Exception:
            pass

    # [1, SID_UPPER_BOUND] を二分探索で「最大の valid sid」を求める。
    # 古い sid (1〜数千) はアーカイブが残っていないことがあるので、
    # 「ある sid 以上は valid」という単調性は保証されないが、
    # 「ある sid 以上はすべて invalid」という上限は単調なので、
    # それを境界として求める。
    lo, hi = 1, SID_UPPER_BOUND
    steps = 0
    while lo + 1 < hi and steps < MAX_BISECT_STEPS:
        mid = (lo + hi) // 2
        if _fetch_detail(mid, cache) is not None:
            lo = mid
        else:
            hi = mid
        steps += 1

    # lo は valid だが、上の二分探索は「mid が invalid なら hi = mid」とするので、
    # 最終 lo が valid である保証は最後に valid を引いたとき。
    # 念のため lo から下方向に確認 (高々数 step)
    while lo > 1 and _fetch_detail(lo, cache) is None:
        lo -= 1

    cache["sid_max"] = lo
    cache["sid_max_at"] = datetime.utcnow().isoformat()
    return lo


def _bisect_sid_for_date(target: date, sid_max: int, cache: dict) -> int | None:
    """target 日のどれか 1 つの sid を返す。

    sid は単調増加だが、古い sid は archive から削除されていることや
    GW 等で連続した無効帯がある。素朴な二分探索は無効帯で迷子になるので、
    既知の (sid, date) アンカー集合を毎ステップ更新し、target を
    straddle する一番近い 2 つから線形補間で seed を絞り込む。
    """
    target_iso = target.isoformat()

    # 既知アンカー集合 (sid, date)
    anchors: dict[int, date] = {}
    by_date = cache["by_date"]
    for d_str, mapping in by_date.items():
        try:
            d = date.fromisoformat(d_str)
        except Exception:
            continue
        for sid_v in mapping.values():
            anchors[int(sid_v)] = d
    # sid_max を最新アンカーとして取り込む
    sid_max_meta = _fetch_detail(sid_max, cache)
    if sid_max_meta:
        try:
            anchors[sid_max] = date.fromisoformat(sid_max_meta["date"])
        except Exception:
            pass
    # cache["detail"] にある valid 全部も拾う (確認済 sid を有効活用)
    for k, v in cache["detail"].items():
        if v is None:
            continue
        try:
            anchors[int(k)] = date.fromisoformat(v["date"])
        except Exception:
            continue

    if not anchors:
        return None
    newest = max(anchors.values())
    if target > newest:
        return None  # 未来 (sid 未割当)

    def _add_anchor(sid: int, rec: dict) -> None:
        try:
            anchors[sid] = date.fromisoformat(rec["date"])
        except Exception:
            pass

    def _bracket() -> tuple[tuple[int, date] | None, tuple[int, date] | None]:
        """target を straddle する (lower, upper) アンカーを返す。"""
        lo_pair = None
        hi_pair = None
        for sid_v, d in anchors.items():
            if d == target:
                return (sid_v, d), (sid_v, d)
            if d < target:
                if lo_pair is None or d > lo_pair[1] or (d == lo_pair[1] and sid_v > lo_pair[0]):
                    lo_pair = (sid_v, d)
            else:
                if hi_pair is None or d < hi_pair[1] or (d == hi_pair[1] and sid_v < hi_pair[0]):
                    hi_pair = (sid_v, d)
        return lo_pair, hi_pair

    for _ in range(MAX_BISECT_STEPS):
        lo_pair, hi_pair = _bracket()
        if lo_pair and lo_pair[1] == target:
            return lo_pair[0]
        if hi_pair and hi_pair[1] == target:
            return hi_pair[0]

        # 補間で seed を決める
        if lo_pair and hi_pair:
            sid_lo, date_lo = lo_pair
            sid_hi, date_hi = hi_pair
            if sid_hi - sid_lo <= 1:
                # 範囲が 1 以下に狭まったのに見つからない → archive 欠落
                return None
            span_days = max(1, (date_hi - date_lo).days)
            ratio = (target - date_lo).days / span_days
            seed = sid_lo + max(1, int((sid_hi - sid_lo) * ratio))
            # 端点を避ける
            seed = min(max(seed, sid_lo + 1), sid_hi - 1)
            search_lo, search_hi = sid_lo + 1, sid_hi - 1
        elif lo_pair:
            sid_lo, date_lo = lo_pair
            seed = sid_lo + max(1, (target - date_lo).days * 4)
            search_lo, search_hi = sid_lo + 1, sid_max
        else:
            assert hi_pair
            sid_hi, date_hi = hi_pair
            seed = max(1, sid_hi - max(1, (date_hi - target).days * 4))
            search_lo, search_hi = 1, sid_hi - 1

        # seed から半径 max((search_hi-search_lo)//2, 5) で valid を探す
        radius = max(5, (search_hi - search_lo) // 2)
        found = None
        for r in range(radius + 1):
            for s in (seed + r, seed - r):
                if s < search_lo or s > search_hi:
                    continue
                if s in anchors:
                    found = (s, {"date": anchors[s].isoformat(), "name": ""})
                    break
                rec = _fetch_detail(s, cache)
                if rec is not None:
                    found = (s, rec)
                    break
            if found:
                break
        if not found:
            return None
        sid, rec = found
        if rec.get("date") == target_iso:
            return sid
        _add_anchor(sid, rec)

    return None


def _walk_neighbors_for_date(seed_sid: int, target: date, cache: dict) -> dict[str, int]:
    """seed_sid から前後に走査して、target 日に開催された全 sid を集める。

    Returns: {meeting_name: sid}
    """
    target_iso = target.isoformat()
    result: dict[str, int] = {}

    rec = _fetch_detail(seed_sid, cache)
    if not rec or rec["date"] != target_iso:
        return result
    result[rec["name"]] = seed_sid

    # 後方走査
    sid = seed_sid - 1
    for _ in range(MAX_NEIGHBOR_WALK):
        if sid < 1:
            break
        r = _fetch_detail(sid, cache)
        if r is None:
            sid -= 1
            continue
        if r["date"] != target_iso:
            break
        result.setdefault(r["name"], sid)
        sid -= 1

    # 前方走査
    sid = seed_sid + 1
    for _ in range(MAX_NEIGHBOR_WALK):
        r = _fetch_detail(sid, cache)
        if r is None:
            sid += 1
            continue
        if r["date"] != target_iso:
            break
        result.setdefault(r["name"], sid)
        sid += 1

    return result


def resolve_sids_for_date(target: date, committee_names: list[str]) -> dict[str, str | None]:
    """指定日の各委員会名に対する sid を返す。

    Returns: {committee_name: sid_string_or_None}
    """
    cache = _load_cache()
    target_iso = target.isoformat()

    # 既にキャッシュ済みか?
    cached_for_day = cache["by_date"].get(target_iso, {})
    out: dict[str, str | None] = {}
    missing = [n for n in committee_names if n not in cached_for_day]

    if missing:
        try:
            sid_max = _probe_sid_max(cache)
            seed = _bisect_sid_for_date(target, sid_max, cache)
            if seed is not None:
                found = _walk_neighbors_for_date(seed, target, cache)
                day_map = cache["by_date"].setdefault(target_iso, {})
                for name, sid in found.items():
                    day_map[name] = sid
                cached_for_day = day_map
        except SidLookupUnavailable:
            # 一時的な通信失敗は「存在しない sid」と区別し、壊れた探索結果を保存しない。
            pass
        else:
            _save_cache(cache)

    for name in committee_names:
        sid = cached_for_day.get(name)
        out[name] = str(sid) if sid is not None else None
    return out
