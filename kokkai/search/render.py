"""検索結果の出力 (human / json / jsonl 3 形式)。"""

from __future__ import annotations

import json as _json
import re
import sys
from datetime import date, datetime


def _format_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _days_since(date_str: str) -> int | None:
    """会議日付からの経過日数 (国会会議録への収録予測のヒント用)。"""
    m = re.match(r"(\d{4})[-年](\d{1,2})[-月](\d{1,2})", date_str)
    if not m:
        return None
    try:
        y, mo, d = (int(x) for x in m.groups())
        meeting = date(y, mo, d)
        return (date.today() - meeting).days
    except ValueError:
        return None


def _jump_url(rec: dict) -> str:
    """公式ページの動画ジャンプ URL (時刻クエリ付き)。"""
    page = rec.get("page_url", "")
    start = int(rec.get("start") or 0)
    if not page:
        return ""
    sep = "&" if "?" in page else "?"
    # sangiin は anchor#start_sec、shugiin は &time=N.N が公式仕様
    if rec.get("house") == "shugiin":
        return f"{page}{sep}time={start}"
    return f"{page}#{start}"


def _kokkai_search_url(rec: dict) -> str:
    """国会会議録検索システムの当該会議クエリ URL (確定版がもし収録済なら参照可能)。"""
    title = rec.get("title", "")
    if not title:
        return ""
    from urllib.parse import quote
    return f"https://kokkai.ndl.go.jp/#/result?keyword={quote(title)}"


def render_human(query: str, hits: list[tuple[dict, float]], *, max_text: int = 140) -> str:
    """ターミナル向けに整形。"""
    if not hits:
        return f"[0 件] 検索: {query!r}\n"

    lines: list[str] = [f"[{len(hits)} 件] 検索: {query!r}", ""]
    for rec, score in hits:
        text = rec["text"]
        if len(text) > max_text:
            text = text[: max_text - 1] + "…"
        sp = rec.get("speaker_name") or "(発言者不明)"
        sp_group = rec.get("speaker_group") or ""
        days = _days_since(rec.get("date") or "")
        days_note = f" / {days} 日前" if days is not None and days >= 0 else ""
        committee_url_label = (
            " ← 公式会議録 (収録済の可能性)" if days is not None and days > 35 else ""
        )

        lines.append(
            f"▶ {rec.get('date', '')} {rec.get('title', '')} ({rec.get('house', '')}){days_note}"
        )
        lines.append(f"   発言者: {sp} ({sp_group}) @ {_format_time(rec.get('start', 0))}")
        lines.append(f"   score={score:.2f}  > {text}")
        lines.append(f"   {_jump_url(rec)}")
        if committee_url_label:
            lines.append(f"   {_kokkai_search_url(rec)}{committee_url_label}")
        lines.append("")
    return "\n".join(lines)


def _hit_to_dict(rec: dict, score: float) -> dict:
    return {
        "score": round(score, 4),
        "house": rec.get("house"),
        "id": rec.get("id"),
        "date": rec.get("date"),
        "title": rec.get("title"),
        "speaker_name": rec.get("speaker_name"),
        "speaker_group": rec.get("speaker_group"),
        "start": rec.get("start"),
        "text": rec.get("text"),
        "jump_url": _jump_url(rec),
        "days_since_meeting": _days_since(rec.get("date") or ""),
        "meta_path": rec.get("meta_path"),
    }


def render_json(query: str, hits: list[tuple[dict, float]]) -> str:
    payload = {
        "ok": True,
        "query": query,
        "n_hits": len(hits),
        "hits": [_hit_to_dict(r, s) for r, s in hits],
    }
    return _json.dumps(payload, ensure_ascii=False)


def render_jsonl(query: str, hits: list[tuple[dict, float]]) -> str:
    out = [_json.dumps({"type": "query", "query": query, "n_hits": len(hits)}, ensure_ascii=False)]
    for r, s in hits:
        d = _hit_to_dict(r, s)
        d["type"] = "hit"
        out.append(_json.dumps(d, ensure_ascii=False))
    return "\n".join(out)
