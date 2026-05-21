"""``kkcap list`` の出力レンダリング (human / json / jsonl)。"""

from __future__ import annotations

import json as _json
from datetime import date


_WEEKDAYS = ["月", "火", "水", "木", "金", "土", "日"]


def _weekday(date_str: str) -> str:
    try:
        y, m, d = (int(x) for x in date_str.split("-"))
        return _WEEKDAYS[date(y, m, d).weekday()]
    except Exception:
        return "?"


def render_human(items: list[dict]) -> str:
    if not items:
        return "[0 件] 該当する中継がありません。\n"
    lines: list[str] = [f"[{len(items)} 件] 国会中継リスト", ""]
    # 日付→院 でソート
    items = sorted(
        items,
        key=lambda it: (it.get("date", ""), it.get("house", ""), it.get("id", "")),
    )
    last_date = None
    for it in items:
        d = it.get("date", "")
        if d != last_date:
            wd = _weekday(d)
            lines.append(f"── {d} ({wd}) ──")
            last_date = d
        house_label = "参議院" if it.get("house") == "sangiin" else "衆議院"
        status = " ✓ 取込済" if it.get("fetched") else ""
        dur = f" / {it['duration']}" if it.get("duration") else ""
        lines.append(f"  [{house_label}] {it.get('title', '')}{dur}{status}")
        item_id = str(it.get("id") or "")
        cmd = "sangiin" if it.get("house") == "sangiin" else "shugiin"
        if item_id:
            lines.append(
                f"    id={item_id}  取込: kkcap fetch {item_id}  (or `{cmd} {item_id}`)"
            )
        else:
            lines.append("    id=(未解決)  取込: page_url から sid/deli_id を確認してください")
        lines.append(f"    {it.get('page_url', '')}")
    lines.append("")
    return "\n".join(lines)


def render_json(items: list[dict]) -> str:
    return _json.dumps(
        {"ok": True, "n_items": len(items), "items": items},
        ensure_ascii=False,
    )


def render_jsonl(items: list[dict]) -> str:
    out = [_json.dumps({"type": "summary", "n_items": len(items)}, ensure_ascii=False)]
    for it in items:
        d = {**it, "type": "item"}
        out.append(_json.dumps(d, ensure_ascii=False))
    return "\n".join(out)
