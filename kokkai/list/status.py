"""``out/`` 配下の meta.json を見て、各会議が「取込済」か判定する。"""

from __future__ import annotations

import json
from pathlib import Path


def load_fetched_index(out_dir: Path) -> dict[tuple[str, str], dict]:
    """(house, id) → meta dict のマップ。取込済の会議を列挙。"""
    out: dict[tuple[str, str], dict] = {}
    if not out_dir.exists():
        return out
    for p in sorted(out_dir.glob("*.meta.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        house = data.get("house", "")
        mid = str(data.get("id", ""))
        if house and mid:
            out[(house, mid)] = {
                "meta_path": str(p),
                "files": data.get("files") or {},
            }
    return out


def annotate(items: list[dict], out_dir: Path) -> list[dict]:
    """``fetch_*`` の結果に ``fetched`` / ``meta_path`` を付与する。"""
    idx = load_fetched_index(out_dir)
    for it in items:
        key = (it.get("house", ""), str(it.get("id", "")))
        info = idx.get(key)
        it["fetched"] = info is not None
        if info:
            it["meta_path"] = info["meta_path"]
            it["files"] = info["files"]
    return items
