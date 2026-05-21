"""
kokkai.search のトークン化キャッシュ (DB なし、単一 JSON ファイル)。

最も重い処理は SudachiPy 形態素解析 (15K cue で約 1-2 秒)。これをファイル単位
で結果保存しておくと、2 回目以降の検索は数十 ms になる。

キャッシュ形式 (out/.kokkai-search-cache.json):
    {
      "version": 2,
      "generated_at": "ISO timestamp",
      "manifest": ["<abs path 1>", "<abs path 2>", ...],
      "files": {
        "<absolute source file path>": {
          "mtime": <float, ファイルの mtime>,
          "cues": [
            {"start": 0.0, "end": 1.0, "text": "...", "tokens": ["..."]},
            ...
          ]
        },
        ...
      }
    }

無効化:
- ソースファイルの mtime が一致しない → そのファイル分だけ再計算
- 現在の `out/` のファイル集合が ``manifest`` と異なる → 死んだエントリを drop、
  新規ファイルは ``_get_cued_tokens`` の通常経路でトークン化 (キャッシュ書き戻し強制)
- version が一致しない → 全部捨てる (token 化ロジック変更時、または v1 → v2 移行)

シンプル化:
- atomic write 不要 (検索処理が並列に走る想定でない)
- 破損時は捨てて再構築 (try/except + 警告 stderr)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


CACHE_FILENAME = ".kokkai-search-cache.json"
CACHE_VERSION = 2


def cache_path(out_dir: Path) -> Path:
    return out_dir / CACHE_FILENAME


def load_cache(out_dir: Path) -> tuple[dict[str, dict], set[str]]:
    """キャッシュをロードして ``(files_dict, manifest_set)`` を返す。

    破損 / version 不一致 / 不在 はすべて ``({}, set())`` を返す (= 再計算が必要)。
    """
    p = cache_path(out_dir)
    if not p.exists():
        return {}, set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[search-cache] WARN: 破損したキャッシュを破棄: {e}", file=sys.stderr)
        return {}, set()
    if data.get("version") != CACHE_VERSION:
        print(
            f"[search-cache] WARN: version 不一致 ({data.get('version')} != {CACHE_VERSION})、"
            f"再構築します",
            file=sys.stderr,
        )
        return {}, set()
    files = data.get("files") or {}
    manifest = set(data.get("manifest") or [])
    return files, manifest


def save_cache(out_dir: Path, files: dict[str, dict]) -> None:
    """キャッシュを保存。manifest は ``files`` の key 集合からそのまま導出。"""
    p = cache_path(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "manifest": sorted(files.keys()),
        "files": files,
    }
    p.write_text(
        json.dumps(payload, ensure_ascii=False, indent=None, separators=(",", ":")),
        encoding="utf-8",
    )


def is_fresh(cache_entry: dict | None, source_path: Path) -> bool:
    """cache_entry のソースファイル mtime と現在の mtime が一致するかチェック。"""
    if cache_entry is None:
        return False
    if not source_path.exists():
        return False
    try:
        return abs(cache_entry.get("mtime", -1) - source_path.stat().st_mtime) < 0.001
    except OSError:
        return False


def make_entry(source_path: Path, cues_with_tokens: list[dict]) -> dict:
    """1 ファイル分のキャッシュエントリを組み立てる。"""
    try:
        mtime = source_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return {"mtime": mtime, "cues": cues_with_tokens}


def clear_cache(out_dir: Path) -> bool:
    """キャッシュファイルを削除。存在しなければ False。"""
    p = cache_path(out_dir)
    if p.exists():
        p.unlink()
        return True
    return False
