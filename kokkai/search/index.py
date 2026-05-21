"""
out/ ディレクトリから「会議 1 件 = meta.json + 字幕/cue ファイル」を読み込み、
cue 単位のフラットなレコードを生成する。

レコード形式:
    {
        "house": "sangiin" | "shugiin",
        "id": "8955" | "56246",
        "date": "2026年4月15日" | "2026-05-15",
        "title": "...委員会",
        "page_url": "...",
        "cue_idx": int,
        "start": float (秒),
        "end": float | None,
        "text": "発言テキスト",
        "speaker_name": str | None,    # cue.start が属する発言者
        "speaker_group": str | None,
        "meta_path": str,
    }

このモジュールはネット I/O 一切なし、純粋にファイルシステムから読む。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from . import cache as _cache
from .tokenize import tokenize


_TS_RE = re.compile(r"(\d+):(\d{2}):(\d{2})\.(\d{3})")


def _ts_to_sec(ts: str) -> float:
    m = _TS_RE.match(ts)
    if not m:
        return 0.0
    h, m_, s, ms = (int(x) for x in m.groups())
    return h * 3600 + m_ * 60 + s + ms / 1000


def _parse_vtt(text: str) -> list[dict]:
    """軽量 WebVTT パーサ (kokkai.sangiin.extract と同等だが import 循環を避けて再実装)。"""
    out: list[dict] = []
    for block in re.split(r"\n\n+", text):
        lines = [l for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        if lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION", "X-TIMESTAMP-MAP")):
            continue
        if "-->" not in lines[0]:
            if len(lines) < 2 or "-->" not in lines[1]:
                continue
            lines = lines[1:]
        m_ts = re.match(r"\s*(\S+)\s*-->\s*(\S+)", lines[0])
        if not m_ts:
            continue
        cue_text = " ".join(lines[1:]).strip()
        if not cue_text:
            continue
        out.append({
            "start": _ts_to_sec(m_ts.group(1)),
            "end": _ts_to_sec(m_ts.group(2)),
            "text": cue_text,
        })
    return out


def _read_cues_for_meta(meta_path: Path, meta: dict) -> list[dict]:
    """meta.json の files から cue 配列を取り出す (sangiin=VTT, shugiin=transcript.json)。"""
    files = meta.get("files") or {}
    parent = meta_path.parent

    # shugiin: transcript.json 優先
    transcript = files.get("transcript")
    if transcript:
        path = parent / transcript
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return []
            return [
                {"start": c.get("start", 0.0), "end": c.get("end"), "text": c.get("text", "")}
                for c in (data.get("cues") or [])
                if c.get("text")
            ]

    # sangiin: VTT
    vtt = files.get("vtt")
    if vtt:
        path = parent / vtt
        if path.exists():
            return _parse_vtt(path.read_text(encoding="utf-8"))

    return []


def _assign_speaker(start: float, speakers: list[dict]) -> tuple[str | None, str | None]:
    """cue.start から属する発言者を線形探索 (発言者数は最大数十なので十分速い)。"""
    if not speakers:
        return None, None
    sorted_sp = sorted(speakers, key=lambda s: s.get("start", 0.0))
    current = None
    for s in sorted_sp:
        if start >= s.get("start", 0.0):
            current = s
        else:
            break
    if current is None:
        current = sorted_sp[0]
    return current.get("name"), current.get("group")


def _get_cued_tokens(
    source_path: Path,
    cues_loader,  # () -> list[dict({start, end, text})]
    file_cache: dict[str, dict],
    use_cache: bool,
    processed_keys: set[str] | None = None,
) -> tuple[list[dict], bool]:
    """ソースファイルから cue 一覧を取り、SudachiPy で tokens を付与する。

    キャッシュが有効かつ source の mtime と一致していれば、tokenize をスキップ。
    ``processed_keys`` を渡すと、今回処理したソースファイル key (絶対パス)
    を集めて返す (manifest drift 検出に使う)。

    Returns:
        (cues_with_tokens, changed):
        cues_with_tokens: ``[{start, end, text, tokens: [...]}, ...]``
        changed: キャッシュを更新する必要があるなら True
    """
    key = str(source_path.resolve())
    if processed_keys is not None:
        processed_keys.add(key)
    cached = file_cache.get(key)
    if use_cache and _cache.is_fresh(cached, source_path):
        return cached["cues"], False
    cues = cues_loader()
    cues_with_tokens = [
        {**c, "tokens": tokenize(c["text"])} for c in cues
    ]
    file_cache[key] = _cache.make_entry(source_path, cues_with_tokens)
    return cues_with_tokens, True


def iter_records(out_dir: Path, *, use_cache: bool = True) -> list[dict]:
    """``out/*.meta.json`` を全部スキャンして cue 単位レコード一覧を返す。

    meta.json が無い古い vtt / transcript.json も fallback で取り込む
    (file 名から date / title を推測、speakers/jump_url は空)。

    SudachiPy 形態素解析の結果は ``out/.kokkai-search-cache.json`` にキャッシュ
    される (ソース mtime で自動無効化、`out/` のファイル集合変化も検知)。
    ``use_cache=False`` で無効化。
    """
    records: list[dict] = []
    if not out_dir.exists():
        return records

    if use_cache:
        file_cache, prev_manifest = _cache.load_cache(out_dir)
    else:
        file_cache, prev_manifest = {}, set()
    cache_changed = False
    processed_keys: set[str] = set()
    seen_bases: set[str] = set()

    # 1. meta.json 付き (本来の経路)
    for meta_path in sorted(out_dir.glob("*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[index] WARN: skip {meta_path.name}: {e}", file=sys.stderr)
            continue

        base = meta_path.name[: -len(".meta.json")]
        seen_bases.add(base)

        source_path = _cue_source_path(meta_path, meta)
        if source_path is None or not source_path.exists():
            continue

        cues_with_tokens, changed = _get_cued_tokens(
            source_path,
            lambda: _read_cues_for_meta(meta_path, meta),
            file_cache,
            use_cache,
            processed_keys,
        )
        if changed:
            cache_changed = True

        speakers = meta.get("speakers") or []
        for i, c in enumerate(cues_with_tokens):
            sp_name, sp_group = _assign_speaker(c["start"], speakers)
            records.append({
                "house": meta.get("house", ""),
                "id": meta.get("id", ""),
                "date": meta.get("date", ""),
                "title": meta.get("title", ""),
                "page_url": meta.get("page_url", ""),
                "cue_idx": i,
                "start": c["start"],
                "end": c.get("end"),
                "text": c["text"],
                "tokens": c.get("tokens") or [],
                "speaker_name": sp_name,
                "speaker_group": sp_group,
                "meta_path": str(meta_path),
            })

    # 2. meta.json 無し vtt / transcript.json fallback
    for vtt_path in sorted(out_dir.glob("*.vtt")):
        base = vtt_path.name[: -len(".vtt")]
        if base in seen_bases:
            continue
        seen_bases.add(base)
        date, title = _parse_base_filename(base)

        cues_with_tokens, changed = _get_cued_tokens(
            vtt_path,
            lambda: _parse_vtt(vtt_path.read_text(encoding="utf-8")),
            file_cache,
            use_cache,
            processed_keys,
        )
        if changed:
            cache_changed = True

        for i, c in enumerate(cues_with_tokens):
            records.append({
                "house": "sangiin",
                "id": "",
                "date": date,
                "title": title,
                "page_url": "",
                "cue_idx": i,
                "start": c["start"],
                "end": c.get("end"),
                "text": c["text"],
                "tokens": c.get("tokens") or [],
                "speaker_name": None,
                "speaker_group": None,
                "meta_path": str(vtt_path),
            })

    for tr_path in sorted(out_dir.glob("*_transcript.json")):
        base = tr_path.name[: -len("_transcript.json")]
        if base in seen_bases:
            continue
        seen_bases.add(base)
        date, title = _parse_base_filename(base)
        try:
            data = json.loads(tr_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        def _load_tr_cues():
            return [
                {"start": c.get("start", 0.0), "end": c.get("end"), "text": c.get("text", "")}
                for c in (data.get("cues") or []) if c.get("text")
            ]

        cues_with_tokens, changed = _get_cued_tokens(
            tr_path, _load_tr_cues, file_cache, use_cache, processed_keys,
        )
        if changed:
            cache_changed = True

        for i, c in enumerate(cues_with_tokens):
            records.append({
                "house": "shugiin",
                "id": data.get("deli_id", ""),
                "date": data.get("date") or date,
                "title": data.get("title") or title,
                "page_url": "",
                "cue_idx": i,
                "start": c["start"],
                "end": c.get("end"),
                "text": c["text"],
                "tokens": c.get("tokens") or [],
                "speaker_name": None,
                "speaker_group": None,
                "meta_path": str(tr_path),
            })

    if use_cache:
        # manifest drift 検出: 期待ファイル集合と現在の glob 結果が違うなら、
        # 死んだエントリを drop してキャッシュを保存し直す
        dead_keys = set(file_cache.keys()) - processed_keys
        if dead_keys:
            for k in dead_keys:
                file_cache.pop(k, None)
            cache_changed = True
        if processed_keys != prev_manifest:
            added = processed_keys - prev_manifest
            removed = prev_manifest - processed_keys
            if prev_manifest:  # 初回構築はうるさいので静かに
                print(
                    f"[search-cache] manifest drift: +{len(added)} added, "
                    f"-{len(removed)} removed (キャッシュ更新)",
                    file=sys.stderr,
                )
            cache_changed = True
        if cache_changed:
            _cache.save_cache(out_dir, file_cache)

    return records


def _cue_source_path(meta_path: Path, meta: dict) -> Path | None:
    """meta.json の files から cue ソースファイルパスを取り出す。"""
    files = meta.get("files") or {}
    parent = meta_path.parent
    transcript = files.get("transcript")
    if transcript:
        p = parent / transcript
        if p.exists():
            return p
    vtt = files.get("vtt")
    if vtt:
        p = parent / vtt
        if p.exists():
            return p
    return None


def _parse_base_filename(base: str) -> tuple[str, str]:
    r"""``2026-05-15_法務委員会_衆56245`` → ("2026-05-15", "法務委員会") のように分解。

    ``2026年4月15日_デジタル...`` (sangiin 旧形式: ID 無し) や、新形式
    ``2026年4月15日_..._参8955`` も拾う。形式は YYYY-MM-DD_TITLE または
    YYYY年M月D日_TITLE。タイトル末尾の ``_(衆|参|shugiin|sangiin|deli)\d+`` は剥がす
    (`shugiin`/`sangiin`/`deli` は後方互換: 旧 filename 形式)。
    """
    m = re.match(r"^(\d{4}-\d{2}-\d{2}|\d{4}年\d{1,2}月\d{1,2}日)_(.+)$", base)
    if not m:
        return "", base
    date, title = m.group(1), m.group(2)
    title = re.sub(r"_(?:衆|参|shugiin|sangiin|deli)\d+$", "", title)
    return date, title
