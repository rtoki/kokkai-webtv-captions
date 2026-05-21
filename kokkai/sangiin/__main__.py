"""CLI: ``sangiin <sid>`` で参議院 → 発言者別 HTML を生成."""

from __future__ import annotations

import argparse
import json as _json
import re
import subprocess
import sys
import urllib.error
from pathlib import Path

from ..errors import CliError, FetchError, InvalidInputError
from .extract import (
    fetch_vtt_segments,
    find_subtitle_playlist,
    http_get,
    resolve_from_sangiin_detail,
    vtt_to_text,
)
from .render import build_sections, render_html, safe_filename


SANGIIN_URL_TMPL = "https://www.webtv.sangiin.go.jp/webtv/detail.php?sid={}"

_QUIET = False


def info(msg: str) -> None:
    if not _QUIET:
        print(msg, file=sys.stderr)


def step(n: int, total: int, msg: str) -> None:
    info(f"[{n}/{total}] {msg}")


def normalize_target(target: str) -> str:
    """sid (数字) でも URL でも受けて detail.php URL を返す。"""
    if target.isdigit():
        return SANGIIN_URL_TMPL.format(target)
    if "detail.php" in target and re.search(r"sid=\d+", target):
        return target
    raise InvalidInputError(
        f"sid または detail.php URL を指定してください: {target}"
    )


def _resolve(target: str) -> tuple[str, str, dict]:
    """target → (detail URL, HLS URL, meta)。"""
    url = normalize_target(target)
    try:
        m3u8_url, meta = resolve_from_sangiin_detail(url)
    except urllib.error.URLError as e:
        raise FetchError(f"参議院ページ取得失敗: {e}") from e
    return url, m3u8_url, meta


def run(
    target: str,
    output_dir: Path,
    force_redownload: bool,
    skip_if_done: bool = False,
) -> tuple[Path, dict, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    step(1, 3, f"会議メタ解決中: target={target}")
    url, m3u8_url, meta = _resolve(target)
    sid = _extract_sid(url)
    base = safe_filename(
        f"{meta.get('date','')}_{meta.get('title','')}_参{sid}".strip("_")
        or "captions"
    )
    vtt_path = output_dir / f"{base}.vtt"
    txt_path = output_dir / f"{base}.txt"
    html_path = output_dir / f"{base}.html"
    meta_path = output_dir / f"{base}.meta.json"
    info(f"      会議: {meta['date']} / {meta['title']}")
    info(f"      発言者: {len(meta.get('speakers') or [])}名")

    # --skip-if-done: 既に meta.json + html が揃っていれば早期 return
    if skip_if_done and meta_path.exists() and html_path.exists():
        info(f"      ✓ 既に取り込み済み (skip): {html_path}")
        return html_path, meta, txt_path, vtt_path

    if vtt_path.exists() and not force_redownload:
        step(2, 3, f"WebVTT キャッシュを使用: {vtt_path.name}")
    else:
        step(2, 3, f"WebVTT を取得中: {m3u8_url}")
        try:
            master_body = http_get(m3u8_url)
            sub_url = find_subtitle_playlist(m3u8_url, master_body)
            if not sub_url:
                raise InvalidInputError(
                    "字幕トラックが見つかりません (2024年8月以降の中継で再試行してください)"
                )
            vtt = fetch_vtt_segments(sub_url)
        except urllib.error.URLError as e:
            raise FetchError(f"WebVTT 取得失敗: {e}") from e
        vtt_path.write_text(vtt, encoding="utf-8")
        text = vtt_to_text(vtt)
        txt_path.write_text(text, encoding="utf-8")
        info(f"      → {vtt_path.name} / {txt_path.name} ({len(text):,}字)")

    step(3, 3, "HTML を生成中 (SudachiPy 発言者検出)")
    data = build_sections(url, vtt_path, meta=meta)
    html = render_html(data)
    html_path.write_text(html, encoding="utf-8")
    info(f"      → {html_path}")

    # 検索 (kokkai.search) 用の meta.json を保存。VTT / TXT / HTML と並列に同名で。
    _write_meta_json(
        output_dir / f"{base}.meta.json",
        house="sangiin",
        target_id=sid,
        page_url=url,
        meta=meta,
        files={
            "vtt": vtt_path.name,
            "text": txt_path.name,
            "html": html_path.name,
        },
    )

    return html_path, meta, txt_path, vtt_path


def _extract_sid(url: str) -> str:
    m = re.search(r"sid=(\d+)", url)
    return m.group(1) if m else ""


def _write_meta_json(
    path: Path,
    *,
    house: str,
    target_id: str,
    page_url: str,
    meta: dict,
    files: dict,
) -> None:
    """検索インデックス用に会議 1 件分のメタを 1 ファイルにまとめる。

    参議院は公式 VTT 字幕 (人間がつけた台本) を使うので AI 校正は無関係。
    pipeline には `{"phase": "vtt"}` を入れて、レンダラがヘッダで「公式 VTT
    (AI 校正なし)」と明示できるようにする。
    """
    import time as _time
    payload = {
        "house": house,
        "id": target_id,
        "date": meta.get("date", ""),
        "title": meta.get("title", ""),
        "page_url": page_url,
        "speakers": meta.get("speakers") or [],
        "files": files,
        "pipeline": {"phase": "vtt"},
        "generated_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.write_text(
        _json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sangiin",
        description="参議院インターネット審議中継 → 発言者別 HTML 文字起こし",
    )
    parser.add_argument("target", nargs="+",
                        help="sid (例: 8955) または detail.php URL。複数指定可: '8955 9012 ...'")
    parser.add_argument("-o", "--output", default="out", help="出力ディレクトリ (既定: ./out)")
    parser.add_argument("--redownload", action="store_true",
                        help="VTT キャッシュを無視して再ダウンロード")
    parser.add_argument("--skip-if-done", action="store_true",
                        help="既に取り込み済み (out/{base}.meta.json + .html 揃っている) ならスキップ")
    parser.add_argument("--resolve-only", action="store_true",
                        help="ページ解決と meta 取得だけして終了 (agent 用 dry-run)")
    parser.add_argument("--open", dest="open_browser", action="store_true",
                        default=None, help="完了時にブラウザを開く")
    parser.add_argument("--no-open", dest="open_browser", action="store_false",
                        help="完了時にブラウザを開かない")
    json_group = parser.add_mutually_exclusive_group()
    json_group.add_argument(
        "--json", dest="emit_json", action="store_true",
        help="完了時 stdout に 1 行 JSON を出力 (compact: files + speakers + 識別情報)",
    )
    json_group.add_argument(
        "--json-full", dest="emit_json_full", action="store_true",
        help="--json と同じだが、caption_url / player_url 等の内部 meta も含めた full 形式",
    )
    json_group.add_argument(
        "--jsonl", dest="emit_jsonl", action="store_true",
        help="進捗・発言者・結果を 1 行ごとの JSONL ストリームで stdout に出力",
    )
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="進捗ログを stderr に出さない")
    return parser


def _emit_json(payload: dict) -> None:
    print(_json.dumps(payload, ensure_ascii=False), flush=True)


def _compact_meta(meta: dict) -> dict:
    """文字起こしツール向けの最小 meta: agent が必要な識別情報 + 出席者情報。

    speakers は配列で含める (agent が「誰が何時から発言」をプログラム的に
    使えるように)。caption_url / player_url 等の実装詳細は --json-full で。
    """
    out: dict = {k: meta[k] for k in ("date", "title", "deli_id") if meta.get(k)}
    if meta.get("speakers"):
        out["speakers"] = [
            {k: s[k] for k in ("start", "name", "group") if s.get(k) is not None}
            for s in meta["speakers"]
        ]
    if meta.get("agenda"):
        out["agenda"] = list(meta["agenda"])
    return out


def _is_emitting_json(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "emit_json", False)
        or getattr(args, "emit_json_full", False)
        or getattr(args, "emit_jsonl", False)
    )


def _emit_jsonl_speakers(meta: dict) -> None:
    """発言者を 1 行ずつ stdout に出す (jsonl ストリーミング)。"""
    for i, sp in enumerate(meta.get("speakers") or []):
        _emit_json({
            "type": "speaker",
            "i": i,
            "start": sp.get("start"),
            "name": sp.get("name"),
            "group": sp.get("group"),
        })


def _open_browser_if_appropriate(args: argparse.Namespace, html_path: Path) -> None:
    if args.open_browser is False:
        return
    if args.open_browser is None and not sys.stdout.isatty():
        return
    if sys.platform == "darwin":
        subprocess.run(["open", str(html_path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(html_path)], check=False)


def _process_one(target: str, args: argparse.Namespace, emitting: bool) -> dict:
    """1 件分の処理。成功時は dict ({"ok": True, ...}) を返す。失敗時は CliError を re-raise。"""
    if args.resolve_only:
        _, _, meta = _resolve(target)
        if emitting:
            if args.emit_jsonl:
                _emit_json({"type": "meta", **_compact_meta(meta)})
                _emit_jsonl_speakers(meta)
                _emit_json({"type": "result", "ok": True, "resolved": True})
            else:
                _emit_json({"ok": True, "resolved": True, "meta": meta})
        else:
            info(f"      会議: {meta['date']} / {meta['title']}")
            info(f"      発言者: {len(meta.get('speakers') or [])}名")
        return {"ok": True, "resolved": True, "target": target}

    html_path, meta, txt_path, vtt_path = run(
        target=target,
        output_dir=Path(args.output),
        force_redownload=args.redownload,
        skip_if_done=args.skip_if_done,
    )
    _open_browser_if_appropriate(args, html_path)
    info(f"\n✓ 完成: {html_path}")

    if emitting:
        files = {"html": str(html_path), "text": str(txt_path), "vtt": str(vtt_path)}
        if args.emit_jsonl:
            _emit_json({"type": "meta", **_compact_meta(meta)})
            _emit_jsonl_speakers(meta)
            _emit_json({"type": "result", "ok": True, "files": files})
        else:
            payload = {"ok": True, "files": files, **_compact_meta(meta)}
            if args.emit_json_full:
                payload["meta"] = meta
            _emit_json(payload)
    return {"ok": True, "target": target, "html": str(html_path)}


def main() -> None:
    global _QUIET
    parser = _build_parser()
    args = parser.parse_args()
    _QUIET = bool(args.quiet or _is_emitting_json(args))
    emitting = _is_emitting_json(args)

    targets: list[str] = args.target
    n = len(targets)
    if n > 1 and emitting and not args.emit_jsonl:
        parser.error("複数 target の JSON 出力は --jsonl を指定してください")
    results: list[dict] = []
    last_exit = 0

    for idx, t in enumerate(targets, 1):
        if n > 1:
            info(f"\n[{idx}/{n}] target={t}")
        try:
            r = _process_one(t, args, emitting)
            results.append(r)
        except KeyboardInterrupt:
            info("\n中断しました")
            if emitting:
                _emit_json({"ok": False, "target": t, "error": "KeyboardInterrupt", "code": 130})
            sys.exit(130)
        except CliError as e:
            info(f"ERROR ({t}): {type(e).__name__}: {e}")
            results.append({"ok": False, "target": t, "error": type(e).__name__, "message": str(e), "code": e.code})
            if emitting:
                _emit_json({"ok": False, "target": t, "error": type(e).__name__, "code": e.code, "message": str(e)})
            last_exit = e.code
            # 複数 target なら 1 件失敗しても続行
            if n == 1:
                sys.exit(e.code)
        except Exception as e:  # noqa: BLE001
            info(f"ERROR ({t}): 予期しない例外: {type(e).__name__}: {e}")
            results.append({"ok": False, "target": t, "error": type(e).__name__, "message": str(e), "code": 1})
            if emitting:
                _emit_json({"ok": False, "target": t, "error": type(e).__name__, "code": 1, "message": str(e)})
            last_exit = 1
            if n == 1:
                sys.exit(1)

    # 複数 target のサマリ
    if n > 1:
        ok = sum(1 for r in results if r.get("ok"))
        info(f"\n=== サマリ: {ok}/{n} 成功 ===")
    if last_exit:
        sys.exit(last_exit)


if __name__ == "__main__":
    main()
