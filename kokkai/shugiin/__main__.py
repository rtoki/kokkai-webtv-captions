"""CLI: ``shugiin <deli_id>`` で衆議院 → タイムライン HTML (Phase 2) / ASR 字幕付き HTML (Phase 3)."""

from __future__ import annotations

import argparse
import json as _json
import os
import re
import subprocess
import sys
import urllib.error
from pathlib import Path

from ..errors import (
    AsrError,
    CliError,
    FetchError,
    InvalidInputError,
    LlmError,
    MissingToolError,
)
from .extract import resolve_from_shugiin_detail
from .render import render_asr_html, render_timeline_html


# ----- 進捗ロガー (--quiet で抑制可能) -----

_QUIET = False


def step(n: int, total: int, msg: str) -> None:
    if not _QUIET:
        print(f"[{n}/{total}] {msg}", file=sys.stderr)


def info(msg: str) -> None:
    if not _QUIET:
        print(msg, file=sys.stderr)


def _safe_filename(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[\s　]+", "_", s.strip())
    s = re.sub(r'[\\/:*?"<>|]+', "", s)
    return s[:max_len] or "shugiin"


def _meta_summary(meta: dict) -> str:
    agenda_info = (
        f" / 議題 {len(meta['agenda'])}件" if meta.get("agenda") else ""
    )
    return (
        f"      会議: {meta['date']} / {meta['title']}\n"
        f"      発言者: {len(meta['speakers'])}名{agenda_info}"
    )


def _resolve(target: str) -> tuple[str, dict]:
    try:
        return resolve_from_shugiin_detail(target)
    except urllib.error.URLError as e:
        raise FetchError(f"shugiin ページ取得失敗: {e}") from e
    except ValueError as e:
        raise InvalidInputError(str(e)) from e


def _run_phase2(
    target: str, output_dir: Path, *, skip_if_done: bool = False
) -> tuple[Path, dict, dict]:
    """Phase 2: 字幕なしタイムライン HTML 生成"""
    output_dir.mkdir(parents=True, exist_ok=True)
    step(1, 2, f"会議メタを解決中: deli_id={target}")
    _, meta = _resolve(target)
    info(_meta_summary(meta))
    base = _safe_filename(
        f"{meta['date']}_{meta['title']}_衆{meta['deli_id']}".strip("_")
        or f"shugiin_{meta['deli_id']}"
    )
    html_path = output_dir / f"{base}.html"
    meta_path = output_dir / f"{base}.meta.json"

    if skip_if_done and meta_path.exists() and html_path.exists():
        info(f"      ✓ 既に取り込み済み (skip): {html_path}")
        return html_path, meta, {"html": str(html_path)}

    step(2, 2, "HTML を生成中 (Phase 2: 字幕なしタイムライン)")
    html = render_timeline_html(meta, pipeline={"phase": "phase2"})
    html_path.write_text(html, encoding="utf-8")
    info(f"      → {html_path}")

    _write_meta_json(
        output_dir / f"{base}.meta.json",
        meta=meta,
        files={"html": html_path.name},
        pipeline={"phase": "phase2"},
    )
    return html_path, meta, {"html": str(html_path)}


def _write_meta_json(
    path: Path, *, meta: dict, files: dict, pipeline: dict | None = None,
) -> None:
    """検索インデックス用に会議 1 件分のメタを 1 ファイルにまとめる。

    pipeline には ASR / 校正パイプラインの記録 (asr_backend, asr_model,
    llm_correct 等) を含める。レンダラがヘッダ表示に使う。

    speakers リストが空のときは ``pipeline["live"] = True`` を入れる。
    衆議院ページの発言者リストはライブ中継の途中だと未確定で空になることが
    あるため、後日 ``kkcap fetch --refresh-pending`` で発言者付き再取得を可能に。
    """
    import time as _time
    speakers = meta.get("speakers") or []
    pipeline = dict(pipeline) if pipeline else {"phase": "phase2"}
    if not speakers:
        pipeline["live"] = True
    payload = {
        "house": "shugiin",
        "id": meta.get("deli_id", ""),
        "date": meta.get("date", ""),
        "title": meta.get("title", ""),
        "page_url": meta.get("page_url", ""),
        "speakers": speakers,
        "agenda": meta.get("agenda") or [],
        "files": files,
        "pipeline": pipeline,
        "generated_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.write_text(
        _json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_phase3(args: argparse.Namespace) -> tuple[Path, dict, dict]:
    """Phase 3+4: ASR 字幕付き HTML 生成 (Whisper + hint + 校正)。

    `--skip-asr` を指定すると ffmpeg / Whisper をスキップし、過去の ASR で
    保存した ``<base>_transcript.json`` を再利用する (低スペック機での再整形・
    glossary 更新後の HTML 再生成、別マシン間での「強い機で ASR、弱い機で閲覧」
    のような用途向け)。

    Returns:
        (html_path, meta, stats)
    """
    from .asr import (
        assign_cues_to_speakers,
        merge_cues_into_sentences,
        transcribe,
    )
    from .audio import hls_to_wav
    from .glossary import apply_glossary, load_glossary, report as report_glossary
    from .hints import build_initial_prompt
    from .llm_context import extract_glossary_terms, merge_into_prompt
    from .llm_correct import correct_cues, drop_hallucinations, preclean_loops
    from .members import load_members

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    step(1, 5, f"会議メタを解決中: deli_id={args.target}")
    m3u8_url, meta = _resolve(args.target)
    info(f"{_meta_summary(meta)}\n      HLS: {m3u8_url}")
    base = _safe_filename(
        f"{meta['date']}_{meta['title']}_衆{meta['deli_id']}".strip("_")
        or f"shugiin_{meta['deli_id']}"
    )
    wav_path = output_dir / f"{base}.wav"
    html_path = output_dir / f"{base}_asr.html"
    transcript_path = output_dir / f"{base}_transcript.json"
    meta_path = output_dir / f"{base}.meta.json"

    # --skip-if-done: 既に ASR 結果 + HTML + meta.json が揃っていれば早期 return
    if getattr(args, "skip_if_done", False) and meta_path.exists() and html_path.exists() and transcript_path.exists():
        info(f"      ✓ 既に取り込み済み (skip): {html_path}")
        return html_path, meta, {"skip_if_done": True}, {
            "html": str(html_path),
            "transcript": str(transcript_path),
        }

    stats: dict = {
        "asr_backend": None if args.skip_asr else args.asr_backend,
        "llm_backend": args.llm_backend if (args.llm_context or args.llm_correct) else None,
        "skip_asr": bool(args.skip_asr),
    }

    if args.skip_asr:
        # ffmpeg / Whisper / hint 構築を全部スキップ → JSON から復元
        if not transcript_path.exists():
            raise InvalidInputError(
                f"--skip-asr が指定されたが transcript.json が見つかりません: "
                f"{transcript_path}\n"
                f"先に同じ deli_id で ASR を 1 回走らせて transcript.json を生成してください (ASR は既定で有効)。"
            )
        info(f"[asr] --skip-asr: transcript.json を再利用 ({transcript_path.name})")
        data = _json.loads(transcript_path.read_text(encoding="utf-8"))
        cues = data.get("cues", [])
        base_prompt = data.get("hint")
        info(
            f"      復元: cue {len(cues)}件 / hint "
            f"{'あり (' + str(len(base_prompt)) + '字)' if base_prompt else 'なし'}"
        )
        stats["restored_from"] = str(transcript_path)
    else:
        step(2, 5, "音声を wav に変換中 (ffmpeg)")
        hls_to_wav(m3u8_url, wav_path, force=args.redownload)

        if args.no_hint:
            base_prompt = None
            info("[hint] スキップ (--no-hint)")
        else:
            step(3, 5, "initial_prompt を構築中 (公式メタ + 議員名簿)")
            members = load_members(refresh=args.refresh_members)
            base_prompt = build_initial_prompt(meta, members)
            info(
                f"      hint ({len(base_prompt)}字): "
                f"{base_prompt[:80]}{'...' if len(base_prompt) > 80 else ''}"
            )

        # ASR (optionally 2-pass with --llm-context)
        if args.llm_context:
            step(4, 5, f"ASR pass1 (固有名詞抽出用): {args.asr_backend}/{args.model}")
            pass1 = transcribe(
                wav_path, initial_prompt=base_prompt,
                backend=args.asr_backend, model_size=args.model,
            )
            joined = " ".join(c["text"] for c in pass1)
            terms = extract_glossary_terms(
                joined,
                backend=args.llm_backend,
                model=args.llm_model,
            )
            info(f"      pass1 から抽出した固有名詞 {len(terms)} 語: {terms}")
            stats["llm_context_terms"] = terms
            augmented = merge_into_prompt(base_prompt, terms)
            info(f"      augmented prompt ({len(augmented)}字)")
            info("[asr] pass2 を実行中…")
            cues = transcribe(
                wav_path, initial_prompt=augmented,
                backend=args.asr_backend, model_size=args.model,
            )
        else:
            step(4, 5, f"ASR 実行中: {args.asr_backend}/{args.model}")
            cues = transcribe(
                wav_path, initial_prompt=base_prompt,
                backend=args.asr_backend, model_size=args.model,
            )

        # raw ASR cues を JSON で保存 → 次回 --skip-asr で再利用可能
        transcript_path.write_text(
            _json.dumps(
                {
                    "deli_id": meta["deli_id"],
                    "title": meta["title"],
                    "date": meta["date"],
                    "asr_backend": args.asr_backend,
                    "asr_model": args.model,
                    "llm_context": bool(args.llm_context),
                    "hint": base_prompt,
                    "cues": [
                        {"start": c["start"], "end": c["end"], "text": c["text"]}
                        for c in cues
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        info(f"      → {transcript_path.name} ({len(cues)} cue 保存)")

    # Whisper hallucination (YouTube 終了テロップ系) を cue ごと除外
    n_halluc = drop_hallucinations(cues)
    if n_halluc:
        info(f"[preclean] hallucination 句 {n_halluc} 件を除外 (無音/BGM 由来)")
    stats["preclean_hallucinations"] = n_halluc

    # Whisper degenerate loop の preclean
    n_loops = preclean_loops(cues)
    if n_loops:
        info(f"[preclean] degenerate loop {n_loops} 件を [音声不明瞭] に置換")
    stats["preclean_loops"] = n_loops

    # cue → 発言者振り分け → 各発言者内で文単位マージ
    groups = assign_cues_to_speakers(cues, meta["speakers"])
    total_after = 0
    for g in groups:
        g["cues"] = merge_cues_into_sentences(g["cues"])
        total_after += len(g["cues"])
    info(f"      cue {len(cues)}件 → 振り分け後 文単位マージで {total_after}件")

    # Phase 4: 静的 glossary 置換
    if not args.no_glossary:
        gloss = load_glossary(args.glossary, include_defaults=True)
        all_cues = [c for g in groups for c in g["cues"]]
        n = apply_glossary(all_cues, gloss)
        report_glossary(n, len(gloss))
        stats["glossary_changes"] = n

    # Phase 4: LLM 校正
    if args.llm_correct:
        all_cues = [c for g in groups for c in g["cues"]]
        n = correct_cues(
            all_cues,
            context_hint=base_prompt,
            backend=args.llm_backend,
            model=args.llm_model,
            base_url=args.llm_base_url,
        )
        stats["llm_correct_changes"] = n

    # 検索 (kokkai.search) 用 meta.json + ヘッダ表示用 pipeline summary を組み立てる
    pipeline = {
        "phase": "asr",
        "asr_backend": args.asr_backend,
        "asr_model": args.model,
        "hint": (not args.no_hint),
        "hint_chars": len(base_prompt) if base_prompt else 0,
        "llm_context": bool(args.llm_context),
        "llm_correct": bool(args.llm_correct),
        "llm_backend": args.llm_backend if (args.llm_context or args.llm_correct) else None,
        "llm_model": args.llm_model,
        "glossary": (not args.no_glossary),
        "glossary_changes": stats.get("glossary_changes", 0),
        "llm_correct_changes": stats.get("llm_correct_changes", 0),
        "llm_context_terms": stats.get("llm_context_terms", []),
        "preclean_hallucinations": stats.get("preclean_hallucinations", 0),
        "preclean_loops": stats.get("preclean_loops", 0),
        "cues": total_after,
        "skip_asr": bool(args.skip_asr),
    }

    step(5, 5, "HTML を生成中 (Phase 3: ASR 字幕付き)")
    html = render_asr_html(meta, groups, pipeline=pipeline)
    html_path.write_text(html, encoding="utf-8")
    info(f"      → {html_path}")
    stats.update({"cues": total_after, "wav": None if args.skip_asr else str(wav_path)})
    files = {
        "html": str(html_path),
        "transcript": str(transcript_path),
    }
    if not args.skip_asr and wav_path.exists():
        files["wav"] = str(wav_path)
    _write_meta_json(
        output_dir / f"{base}.meta.json",
        meta=meta,
        files={
            "html": html_path.name,
            "transcript": transcript_path.name,
        },
        pipeline=pipeline,
    )

    return html_path, meta, stats, files


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shugiin",
        description=(
            "衆議院インターネット審議中継 → 発言者別 HTML。"
            " デフォルトで Whisper による字幕付き HTML (Phase 3) を生成。"
            " --no-asr でタイムラインのみ (Phase 2)。"
        ),
    )
    parser.add_argument("target", nargs="+",
                        help="deli_id (例: 56246) または視聴ページ URL。複数指定可: '56246 56245 ...'")

    g_io = parser.add_argument_group("入力 / 出力")
    g_io.add_argument("-o", "--output", default="out", help="出力ディレクトリ (既定: ./out)")
    g_io.add_argument("--redownload", action="store_true", help="wav キャッシュを無視して再変換")
    g_io.add_argument("--skip-if-done", action="store_true",
                      help="既に取り込み済み (out/{base}.meta.json + .html / _asr.html 揃っている) ならスキップ")
    g_io.add_argument("--resolve-only", action="store_true",
                      help="ページ解決と meta 取得だけして終了 (agent 用 dry-run)")
    g_io.add_argument("--open", dest="open_browser", action="store_true",
                      default=None, help="完了時にブラウザを開く")
    g_io.add_argument("--no-open", dest="open_browser", action="store_false",
                      help="完了時にブラウザを開かない")
    g_json = g_io.add_mutually_exclusive_group()
    g_json.add_argument("--json", dest="emit_json", action="store_true",
                        help="完了時 stdout に 1 行 JSON (compact: files + speakers + agenda + 識別情報)")
    g_json.add_argument("--json-full", dest="emit_json_full", action="store_true",
                        help="--json と同じだが、HLS URL 等の内部 meta も含めた full 形式")
    g_json.add_argument("--jsonl", dest="emit_jsonl", action="store_true",
                        help="meta / 各 speaker / 結果を 1 行ごとの JSONL ストリームで出力")
    g_io.add_argument("-q", "--quiet", action="store_true",
                      help="進捗ログ (`[N/M]` 等) を stderr に出さない")

    g_asr = parser.add_argument_group("ASR (Whisper)")
    g_asr.add_argument("--asr", dest="asr", action="store_true", default=True,
                       help="ASR を実行して字幕付き HTML を生成 (既定で有効、要 [shugiin-asr] + ffmpeg)")
    g_asr.add_argument("--no-asr", dest="asr", action="store_false",
                       help="ASR を実行せず Phase 2 のタイムラインのみ生成")
    g_asr.add_argument("--asr-backend", choices=["faster", "mlx"], default=None,
                       help=("ASR backend (省略時は Apple Silicon + mlx_whisper 利用可なら "
                             "mlx、それ以外は faster を自動選択)"))
    g_asr.add_argument("--model", default="turbo",
                       help=("Whisper モデル (faster: turbo/large-v3/medium/small/base/tiny..., "
                             "mlx: HF model id)。tiny/base は CPU 機・低スペック向け"))
    g_asr.add_argument("--skip-asr", action="store_true",
                       help=("ASR を実行せず <base>_transcript.json (前回 ASR 結果) を再利用。"
                             " 強い機で 1 回 ASR → 弱い機で HTML 再生成、glossary 更新後の再 render 等"))
    g_asr.add_argument("--no-hint", action="store_true",
                       help="initial_prompt 注入を無効化 (精度比較・デバッグ用)")
    g_asr.add_argument("--refresh-members", action="store_true",
                       help="衆議院議員名簿キャッシュを無視して再取得")

    g_corr = parser.add_argument_group("校正 (--asr 時のみ)")
    g_corr.add_argument("--no-glossary", action="store_true",
                        help="デフォルトの議会用語 glossary 静的置換を無効化")
    g_corr.add_argument("--glossary", type=Path, default=None,
                        help="追加 glossary ファイル (1行に 「誤 → 正」)")
    g_corr.add_argument("--llm-context", action="store_true",
                        help="Whisper 2-pass: pass1 から LLM で固有名詞抽出 → pass2 prompt に注入")
    g_corr.add_argument("--llm-correct", action="store_true",
                        help="LLM で context-aware に cue を校正")
    g_corr.add_argument("--llm-backend", choices=["mlx", "openai"],
                        default="mlx",
                        help="LLM backend (mlx: ローカル Outlines, openai: OpenAI 互換 HTTP)")
    g_corr.add_argument("--llm-model", default=None,
                        help="LLM モデル ID (省略時 backend 既定)")
    g_corr.add_argument("--llm-base-url", default="http://localhost:8000/v1",
                        help="LLM サーバ endpoint (openai backend のみ)")
    return parser


def _emit_json(payload: dict) -> None:
    print(_json.dumps(payload, ensure_ascii=False), flush=True)


def _compact_meta(meta: dict) -> dict:
    """文字起こしツール向けの最小 meta: 識別情報 + 出席者情報 + 議題。

    speakers / agenda は配列で含める (agent が「誰がいつ何の議題で発言」を
    プログラム的に使えるように)。HLS URL 等の実装詳細は --json-full で。
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
    # 既定: TTY なら開く、非対話 (--json 経由など) なら開かない
    if args.open_browser is None and not sys.stdout.isatty():
        return
    if sys.platform == "darwin":
        subprocess.run(["open", str(html_path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(html_path)], check=False)


def _process_one(target: str, args: argparse.Namespace, emitting: bool) -> dict:
    """1 件分の処理 (args.target を一時的にこの target に差し替えて内部関数を呼ぶ)。"""
    saved_target = args.target
    args.target = target
    try:
        if args.resolve_only:
            _, meta = _resolve(target)
            if emitting:
                if args.emit_jsonl:
                    _emit_json({"type": "meta", **_compact_meta(meta)})
                    _emit_jsonl_speakers(meta)
                    _emit_json({"type": "result", "ok": True, "resolved": True})
                else:
                    _emit_json({"ok": True, "resolved": True, "meta": meta})
            else:
                info(_meta_summary(meta))
            return {"ok": True, "resolved": True, "target": target}

        if args.asr or args.skip_asr:
            html_path, meta, stats, files = _run_phase3(args)
        else:
            html_path, meta, files = _run_phase2(
                target, Path(args.output), skip_if_done=args.skip_if_done,
            )
            stats = {}
    finally:
        args.target = saved_target

    _open_browser_if_appropriate(args, html_path)
    info(f"\n✓ 完成: {html_path}")

    if emitting:
        if args.emit_jsonl:
            _emit_json({"type": "meta", **_compact_meta(meta)})
            _emit_jsonl_speakers(meta)
            _emit_json({"type": "result", "ok": True, "files": files, "stats": stats})
        else:
            payload = {"ok": True, "files": files, "stats": stats, **_compact_meta(meta)}
            if args.emit_json_full:
                payload["meta"] = meta
            _emit_json(payload)
    return {"ok": True, "target": target, "html": str(html_path)}


def main() -> None:
    global _QUIET
    parser = _build_parser()
    args = parser.parse_args()
    emitting = _is_emitting_json(args)
    if args.llm_context and args.llm_backend != "mlx":
        parser.error("--llm-context は --llm-backend mlx のみ対応しています")
    _QUIET = bool(args.quiet or emitting)

    # --asr-backend 未指定なら Apple Silicon かどうかで自動選択
    if args.asr_backend is None:
        from .asr import recommended_asr_backend
        args.asr_backend = recommended_asr_backend()
        if not _QUIET:
            info(f"[asr] backend 自動選択: {args.asr_backend}")

    targets: list[str] = args.target
    n = len(targets)
    if n > 1 and emitting and not args.emit_jsonl:
        parser.error("複数 target の JSON 出力は --jsonl を指定してください")
    last_exit = 0

    for idx, t in enumerate(targets, 1):
        if n > 1:
            info(f"\n[{idx}/{n}] target={t}")
        try:
            _process_one(t, args, emitting)
        except KeyboardInterrupt:
            info("\n中断しました")
            if emitting:
                _emit_json({"ok": False, "target": t, "error": "KeyboardInterrupt", "code": 130})
            sys.exit(130)
        except CliError as e:
            info(f"ERROR ({t}): {type(e).__name__}: {e}")
            if emitting:
                _emit_json({"ok": False, "target": t, "error": type(e).__name__, "code": e.code, "message": str(e)})
            last_exit = e.code
            if n == 1:
                sys.exit(e.code)
        except Exception as e:  # noqa: BLE001
            info(f"ERROR ({t}): 予期しない例外: {type(e).__name__}: {e}")
            if emitting:
                _emit_json({"ok": False, "target": t, "error": type(e).__name__, "code": 1, "message": str(e)})
            last_exit = 1
            if n == 1:
                sys.exit(1)

    if n > 1:
        info(f"\n=== サマリ: {n} 件処理完了 ===")
    if last_exit:
        sys.exit(last_exit)


if __name__ == "__main__":
    main()
