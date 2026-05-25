"""
Whisper による wav 文字起こし + 公式発言者リストへの cue 割付。

サポート backend:
- ``faster`` (既定): ``faster-whisper`` の Whisper turbo。CPU/Metal/CUDA portable。
- ``mlx``: ``mlx-whisper`` を subprocess で起動。Apple Silicon でより速い。

``initial_prompt`` は ``kokkai.shugiin.hints.build_initial_prompt`` で構築する。

公開関数:
- ``transcribe(wav, initial_prompt=None, backend="faster", model_size="turbo", ...) -> list[dict]``
- ``assign_cues_to_speakers(cues, speakers) -> list[dict]``
"""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ASR_BACKENDS = ("faster", "mlx")
# ライブラリ層の portable default。CLI 既定は recommended_asr_backend() で
# Apple Silicon なら "mlx" に切り替わる。
DEFAULT_ASR_BACKEND = "faster"
DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"

# faster-whisper の model_size → HF repo_id (cache 存在確認に使う)。
# faster_whisper.utils._MODELS と同じ内容のフリーズコピー (private API への依存を
# 避ける + faster-whisper 未インストールでも CLI 補助が動くように)。
_FASTER_MODEL_REPOS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

# 既知モデルの目安サイズ (Y/N プロンプトで表示)。
_KNOWN_MODEL_SIZES = {
    "mlx-community/whisper-large-v3-turbo": "1.5 GB",
    "mobiuslabsgmbh/faster-whisper-large-v3-turbo": "1.5 GB",
    "Systran/faster-whisper-large-v3": "3 GB",
    "Systran/faster-whisper-medium": "1.5 GB",
    "Systran/faster-whisper-small": "500 MB",
    "Systran/faster-whisper-base": "150 MB",
    "Systran/faster-whisper-tiny": "75 MB",
}


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _mlx_whisper_available() -> bool:
    """mlx-whisper を起動できるか (PATH / importable metadata / uvx / uv tool)。

    NOTE: 実 ``import mlx_whisper`` は意図的に呼ばない。mlx_whisper 経由で
    llvmlite.dylib (116MB) の初回 ``dlopen`` が走ると Norton 等の AV に
    数分ブロックされるため、``find_spec`` で「import 可能か」だけ覗いて
    モジュール本体はロードしない。
    """
    if shutil.which("mlx_whisper"):
        return True
    try:
        if importlib.util.find_spec("mlx_whisper") is not None:
            return True
    except (ImportError, ValueError):
        pass
    # uvx / uv tool 経由で取得可能なら mlx-whisper を使える (要 Apple Silicon)
    return bool(shutil.which("uvx") or shutil.which("uv"))


def recommended_asr_backend() -> str:
    """CLI から使う既定 backend。Apple Silicon + mlx_whisper 可なら ``"mlx"``。"""
    if _is_apple_silicon() and _mlx_whisper_available():
        return "mlx"
    return DEFAULT_ASR_BACKEND


def _load_faster_whisper():
    """faster-whisper の遅延 import。`[shugiin-asr]` extras 未インストールでも
    Phase 2 の機能 (字幕無しタイムライン) は動くようにするための分離。"""
    from ..errors import MissingToolError
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise MissingToolError(
            "faster-whisper がインストールされていません。\n"
            "  uv pip install -e \".[shugiin-asr]\"\n"
            "または\n"
            "  pip install -e \".[shugiin-asr]\""
        ) from e
    return WhisperModel


def _mlx_whisper_runner() -> list[str]:
    """``mlx_whisper`` CLI を起動するためのコマンド列を返す。"""
    from ..errors import MissingToolError
    if shutil.which("mlx_whisper"):
        return ["mlx_whisper"]
    if shutil.which("uvx"):
        return ["uvx", "--from", "mlx-whisper", "mlx_whisper"]
    if shutil.which("uv"):
        return ["uv", "tool", "run", "--from", "mlx-whisper", "mlx_whisper"]
    raise MissingToolError(
        "mlx_whisper を起動できません。"
        "`pip install mlx-whisper` または `uv tool install mlx-whisper` を実行してください。"
    )


def _transcribe_mlx(
    wav_path: Path,
    initial_prompt: str | None,
    model_id: str,
    language: str,
    *,
    clip_start: float = 0.0,
    clip_end: float | None = None,
) -> list[dict]:
    """mlx-whisper を subprocess で実行し JSON segments を cue リストに変換する。

    ``clip_start`` / ``clip_end`` を指定すると mlx-whisper の ``--clip-timestamps``
    を介して処理対象を時刻範囲に絞れる (冒頭/末尾の長尺無音をスキップする用途)。
    cue の start/end は元 wav 上の絶対時刻なのでオフセット調整は不要。
    """
    from ..errors import AsrError
    from ._models import ensure_model_downloaded

    # mlx-whisper の DL は subprocess の内部で走るので、対話 TTY なら subprocess
    # 起動前に Y/N を聞く。
    ensure_model_downloaded(
        model_id,
        label="asr",
        size_hint=_KNOWN_MODEL_SIZES.get(model_id),
    )

    out_dir = wav_path.parent
    out_name = wav_path.stem + "_mlx"
    cmd = _mlx_whisper_runner() + [
        "--model", model_id,
        "--task", "transcribe",
        "--output-format", "json",
        "--output-dir", str(out_dir),
        "--output-name", out_name,
        # cascade hallucination 抑制 (gov-online と同じ理由)
        "--condition-on-previous-text", "False",
        # 圧縮率が高い (= 「ご視聴ありがとうございました」等の繰り返し) は
        # 復号失敗扱いにして fallback サンプリングへ。faster backend と同じ値。
        "--compression-ratio-threshold", "2.0",
    ]
    if initial_prompt:
        cmd += ["--initial-prompt", initial_prompt]
    if language and language.lower() not in ("auto", "none", ""):
        cmd += ["--language", language]
    # 冒頭/末尾の無音スキップ。clip_end=None なら end-of-audio まで。
    if clip_start > 0.0 or clip_end is not None:
        clip_value = f"{clip_start:.2f}"
        if clip_end is not None:
            clip_value += f",{clip_end:.2f}"
        cmd += ["--clip-timestamps", clip_value]
    cmd.append(str(wav_path))

    print(f"[asr] mlx_whisper 起動: {model_id}", file=sys.stderr)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise AsrError(f"mlx_whisper 実行失敗 (exit {e.returncode})") from e

    json_path = out_dir / f"{out_name}.json"
    if not json_path.exists():
        raise AsrError(f"mlx_whisper が JSON を出力しませんでした: {json_path}")
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])
    cues = [
        {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
        for s in segments
    ]
    print(f"[asr] 完了 (mlx): cue数={len(cues)}", file=sys.stderr)
    return cues


def transcribe(
    wav_path: Path,
    initial_prompt: str | None = None,
    *,
    backend: str = DEFAULT_ASR_BACKEND,
    model_size: str = "turbo",
    language: str = "ja",
    compute_type: str = "auto",
    device: str = "auto",
    beam_size: int = 5,
    clip_start: float = 0.0,
    clip_end: float | None = None,
) -> list[dict]:
    """
    wav を Whisper で文字起こしして cue リストを返す。

    Args:
        wav_path: 16kHz mono wav (audio.hls_to_wav の出力)
        initial_prompt: bias 用プロンプト (None なら hint 無し)
        backend: ``faster`` (CPU/Metal/CUDA portable) / ``mlx`` (Apple Silicon ネイティブ)
        model_size: モデル名。faster: ``turbo``/``large-v3``/``medium`` 等、
                    mlx: ``mlx-community/whisper-large-v3-turbo`` 等
        language: "ja" 固定推奨 (自動言語判定は不要)
        compute_type, device, beam_size: faster backend 専用
        clip_start: 処理開始時刻 (秒、wav 先頭からのオフセット)。0.0 で先頭から。
        clip_end: 処理終了時刻 (秒)。None で wav 末尾まで。``clip_start`` と組み合わせて
            冒頭/末尾の長尺無音を Whisper の処理対象から外す (``audio.detect_edge_silence``
            と併用)。cue の start/end は元 wav 上の絶対時刻なので、後段で
            オフセット調整は不要。

    Returns:
        ``[{"start": float, "end": float, "text": str}, ...]`` (時刻昇順)
    """
    from ..errors import AsrError

    if backend not in ASR_BACKENDS:
        raise AsrError(f"未対応の ASR backend: {backend} (使えるのは {ASR_BACKENDS})")

    if backend == "mlx":
        model_id = model_size if "/" in model_size else DEFAULT_MLX_MODEL
        return _transcribe_mlx(
            wav_path, initial_prompt, model_id, language,
            clip_start=clip_start, clip_end=clip_end,
        )

    WhisperModel = _load_faster_whisper()
    # 初回 DL 前に Y/N (対話 TTY のみ)。model_size が "turbo" 等のショート名なら
    # 対応する HF repo_id を逆引き。"org/name" 直指定ならそのまま。
    from ._models import ensure_model_downloaded
    repo_id = model_size if "/" in model_size else _FASTER_MODEL_REPOS.get(model_size)
    if repo_id:
        ensure_model_downloaded(
            repo_id,
            label="asr",
            size_hint=_KNOWN_MODEL_SIZES.get(repo_id),
        )
    print(
        f"[asr] モデルロード中: {model_size} (compute={compute_type}, device={device})",
        file=sys.stderr,
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(
        f"[asr] 文字起こし開始: {wav_path.name}"
        f"{' (hint あり)' if initial_prompt else ' (hint なし)'}",
        file=sys.stderr,
    )

    # repetition hallucination 対策:
    # - condition_on_previous_text=False: 前 chunk の出力を次 chunk の prompt に
    #   使わない。代わりに毎 chunk で initial_prompt が再注入されるので、hint の
    #   bias は会議全体に維持される。「同じフレーズを延々と繰り返す」典型的な
    #   Whisper の故障モードを防ぐ。
    # - no_repeat_ngram_size=5: 同じ 5-gram を 1 chunk 内で繰り返さない。
    # - compression_ratio_threshold=2.0 (default 2.4 から引き締め): chunk 出力の
    #   gzip 圧縮率が高い (=繰り返しが多い) ものは hallucination とみなして
    #   再生成 or 破棄させる。
    # 冒頭/末尾の無音をスキップ (audio.detect_edge_silence で得た範囲を渡す)。
    # faster-whisper の clip_timestamps は "start,end" / "start" 形式の文字列。
    # cue start/end は元 wav 上の絶対時刻なので、後段の調整は不要。
    if clip_start > 0.0 or clip_end is not None:
        clip_timestamps = (
            f"{clip_start:.2f},{clip_end:.2f}" if clip_end is not None
            else f"{clip_start:.2f}"
        )
    else:
        clip_timestamps = "0"

    segments, info = model.transcribe(
        str(wav_path),
        language=language,
        initial_prompt=initial_prompt,
        beam_size=beam_size,
        vad_filter=True,                       # 無音区間を切って高速化
        word_timestamps=False,                 # 文単位の segment (UI 互換性)
        condition_on_previous_text=False,
        no_repeat_ngram_size=5,
        compression_ratio_threshold=2.0,
        clip_timestamps=clip_timestamps,
    )

    print(
        f"[asr]   検出言語={info.language} 確信度={info.language_probability:.2f}"
        f" 音声長={info.duration:.1f}s",
        file=sys.stderr,
    )

    cues: list[dict] = []
    last_logged = -10.0
    for seg in segments:
        cues.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
        })
        # 10 秒進むごとに進捗を出す (stderr が静かすぎると不安になるので)
        if seg.start - last_logged > 60:
            mins = int(seg.start // 60)
            print(f"[asr]   進捗 {mins} 分処理済", file=sys.stderr)
            last_logged = seg.start

    print(f"[asr] 完了: cue数={len(cues)}", file=sys.stderr)
    return cues


# Whisper はしばしば句点を入れず、5-10 秒の VAD chunk 単位で cue を返してくる
# (例: 「この法律案は高齢化の進展」「単身高齢者世帯の増加等の社会経済情勢の変化に鑑み」
# のように 1 文が複数 cue に分割される)。下記文末パターンが cue 末尾に現れたら
# その cue で文を確定し、それ以外は次の cue と連結することで読みやすい単位にまとめる。
import re as _re

_SENTENCE_END_RE = _re.compile(
    r"(?:"
    r"ます|です|ました|でした|あります|"
    r"おります|ております|います|ていました|"
    r"いたします|致します|思います|考えます|存じます|"
    r"ございます|である|であります|だ|である|"
    r"ましょう|でしょう|でしょうか|ますか|ですか|"
    r"とのこと|と聞いて|と思って|お願いいたします"
    r")"
    r"[。！？]?$"
)


def _normalize_jp_text(text: str) -> str:
    """日本語テキスト中の半角スペース (Whisper が時々挿入する) を除去。

    Whisper の日本語出力には ``民法等の 一部を改正`` のように半角空白が
    トークン境界として残ることがある。日本語的に不要なので除去する。
    """
    return _re.sub(r"[ \t]+", "", text)


# Whisper は日本語の句点 (。) をあまり挿入しないので、強い文末パターンの直後に
# 自動挿入する。「ます」単独だと「ますが」「ますと」など mid-sentence で false
# positive になるため、議会で頻出する確実な文末語に限定する。
_PERIOD_INSERT_RE = _re.compile(
    r"(であります|でございます|いたします|致します|"
    r"ました|でした|おります|ております|"
    r"思います|考えます|存じます|"
    r"いたしました|してまいります|"
    r"お願いいたします|お願いします)"
    # 直後が新しい文の頭らしいもの (漢字・カタカナ、または接続詞・副詞) なら挿入
    r"(?=[一-龥ァ-ヶー]|また|さらに|ただし|なお|しかし|"
    r"続いて|次に|以上|以下|ところで|そこで|"
    r"第[一二三四五六七八九十]+に|"
    r"第[一二三四五六七八九十]+条|"
    r"ほか[、。]?[一-龥ァ-ヶー])"
)


def _insert_periods(text: str) -> str:
    """強い文末パターン + 新文起点を検出して 。 を挿入。

    既に直後に 。 ！ ？ や 助詞 (が/の/を/に/で/と/か/も) がある場合は挿入しない。
    """
    return _PERIOD_INSERT_RE.sub(r"\1。", text)


# 文頭になりやすい接続詞・副詞の直後に 、 を挿入。「また」「さらに」のような
# 議会頻出語に限定 (mid-sentence false positive を避けるため、直前が 。 か
# 文頭の場合のみマッチさせる)。
_COMMA_INSERT_RE = _re.compile(
    r"(?:^|。)(また|さらに|ただし|なお|しかし|"
    r"続いて|次に|以上|以下|ところで|そこで|"
    r"第[一二三四五六七八九十]+に|"
    r"その上で|したがって|よって)"
    r"(?=[一-龥ァ-ヶー])"
)


def _insert_commas(text: str) -> str:
    """文頭の接続詞・副詞の直後に 、 を挿入。"""
    def repl(m: "_re.Match[str]") -> str:
        prefix = "。" if m.group(0).startswith("。") else ""
        return f"{prefix}{m.group(1)}、"
    return _COMMA_INSERT_RE.sub(repl, text)


def merge_cues_into_sentences(cues: list[dict]) -> list[dict]:
    """
    Whisper の cue を文単位 (まとまった発話) にマージする。

    cue 末尾が ``_SENTENCE_END_RE`` (「〜ます」「〜です」「〜あります」等) に
    マッチしたらそこで区切り、それ以外は次 cue と連結する。マージ後の start は
    最初 cue の start、end は最後 cue の end、text は連結 (半角スペースは除去)。

    Whisper は句点 (。) をあまり挿入しないため、語尾パターンマッチを採用。
    """
    if not cues:
        return []
    merged: list[dict] = [dict(cues[0])]
    for c in cues[1:]:
        prev = merged[-1]
        if _SENTENCE_END_RE.search(_normalize_jp_text(prev["text"])):
            merged.append(dict(c))
        else:
            prev["text"] += c["text"]
            prev["end"] = c["end"]
    # 最終 normalize: 連結後の text から半角空白除去 → 句点挿入 → 読点挿入
    for m in merged:
        t = _normalize_jp_text(m["text"])
        t = _insert_periods(t)
        t = _insert_commas(t)
        # 末尾に句点が無ければ補う (cue 全体が確実に文末で終わるように)
        if t and t[-1] not in "。！？":
            t = t + "。"
        m["text"] = t
    return merged


def assign_cues_to_speakers(
    cues: list[dict],
    speakers: list[dict],
) -> list[dict]:
    """
    cue を発言者の time 範囲で振り分ける。

    各 cue の start が ``speakers[i]["start"]`` 以上かつ ``speakers[i+1]["start"]``
    未満ならその発言者に帰属。最後の発言者は末尾まで全部。

    Returns:
        ``[{**speaker, "cues": [cue, ...]}, ...]`` (speakers と同じ順序)
    """
    if not speakers:
        return [{"start": 0.0, "name": "(発言者リスト無し)", "group": "", "cues": cues}]

    sorted_sp = sorted(speakers, key=lambda s: s["start"])
    groups: list[dict] = [
        {**sp, "cues": []} for sp in sorted_sp
    ]
    boundaries = [sp["start"] for sp in sorted_sp]

    for c in cues:
        # 発言者リストの最初の start より前の cue (冒頭挨拶等) は先頭発言者に寄せる
        if c["start"] < boundaries[0]:
            groups[0]["cues"].append(c)
            continue
        # 2 分探索でいいが、議員数は最大 ~20 名程度なので線形で十分
        idx = 0
        for i, b in enumerate(boundaries):
            if c["start"] >= b:
                idx = i
            else:
                break
        groups[idx]["cues"].append(c)

    return groups
