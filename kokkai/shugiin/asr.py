"""
wav 文字起こし (Whisper / VibeVoice) + 公式発言者リストへの cue 割付。

サポート backend:
- ``faster`` (既定): ``faster-whisper`` の Whisper turbo。CPU/Metal/CUDA portable。
- ``mlx``: ``mlx-whisper`` を subprocess で起動。Apple Silicon でより速い。
- ``vibevoice``: Microsoft VibeVoice-ASR (HF transformers >= 5.3)。7B クラスの
  長時間 ASR で、60 分までの音声を 1 パスで転写する。CUDA GPU 推奨。

``initial_prompt`` は ``kokkai.shugiin.hints.build_initial_prompt`` で構築する。
vibevoice backend では initial_prompt を context prompt (hotwords 相当) として渡す。

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


ASR_BACKENDS = ("faster", "mlx", "vibevoice")
# ライブラリ層の portable default。CLI 既定は recommended_asr_backend() で
# Apple Silicon なら "mlx" に切り替わる。
DEFAULT_ASR_BACKEND = "faster"
DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_VIBEVOICE_MODEL = "microsoft/VibeVoice-ASR-HF"

# VibeVoice-ASR の入力仕様: 24kHz mono、最長 60 分 / 64K トークン。
# 国会の会議は数時間になるので、上限に余裕を見た chunk に分割して順に流す。
VIBEVOICE_SAMPLE_RATE = 24000
VIBEVOICE_CHUNK_SEC = 3000  # 50 分

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
    "microsoft/VibeVoice-ASR-HF": "18 GB",
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
) -> list[dict]:
    """mlx-whisper を subprocess で実行し JSON segments を cue リストに変換する。"""
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


def _load_vibevoice():
    """transformers (VibeVoice-ASR 統合版) の遅延 import。"""
    from ..errors import MissingToolError
    try:
        import torch  # noqa: F401
        from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration
    except ImportError as e:
        raise MissingToolError(
            "vibevoice backend には transformers >= 5.3 と torch が必要です。\n"
            "  uv pip install -e \".[shugiin-asr-vibevoice]\"\n"
            "または\n"
            "  pip install -e \".[shugiin-asr-vibevoice]\"\n"
            "(transformers が古い場合は `pip install -U 'transformers>=5.3'` で更新)"
        ) from e
    return AutoProcessor, VibeVoiceAsrForConditionalGeneration


def _wav_duration_sec(wav_path: Path) -> float:
    """PCM wav の長さ (秒)。標準 ``wave`` モジュールのみ使用。"""
    import wave
    with wave.open(str(wav_path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _ffmpeg_run(cmd: list[str], what: str) -> None:
    from ..errors import AsrError
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AsrError(
            f"{what} に失敗しました (ffmpeg returncode={proc.returncode}):\n"
            f"  stderr: {proc.stderr.strip()[:500]}"
        )


def _prepare_vibevoice_wav(wav_path: Path) -> Path:
    """パイプラインの 16kHz wav を VibeVoice 用の 24kHz mono wav に変換する。

    ``<stem>_24k.wav`` として wav と同じディレクトリにキャッシュし、
    既に存在すれば再利用する (hls_to_wav の wav キャッシュと同じ流儀)。
    """
    from .audio import ensure_ffmpeg
    ensure_ffmpeg()
    out = wav_path.with_name(wav_path.stem + "_24k.wav")
    if out.exists():
        print(f"[asr] 24kHz wav キャッシュを使用: {out.name}", file=sys.stderr)
        return out
    print(f"[asr] {VIBEVOICE_SAMPLE_RATE}Hz mono に変換中 (VibeVoice 入力仕様)", file=sys.stderr)
    _ffmpeg_run(
        [
            "ffmpeg", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(wav_path),
            "-ac", "1", "-ar", str(VIBEVOICE_SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(out),
        ],
        "24kHz への resample",
    )
    return out


def _split_wav_chunks(wav_path: Path, chunk_sec: int = VIBEVOICE_CHUNK_SEC) -> list[Path]:
    """wav を chunk_sec 単位に分割して chunk パスのリストを返す。

    モデル上限 (60 分) 以内ならそのまま 1 要素で返す。分割は ffmpeg の
    segment muxer (`-c copy`) で行うので PCM の再エンコードは走らない。
    chunk 境界で文が切れることは許容する (Whisper backend の VAD chunk と
    同様、後段の merge_cues_into_sentences で吸収される)。
    """
    from ..errors import AsrError
    if _wav_duration_sec(wav_path) <= chunk_sec:
        return [wav_path]
    chunk_dir = wav_path.with_name(wav_path.stem + "_chunks")
    chunk_dir.mkdir(exist_ok=True)
    pattern = chunk_dir / "chunk_%04d.wav"
    print(f"[asr] {chunk_sec // 60} 分単位に分割中 (VibeVoice の 60 分上限対応)", file=sys.stderr)
    _ffmpeg_run(
        [
            "ffmpeg", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(wav_path),
            "-f", "segment", "-segment_time", str(chunk_sec),
            "-c", "copy",
            str(pattern),
        ],
        "wav の chunk 分割",
    )
    chunks = sorted(chunk_dir.glob("chunk_*.wav"))
    if not chunks:
        raise AsrError(f"wav の chunk 分割結果が空です: {chunk_dir}")
    return chunks


def _vibevoice_segments_to_cues(segments: list, offset: float = 0.0) -> list[dict]:
    """VibeVoice の parsed segment を cue リストに変換する。

    parsed 形式は ``{"Start": float, "End": float, "Speaker": ..., "Content": str}``。
    Speaker (モデルの話者分離結果) はここでは捨てる — 発言者の帰属は公式
    発言者リスト (assign_cues_to_speakers) が一次ソースなので二重管理しない。
    """
    cues: list[dict] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("Content") or "").strip()
        if not text:
            continue
        try:
            start = float(seg.get("Start", 0.0))
        except (TypeError, ValueError):
            continue
        try:
            end = float(seg.get("End", start))
        except (TypeError, ValueError):
            end = start
        cues.append({"start": start + offset, "end": end + offset, "text": text})
    cues.sort(key=lambda c: c["start"])
    return cues


def _transcribe_vibevoice(
    wav_path: Path,
    initial_prompt: str | None,
    model_id: str,
    language: str,
) -> list[dict]:
    """VibeVoice-ASR (HF transformers) で文字起こしして cue リストを返す。

    - 言語は自動判定 (50+ 言語対応) なので ``language`` は使わない。
    - ``initial_prompt`` は context prompt (hotwords 相当) として渡す。
    - 60 分超の音声は VIBEVOICE_CHUNK_SEC 単位の chunk に分割し、
      chunk ごとのタイムスタンプに累積 offset を足して連結する。
    """
    from ..errors import AsrError
    from ._models import ensure_model_downloaded

    AutoProcessor, VibeVoiceAsrForConditionalGeneration = _load_vibevoice()
    ensure_model_downloaded(
        model_id,
        label="asr",
        size_hint=_KNOWN_MODEL_SIZES.get(model_id),
    )

    wav24k = _prepare_vibevoice_wav(wav_path)
    chunks = _split_wav_chunks(wav24k)

    print(f"[asr] VibeVoice モデルロード中: {model_id}", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(model_id)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        model_id, dtype="auto", device_map="auto",
    )

    print(
        f"[asr] 文字起こし開始 (vibevoice): {wav_path.name}"
        f" / {len(chunks)} chunk"
        f"{' (hint あり)' if initial_prompt else ' (hint なし)'}",
        file=sys.stderr,
    )

    cues: list[dict] = []
    offset = 0.0
    for i, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            print(f"[asr]   chunk {i}/{len(chunks)} ({int(offset) // 60} 分〜)", file=sys.stderr)
        kwargs = {"prompt": initial_prompt} if initial_prompt else {}
        inputs = processor.apply_transcription_request(
            audio=str(chunk), **kwargs,
        ).to(model.device, model.dtype)
        output_ids = model.generate(**inputs, max_new_tokens=32768, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        decoded = processor.decode(generated_ids, return_format="parsed")
        # decode は batch ごとの list を返す (batch=1 なので先頭)。古い/将来の
        # transformers で segment list が直接返るケースも許容する。
        segments = decoded[0] if decoded and isinstance(decoded[0], list) else decoded
        if not isinstance(segments, list):
            raise AsrError(
                f"VibeVoice の parsed 出力を解釈できません: {type(segments).__name__}"
            )
        cues.extend(_vibevoice_segments_to_cues(segments, offset=offset))
        offset += _wav_duration_sec(chunk)

    print(f"[asr] 完了 (vibevoice): cue数={len(cues)}", file=sys.stderr)
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
) -> list[dict]:
    """
    wav を文字起こしして cue リストを返す。

    Args:
        wav_path: 16kHz mono wav (audio.hls_to_wav の出力)
        initial_prompt: bias 用プロンプト (None なら hint 無し)。
                        vibevoice では context prompt (hotwords 相当) として渡る
        backend: ``faster`` (CPU/Metal/CUDA portable) / ``mlx`` (Apple Silicon ネイティブ)
                 / ``vibevoice`` (VibeVoice-ASR、CUDA GPU 推奨)
        model_size: モデル名。faster: ``turbo``/``large-v3``/``medium`` 等、
                    mlx: ``mlx-community/whisper-large-v3-turbo`` 等、
                    vibevoice: ``microsoft/VibeVoice-ASR-HF`` (既定)。
                    ``org/name`` を含まない値は各 backend の既定モデルになる
        language: "ja" 固定推奨 (vibevoice は自動判定のため未使用)
        compute_type, device, beam_size: faster backend 専用

    Returns:
        ``[{"start": float, "end": float, "text": str}, ...]`` (時刻昇順)
    """
    from ..errors import AsrError

    if backend not in ASR_BACKENDS:
        raise AsrError(f"未対応の ASR backend: {backend} (使えるのは {ASR_BACKENDS})")

    if backend == "mlx":
        model_id = model_size if "/" in model_size else DEFAULT_MLX_MODEL
        return _transcribe_mlx(wav_path, initial_prompt, model_id, language)

    if backend == "vibevoice":
        model_id = model_size if "/" in model_size else DEFAULT_VIBEVOICE_MODEL
        return _transcribe_vibevoice(wav_path, initial_prompt, model_id, language)

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
