"""
HLS ストリームを ffmpeg で 16kHz mono PCM wav に変換するモジュール。

faster-whisper は内部で同じことを ffmpeg で行うが、一度ローカルに wav を
作り置きしておくと:
- 同じ会議で再 ASR (モデル変更や hint 調整) する時にダウンロードが不要
- メモリでなくファイル経由で渡せるので chunk 単位処理がしやすい
- ffmpeg の失敗が ASR 失敗と分離できる

必要なシステム依存: ``ffmpeg`` バイナリ (``brew install ffmpeg`` / ``apt install ffmpeg``)
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


def ensure_ffmpeg() -> None:
    """ffmpeg バイナリが PATH に存在することを確認。無ければ判りやすく abort。"""
    from ..errors import MissingToolError
    if shutil.which("ffmpeg") is None:
        raise MissingToolError(
            "ffmpeg が見つかりません。\n"
            "  macOS:  brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg"
        )


def hls_to_wav(
    hls_url: str,
    out_path: Path,
    force: bool = False,
    sample_rate: int = 16000,
) -> Path:
    """
    HLS m3u8 URL を 16kHz mono PCM wav にダウンロード&変換する。

    Args:
        hls_url: HLS マスター m3u8 URL
        out_path: 出力 wav パス
        force: True なら既存ファイルを上書き
        sample_rate: 出力サンプリングレート (Whisper の標準は 16000)

    Returns:
        書き出した wav パス (= out_path)
    """
    ensure_ffmpeg()
    if out_path.exists() and not force:
        print(
            f"[audio] wav キャッシュを使用: {out_path.name}"
            f" ({out_path.stat().st_size / 1024 / 1024:.1f} MB)",
            file=sys.stderr,
        )
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[audio] HLS → wav 変換中 ({sample_rate}Hz mono)", file=sys.stderr)

    # -loglevel error: 進捗バー等を抑制 (大文字ログのみ)
    # -nostdin: ffmpeg が stdin を読みに行かないように
    # -y: 上書き許可 (force チェックは事前に済ませているので OK)
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", hls_url,
        "-vn",                  # 映像捨てる
        "-ac", "1",             # mono
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    from ..errors import FetchError
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # 失敗時は中途半端な wav を残さない
        if out_path.exists():
            out_path.unlink()
        raise FetchError(
            f"ffmpeg に失敗しました (returncode={proc.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {proc.stderr.strip()[:500]}"
        )

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[audio] 完了: {out_path.name} ({size_mb:.1f} MB)", file=sys.stderr)
    return out_path


# ffmpeg silencedetect の出力パース用 (stderr に書かれる):
#   [silencedetect @ 0x...] silence_start: 0
#   [silencedetect @ 0x...] silence_end: 1200.5 | silence_duration: 1200.5
_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)")
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):([\d.]+)")


def detect_edge_silence(
    wav_path: Path,
    *,
    threshold_db: float = -45.0,
    min_silence_sec: float = 2.0,
) -> tuple[float, float]:
    """wav の冒頭/末尾の長い無音区間を検出し、(first_speech_sec, last_speech_sec) を返す。

    衆議院 webtv は開議の十数分〜数十分前から HLS が流れるため、wav の冒頭が長い
    無音区間になることが多い。Whisper は無音区間に対して「ご視聴ありがとうございました」
    系の YouTube 系幻覚を高頻度で吐く (``llm_correct.drop_hallucinations`` で除去
    されるが、ASR の処理時間自体は消費される)。

    本関数は ffmpeg の ``silencedetect`` フィルタで無音を検出するだけで、wav 自体は
    改変しない。返り値の (start, end) を ``--clip-timestamps`` 系で Whisper に
    渡すことで、ASR の処理対象を発話区間だけに絞る。

    Args:
        wav_path: 検査対象の 16kHz mono PCM wav。
        threshold_db: 無音判定の音量閾値 (dBFS)。-45 dB は会議音声の暗騒音に
            十分まで余裕がある (議場マイクは無発話時でも -55 〜 -50 dB 程度)。
        min_silence_sec: この長さ以上の無音だけを検出対象とする (短い言間は無視)。

    Returns:
        ``(first_speech_sec, last_speech_sec)``。

        - 発話を検出できない場合: ``(0.0, duration)`` (=trim 無し相当)
        - 冒頭に長い無音が無い場合: ``first_speech_sec = 0.0``
        - 末尾に長い無音が無い場合: ``last_speech_sec = duration``
    """
    ensure_ffmpeg()

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-i", str(wav_path),
        "-af", f"silencedetect=noise={threshold_db}dB:duration={min_silence_sec}",
        "-f", "null", "-",
    ]
    # silencedetect の出力は stderr に出る。-loglevel info 以上が必要 (既定で OK)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stderr = proc.stderr

    # 全体 duration を取得 (HH:MM:SS.ms)
    duration = 0.0
    m = _DURATION_RE.search(stderr)
    if m:
        duration = (
            int(m.group(1)) * 3600
            + int(m.group(2)) * 60
            + float(m.group(3))
        )

    starts = [float(x) for x in _SILENCE_START_RE.findall(stderr)]
    ends = [float(x) for x in _SILENCE_END_RE.findall(stderr)]

    # 冒頭の無音: silence_start が 0 付近 (0 以下も含む) で開始する pair の
    # silence_end[0] が最初の発話開始。閾値 0.1s 未満なら「冒頭無音」と判定。
    first_speech = 0.0
    if starts and ends and starts[0] < 0.1:
        first_speech = ends[0]

    # 末尾の無音: silence_start が duration 近くで silence_end が無いケース
    # (ffmpeg は末尾無音の silence_end を出さない)。starts > ends ならそれ。
    last_speech = duration
    if duration > 0 and len(starts) > len(ends):
        last_speech = starts[-1]

    # 異常入力時の安全側フォールバック (発話区間が逆転している等)
    if last_speech <= first_speech:
        return (0.0, duration)
    return (first_speech, last_speech)
