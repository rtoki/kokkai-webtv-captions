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
