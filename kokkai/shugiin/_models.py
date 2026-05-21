"""HuggingFace モデルの cache 確認 + 初回 DL 前の Y/N プロンプト。

mlx-whisper / faster-whisper / mlx-lm はいずれも ``~/.cache/huggingface/hub``
にモデルを置く。初回起動で数 GB が黙って降ってくる UX を避けるため、
対話 TTY のときだけ事前に Y/N で確認する。

非対話環境 (CI、agent からの subprocess、パイプ越し起動) では従来どおり通過し、
上位の自動 DL に任せる。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def hf_cache_dir() -> Path:
    """HF Hub の hub cache ディレクトリ。

    ``HUGGINGFACE_HUB_CACHE`` を直接指す環境変数を最優先。
    ``HF_HOME`` は HF の規約どおり ``<HF_HOME>/hub`` を返す。
    どちらも未設定なら ``~/.cache/huggingface/hub``。
    """
    if v := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(v)
    if v := os.environ.get("HF_HOME"):
        return Path(v) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_model_cache_path(repo_id: str) -> Path:
    """``org/name`` → ``<hf_cache>/models--org--name``。"""
    safe = "models--" + repo_id.replace("/", "--")
    return hf_cache_dir() / safe


def is_model_cached(repo_id: str) -> bool:
    """snapshots/ に実ファイルが 1 つでもあれば cached とみなす。

    空ディレクトリだけ残るケース (DL を途中で中断した残骸) を弾くため
    `any(sub.iterdir())` でファイルの存在まで確認する。
    """
    snap = hf_model_cache_path(repo_id) / "snapshots"
    if not snap.is_dir():
        return False
    for sub in snap.iterdir():
        if sub.is_dir() and any(sub.iterdir()):
            return True
    return False


def _is_interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def ensure_model_downloaded(
    repo_id: str,
    *,
    label: str,
    size_hint: str | None = None,
) -> None:
    """未 DL なら対話 TTY で Y/N を聞く。

    - 既に cached: 即 return
    - 非対話 (TTY ではない): 何もせず return (上位の自動 DL に委ねる)
    - 対話: stderr に確認を出して input()。空 Enter / "y" → 続行、それ以外 → 中断
    """
    from ..errors import MissingToolError

    if is_model_cached(repo_id):
        return
    if not _is_interactive():
        return

    size = f" (約 {size_hint})" if size_hint else ""
    print(
        f"\n[{label}] モデル '{repo_id}' が未ダウンロードです{size}。\n"
        f"  HuggingFace から {hf_cache_dir()} にダウンロードします。",
        file=sys.stderr,
    )
    print("  続行しますか? [Y/n]: ", end="", file=sys.stderr, flush=True)
    try:
        ans = input().strip().lower()
    except EOFError:
        ans = ""
    if ans and not ans.startswith("y"):
        raise MissingToolError(
            f"モデル '{repo_id}' のダウンロードを拒否したため中断しました。"
        )
